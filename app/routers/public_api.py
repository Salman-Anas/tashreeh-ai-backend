"""Public, API-key authenticated endpoints (``/v1``) for other apps.

Authenticate with ``Authorization: Bearer tsh_live_...`` (or ``X-API-Key``).
Every call is metered: tokens, Gemini cost, and the price charged to the key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, Header

from app.config import get_settings
from app.deps import RateLimiter, get_explainer, get_pipeline, glossary
from app.errors import AppError
from app.models import (
    ApiKeyStats,
    GlossaryTerm,
    PublicExplainRequest,
    PublicExplainResponse,
    PublicTerm,
    PublicTranslateRequest,
    PublicTranslateResponse,
    PublicUsage,
    PublicUsageSummary,
)
from app.routers.translate import check_direction
from app.services.api_keys import empty_stats, get_key_store, looks_like_key, lookup_key
from app.services.gemini_client import GeminiError, GeminiRateLimitError, GeminiTimeoutError, get_gemini
from app.services.pricing import compute_cost, price_sheet
from app.services.usage import UsageMeter, track_usage

log = logging.getLogger("tashreeh.public_api")
router = APIRouter(prefix="/v1", tags=["Public API v1"])

key_rate_limiter = RateLimiter(get_settings().api_rate_limit_per_min)


async def require_api_key(
    authorization: str | None = Header(default=None, description="Bearer tsh_live_..."),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    key = (x_api_key or "").strip()
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key:
        raise AppError(401, "missing_api_key", "Missing API key. Send 'Authorization: Bearer <your key>'.")
    if not looks_like_key(key):
        raise AppError(401, "invalid_api_key", "Invalid API key.")
    rec = await asyncio.to_thread(lookup_key, key)
    if rec is None:
        raise AppError(401, "invalid_api_key", "Invalid or revoked API key.")
    wait = key_rate_limiter.check(f"key:{rec['id']}")
    if wait is not None:
        raise AppError(429, "rate_limited", f"Rate limit is {key_rate_limiter.limit} requests/minute per key. Retry in {int(wait) + 1}s.")
    return rec


def _record(row: dict[str, Any], key_id: str) -> None:
    store = get_key_store()
    try:
        store.record_usage(row)
        store.touch(key_id)
    except Exception as exc:  # noqa: BLE001 — metering must never break the response
        log.error("could not record API usage: %s", exc)


def _require_gemini() -> None:
    if not get_gemini().configured:
        raise AppError(503, "gemini_not_configured", "The translation model is not configured on the server.")


@dataclass
class Metered:
    """What a metered call used; filled in when the ``metered`` block exits."""

    meter: UsageMeter = field(default_factory=UsageMeter)
    gemini_cost_usd: float = 0.0
    price_usd: float = 0.0
    latency_ms: int = 0

    def usage(self) -> PublicUsage:
        m = self.meter
        return PublicUsage(
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            embedding_tokens=m.embedding_tokens,
            total_tokens=m.total_tokens,
            gemini_cost_usd=self.gemini_cost_usd,
            price_usd=self.price_usd,
        )


@asynccontextmanager
async def metered(key: dict[str, Any], endpoint: str, direction: str | None, input_chars: int) -> AsyncIterator[Metered]:
    """Meter every Gemini call inside the block and log one usage row for it.

    Failed calls are recorded with their status code but billed at zero, even
    though Gemini may have charged us for part of the work.
    """
    s = get_settings()
    sheet = price_sheet(s)
    started = time.perf_counter()
    status, error_code = 200, None
    result = Metered()
    with track_usage() as meter:
        result.meter = meter
        try:
            yield result
        except AppError as exc:
            status, error_code = exc.status, exc.code
            raise
        except GeminiRateLimitError:
            status, error_code = 429, "gemini_rate_limited"
            raise
        except GeminiTimeoutError:
            status, error_code = 504, "gemini_timeout"
            raise
        except GeminiError:
            status, error_code = 502, "gemini_error"
            raise
        finally:
            cost = compute_cost(meter, sheet)
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            result.gemini_cost_usd = cost.gemini_cost_usd
            result.price_usd = cost.price_usd if status == 200 else 0.0
            row = {
                "api_key_id": key["id"],
                "endpoint": endpoint,
                "direction": direction,
                "input_chars": input_chars,
                "input_tokens": meter.input_tokens,
                "output_tokens": meter.output_tokens,
                "embedding_tokens": meter.embedding_tokens,
                "total_tokens": meter.total_tokens,
                "gemini_calls": meter.gemini_calls,
                "gemini_cost_usd": result.gemini_cost_usd,
                "billed_usd": result.price_usd,
                "model": s.gemini_model,
                "status_code": status,
                "error_code": error_code,
                "latency_ms": result.latency_ms,
            }
            await asyncio.to_thread(_record, row, key["id"])
            log.info(
                json.dumps(
                    {
                        "event": endpoint,
                        "key": key["key_prefix"],
                        **{k: row[k] for k in ("status_code", "total_tokens", "gemini_cost_usd", "billed_usd", "latency_ms")},
                    }
                )
            )


@router.get("/health")
async def v1_health() -> dict[str, object]:
    return {"status": "ok", "version": "v1", "model": get_settings().gemini_model}


@router.post("/translate", response_model=PublicTranslateResponse)
async def v1_translate(body: PublicTranslateRequest, key: dict[str, Any] = Depends(require_api_key)) -> PublicTranslateResponse:
    s = get_settings()
    text = body.text.strip()
    explanations = None

    async with metered(key, "/v1/translate", body.direction, len(text)) as m:
        if not text:
            raise AppError(422, "empty_input", "'text' is empty.")
        if len(text) > s.max_input_chars:
            raise AppError(413, "input_too_long", f"'text' is {len(text):,} characters; the limit is {s.max_input_chars:,}.")
        check_direction(text, body.direction)
        _require_gemini()
        result = await get_pipeline().translate(text, body.direction)
        if body.explain and result.terms_required:
            terms = glossary.by_ids([t.term_id for t in result.terms_required])
            explanations, _ = await get_explainer().explain(terms)

    missing_ids = {t.term_id for t in result.terms_missing}
    return PublicTranslateResponse(
        id=str(uuid.uuid4()),
        direction=body.direction,
        translation=result.translation,
        terms=[
            PublicTerm(source=t.source, target=t.target, category=t.category, found=t.term_id not in missing_ids)
            for t in result.terms_required
        ],
        terms_missing=len(missing_ids),
        notes=result.notes,
        model=s.gemini_model,
        latency_ms=m.latency_ms,
        usage=m.usage(),
        explanations=explanations if body.explain else None,
    )


@router.post("/explain", response_model=PublicExplainResponse)
async def v1_explain(body: PublicExplainRequest, key: dict[str, Any] = Depends(require_api_key)) -> PublicExplainResponse:
    """Explain glossary terms (given in English or Urdu) in simple English and Urdu, with an example."""
    found: list[GlossaryTerm] = []
    not_found: list[str] = []
    for text in body.terms:
        term = glossary.find_by_text(text)
        if term is None:
            not_found.append(text)
        elif term.id not in {t.id for t in found}:
            found.append(term)

    async with metered(key, "/v1/explain", None, sum(len(t) for t in body.terms)) as m:
        if not found:
            raise AppError(404, "terms_not_found", "None of these terms are in the Tashreeh AI legal glossary.")
        _require_gemini()
        explanations, missing = await get_explainer().explain(found)

    by_id = {t.id: t for t in found}
    not_found += [by_id[i].term_en for i in missing]
    return PublicExplainResponse(
        id=str(uuid.uuid4()),
        explanations=explanations,
        not_found=not_found,
        model=get_settings().gemini_model,
        latency_ms=m.latency_ms,
        usage=m.usage(),
    )


@router.get("/usage", response_model=PublicUsageSummary)
async def v1_usage(key: dict[str, Any] = Depends(require_api_key)) -> PublicUsageSummary:
    stats = (await asyncio.to_thread(get_key_store().stats, [key["id"]])).get(key["id"], empty_stats())
    return PublicUsageSummary(key_name=key["name"], key_prefix=key["key_prefix"], usage=ApiKeyStats(**stats))
