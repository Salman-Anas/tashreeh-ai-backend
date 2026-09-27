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
from typing import Any

from fastapi import APIRouter, Depends, Header

from app.config import get_settings
from app.deps import RateLimiter, get_pipeline
from app.errors import AppError
from app.models import (
    ApiKeyStats,
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
from app.services.usage import track_usage

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


@router.get("/health")
async def v1_health() -> dict[str, object]:
    return {"status": "ok", "version": "v1", "model": get_settings().gemini_model}


@router.post("/translate", response_model=PublicTranslateResponse)
async def v1_translate(body: PublicTranslateRequest, key: dict[str, Any] = Depends(require_api_key)) -> PublicTranslateResponse:
    s = get_settings()
    sheet = price_sheet(s)
    started = time.perf_counter()
    text = body.text.strip()
    status, error_code = 200, None

    with track_usage() as meter:
        try:
            if not text:
                raise AppError(422, "empty_input", "'text' is empty.")
            if len(text) > s.max_input_chars:
                raise AppError(413, "input_too_long", f"'text' is {len(text):,} characters; the limit is {s.max_input_chars:,}.")
            check_direction(text, body.direction)
            if not get_gemini().configured:
                raise AppError(503, "gemini_not_configured", "The translation model is not configured on the server.")
            result = await get_pipeline().translate(text, body.direction)
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
            latency_ms = int((time.perf_counter() - started) * 1000)
            cost = compute_cost(meter, sheet)
            # Failed calls are not charged to the client, but Gemini's cost is still recorded.
            billed = cost.price_usd if status == 200 else 0.0
            row = {
                "api_key_id": key["id"],
                "endpoint": "/v1/translate",
                "direction": body.direction,
                "input_chars": len(text),
                "input_tokens": meter.input_tokens,
                "output_tokens": meter.output_tokens,
                "embedding_tokens": meter.embedding_tokens,
                "total_tokens": meter.total_tokens,
                "gemini_calls": meter.gemini_calls,
                "gemini_cost_usd": cost.gemini_cost_usd,
                "billed_usd": billed,
                "model": s.gemini_model,
                "status_code": status,
                "error_code": error_code,
                "latency_ms": latency_ms,
            }
            await asyncio.to_thread(_record, row, key["id"])
            log.info(json.dumps({"event": "v1_translate", "key": key["key_prefix"], **{k: row[k] for k in ("status_code", "total_tokens", "gemini_cost_usd", "billed_usd", "latency_ms")}}))

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
        latency_ms=latency_ms,
        usage=PublicUsage(
            input_tokens=meter.input_tokens,
            output_tokens=meter.output_tokens,
            embedding_tokens=meter.embedding_tokens,
            total_tokens=meter.total_tokens,
            gemini_cost_usd=cost.gemini_cost_usd,
            price_usd=billed,
        ),
    )


@router.get("/usage", response_model=PublicUsageSummary)
async def v1_usage(key: dict[str, Any] = Depends(require_api_key)) -> PublicUsageSummary:
    stats = (await asyncio.to_thread(get_key_store().stats, [key["id"]])).get(key["id"], empty_stats())
    return PublicUsageSummary(key_name=key["name"], key_prefix=key["key_prefix"], usage=ApiKeyStats(**stats))
