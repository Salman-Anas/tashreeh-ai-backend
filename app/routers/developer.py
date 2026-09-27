"""Dashboard endpoints for the Developer API page: keys, usage, pricing.

Keys belong to the browser session that created them (same model as History).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from app.config import get_settings
from app.errors import AppError
from app.models import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyInfo,
    ApiKeyListResponse,
    ApiKeyStats,
    PricingResponse,
    UsageRecord,
    UsageResponse,
)
from app.services.api_keys import empty_stats, forget_key, get_key_store
from app.services.pricing import price_sheet

router = APIRouter(prefix="/api/developer", tags=["developer"])

SessionQ = Query(min_length=8, max_length=64)


def _with_stats(rec: dict[str, Any], stats: dict[str, float] | None) -> dict[str, Any]:
    return {**rec, "stats": ApiKeyStats(**{k: v for k, v in (stats or empty_stats()).items()})}


@router.get("/pricing", response_model=PricingResponse)
async def pricing() -> PricingResponse:
    s = get_settings()
    sheet = price_sheet(s)
    gemini = {"input": sheet.input_per_m, "output": sheet.output_per_m, "embedding": sheet.embed_per_m}
    return PricingResponse(
        model=sheet.model,
        embed_model=sheet.embed_model,
        markup=sheet.markup,
        gemini=gemini,
        client={k: sheet.charge(v) for k, v in gemini.items()},
        gemini_keys=len(s.gemini_keys),
        note="Gemini paid-tier list prices per 1M tokens. Output includes thinking tokens; embedding tokens are estimated.",
    )


@router.post("/keys", response_model=ApiKeyCreated, status_code=201)
async def create_key(body: ApiKeyCreate) -> ApiKeyCreated:
    store = get_key_store()
    rec, key = await asyncio.to_thread(store.create, body.name.strip(), body.session_id)
    return ApiKeyCreated(**_with_stats(rec, None), key=key)


@router.get("/keys", response_model=ApiKeyListResponse)
async def list_keys(session_id: str = SessionQ) -> ApiKeyListResponse:
    store = get_key_store()
    recs = await asyncio.to_thread(store.list, session_id)
    stats = await asyncio.to_thread(store.stats, [r["id"] for r in recs])
    return ApiKeyListResponse(
        items=[ApiKeyInfo(**_with_stats(r, stats.get(r["id"]))) for r in recs], persistent=store.persistent
    )


@router.delete("/keys/{key_id}", status_code=204)
async def revoke_key(key_id: str, session_id: str = SessionQ) -> None:
    if not await asyncio.to_thread(get_key_store().revoke, key_id, session_id):
        raise AppError(404, "not_found", "API key not found or already revoked.")
    forget_key(key_id)


@router.get("/usage", response_model=UsageResponse)
async def usage(
    session_id: str = SessionQ,
    key_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> UsageResponse:
    store = get_key_store()
    recs = await asyncio.to_thread(store.list, session_id)
    ids = [r["id"] for r in recs if not key_id or r["id"] == key_id]
    stats = await asyncio.to_thread(store.stats, ids)
    summary = empty_stats()
    for st in stats.values():
        for k, v in st.items():
            summary[k] += v
    rows = await asyncio.to_thread(store.usage, session_id, key_id, limit)
    return UsageResponse(
        summary=ApiKeyStats(**{k: round(v, 6) for k, v in summary.items()}),
        items=[UsageRecord.model_validate(r) for r in rows],
    )
