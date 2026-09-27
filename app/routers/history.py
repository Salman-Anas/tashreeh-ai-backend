from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from app.db import db_configured, get_db
from app.errors import AppError
from app.models import HistoryItem, HistoryResponse

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def history(
    session_id: str = Query(min_length=1, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
) -> HistoryResponse:
    if not db_configured():
        raise AppError(503, "history_unavailable", "History needs Supabase; it is not configured on the server.")

    def fetch() -> list[dict]:  # type: ignore[type-arg]
        res = (
            get_db()
            .table("translations")
            .select("id,direction,source_text,output_text,terms_required,terms_missing,model,latency_ms,created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(res.data or [])

    rows = await asyncio.to_thread(fetch)
    return HistoryResponse(items=[HistoryItem.model_validate(r) for r in rows])
