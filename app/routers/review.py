"""Review queue: a human approves or rejects readers' corrections before the system learns from them."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query

from app.db import MIGRATION_HINT, db_configured, get_db, is_outdated_schema
from app.deps import glossary, load_glossary
from app.errors import AppError
from app.models import (
    GlossaryChange,
    ReviewApproveRequest,
    ReviewApproveResponse,
    ReviewItem,
    ReviewListResponse,
    ReviewRejectRequest,
    ReviewStatus,
    ReviewTranslation,
)
from app.services import review
from app.services.gemini_client import GeminiError, get_gemini
from app.services.tm import add_pairs

log = logging.getLogger("tashreeh.review")
router = APIRouter(prefix="/api/review", tags=["review"])

_COLUMNS = (
    "id,status,rating,comment,corrected_text,term_corrections,review_note,created_at,reviewed_at,"
    "translations(id,direction,source_text,output_text)"
)
def _require_db() -> None:
    if not db_configured():
        raise AppError(503, "review_unavailable", "Reviewing corrections needs Supabase; it is not configured on the server.")


async def _db(fn: Any) -> Any:
    try:
        return await asyncio.to_thread(fn)
    except Exception as exc:  # noqa: BLE001
        if is_outdated_schema(exc):
            raise AppError(503, "schema_outdated", MIGRATION_HINT) from exc
        raise


def _item(row: dict[str, Any]) -> ReviewItem:
    tr = row.get("translations")
    return ReviewItem.model_validate(
        {
            **row,
            "term_corrections": row.get("term_corrections") or [],
            "translation": ReviewTranslation.model_validate(tr) if tr else None,
        }
    )


async def _get(feedback_id: int) -> ReviewItem:
    def fetch() -> list[dict[str, Any]]:
        return list(get_db().table("feedback").select(_COLUMNS).eq("id", feedback_id).limit(1).execute().data or [])

    rows = await _db(fetch)
    if not rows or rows[0].get("status") is None:
        raise AppError(404, "not_found", "Correction not found.")
    item = _item(rows[0])
    if item.status != "pending":
        raise AppError(409, "already_reviewed", f"This correction was already {item.status}.")
    if item.translation is None:
        raise AppError(409, "translation_missing", "The translation this correction belongs to no longer exists.")
    return item


@router.get("", response_model=ReviewListResponse)
async def list_corrections(
    status: ReviewStatus = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=200),
) -> ReviewListResponse:
    _require_db()

    def fetch() -> tuple[list[dict[str, Any]], dict[str, int]]:
        db = get_db()
        rows = (
            db.table("feedback")
            .select(_COLUMNS)
            .eq("status", status)
            .order("created_at", desc=status == "pending")
            .limit(limit)
            .execute()
            .data
        )
        counts = {
            s: db.table("feedback").select("id", count="exact").eq("status", s).limit(1).execute().count or 0
            for s in ("pending", "approved", "rejected")
        }
        return list(rows or []), counts

    rows, counts = await _db(fetch)
    return ReviewListResponse(items=[_item(r) for r in rows], counts=counts)


@router.post("/{feedback_id}/approve", response_model=ReviewApproveResponse)
async def approve(feedback_id: int, body: ReviewApproveRequest) -> ReviewApproveResponse:
    """Accept a correction and learn from it: translation memory + glossary."""
    _require_db()
    item = await _get(feedback_id)
    tr = item.translation
    assert tr is not None
    corrections = body.term_corrections if body.term_corrections is not None else item.term_corrections
    corrected = (body.corrected_text if body.corrected_text is not None else item.corrected_text or "").strip()
    if not corrected and corrections:
        # Word fixes only: learn the machine sentence with the new words in place.
        corrected = review.apply_term_fixes(tr.output_text, corrections)
    skipped: list[str] = []

    # 1. Translation memory (Gemini embeddings) — first, so a failure changes nothing.
    pairs: list[tuple[str, str]] = []
    if corrected and corrected != tr.output_text.strip():
        pairs, why = review.tm_pairs(tr.source_text, corrected, tr.direction)
        skipped += why
    elif corrected:
        skipped.append("translation not added to memory: it is the same as the machine translation")
    gemini = get_gemini()
    if pairs and not gemini.configured:
        skipped.append("translation not added to memory: the embedding model is not configured")
        pairs = []
    try:
        tm_added = await add_pairs(gemini, pairs, review.TM_SOURCE)
    except GeminiError as exc:
        raise AppError(502, "embedding_failed", f"Could not add the correction to memory; nothing was changed. {exc}") from exc

    # 2. Glossary.
    plan = review.plan_glossary_changes(corrections, glossary, tr.direction)
    skipped += plan.skipped
    changes: list[GlossaryChange] = []

    def apply_glossary() -> None:
        db = get_db()
        for term_id, cols in plan.updates:
            for row in db.table("glossary_terms").update(cols).eq("id", term_id).execute().data or []:
                changes.append(GlossaryChange(action="updated", term_id=row["id"], **_names(row)))
        for new in plan.inserts:
            try:
                row = db.table("glossary_terms").insert(new).execute().data[0]
            except Exception as exc:  # noqa: BLE001
                if "23505" in str(exc) or "duplicate" in str(exc).lower():
                    skipped.append(f"“{new['term_en']}” is already in the glossary with this Urdu term")
                    continue
                raise
            changes.append(GlossaryChange(action="added", term_id=row["id"], **_names(row)))

    await _db(apply_glossary)
    if changes:
        await asyncio.to_thread(load_glossary)

    # 3. Close the item, recording exactly what was accepted.
    record = {
        "status": "approved",
        "approved_for_tm": tm_added > 0,
        "corrected_text": corrected or None,
        "term_corrections": [c.model_dump() for c in corrections] or None,
        "review_note": (body.note or "").strip() or None,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    await _db(lambda: get_db().table("feedback").update(record).eq("id", feedback_id).execute())
    log.info("review: approved %s (tm +%d, glossary %d changes, %d skipped)", feedback_id, tm_added, len(changes), len(skipped))
    return ReviewApproveResponse(id=feedback_id, tm_pairs_added=tm_added, glossary_changes=changes, skipped=skipped)


def _names(row: dict[str, Any]) -> dict[str, Any]:
    return {"term_en": row["term_en"], "term_ur": row["term_ur"], "term_ur_common": row.get("term_ur_common")}


@router.post("/{feedback_id}/reject", response_model=ReviewItem)
async def reject(feedback_id: int, body: ReviewRejectRequest) -> ReviewItem:
    _require_db()
    item = await _get(feedback_id)
    record = {
        "status": "rejected",
        "review_note": (body.note or "").strip() or None,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    await _db(lambda: get_db().table("feedback").update(record).eq("id", feedback_id).execute())
    return item.model_copy(update={"status": "rejected", "review_note": record["review_note"]})
