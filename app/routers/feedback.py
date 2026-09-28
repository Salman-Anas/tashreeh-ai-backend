from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter

from app.db import MIGRATION_HINT, db_configured, get_db, is_outdated_schema
from app.errors import AppError
from app.models import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    corrected = (body.corrected_text or "").strip() or None
    term_fixes = [c for c in body.term_corrections or [] if c.new_target.strip() and c.new_target.strip() != c.old_target.strip()]
    if body.rating is None and not corrected and not term_fixes and not (body.comment or "").strip():
        raise AppError(422, "empty_feedback", "Provide a rating, a correction, or a comment.")
    try:
        uuid.UUID(body.translation_id)
    except ValueError as exc:
        raise AppError(422, "invalid_translation_id", "translation_id must be a UUID.") from exc
    if not db_configured():
        raise AppError(503, "feedback_unavailable", "Feedback needs Supabase; it is not configured on the server.")

    row = {
        "translation_id": body.translation_id,
        "rating": body.rating,
        "corrected_text": corrected,
        "comment": (body.comment or "").strip() or None,
        "term_corrections": [c.model_dump() for c in term_fixes] or None,
        # Corrections wait for a human reviewer (see /api/review) before anything learns from them.
        "status": "pending" if corrected or term_fixes else None,
        "approved_for_tm": False,
    }

    def insert() -> int:
        res = get_db().table("feedback").insert(row).execute()
        return int(res.data[0]["id"])

    try:
        new_id = await asyncio.to_thread(insert)
    except Exception as exc:  # noqa: BLE001
        if "23503" in str(exc) or "foreign key" in str(exc).lower():
            raise AppError(
                404, "translation_not_found", "That translation was not saved, so feedback cannot be linked to it."
            ) from exc
        if is_outdated_schema(exc):
            raise AppError(503, "schema_outdated", MIGRATION_HINT) from exc
        raise
    return FeedbackResponse(id=new_id)
