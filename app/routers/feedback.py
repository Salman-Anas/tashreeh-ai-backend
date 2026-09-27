from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter

from app.db import db_configured, get_db
from app.errors import AppError
from app.models import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    if body.rating is None and not (body.corrected_text or "").strip() and not (body.comment or "").strip():
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
        "corrected_text": (body.corrected_text or "").strip() or None,
        "comment": (body.comment or "").strip() or None,
        # Corrections are reviewed by a human before they enter the TM.
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
        raise
    return FeedbackResponse(id=new_id)
