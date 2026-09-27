from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request

from app.deps import enforce_rate_limit, get_explainer, glossary
from app.errors import AppError
from app.models import ExplainRequest, ExplainResponse
from app.services.gemini_client import get_gemini

log = logging.getLogger("tashreeh.explain")
router = APIRouter(prefix="/api/explain", tags=["explain"])


@router.post("", response_model=ExplainResponse)
async def explain_terms(body: ExplainRequest, request: Request) -> ExplainResponse:
    """Explain glossary terms (by id) in simple English and Urdu, with an example."""
    terms = glossary.by_ids(body.term_ids)
    if not terms:
        raise AppError(404, "terms_not_found", "None of these terms are in the glossary.")
    enforce_rate_limit(request, body.session_id)
    if not get_gemini().configured:
        raise AppError(503, "gemini_not_configured", "The explanation model is not configured on the server.")

    started = time.perf_counter()
    explanations, missing = await get_explainer().explain(terms)
    unknown = [i for i in body.term_ids if i not in {t.id for t in terms}]
    log.info(
        json.dumps(
            {
                "event": "explain",
                "terms": len(terms),
                "cached": sum(e.cached for e in explanations),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
    )
    return ExplainResponse(explanations=explanations, missing=missing + unknown)
