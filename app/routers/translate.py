from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.config import get_settings
from app.db import db_configured, get_db
from app.deps import enforce_rate_limit, get_pipeline
from app.errors import AppError
from app.models import (
    Direction,
    DocumentSegmentResult,
    DocumentTranslateResponse,
    ExportRequest,
    TranslateRequest,
    TranslateResponse,
)
from app.services import documents
from app.services.gemini_client import GeminiError, get_gemini
from app.services.normalize import urdu_ratio

log = logging.getLogger("tashreeh.translate")
router = APIRouter(prefix="/api/translate", tags=["translate"])

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _require_gemini() -> None:
    if not get_gemini().configured:
        raise AppError(503, "gemini_not_configured", "The translation model is not configured on the server.")


def check_direction(text: str, direction: Direction) -> None:
    ratio = urdu_ratio(text)
    if direction == "en-ur" and ratio > 0.6:
        raise AppError(422, "direction_mismatch", "This text looks like Urdu. Switch the direction to Urdu → English.")
    if direction == "ur-en" and ratio < 0.2:
        raise AppError(422, "direction_mismatch", "This text looks like English. Switch the direction to English → Urdu.")


def _save_translation(row: dict) -> str | None:  # type: ignore[type-arg]
    if not db_configured():
        return None
    try:
        res = get_db().table("translations").insert(row).execute()
        return str(res.data[0]["id"]) if res.data else None
    except Exception as exc:  # noqa: BLE001
        log.warning("could not save translation history: %s", exc)
        return None


@router.post("", response_model=TranslateResponse)
async def translate(body: TranslateRequest, request: Request) -> TranslateResponse:
    s = get_settings()
    text = body.text.strip()
    if not text:
        raise AppError(422, "empty_input", "Please enter some text to translate.")
    if len(text) > s.max_input_chars:
        raise AppError(
            413, "input_too_long", f"Text is {len(text):,} characters; the limit is {s.max_input_chars:,}."
        )
    check_direction(text, body.direction)
    enforce_rate_limit(request, body.session_id)
    _require_gemini()

    result = await get_pipeline().translate(text, body.direction)

    row = {
        "session_id": body.session_id,
        "direction": body.direction,
        "source_text": text,
        "output_text": result.translation,
        "terms_required": [t.model_dump() for t in result.terms_required],
        "terms_missing": [t.model_dump() for t in result.terms_missing],
        "model": s.gemini_model,
        "latency_ms": result.latency_ms,
    }
    saved_id = await asyncio.to_thread(_save_translation, row)

    log.info(
        json.dumps(
            {
                "event": "translate",
                "direction": body.direction,
                "chars": len(text),
                "paragraphs": len(result.segments),
                "terms_required": len(result.terms_required),
                "terms_missing": len(result.terms_missing),
                "tm_examples": len(result.tm_examples_used),
                "retried": result.retried,
                "latency_ms": result.latency_ms,
                "saved": saved_id is not None,
            }
        )
    )

    return TranslateResponse(
        id=saved_id or str(uuid.uuid4()),
        direction=body.direction,
        source_text=text,
        translation=result.translation,
        terms_required=result.terms_required,
        terms_found=result.terms_found,
        terms_missing=result.terms_missing,
        glossary_matches=result.glossary_matches,
        output_matches=result.output_matches,
        tm_examples_used=result.tm_examples_used,
        notes=result.notes,
        model=s.gemini_model,
        latency_ms=result.latency_ms,
        retried=result.retried,
        saved=saved_id is not None,
    )


@router.post("/export")
async def export_docx(body: ExportRequest) -> Response:
    data = await asyncio.to_thread(documents.build_text_docx, body.translation, body.direction, body.source_text)
    return Response(
        content=data,
        media_type=DOCX_MIME,
        headers={"Content-Disposition": 'attachment; filename="tashreeh-translation.docx"'},
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
async def _translate_segments(
    extracted: documents.ExtractedDocument, direction: Direction
) -> AsyncIterator[DocumentSegmentResult]:
    """Yield segment results as they complete (unordered)."""
    pipeline = get_pipeline()
    sem = asyncio.Semaphore(get_settings().translate_concurrency)

    async def one(i: int) -> DocumentSegmentResult:
        seg = extracted.segments[i]
        async with sem:
            try:
                r = await pipeline.translate_segment(seg.text, direction)
                return DocumentSegmentResult(
                    index=i,
                    kind=seg.kind,
                    source=seg.text,
                    translation=r.translation,
                    terms_required=r.check.terms_required,
                    terms_missing=r.check.terms_missing,
                )
            except GeminiError as exc:
                return DocumentSegmentResult(
                    index=i, kind=seg.kind, source=seg.text, translation="", terms_required=[],
                    terms_missing=[], error=str(exc),
                )

    for fut in asyncio.as_completed([one(i) for i in range(len(extracted.segments))]):
        yield await fut


def _docx_name(filename: str, direction: Direction) -> str:
    stem = Path(filename).stem or "document"
    return f"{stem}.{'ur' if direction == 'en-ur' else 'en'}.docx"


@router.post("/document", response_model=DocumentTranslateResponse)
async def translate_document(
    request: Request,
    file: UploadFile = File(...),
    direction: Direction = Form(...),
    session_id: str | None = Form(default=None),
    stream: bool = Query(default=False, description="Stream NDJSON progress events"),
) -> Response:
    s = get_settings()
    filename = file.filename or "document"
    data = await file.read()
    if not data:
        raise AppError(422, "empty_file", "The uploaded file is empty.")
    if len(data) > s.max_upload_bytes:
        raise AppError(413, "file_too_large", f"Files up to {s.max_upload_bytes // (1024 * 1024)} MB are supported.")

    extracted = await asyncio.to_thread(documents.extract, filename, data)
    if extracted.total_chars > s.max_document_chars:
        raise AppError(
            413,
            "document_too_long",
            f"This document has {extracted.total_chars:,} characters; the limit is {s.max_document_chars:,}.",
        )
    enforce_rate_limit(request, session_id)
    _require_gemini()
    started = time.perf_counter()
    total = len(extracted.segments)

    async def finish(results: list[DocumentSegmentResult]) -> DocumentTranslateResponse:
        results.sort(key=lambda r: r.index)
        if all(r.error for r in results):
            raise GeminiError(results[0].error or "Translation failed.")
        # Untranslated segments fall back to the source text so the DOCX stays complete.
        texts = [r.translation or r.source for r in results]
        docx_bytes = await asyncio.to_thread(documents.build_docx, extracted, texts, direction)
        latency = int((time.perf_counter() - started) * 1000)
        log.info(json.dumps({"event": "translate_document", "segments": total, "latency_ms": latency}))
        return DocumentTranslateResponse(
            filename=filename,
            direction=direction,
            segments=results,
            docx_base64=base64.b64encode(docx_bytes).decode("ascii"),
            docx_filename=_docx_name(filename, direction),
            latency_ms=latency,
        )

    if not stream:
        results = [r async for r in _translate_segments(extracted, direction)]
        return Response(content=(await finish(results)).model_dump_json(), media_type="application/json")

    async def events() -> AsyncIterator[bytes]:
        def line(obj: dict) -> bytes:  # type: ignore[type-arg]
            return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        yield line(
            {
                "type": "start",
                "total": total,
                "segments": [{"index": i, "kind": seg.kind, "source": seg.text} for i, seg in enumerate(extracted.segments)],
            }
        )
        results: list[DocumentSegmentResult] = []
        async for r in _translate_segments(extracted, direction):
            results.append(r)
            yield line({"type": "segment", "done": len(results), "total": total, "segment": r.model_dump()})
        try:
            final = await finish(results)
            yield line({"type": "done", "result": final.model_dump()})
        except Exception as exc:  # noqa: BLE001
            log.exception("document finish failed")
            yield line({"type": "error", "message": str(exc) or "Could not build the translated document."})

    return StreamingResponse(events(), media_type="application/x-ndjson")
