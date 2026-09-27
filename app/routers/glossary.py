from __future__ import annotations

import asyncio
from collections import Counter

from fastapi import APIRouter, Query

from app.db import db_configured, get_db
from app.deps import glossary, load_glossary
from app.errors import AppError
from app.models import GlossaryListResponse, GlossaryTerm, GlossaryTermCreate
from app.services.normalize import normalize_for_matching, normalize_urdu

router = APIRouter(prefix="/api/glossary", tags=["glossary"])


def _require_db() -> None:
    if not db_configured():
        raise AppError(
            503,
            "read_only",
            "The glossary is read-only because Supabase is not configured. Edit data/glossary.csv instead.",
        )


@router.get("", response_model=GlossaryListResponse)
async def list_terms(
    q: str | None = Query(default=None, max_length=200),
    category: str | None = Query(default=None, max_length=50),
) -> GlossaryListResponse:
    terms = glossary.terms
    counts = Counter((t.category or "uncategorised") for t in terms)
    items = terms
    if category:
        items = [t for t in items if (t.category or "uncategorised") == category]
    if q and q.strip():
        needle_en = q.strip().lower()
        needle_ur = normalize_for_matching(q)
        items = [
            t
            for t in items
            if needle_en in t.term_en.lower()
            or (needle_ur and needle_ur in normalize_for_matching(t.term_ur))
            or (t.notes and needle_en in t.notes.lower())
        ]
    items = sorted(items, key=lambda t: t.term_en.lower())
    return GlossaryListResponse(
        items=items, total=len(terms), category_counts=dict(counts), read_only=not db_configured()
    )


@router.post("", response_model=GlossaryTerm, status_code=201)
async def add_term(body: GlossaryTermCreate) -> GlossaryTerm:
    _require_db()
    row = {
        "term_en": " ".join(body.term_en.split()),
        "term_ur": normalize_urdu(body.term_ur, preserve_newlines=False),
        "category": (body.category or "").strip().lower() or None,
        "notes": (body.notes or "").strip() or None,
        "source": (body.source or "").strip() or "user",
    }

    def insert() -> dict:  # type: ignore[type-arg]
        res = get_db().table("glossary_terms").insert(row).execute()
        return res.data[0]

    try:
        created = await asyncio.to_thread(insert)
    except Exception as exc:  # noqa: BLE001
        if "duplicate" in str(exc).lower() or "23505" in str(exc):
            raise AppError(409, "duplicate_term", "This English–Urdu term pair already exists.") from exc
        raise
    await asyncio.to_thread(load_glossary)
    return GlossaryTerm.model_validate(created)


@router.delete("/{term_id}", status_code=204)
async def delete_term(term_id: int) -> None:
    _require_db()

    def delete() -> int:
        res = get_db().table("glossary_terms").delete().eq("id", term_id).execute()
        return len(res.data or [])

    if await asyncio.to_thread(delete) == 0:
        raise AppError(404, "not_found", "Glossary term not found.")
    await asyncio.to_thread(load_glossary)


@router.post("/reload")
async def reload_glossary() -> dict[str, object]:
    await asyncio.to_thread(load_glossary)
    return {"ok": True, "terms": len(glossary), "source": glossary.source}
