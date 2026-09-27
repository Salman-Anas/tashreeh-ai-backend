"""Shared singletons: glossary matcher, pipeline, rate limiter."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from app.config import BACKEND_DIR, get_settings
from app.db import db_configured, get_db
from app.errors import AppError
from app.models import GlossaryTerm
from app.services.explain import TermExplainer
from app.services.gemini_client import get_gemini
from app.services.glossary import GlossaryMatcher, load_terms_from_csv
from app.services.pipeline import TranslationPipeline
from app.services.tm import SupabaseTM

log = logging.getLogger(__name__)

GLOSSARY_CSV = BACKEND_DIR / "data" / "glossary.csv"

glossary = GlossaryMatcher()
_pipeline: TranslationPipeline | None = None
_explainer: TermExplainer | None = None


def _fetch_glossary_from_db() -> list[GlossaryTerm]:
    db = get_db()
    rows: list[dict] = []  # type: ignore[type-arg]
    page, size = 0, 1000
    while True:
        res = (
            db.table("glossary_terms")
            .select("id,term_en,term_ur,category,notes,source")
            .order("id")
            .range(page * size, page * size + size - 1)
            .execute()
        )
        batch = list(res.data or [])
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
    return [GlossaryTerm.model_validate(r) for r in rows]


def load_glossary() -> None:
    """Load the glossary from Supabase, falling back to the bundled CSV."""
    if db_configured():
        try:
            terms = _fetch_glossary_from_db()
            glossary.set_terms(terms)
            glossary.source = "supabase"
            log.info("glossary loaded from supabase: %d terms", len(terms))
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("glossary: supabase load failed (%s); falling back to CSV", exc)
    terms = load_terms_from_csv(GLOSSARY_CSV) if GLOSSARY_CSV.exists() else []
    glossary.set_terms(terms)
    glossary.source = "csv"
    log.info("glossary loaded from csv: %d terms", len(terms))


def get_pipeline() -> TranslationPipeline:
    global _pipeline
    if _pipeline is None:
        gemini = get_gemini()
        _pipeline = TranslationPipeline(
            llm=gemini,
            glossary=glossary,
            tm=SupabaseTM(gemini),
            concurrency=get_settings().translate_concurrency,
        )
    return _pipeline


def get_explainer() -> TermExplainer:
    global _explainer
    if _explainer is None:
        _explainer = TermExplainer(get_gemini())
    return _explainer


class RateLimiter:
    """Sliding-window, in-memory, per-key limiter (single process only)."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit; return seconds to wait if the key is over the limit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return self.window_s - (now - q[0])
            q.append(now)
            return None


rate_limiter = RateLimiter(get_settings().rate_limit_per_min)


def enforce_rate_limit(request: Request, session_id: str | None) -> None:
    client = request.client.host if request.client else "unknown"
    key = f"s:{session_id}" if session_id else f"ip:{client}"
    wait = rate_limiter.check(key)
    if wait is not None:
        raise AppError(
            429,
            "rate_limited",
            f"Too many requests. Limit is {rate_limiter.limit} per minute; try again in {int(wait) + 1}s.",
        )
