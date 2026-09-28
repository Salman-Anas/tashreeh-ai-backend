"""Translation-memory retrieval (pgvector via the ``match_translation_memory`` RPC)."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from app.config import get_settings
from app.db import DatabaseUnavailable, db_configured, get_db
from app.models import Direction, TMExample
from app.services.gemini_client import GeminiClient, GeminiError
from app.services.normalize import normalize_for_matching, urdu_ratio

log = logging.getLogger(__name__)

MIN_RATIO, MAX_RATIO = 0.4, 2.5  # len(ur) / len(en), in characters
MAX_CHARS = 3000


def validate_pair(en: str, ur: str) -> str | None:
    """Return a rejection reason, or None if the pair looks usable as a TM example."""
    if not en or not ur:
        return "empty side"
    if len(en) > MAX_CHARS or len(ur) > MAX_CHARS:
        return "too long"
    ratio = len(ur) / len(en)
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        return "length ratio out of range"
    if urdu_ratio(ur) < 0.6:
        return "text_ur is not mostly Urdu"
    if urdu_ratio(en) > 0.2:
        return "text_en contains Urdu script"
    return None


class TMRetriever(Protocol):
    async def retrieve(self, text: str, direction: Direction) -> list[TMExample]: ...


def embedding_text(text: str, lang: str) -> str:
    """Text as it should be embedded (Urdu is matching-normalized)."""
    return normalize_for_matching(text) if lang == "ur" else " ".join(text.split())


class SupabaseTM:
    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def retrieve(self, text: str, direction: Direction) -> list[TMExample]:
        """Top-k similar TM pairs. Never raises: TM is an enhancement, not a hard dependency."""
        if not db_configured() or not self.gemini.configured:
            return []
        s = get_settings()
        src_lang = "en" if direction == "en-ur" else "ur"
        try:
            vector = await self.gemini.embed_query(embedding_text(text, src_lang))
        except GeminiError as exc:
            log.warning("tm: embedding failed, continuing without examples: %s", exc)
            return []

        def rpc() -> list[dict]:  # type: ignore[type-arg]
            res = (
                get_db()
                .rpc(
                    "match_translation_memory",
                    {
                        "query_embedding": vector,
                        "direction": direction,
                        "match_count": s.tm_match_count,
                        "min_similarity": s.tm_min_similarity,
                    },
                )
                .execute()
            )
            return list(res.data or [])

        try:
            rows = await asyncio.to_thread(rpc)
        except (DatabaseUnavailable, Exception) as exc:  # noqa: BLE001
            log.warning("tm: supabase rpc failed, continuing without examples: %s", exc)
            return []
        return [TMExample.model_validate(r) for r in rows]


async def add_pairs(gemini: GeminiClient, pairs: list[tuple[str, str]], source: str) -> int:
    """Embed both sides of ``(text_en, text_ur)`` pairs and upsert them into the TM.

    Pairs must already be validated and normalized. Returns the number stored.
    Raises ``GeminiError`` / database errors: the caller decides how to report them.
    """
    if not pairs:
        return 0
    en_vecs = await gemini.embed_documents([embedding_text(en, "en") for en, _ in pairs])
    ur_vecs = await gemini.embed_documents([embedding_text(ur, "ur") for _, ur in pairs])
    rows = [
        {"text_en": en, "text_ur": ur, "source": source, "embedding_en": ve, "embedding_ur": vu}
        for (en, ur), ve, vu in zip(pairs, en_vecs, ur_vecs, strict=True)
    ]

    def upsert() -> None:
        get_db().table("translation_memory").upsert(rows, on_conflict="text_en,text_ur").execute()

    await asyncio.to_thread(upsert)
    return len(rows)


class NullTM:
    async def retrieve(self, text: str, direction: Direction) -> list[TMExample]:
        return []
