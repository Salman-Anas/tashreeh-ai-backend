"""Translation-memory retrieval (pgvector via the ``match_translation_memory`` RPC)."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from app.config import get_settings
from app.db import DatabaseUnavailable, db_configured, get_db
from app.models import Direction, TMExample
from app.services.gemini_client import GeminiClient, GeminiError
from app.services.normalize import normalize_for_matching

log = logging.getLogger(__name__)


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


class NullTM:
    async def retrieve(self, text: str, direction: Direction) -> list[TMExample]:
        return []
