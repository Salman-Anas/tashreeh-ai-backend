"""Gemini generation + embeddings via the ``google-genai`` SDK.

Several API keys can be configured (``GEMINI_API_KEY``, ``GEMINI_API_KEY_2``,
``GEMINI_API_KEYS``). Calls go to the current key; if it is rate-limited or
rejected, the call is retried on the next key and the failing key is put on a
cooldown so later requests skip it until it recovers.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.services.usage import record_embedding, record_generation

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "

RATE_LIMIT_COOLDOWN_S = 60.0
INVALID_KEY_COOLDOWN_S = 15 * 60.0


class GeminiError(Exception):
    """Generic Gemini failure (bad response, server error, misconfiguration)."""


class GeminiRateLimitError(GeminiError):
    pass


class GeminiTimeoutError(GeminiError):
    pass


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, genai_errors.APIError) and exc.code in (500, 502, 503, 504)


def _key_failure(exc: Exception) -> float | None:
    """Cooldown (seconds) if ``exc`` means *this key* is unusable right now, else None."""
    if not isinstance(exc, genai_errors.APIError):
        return None
    if exc.code == 429:
        return RATE_LIMIT_COOLDOWN_S
    if exc.code in (401, 403):
        return INVALID_KEY_COOLDOWN_S
    if exc.code == 400 and "api key" in (exc.message or "").lower():
        return INVALID_KEY_COOLDOWN_S
    return None


def _translate_error(exc: Exception) -> GeminiError:
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return GeminiRateLimitError("Gemini rate limit reached on all API keys. Please wait a moment and try again.")
        if exc.code in (401, 403) or (exc.code == 400 and "api key" in (exc.message or "").lower()):
            return GeminiError("Gemini rejected the API key(s). Check GEMINI_API_KEY / GEMINI_API_KEY_2 on the server.")
        if exc.code == 404:
            return GeminiError("Gemini model not found. Check GEMINI_MODEL / GEMINI_EMBED_MODEL.")
        return GeminiError(f"Gemini error ({exc.code}): {exc.message or 'unknown error'}")
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return GeminiTimeoutError("Gemini took too long to respond. Please try again.")
    return GeminiError(str(exc) or exc.__class__.__name__)


def _mask(key: str) -> str:
    return f"…{key[-4:]}" if len(key) > 4 else "…"


class GeminiClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.keys: list[str] = self.settings.gemini_keys
        self._clients: dict[int, genai.Client] = {}
        self._cooldown_until: list[float] = [0.0] * len(self.keys)
        self._active = 0

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    def _client(self, i: int) -> genai.Client:
        if i not in self._clients:
            self._clients[i] = genai.Client(api_key=self.keys[i])
        return self._clients[i]

    def _key_order(self) -> list[int]:
        """Keys to try: healthy keys first (starting from the active one), then cooling-down keys."""
        n = len(self.keys)
        order = [(self._active + k) % n for k in range(n)]
        now = time.monotonic()
        healthy = [i for i in order if self._cooldown_until[i] <= now]
        cooling = sorted((i for i in order if self._cooldown_until[i] > now), key=lambda i: self._cooldown_until[i])
        return healthy + cooling

    def key_status(self) -> list[dict[str, object]]:
        now = time.monotonic()
        return [
            {
                "key": _mask(k),
                "active": i == self._active,
                "cooldown_s": max(0, round(self._cooldown_until[i] - now)),
            }
            for i, k in enumerate(self.keys)
        ]

    async def _call(
        self,
        fn: Callable[[genai.Client], Awaitable[R]],
        *,
        transient_attempts: int = 2,
        base_delay: float = 1.0,
        rate_limit_waits: int = 0,
    ) -> R:
        """Run ``fn`` on the best available key, failing over to the next key when one
        is rate-limited or rejected. With ``rate_limit_waits`` > 0 (batch jobs), wait
        for a key to come off cooldown instead of giving up when all are rate-limited."""
        if not self.configured:
            raise GeminiError("GEMINI_API_KEY is not set on the server.")
        for round_ in range(rate_limit_waits + 1):
            try:
                return await self._call_once(fn, transient_attempts, base_delay)
            except GeminiRateLimitError:
                if round_ == rate_limit_waits:
                    raise
                wait = max(1.0, min(self._cooldown_until) - time.monotonic())
                log.warning("gemini: all keys rate-limited; waiting %.0fs", wait)
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def _call_once(
        self, fn: Callable[[genai.Client], Awaitable[R]], transient_attempts: int, base_delay: float
    ) -> R:
        last: Exception | None = None
        for idx in self._key_order():
            client = self._client(idx)
            for attempt in range(transient_attempts):
                try:
                    result = await asyncio.wait_for(fn(client), timeout=self.settings.gemini_timeout_s)
                    if idx != self._active:
                        log.warning("gemini: switched to API key #%d (%s)", idx + 1, _mask(self.keys[idx]))
                        self._active = idx
                    return result
                except Exception as exc:  # noqa: BLE001 - normalised below
                    last = exc
                    cooldown = _key_failure(exc)
                    if cooldown is not None:
                        self._cooldown_until[idx] = time.monotonic() + cooldown
                        log.warning(
                            "gemini: key #%d (%s) unavailable (%s); trying next key",
                            idx + 1, _mask(self.keys[idx]), getattr(exc, "code", exc),
                        )
                        break  # next key
                    if _is_transient(exc) and attempt < transient_attempts - 1:
                        delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                        log.warning("gemini: transient error, retry in %.1fs: %s", delay, exc)
                        await asyncio.sleep(delay)
                        continue
                    raise _translate_error(exc) from exc
        assert last is not None
        raise _translate_error(last) from last

    # -- generation ----------------------------------------------------------
    async def generate_structured(
        self,
        *,
        system_instruction: str,
        contents: str,
        schema: type[T],
        temperature: float | None = None,
    ) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.settings.temperature if temperature is None else temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )

        async def call(client: genai.Client) -> types.GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=self.settings.gemini_model, contents=contents, config=config
            )

        response = await self._call(call)
        record_generation(getattr(response, "usage_metadata", None))
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise GeminiError("Gemini returned an empty response (it may have been blocked).")
        try:
            return schema.model_validate_json(text)
        except Exception as exc:  # noqa: BLE001
            raise GeminiError("Gemini returned malformed JSON.") from exc

    # -- embeddings ----------------------------------------------------------
    async def _embed(self, texts: list[str], *, patient: bool = False) -> list[list[float]]:
        # One Content per string → one embedding per string. Passing plain strings
        # would make gemini-embedding-2 return a single aggregated embedding.
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self.settings.embed_dim)

        async def call(client: genai.Client) -> types.EmbedContentResponse:
            return await client.aio.models.embed_content(
                model=self.settings.gemini_embed_model, contents=contents, config=config
            )

        response = await self._call(call, transient_attempts=4, base_delay=2.0, rate_limit_waits=5 if patient else 0)
        record_embedding(texts)
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise GeminiError(f"Expected {len(texts)} embeddings, got {len(embeddings)}.")
        return [list(e.values or []) for e in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([QUERY_PREFIX + text]))[0]

    async def embed_documents(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [DOCUMENT_PREFIX + t for t in texts[i : i + batch_size]]
            out.extend(await self._embed(batch, patient=True))
        return out


_default: GeminiClient | None = None


def get_gemini() -> GeminiClient:
    global _default
    if _default is None:
        _default = GeminiClient()
    return _default
