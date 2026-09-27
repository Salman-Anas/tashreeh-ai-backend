"""Per-request token metering.

``track_usage()`` installs a ``UsageMeter`` in a context variable; every Gemini
call made inside that context (including in child asyncio tasks, which inherit
the context) adds its token counts to the same meter.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageMeter:
    input_tokens: int = 0  # prompt tokens (generation)
    output_tokens: int = 0  # response + thinking tokens (both billed as output)
    thinking_tokens: int = 0  # subset of output_tokens
    embedding_tokens: int = 0  # estimated; the Gemini API does not report them
    gemini_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.embedding_tokens


_meter: ContextVar[UsageMeter | None] = ContextVar("usage_meter", default=None)


@contextmanager
def track_usage() -> Iterator[UsageMeter]:
    meter = UsageMeter()
    token = _meter.set(meter)
    try:
        yield meter
    finally:
        _meter.reset(token)


def record_generation(usage_metadata: Any) -> None:
    meter = _meter.get()
    if meter is None:
        return
    meter.gemini_calls += 1
    if usage_metadata is None:
        return
    prompt = getattr(usage_metadata, "prompt_token_count", None) or 0
    candidates = getattr(usage_metadata, "candidates_token_count", None) or 0
    thoughts = getattr(usage_metadata, "thoughts_token_count", None) or 0
    meter.input_tokens += prompt
    meter.output_tokens += candidates + thoughts
    meter.thinking_tokens += thoughts


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for Latin script, ~2 for Urdu."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return max(1, round(ascii_chars / 4 + (len(text) - ascii_chars) / 2))


def record_embedding(texts: list[str]) -> None:
    meter = _meter.get()
    if meter is None:
        return
    meter.gemini_calls += 1
    meter.embedding_tokens += sum(estimate_tokens(t) for t in texts)
