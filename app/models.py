from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["en-ur", "ur-en"]


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------
class GlossaryTerm(BaseModel):
    id: int
    term_en: str
    term_ur: str
    category: str | None = None
    notes: str | None = None
    source: str | None = None


class GlossaryTermCreate(BaseModel):
    term_en: str = Field(min_length=1, max_length=200)
    term_ur: str = Field(min_length=1, max_length=200)
    category: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=500)
    source: str | None = Field(default="user", max_length=100)


class GlossaryListResponse(BaseModel):
    items: list[GlossaryTerm]
    total: int
    category_counts: dict[str, int]
    read_only: bool


class GlossaryMatch(BaseModel):
    """A glossary term found in a piece of text, with its character span."""

    term_id: int
    term_en: str
    term_ur: str
    category: str | None = None
    notes: str | None = None
    span_start: int
    span_end: int


class RequiredTerm(BaseModel):
    """A glossary term the translation is required to use."""

    term_id: int
    source: str  # term in the source language
    target: str  # required term in the target language
    term_en: str
    term_ur: str
    category: str | None = None
    alternatives: list[str] = Field(default_factory=list)  # other accepted targets


# ---------------------------------------------------------------------------
# Translation memory
# ---------------------------------------------------------------------------
class TMExample(BaseModel):
    id: int
    text_en: str
    text_ur: str
    source: str | None = None
    similarity: float


# ---------------------------------------------------------------------------
# LLM structured output
# ---------------------------------------------------------------------------
class LLMTermUsed(BaseModel):
    source: str
    target: str


class LLMTranslation(BaseModel):
    translation: str
    terms_used: list[LLMTermUsed]
    notes: list[str]


# ---------------------------------------------------------------------------
# Translate API
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    direction: Direction
    session_id: str | None = Field(default=None, max_length=64)


class TranslateResponse(BaseModel):
    id: str
    direction: Direction
    source_text: str
    translation: str
    terms_required: list[RequiredTerm]
    terms_found: list[RequiredTerm]
    terms_missing: list[RequiredTerm]
    glossary_matches: list[GlossaryMatch]  # spans in source_text
    output_matches: list[GlossaryMatch]  # spans in translation
    tm_examples_used: list[TMExample]
    notes: list[str]
    model: str
    latency_ms: int
    retried: bool = False
    saved: bool = True


class ExportRequest(BaseModel):
    translation: str = Field(min_length=1)
    direction: Direction
    source_text: str | None = None


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class DocumentSegmentResult(BaseModel):
    index: int
    kind: Literal["paragraph", "heading", "table_cell"]
    source: str
    translation: str
    terms_required: list[RequiredTerm]
    terms_missing: list[RequiredTerm]
    error: str | None = None


class DocumentTranslateResponse(BaseModel):
    filename: str
    direction: Direction
    segments: list[DocumentSegmentResult]
    docx_base64: str
    docx_filename: str
    latency_ms: int


# ---------------------------------------------------------------------------
# History & feedback
# ---------------------------------------------------------------------------
class HistoryItem(BaseModel):
    id: str
    direction: Direction
    source_text: str
    output_text: str
    terms_required: list[RequiredTerm] | None = None
    terms_missing: list[RequiredTerm] | None = None
    model: str | None = None
    latency_ms: int | None = None
    created_at: datetime | None = None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


class FeedbackRequest(BaseModel):
    translation_id: str
    rating: Literal[-1, 1] | None = None
    corrected_text: str | None = Field(default=None, max_length=20000)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: int
    ok: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    model: str
    embed_model: str
    supabase: bool
    gemini: bool
    gemini_keys: int = 0
    glossary_terms: int
    glossary_source: Literal["supabase", "csv"]


# ---------------------------------------------------------------------------
# Developer API
# ---------------------------------------------------------------------------
class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    session_id: str = Field(min_length=8, max_length=64)


class ApiKeyStats(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0
    gemini_cost_usd: float = 0.0
    billed_usd: float = 0.0


class ApiKeyInfo(BaseModel):
    id: str
    name: str
    key_prefix: str
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    stats: ApiKeyStats = Field(default_factory=ApiKeyStats)


class ApiKeyCreated(ApiKeyInfo):
    key: str  # full secret — returned once


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeyInfo]
    persistent: bool  # False = in-memory store (Supabase not configured)


class UsageRecord(BaseModel):
    id: int
    api_key_id: str
    key_name: str | None = None
    key_prefix: str | None = None
    endpoint: str
    direction: str | None = None
    input_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0
    gemini_calls: int = 0
    gemini_cost_usd: float = 0.0
    billed_usd: float = 0.0
    model: str | None = None
    status_code: int
    error_code: str | None = None
    latency_ms: int = 0
    created_at: datetime | None = None


class UsageResponse(BaseModel):
    summary: ApiKeyStats
    items: list[UsageRecord]


class PricingResponse(BaseModel):
    currency: Literal["USD"] = "USD"
    model: str
    embed_model: str
    markup: float
    gemini: dict[str, float]  # per 1M tokens: input, output, embedding
    client: dict[str, float]  # per 1M tokens after markup
    gemini_keys: int
    note: str


class PublicTranslateRequest(BaseModel):
    text: str = Field(description="Legal text to translate (max MAX_INPUT_CHARS characters).")
    direction: Direction = Field(description='"en-ur" (English → Urdu) or "ur-en" (Urdu → English).')


class PublicTerm(BaseModel):
    source: str
    target: str
    category: str | None = None
    found: bool


class PublicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    total_tokens: int
    gemini_cost_usd: float
    price_usd: float
    currency: Literal["USD"] = "USD"


class PublicTranslateResponse(BaseModel):
    id: str
    direction: Direction
    translation: str
    terms: list[PublicTerm]
    terms_missing: int
    notes: list[str]
    model: str
    latency_ms: int
    usage: PublicUsage


class PublicUsageSummary(BaseModel):
    key_name: str
    key_prefix: str
    usage: ApiKeyStats
    currency: Literal["USD"] = "USD"
