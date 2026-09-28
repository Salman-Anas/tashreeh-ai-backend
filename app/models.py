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
    term_ur: str  # official legal Urdu (as in statutes), e.g. ابتدائی اطلاعی رپورٹ
    term_ur_common: str | None = None  # everyday Urdu used in translations, e.g. ایف آئی آر
    category: str | None = None
    notes: str | None = None
    source: str | None = None

    @property
    def ur_target(self) -> str:
        """The Urdu a translation should use: the everyday form when there is one."""
        return self.term_ur_common or self.term_ur

    @property
    def ur_forms(self) -> list[str]:
        """Every Urdu spelling that refers to this term (everyday first)."""
        return list(dict.fromkeys(f for f in (self.term_ur_common, self.term_ur) if f))


class GlossaryTermCreate(BaseModel):
    term_en: str = Field(min_length=1, max_length=200)
    term_ur: str = Field(min_length=1, max_length=200)
    term_ur_common: str | None = Field(default=None, max_length=200)
    category: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=500)
    source: str | None = Field(default="user", max_length=100)


class GlossaryTermUpdate(BaseModel):
    """Fields to change; omitted fields stay as they are, empty strings clear optional ones."""

    term_en: str | None = Field(default=None, min_length=1, max_length=200)
    term_ur: str | None = Field(default=None, min_length=1, max_length=200)
    term_ur_common: str | None = Field(default=None, max_length=200)
    category: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=500)


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
    term_ur_common: str | None = None
    category: str | None = None
    notes: str | None = None
    span_start: int
    span_end: int


class RequiredTerm(BaseModel):
    """A glossary term the translation is required to use."""

    term_id: int
    source: str  # term in the source language
    target: str  # required term in the target language (everyday Urdu when translating into Urdu)
    term_en: str
    term_ur: str  # official legal Urdu
    term_ur_common: str | None = None
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


class LLMTermExplanation(BaseModel):
    term_id: int
    legal_en: str
    legal_ur: str
    simple_en: str
    simple_ur: str
    example_en: str
    example_ur: str


class LLMExplanations(BaseModel):
    explanations: list[LLMTermExplanation]


# ---------------------------------------------------------------------------
# Plain-language term explanations ("Explain in simple words")
# ---------------------------------------------------------------------------
class TermExplanation(BaseModel):
    term_id: int
    term_en: str
    term_ur: str
    term_ur_common: str | None = None
    category: str | None = None
    legal_en: str = ""  # the formal legal meaning under Pakistani law, in English
    legal_ur: str = ""  # the same in Urdu
    simple_en: str  # what it means, for a non-lawyer, in plain English
    simple_ur: str  # the same in everyday Urdu
    example_en: str  # a short real-life example
    example_ur: str
    cached: bool = False


EXPLAIN_DISCLAIMER_EN = "General information to help you understand the term, not legal advice."
EXPLAIN_DISCLAIMER_UR = "یہ اصطلاح سمجھانے کے لیے عمومی معلومات ہیں، قانونی مشورہ نہیں۔"


class ExplainRequest(BaseModel):
    term_ids: list[int] = Field(min_length=1, max_length=12)
    session_id: str | None = Field(default=None, max_length=64)


class ExplainResponse(BaseModel):
    explanations: list[TermExplanation]
    missing: list[int] = Field(default_factory=list)  # ids that could not be explained
    disclaimer_en: str = EXPLAIN_DISCLAIMER_EN
    disclaimer_ur: str = EXPLAIN_DISCLAIMER_UR


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


class TermCorrection(BaseModel):
    """A reader's better word for one glossary term, in the target language."""

    term_id: int
    source: str = Field(max_length=200)  # the term as it appears in the source language
    old_target: str = Field(max_length=200)  # what the glossary asked for
    new_target: str = Field(min_length=1, max_length=200)  # what the reader suggests


class FeedbackRequest(BaseModel):
    translation_id: str
    rating: Literal[-1, 1] | None = None
    corrected_text: str | None = Field(default=None, max_length=20000)
    comment: str | None = Field(default=None, max_length=2000)
    term_corrections: list[TermCorrection] | None = Field(default=None, max_length=30)


class FeedbackResponse(BaseModel):
    id: int
    ok: bool = True


# ---------------------------------------------------------------------------
# Human review of corrections
# ---------------------------------------------------------------------------
ReviewStatus = Literal["pending", "approved", "rejected"]


class ReviewTranslation(BaseModel):
    id: str
    direction: Direction
    source_text: str
    output_text: str


class ReviewItem(BaseModel):
    id: int
    status: ReviewStatus
    rating: int | None = None
    comment: str | None = None
    corrected_text: str | None = None
    term_corrections: list[TermCorrection] = Field(default_factory=list)
    review_note: str | None = None
    created_at: datetime | None = None
    reviewed_at: datetime | None = None
    translation: ReviewTranslation | None = None


class ReviewListResponse(BaseModel):
    items: list[ReviewItem]
    counts: dict[str, int]


class ReviewApproveRequest(BaseModel):
    """What the reviewer accepts. Omitted fields keep what the reader submitted."""

    corrected_text: str | None = Field(default=None, max_length=20000)
    term_corrections: list[TermCorrection] | None = Field(default=None, max_length=30)
    note: str | None = Field(default=None, max_length=500)


class ReviewRejectRequest(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class GlossaryChange(BaseModel):
    term_id: int
    action: Literal["updated", "added"]
    term_en: str
    term_ur: str
    term_ur_common: str | None = None


class ReviewApproveResponse(BaseModel):
    id: int
    status: ReviewStatus = "approved"
    tm_pairs_added: int = 0
    glossary_changes: list[GlossaryChange] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # parts that could not be learned, with the reason


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
    explain: bool = Field(default=False, description="Also explain each legal term in simple English and Urdu, with an example.")


class PublicTerm(BaseModel):
    source: str
    target: str
    official: str | None = None  # official legal term, when the target is an everyday form
    category: str | None = None
    found: bool


class PublicHighlight(BaseModel):
    """A legal term's character span, for highlighting (``text[start:end]``)."""

    start: int
    end: int
    term_en: str
    term_ur: str  # official legal Urdu
    term_ur_common: str | None = None  # everyday Urdu used in translations
    category: str | None = None


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
    source_highlights: list[PublicHighlight] = Field(default_factory=list)  # spans in the request text (after trimming)
    output_highlights: list[PublicHighlight] = Field(default_factory=list)  # spans in "translation"
    references: list[TMExample] = Field(default_factory=list)  # translation-memory pairs used as examples
    explanations: list[TermExplanation] | None = None  # present when explain=true


class PublicExplainRequest(BaseModel):
    terms: list[str] = Field(
        min_length=1, max_length=12, description="Glossary terms in English or Urdu, e.g. [\"bail\", \"مدعی\"]."
    )


class PublicExplainResponse(BaseModel):
    id: str
    explanations: list[TermExplanation]
    not_found: list[str]  # terms that are not in the legal glossary
    disclaimer_en: str = EXPLAIN_DISCLAIMER_EN
    disclaimer_ur: str = EXPLAIN_DISCLAIMER_UR
    model: str
    latency_ms: int
    usage: PublicUsage


class PublicUsageSummary(BaseModel):
    key_name: str
    key_prefix: str
    usage: ApiKeyStats
    currency: Literal["USD"] = "USD"
