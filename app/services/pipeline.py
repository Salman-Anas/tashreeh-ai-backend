"""The translation pipeline.

1. Glossary lookup   → required terms for the input
2. TM retrieval      → similar, already-translated legal sentence pairs
3. Gemini            → structured JSON translation using terms + examples
4. Terminology check → verify required terms, retry once with a stricter prompt
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from app.models import Direction, GlossaryMatch, LLMTranslation, RequiredTerm, TMExample
from app.services.glossary import GlossaryMatcher, ur_keys
from app.services.term_check import TermCheckResult, check_terms
from app.services.tm import TMRetriever

log = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """\
You are an expert legal translator specialising in Pakistani law, translating between English and Urdu \
for ordinary Pakistani readers, not lawyers.
Rules:
1. Translate faithfully. Do not summarise, explain, add, or omit content. The legal meaning must stay exact.
2. You MUST use the exact target-language term given in REQUIRED TERMS for each listed source term. \
You may inflect it only where grammar strictly requires (e.g. plural), keeping the term itself recognisable. \
These are the words people actually use (e.g. FIR → ایف آئی آر, High Court → ہائی کورٹ); \
do not replace them with a more formal synonym.
3. When translating into Urdu, write clear, everyday Urdu (عام فہم اردو) that a reader with ordinary schooling \
understands at first reading: natural word order, common words, short sentences where the source allows. \
Avoid archaic or heavily Persianised court Urdu. Where an English legal word is commonly spoken in Urdu \
(FIR, challan, remand, stay order), write it in Urdu script. When translating into English, use plain, clear English.
4. Preserve section numbers, article numbers, clause labels like (a), (b), (i), dates, names, and citations \
such as "PLD 2024 SC 337" exactly.
5. Keep Pakistani statute names recognisable, using the REQUIRED TERMS form when one is given \
(e.g. PPC → تعزیرات پاکستان).
6. Use the EXAMPLES for meaning and terminology. Some are in formal court Urdu: keep their meaning but prefer \
everyday wording and the REQUIRED TERMS. The examples are references, not text to translate.
7. If a phrase is ambiguous, choose the most likely legal meaning and mention it in "notes".
8. Preserve line breaks of the SOURCE TEXT. Output only the translation in "translation".
9. In "terms_used" list each REQUIRED TERM you applied as {"source", "target"}. \
"notes" may be an empty list; write notes in English.
"""

LANG_NAME = {"en": "English", "ur": "Urdu"}
_PARA_SPLIT = re.compile(r"\n[^\S\n]*\n\s*")


class StructuredLLM(Protocol):
    async def generate_structured(
        self, *, system_instruction: str, contents: str, schema: type[LLMTranslation], temperature: float | None = None
    ) -> LLMTranslation: ...


def langs(direction: Direction) -> tuple[str, str]:
    return ("en", "ur") if direction == "en-ur" else ("ur", "en")


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; single line breaks stay inside a paragraph."""
    return [p.strip() for p in _PARA_SPLIT.split(text.strip()) if p.strip()]


def _format_terms(terms: list[RequiredTerm]) -> str:
    if not terms:
        return "(none)"
    return "\n".join(f"- {t.source} → {t.target}" for t in terms)


def _format_examples(examples: list[TMExample], direction: Direction) -> str:
    if not examples:
        return "(none)"
    src, tgt = langs(direction)
    lines: list[str] = []
    for i, ex in enumerate(examples, start=1):
        s_text = ex.text_en if src == "en" else ex.text_ur
        t_text = ex.text_ur if tgt == "ur" else ex.text_en
        lines.append(f"[{i}] {LANG_NAME[src]}: {s_text}\n    {LANG_NAME[tgt]}: {t_text}")
    return "\n".join(lines)


def build_prompt(text: str, direction: Direction, terms: list[RequiredTerm], examples: list[TMExample]) -> str:
    src, tgt = langs(direction)
    return (
        f"DIRECTION: {LANG_NAME[src]} → {LANG_NAME[tgt]}\n\n"
        f"REQUIRED TERMS (source → target):\n{_format_terms(terms)}\n\n"
        f"EXAMPLES:\n{_format_examples(examples, direction)}\n\n"
        f"SOURCE TEXT ({LANG_NAME[src]}):\n<<<\n{text}\n>>>"
    )


def build_retry_prompt(text: str, direction: Direction, previous: str, missing: list[RequiredTerm]) -> str:
    src, tgt = langs(direction)
    return (
        f"DIRECTION: {LANG_NAME[src]} → {LANG_NAME[tgt]}\n\n"
        "Your previous translation did NOT use the following REQUIRED TERMS. Revise it so that each source "
        "term below is rendered with EXACTLY the given target term (verbatim spelling). Keep everything "
        "else as close to the previous translation as possible.\n\n"
        f"MISSING REQUIRED TERMS (source → target):\n{_format_terms(missing)}\n\n"
        f"SOURCE TEXT ({LANG_NAME[src]}):\n<<<\n{text}\n>>>\n\n"
        f"PREVIOUS TRANSLATION ({LANG_NAME[tgt]}):\n<<<\n{previous}\n>>>"
    )


@dataclass
class SegmentResult:
    source: str
    translation: str
    check: TermCheckResult
    examples: list[TMExample]
    notes: list[str]
    retried: bool


@dataclass
class PipelineResult:
    translation: str
    glossary_matches: list[GlossaryMatch]
    output_matches: list[GlossaryMatch]
    terms_required: list[RequiredTerm]
    terms_found: list[RequiredTerm]
    terms_missing: list[RequiredTerm]
    tm_examples_used: list[TMExample]
    notes: list[str]
    latency_ms: int
    retried: bool
    segments: list[SegmentResult] = field(default_factory=list)


class TranslationPipeline:
    def __init__(
        self, *, llm: StructuredLLM, glossary: GlossaryMatcher, tm: TMRetriever, concurrency: int = 4
    ) -> None:
        self.llm = llm
        self.glossary = glossary
        self.tm = tm
        self.concurrency = max(1, concurrency)

    async def translate_segment(self, text: str, direction: Direction) -> SegmentResult:
        src, _ = langs(direction)
        matches = self.glossary.match(text, src)
        required = self.glossary.required_terms(matches, direction, text)
        examples = await self.tm.retrieve(text, direction)

        out = await self.llm.generate_structured(
            system_instruction=SYSTEM_INSTRUCTION,
            contents=build_prompt(text, direction, required, examples),
            schema=LLMTranslation,
        )
        translation = out.translation.strip()
        notes = [n.strip() for n in out.notes if n.strip()]
        check = check_terms(translation, required, direction)
        retried = False

        if check.terms_missing:
            retried = True
            log.info("term check: %d missing, retrying once", len(check.terms_missing))
            retry = await self.llm.generate_structured(
                system_instruction=SYSTEM_INSTRUCTION,
                contents=build_retry_prompt(text, direction, translation, check.terms_missing),
                schema=LLMTranslation,
                temperature=0.0,
            )
            retry_text = retry.translation.strip()
            retry_check = check_terms(retry_text, required, direction)
            # Keep the retry only if it is at least as good terminologically.
            if retry_text and len(retry_check.terms_missing) <= len(check.terms_missing):
                translation, check = retry_text, retry_check
                notes = [n.strip() for n in retry.notes if n.strip()] or notes

        return SegmentResult(text, translation, check, examples, notes, retried)

    async def translate(self, text: str, direction: Direction) -> PipelineResult:
        started = time.perf_counter()
        src, tgt = langs(direction)
        paragraphs = split_paragraphs(text)
        sem = asyncio.Semaphore(self.concurrency)

        async def run(p: str) -> SegmentResult:
            async with sem:
                return await self.translate_segment(p, direction)

        segments = list(await asyncio.gather(*(run(p) for p in paragraphs)))
        translation = "\n\n".join(s.translation for s in segments)

        # Aggregate terms across paragraphs (a term missing anywhere counts as missing).
        required: dict[int, RequiredTerm] = {}
        missing_ids: set[int] = set()
        for s in segments:
            for t in s.check.terms_required:
                required.setdefault(t.term_id, t)
            missing_ids.update(t.term_id for t in s.check.terms_missing)
        terms_required = list(required.values())

        examples: dict[int, TMExample] = {}
        for s in segments:
            for ex in s.examples:
                if ex.id not in examples or ex.similarity > examples[ex.id].similarity:
                    examples[ex.id] = ex

        notes: list[str] = []
        for s in segments:
            for n in s.notes:
                if n not in notes:
                    notes.append(n)

        return PipelineResult(
            translation=translation,
            glossary_matches=self.glossary.match(text, src),
            output_matches=self._output_matches(translation, terms_required, direction),
            terms_required=terms_required,
            terms_found=[t for t in terms_required if t.term_id not in missing_ids],
            terms_missing=[t for t in terms_required if t.term_id in missing_ids],
            tm_examples_used=sorted(examples.values(), key=lambda e: -e.similarity),
            notes=notes,
            latency_ms=int((time.perf_counter() - started) * 1000),
            retried=any(s.retried for s in segments),
            segments=segments,
        )

    def _output_matches(
        self, output: str, required: list[RequiredTerm], direction: Direction
    ) -> list[GlossaryMatch]:
        """Glossary spans in the output, limited to the terms that were required."""
        _, tgt = langs(direction)
        if direction == "en-ur":
            keys = {t.term_en.strip().lower() for t in required}
            return [m for m in self.glossary.match(output, tgt) if m.term_en.strip().lower() in keys]
        keys = set().union(*(ur_keys(t) for t in required)) if required else set()
        return [m for m in self.glossary.match(output, tgt) if ur_keys(m) & keys]
