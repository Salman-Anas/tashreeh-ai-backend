"""Plain-language explanations of legal terms ("Explain in simple words").

For each glossary term, Gemini writes its formal legal meaning under Pakistani
law, what it means for an ordinary person in simple English and everyday Urdu,
and a short real-life example in both.
Only glossary terms are explained (never free text from the request), and
explanations are cached per term: a term explained once costs nothing after.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.models import GlossaryTerm, LLMExplanations, TermExplanation
from app.services.normalize import normalize_for_matching, normalize_urdu

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_TERMS_PER_CALL = 12
CACHE_SIZE = 2000

EXPLAIN_INSTRUCTION = """\
You explain Pakistani legal terms to ordinary people who have no legal training.
For every term you are given, write:
- legal_en: the term's formal legal meaning under Pakistani law (Constitution, statutes, court usage) in one or two \
precise sentences. Name the law it comes from (e.g. "Code of Criminal Procedure, 1898") and a section or article \
number only if you are certain of it.
- legal_ur: the same legal meaning in Urdu script; you may use the official legal term here.
- simple_en: one or two short sentences in plain English saying what the term means in practice in Pakistan. No legal jargon; if you must use a legal word, explain it.
- simple_ur: the same meaning in simple, everyday Urdu (عام فہم اردو) in Urdu script, not formal legal Urdu.
- example_en: one short, realistic example situation in Pakistan (one or two sentences) that shows the term in use.
- example_ur: the same example in simple Urdu.
Rules:
1. Be accurate to how the term is used in Pakistani law and courts. Use the category given to pick the legal meaning.
2. Do not give legal advice, and do not cite section numbers unless you are certain of them.
3. Keep each field under 45 words. Use simple names such as Ali, Ayesha or Ahmed in examples.
4. Return one entry per term, with the same term_id you were given.
"""


class StructuredLLM(Protocol):
    async def generate_structured(
        self, *, system_instruction: str, contents: str, schema: type[T], temperature: float | None = None
    ) -> T: ...


def _key(term: GlossaryTerm) -> tuple[str, str]:
    return term.term_en.strip().lower(), normalize_for_matching(term.term_ur)


def _prompt(terms: list[GlossaryTerm]) -> str:
    lines = ["TERMS TO EXPLAIN:"]
    for t in terms:
        common = f" | everyday Urdu: {t.term_ur_common}" if t.term_ur_common else ""
        extra = f" | category: {t.category}" if t.category else ""
        note = f" | note: {t.notes}" if t.notes else ""
        lines.append(f"- term_id {t.id}: {t.term_en} | official Urdu: {t.term_ur}{common}{extra}{note}")
    return "\n".join(lines)


class TermExplainer:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm
        self._cache: OrderedDict[tuple[str, str], TermExplanation] = OrderedDict()
        self._lock = threading.Lock()

    def _cached(self, term: GlossaryTerm) -> TermExplanation | None:
        with self._lock:
            hit = self._cache.get(_key(term))
            if hit is None:
                return None
            self._cache.move_to_end(_key(term))
        # Same words, but report the id and names the caller's glossary uses.
        return hit.model_copy(
            update={
                "term_id": term.id,
                "term_en": term.term_en,
                "term_ur": term.term_ur,
                "term_ur_common": term.term_ur_common,
                "category": term.category,
                "cached": True,
            }
        )

    def _store(self, exp: TermExplanation, term: GlossaryTerm) -> None:
        with self._lock:
            self._cache[_key(term)] = exp
            self._cache.move_to_end(_key(term))
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)

    async def explain(self, terms: list[GlossaryTerm]) -> tuple[list[TermExplanation], list[int]]:
        """Explanations for ``terms`` (in order) and the ids that could not be explained."""
        terms = terms[:MAX_TERMS_PER_CALL]
        found: dict[int, TermExplanation] = {}
        todo: list[GlossaryTerm] = []
        for t in terms:
            hit = self._cached(t)
            if hit:
                found[t.id] = hit
            else:
                todo.append(t)

        if todo:
            out = await self.llm.generate_structured(
                system_instruction=EXPLAIN_INSTRUCTION,
                contents=_prompt(todo),
                schema=LLMExplanations,
                temperature=0.3,
            )
            by_id = {t.id: t for t in todo}
            for item in out.explanations:
                term = by_id.get(item.term_id)
                if term is None or item.term_id in found:
                    continue
                if not (item.simple_en.strip() and item.simple_ur.strip()):
                    continue
                exp = TermExplanation(
                    term_id=term.id,
                    term_en=term.term_en,
                    term_ur=term.term_ur,
                    term_ur_common=term.term_ur_common,
                    category=term.category,
                    legal_en=item.legal_en.strip(),
                    legal_ur=normalize_urdu(item.legal_ur, preserve_newlines=False),
                    simple_en=item.simple_en.strip(),
                    simple_ur=normalize_urdu(item.simple_ur, preserve_newlines=False),
                    example_en=item.example_en.strip(),
                    example_ur=normalize_urdu(item.example_ur, preserve_newlines=False),
                )
                self._store(exp, term)
                found[term.id] = exp

        ordered = [found[t.id] for t in terms if t.id in found]
        missing = [t.id for t in terms if t.id not in found]
        if missing:
            log.warning("explain: model returned no explanation for term ids %s", missing)
        return ordered, missing
