"""In-memory glossary matcher.

English: case-insensitive (acronyms such as FIR are case-sensitive), whole
word/phrase, simple plurals, longest match first.
Urdu: on matching-normalized text (see ``normalize.py``), longest match first,
with common plural/oblique endings; spans map back to the original text. A term
matches in Urdu by its official legal form *or* its everyday form.

Translations into Urdu use a term's everyday form (``term_ur_common``, e.g.
ایف آئی آر) when it has one, else the official legal form (``term_ur``).
"""

from __future__ import annotations

import csv
import functools
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.models import Direction, GlossaryMatch, GlossaryTerm, RequiredTerm
from app.services.normalize import HEH_GOAL, normalize_for_matching, normalize_with_map

log = logging.getLogger(__name__)

# Urdu suffixes that still count as the same term (plural / oblique forms).
_UR_SUFFIXES = ("وں", "ان", "یں", "ات", "ین")
# Words ending in ہ often take ے / وں in place of ہ (فیصلہ → فیصلے, فیصلوں).
_UR_HEH_REPLACEMENTS = ("ے", "وں", "ات")
# Glossary rows written by a human reviewer win when several rows match the same text.
REVIEWED_SOURCE = "reviewed"
_ACRONYM = re.compile(r"[A-Z][A-Z.&]{1,7}")


def _plural_pattern(word: str) -> str:
    w = re.escape(word)
    lower = word.lower()
    if len(lower) > 1 and lower.endswith("y") and lower[-2] not in "aeiou":
        return re.escape(word[:-1]) + r"(?:y|ies)"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return w + r"(?:es)?"
    return w + r"(?:s|es)?"


@functools.lru_cache(maxsize=4096)
def english_term_pattern(term: str) -> re.Pattern[str]:
    """Whole-phrase pattern; the last word may be pluralised.

    Case-insensitive, except for acronyms ("FIR", "PPC", "NAB") so they do not
    match ordinary words ("fir", "nab").
    """
    words = term.split()
    if not words:
        raise ValueError("empty term")
    if len(words) == 1 and _ACRONYM.fullmatch(term):
        return re.compile(rf"(?<![\w]){re.escape(term)}(?:s|'s)?(?![\w])")
    parts = [re.escape(w) for w in words[:-1]] + [_plural_pattern(words[-1])]
    body = r"[\s\-]+".join(parts)
    return re.compile(rf"(?<![\w]){body}(?![\w])", re.IGNORECASE)


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("L", "M")


def find_urdu(norm_text: str, norm_term: str) -> list[tuple[int, int]]:
    """Find ``norm_term`` in ``norm_text`` (both matching-normalized).

    Returns ``(start, end)`` spans in ``norm_text``. Matches must sit on word
    boundaries, optionally followed by a plural/oblique ending.
    """
    if not norm_term:
        return []
    spans: list[tuple[int, int]] = []
    stems: list[tuple[str, tuple[str, ...]]] = [(norm_term, _UR_SUFFIXES)]
    if norm_term.endswith(HEH_GOAL) and len(norm_term) > 1:
        stems.append((norm_term[:-1], _UR_HEH_REPLACEMENTS))

    for stem, endings in stems:
        start = norm_text.find(stem)
        while start != -1:
            end = start + len(stem)
            ok_before = start == 0 or not _is_letter(norm_text[start - 1])
            if ok_before:
                candidates = ([""] if stem == norm_term else []) + list(endings)
                for suf in sorted(candidates, key=len, reverse=True):
                    stop = end + len(suf)
                    if norm_text.startswith(suf, end) and (
                        stop == len(norm_text) or not _is_letter(norm_text[stop])
                    ):
                        spans.append((start, stop))
                        break
            start = norm_text.find(stem, start + 1)
    return spans


def contains_term(text: str, term: str, lang: str) -> bool:
    """True if ``term`` (in ``lang``) occurs in ``text`` as a whole word/phrase."""
    if lang == "en":
        return english_term_pattern(term).search(text) is not None
    return bool(find_urdu(normalize_for_matching(text), normalize_for_matching(term)))


@dataclass(frozen=True)
class _Compiled:
    term: GlossaryTerm
    en_pattern: re.Pattern[str]
    ur_norms: tuple[str, ...]  # official and everyday forms, matching-normalized


def _rank(term: GlossaryTerm) -> tuple[int, int]:
    """Tie-break for rows matching the same text: reviewed rows first, then lowest id."""
    return (0 if term.source == REVIEWED_SOURCE else 1, term.id)


def ur_keys(term: GlossaryTerm | RequiredTerm | GlossaryMatch) -> set[str]:
    """Matching-normalized Urdu forms (official + everyday) of a term."""
    return {k for k in (normalize_for_matching(term.term_ur), normalize_for_matching(term.term_ur_common or "")) if k}


def _written_form(term: GlossaryTerm, matched: str | None) -> str:
    """The Urdu form of ``term`` (official or everyday) that ``matched`` was written in."""
    if matched and len(term.ur_forms) > 1:
        norm = normalize_for_matching(matched)
        for form in sorted(term.ur_forms, key=len, reverse=True):
            stem = normalize_for_matching(form)
            if norm.startswith(stem.removesuffix(HEH_GOAL) if len(stem) > 1 else stem):
                return form
    return term.term_ur


class GlossaryMatcher:
    def __init__(self, terms: list[GlossaryTerm] | None = None) -> None:
        self._lock = threading.Lock()
        self._terms: list[GlossaryTerm] = []
        self._compiled: list[_Compiled] = []
        self.source: str = "csv"
        self.set_terms(terms or [])

    # -- loading -------------------------------------------------------------
    def set_terms(self, terms: list[GlossaryTerm]) -> None:
        compiled: list[_Compiled] = []
        for t in terms:
            try:
                norms = tuple(dict.fromkeys(n for n in (normalize_for_matching(f) for f in t.ur_forms) if n))
                compiled.append(_Compiled(t, english_term_pattern(t.term_en.strip()), norms))
            except ValueError:
                log.warning("skipping empty glossary term id=%s", t.id)
        with self._lock:
            self._terms = sorted(terms, key=lambda t: t.id)
            self._compiled = compiled

    @property
    def terms(self) -> list[GlossaryTerm]:
        return list(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def get(self, term_id: int) -> GlossaryTerm | None:
        return next((t for t in self._terms if t.id == term_id), None)

    def by_ids(self, ids: list[int]) -> list[GlossaryTerm]:
        """Terms for ``ids`` in the order given; unknown ids are skipped."""
        by_id = {t.id: t for t in self._terms}
        return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]

    def find_by_text(self, text: str) -> GlossaryTerm | None:
        """The glossary entry whose English or Urdu side is exactly ``text`` (case/spelling-insensitive)."""
        en = " ".join(text.split()).lower()
        ur = normalize_for_matching(text)
        for t in sorted(self._terms, key=_rank):
            if t.term_en.strip().lower() == en or (ur and ur in ur_keys(t)):
                return t
        return None

    # -- matching ------------------------------------------------------------
    def _candidates(self, text: str, lang: str) -> list[tuple[int, int, _Compiled]]:
        cands: list[tuple[int, int, _Compiled]] = []
        if lang == "en":
            for c in self._compiled:
                for m in c.en_pattern.finditer(text):
                    cands.append((m.start(), m.end(), c))
        else:
            norm, idx = normalize_with_map(text)
            if not norm:
                return []
            for c in self._compiled:
                for ur_norm in c.ur_norms:
                    for s, e in find_urdu(norm, ur_norm):
                        end = idx[e - 1] + 1
                        # Keep trailing diacritics attached to the highlighted word.
                        while end < len(text) and unicodedata.category(text[end]) == "Mn":
                            end += 1
                        cands.append((idx[s], end, c))
        return cands

    def match(self, text: str, lang: str) -> list[GlossaryMatch]:
        """Non-overlapping glossary matches in ``text`` ("en" or "ur").

        Longest match wins; ties go to the earlier position, then to a reviewed
        row, then to the lower term id. When one span matches several terms (the
        same source term with alternative translations) only one is returned.
        """
        with self._lock:
            cands = self._candidates(text, lang)
        cands.sort(key=lambda x: (-(x[1] - x[0]), x[0], _rank(x[2].term)))
        taken: list[tuple[int, int]] = []
        chosen: list[tuple[int, int, _Compiled]] = []
        for s, e, c in cands:
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            chosen.append((s, e, c))
        chosen.sort(key=lambda x: x[0])
        return [
            GlossaryMatch(
                term_id=c.term.id,
                term_en=c.term.term_en,
                term_ur=c.term.term_ur,
                term_ur_common=c.term.term_ur_common,
                category=c.term.category,
                notes=c.term.notes,
                span_start=s,
                span_end=e,
            )
            for s, e, c in chosen
        ]

    def alternatives(self, term: GlossaryTerm, direction: Direction) -> list[str]:
        """Other accepted targets for the same source term."""
        if direction == "en-ur":
            key = term.term_en.strip().lower()
            alts = [t.ur_target for t in self._terms if t.id != term.id and t.term_en.strip().lower() == key]
            return [a for a in dict.fromkeys(alts) if a != term.ur_target]
        keys = ur_keys(term)
        alts = [t.term_en for t in self._terms if t.id != term.id and ur_keys(t) & keys]
        return [a for a in dict.fromkeys(alts) if a.lower() != term.term_en.lower()]

    def required_terms(
        self, matches: list[GlossaryMatch], direction: Direction, text: str | None = None
    ) -> list[RequiredTerm]:
        """Unique required terms (in first-appearance order) for ``matches``.

        ``text`` is the text the matches came from; for Urdu sources it tells which
        Urdu form (official or everyday) the writer used.
        """
        by_id = {t.id: t for t in self._terms}
        seen: set[int] = set()
        out: list[RequiredTerm] = []
        for m in matches:
            if m.term_id in seen:
                continue
            seen.add(m.term_id)
            term = by_id.get(m.term_id)
            if term is None:
                continue
            if direction == "en-ur":
                src, tgt = term.term_en, term.ur_target
            else:
                src, tgt = _written_form(term, text[m.span_start : m.span_end] if text else None), term.term_en
            out.append(
                RequiredTerm(
                    term_id=term.id,
                    source=src,
                    target=tgt,
                    term_en=term.term_en,
                    term_ur=term.term_ur,
                    term_ur_common=term.term_ur_common,
                    category=term.category,
                    alternatives=self.alternatives(term, direction),
                )
            )
        return out


def load_terms_from_csv(path: Path) -> list[GlossaryTerm]:
    terms: list[GlossaryTerm] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            en = (row.get("term_en") or "").strip()
            ur = (row.get("term_ur") or "").strip()
            if not en or not ur:
                continue
            common = (row.get("term_ur_common") or "").strip()
            terms.append(
                GlossaryTerm(
                    id=i,
                    term_en=en,
                    term_ur=ur,
                    term_ur_common=common if common and common != ur else None,
                    category=(row.get("category") or "").strip() or None,
                    notes=(row.get("notes") or "").strip() or None,
                    source=(row.get("source") or "").strip() or None,
                )
            )
    return terms
