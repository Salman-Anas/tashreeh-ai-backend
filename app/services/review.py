"""Human-in-the-loop learning from reviewed corrections.

Readers suggest a corrected translation and/or a better word for a legal term.
Nothing changes until a reviewer approves. On approval:

* the corrected translation is stored in the translation memory (paragraph-aligned
  with the source), so similar future inputs get it as a reference example;
* each term fix updates the glossary, so every future translation must use the
  new word:
  - English → Urdu: the new word becomes the term's everyday Urdu form;
  - Urdu → English: a reviewed glossary row is added (or promoted), which wins
    over the old English term when both match.

This module only plans the changes; ``routers/review.py`` applies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models import Direction, GlossaryTerm, TermCorrection
from app.services.glossary import REVIEWED_SOURCE, GlossaryMatcher, ur_keys
from app.services.normalize import normalize_for_matching, normalize_urdu, urdu_ratio
from app.services.pipeline import split_paragraphs
from app.services.tm import validate_pair

TM_SOURCE = "user-approved"


def apply_term_fixes(output: str, corrections: list[TermCorrection]) -> str:
    """The machine output with each fixed term's old word replaced by the new one."""
    for c in corrections:
        if c.old_target and c.old_target in output:
            output = output.replace(c.old_target, c.new_target)
    return output


def tm_pairs(source_text: str, corrected: str, direction: Direction) -> tuple[list[tuple[str, str]], list[str]]:
    """``(text_en, text_ur)`` pairs to learn from, and reasons for anything skipped.

    Paragraphs are paired one-to-one when source and correction have the same
    number of paragraphs; otherwise the whole text is one pair.
    """
    src, tgt = split_paragraphs(source_text), split_paragraphs(corrected)
    aligned = list(zip(src, tgt)) if len(src) == len(tgt) else [("\n\n".join(src), "\n\n".join(tgt))]
    pairs: list[tuple[str, str]] = []
    skipped: list[str] = []
    for i, (s, t) in enumerate(aligned, start=1):
        en, ur = (s, t) if direction == "en-ur" else (t, s)
        en = " ".join(en.split())
        ur = normalize_urdu(ur, preserve_newlines=False)
        reason = validate_pair(en, ur)
        label = f"paragraph {i}" if len(aligned) > 1 else "translation"
        if reason:
            skipped.append(f"{label} not added to memory: {reason}")
        elif (en, ur) not in pairs:
            pairs.append((en, ur))
    return pairs, skipped


@dataclass
class GlossaryPlan:
    updates: list[tuple[int, dict[str, Any]]] = field(default_factory=list)  # (term id, changed columns)
    inserts: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _find_row(glossary: GlossaryMatcher, term_en: str, like: GlossaryTerm) -> GlossaryTerm | None:
    """An existing row with this English term and the same Urdu meaning as ``like``."""
    key = term_en.lower()
    keys = ur_keys(like)
    return next((t for t in glossary.terms if t.term_en.lower() == key and ur_keys(t) & keys), None)


def plan_glossary_changes(
    corrections: list[TermCorrection], glossary: GlossaryMatcher, direction: Direction
) -> GlossaryPlan:
    plan = GlossaryPlan()
    for c in corrections:
        term = glossary.get(c.term_id)
        if term is None:
            plan.skipped.append(f"“{c.source}”: this term is no longer in the glossary")
            continue
        if direction == "en-ur":
            new = normalize_urdu(c.new_target, preserve_newlines=False)
            if urdu_ratio(new) < 0.8:
                plan.skipped.append(f"“{c.source}”: the new word “{c.new_target}” is not in Urdu script")
                continue
            if normalize_for_matching(new) == normalize_for_matching(term.ur_target):
                plan.skipped.append(f"“{c.source}”: “{new}” is already the word in use")
                continue
            # Choosing the official form itself clears the everyday form.
            same_as_official = normalize_for_matching(new) == normalize_for_matching(term.term_ur)
            plan.updates.append((term.id, {"term_ur_common": None if same_as_official else new}))
        else:
            new = " ".join(c.new_target.split())
            if urdu_ratio(new) > 0.2:
                plan.skipped.append(f"“{c.source}”: the new word “{c.new_target}” is not in English")
                continue
            if new.lower() == term.term_en.lower():
                plan.skipped.append(f"“{c.source}”: “{new}” is already the word in use")
                continue
            existing = _find_row(glossary, new, term)
            if existing is not None:
                if existing.source != REVIEWED_SOURCE:
                    plan.updates.append((existing.id, {"source": REVIEWED_SOURCE}))
                continue
            plan.inserts.append(
                {
                    "term_en": new,
                    "term_ur": term.term_ur,
                    "term_ur_common": term.term_ur_common,
                    "category": term.category,
                    "notes": f"Reviewed correction of “{term.term_en}”",
                    "source": REVIEWED_SOURCE,
                }
            )
    return plan
