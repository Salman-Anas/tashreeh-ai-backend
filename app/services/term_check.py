"""Post-translation terminology check."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models import Direction, RequiredTerm
from app.services.glossary import contains_term


@dataclass
class TermCheckResult:
    terms_required: list[RequiredTerm]
    terms_found: list[RequiredTerm] = field(default_factory=list)
    terms_missing: list[RequiredTerm] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.terms_missing


def target_lang(direction: Direction) -> str:
    return "ur" if direction == "en-ur" else "en"


def check_terms(output: str, required: list[RequiredTerm], direction: Direction) -> TermCheckResult:
    """Check that each required target term (or an accepted alternative) is in ``output``."""
    lang = target_lang(direction)
    result = TermCheckResult(terms_required=list(required))
    for term in required:
        targets = [term.target, *term.alternatives]
        if any(contains_term(output, t, lang) for t in targets):
            result.terms_found.append(term)
        else:
            result.terms_missing.append(term)
    return result
