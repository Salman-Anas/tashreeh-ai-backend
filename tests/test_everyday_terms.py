"""Everyday vs literal/official Urdu forms, and learning from reviewed corrections."""

from __future__ import annotations

import pytest

from app.models import GlossaryTerm, TermCorrection
from app.services import review
from app.services.glossary import REVIEWED_SOURCE, GlossaryMatcher
from app.services.term_check import check_terms

PPC_OFFICIAL, PPC_EVERYDAY = "مجموعہ تعزیرات پاکستان", "پاکستان پینل کوڈ"


@pytest.fixture
def everyday() -> GlossaryMatcher:
    return GlossaryMatcher(
        [
            GlossaryTerm(id=1, term_en="Pakistan Penal Code", term_ur=PPC_OFFICIAL, term_ur_common=PPC_EVERYDAY, category="statute"),
            GlossaryTerm(id=2, term_en="FIR", term_ur="ابتدائی اطلاعی رپورٹ", term_ur_common="ایف آئی آر", category="criminal"),
            GlossaryTerm(id=3, term_en="bail", term_ur="ضمانت", category="criminal"),
            GlossaryTerm(id=4, term_en="PPC", term_ur=PPC_OFFICIAL, term_ur_common="پی پی سی", category="statute"),
        ]
    )


def test_translation_into_urdu_requires_the_everyday_form(everyday: GlossaryMatcher) -> None:
    text = "An FIR was lodged under the Pakistan Penal Code."
    req = everyday.required_terms(everyday.match(text, "en"), "en-ur", text)
    assert [(r.source, r.target, r.term_ur) for r in req] == [
        ("FIR", "ایف آئی آر", "ابتدائی اطلاعی رپورٹ"),
        ("Pakistan Penal Code", PPC_EVERYDAY, PPC_OFFICIAL),
    ]
    # The literal/official translation alone does not satisfy the check.
    assert check_terms(f"{PPC_OFFICIAL} کے تحت ایف آئی آر", req, "en-ur").terms_missing == [req[1]]
    assert check_terms(f"{PPC_EVERYDAY} کے تحت ایف آئی آر درج ہوئی", req, "en-ur").ok


def test_acronyms_are_case_sensitive(everyday: GlossaryMatcher) -> None:
    assert [m.term_en for m in everyday.match("Two FIRs were registered", "en")] == ["FIR"]
    assert everyday.match("a fir tree", "en") == []


def test_urdu_source_matches_either_form(everyday: GlossaryMatcher) -> None:
    for form in (PPC_OFFICIAL, PPC_EVERYDAY):
        text = f"ملزم پر {form} کے تحت مقدمہ ہے"
        m = everyday.match(text, "ur")
        assert [x.term_en for x in m] == ["Pakistan Penal Code"]
        req = everyday.required_terms(m, "ur-en", text)
        assert req[0].source == form and req[0].target == "Pakistan Penal Code"
        assert req[0].alternatives == ["PPC"]


def test_output_highlights_everyday_form(everyday: GlossaryMatcher) -> None:
    m = everyday.match(f"{PPC_EVERYDAY} کی دفعہ", "ur")
    assert m[0].term_ur_common == PPC_EVERYDAY and m[0].span_start == 0


def test_reviewed_row_wins_a_tie() -> None:
    g = GlossaryMatcher(
        [
            GlossaryTerm(id=1, term_en="bail", term_ur="ضمانت"),
            GlossaryTerm(id=9, term_en="surety release", term_ur="ضمانت", source=REVIEWED_SOURCE),
        ]
    )
    req = g.required_terms(g.match("ضمانت منظور", "ur"), "ur-en", "ضمانت منظور")
    assert req[0].target == "surety release" and req[0].alternatives == ["bail"]


# -- review planning -----------------------------------------------------------
def _fix(term_id: int, source: str, old: str, new: str) -> TermCorrection:
    return TermCorrection(term_id=term_id, source=source, old_target=old, new_target=new)


def test_plan_en_ur_sets_everyday_form(everyday: GlossaryMatcher) -> None:
    plan = review.plan_glossary_changes([_fix(3, "bail", "ضمانت", "بیل")], everyday, "en-ur")
    assert plan.updates == [(3, {"term_ur_common": "بیل"})] and not plan.skipped


def test_plan_choosing_official_form_clears_everyday(everyday: GlossaryMatcher) -> None:
    plan = review.plan_glossary_changes([_fix(1, "Pakistan Penal Code", PPC_EVERYDAY, PPC_OFFICIAL)], everyday, "en-ur")
    assert plan.updates == [(1, {"term_ur_common": None})]


def test_plan_skips_bad_or_unchanged_fixes(everyday: GlossaryMatcher) -> None:
    plan = review.plan_glossary_changes(
        [_fix(3, "bail", "ضمانت", "bail money"), _fix(2, "FIR", "ایف آئی آر", "ایف آئی آر"), _fix(99, "x", "a", "ب")],
        everyday,
        "en-ur",
    )
    assert plan.updates == [] and len(plan.skipped) == 3


def test_plan_ur_en_adds_reviewed_row(everyday: GlossaryMatcher) -> None:
    plan = review.plan_glossary_changes([_fix(3, "ضمانت", "bail", "surety")], everyday, "ur-en")
    assert plan.inserts[0]["term_en"] == "surety" and plan.inserts[0]["term_ur"] == "ضمانت"
    assert plan.inserts[0]["source"] == REVIEWED_SOURCE


def test_plan_ur_en_promotes_existing_row(everyday: GlossaryMatcher) -> None:
    plan = review.plan_glossary_changes([_fix(1, PPC_EVERYDAY, "Pakistan Penal Code", "PPC")], everyday, "ur-en")
    assert plan.inserts == [] and plan.updates == [(4, {"source": REVIEWED_SOURCE})]


def test_tm_pairs_align_paragraphs() -> None:
    src = "The accused was granted bail.\n\nThe FIR was quashed."
    fixed = "ملزم کی ضمانت منظور ہو گئی۔\n\nایف آئی آر ختم کر دی گئی۔"
    pairs, skipped = review.tm_pairs(src, fixed, "en-ur")
    assert pairs == [
        ("The accused was granted bail.", "ملزم کی ضمانت منظور ہو گئی۔"),
        ("The FIR was quashed.", "ایف آئی آر ختم کر دی گئی۔"),
    ]
    assert skipped == []
    pairs, _ = review.tm_pairs(fixed, "The accused got bail. The FIR was quashed.", "ur-en")
    assert pairs == [("The accused got bail. The FIR was quashed.", "ملزم کی ضمانت منظور ہو گئی۔ ایف آئی آر ختم کر دی گئی۔")]


def test_tm_pairs_rejects_wrong_script() -> None:
    pairs, skipped = review.tm_pairs("The accused was granted bail.", "The accused was granted bail.", "en-ur")
    assert pairs == [] and "not mostly Urdu" in skipped[0]


def test_word_fixes_rewrite_the_machine_output() -> None:
    out = review.apply_term_fixes(f"{PPC_OFFICIAL} کی دفعہ 302", [_fix(1, "Pakistan Penal Code", PPC_OFFICIAL, PPC_EVERYDAY)])
    assert out == f"{PPC_EVERYDAY} کی دفعہ 302"
