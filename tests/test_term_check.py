from app.services.glossary import GlossaryMatcher
from app.services.term_check import check_terms


def test_all_terms_found_en_ur(matcher: GlossaryMatcher) -> None:
    req = matcher.required_terms(matcher.match("The accused was granted bail.", "en"), "en-ur")
    res = check_terms("ملزم کی ضمانت منظور کر لی گئی۔", req, "en-ur")
    assert res.ok
    assert [t.source for t in res.terms_found] == ["accused", "bail"]


def test_missing_term_is_reported(matcher: GlossaryMatcher) -> None:
    req = matcher.required_terms(matcher.match("The accused was granted bail.", "en"), "en-ur")
    res = check_terms("مشتبہ شخص کی ضمانت منظور کر لی گئی۔", req, "en-ur")
    assert not res.ok
    assert [t.source for t in res.terms_missing] == ["accused"]
    assert [t.source for t in res.terms_found] == ["bail"]


def test_found_despite_arabic_letters_or_diacritics(matcher: GlossaryMatcher) -> None:
    req = matcher.required_terms(matcher.match("The High Court", "en"), "en-ur")
    assert check_terms("عدالتِ عاليه نے", req, "en-ur").ok


def test_alternative_target_is_accepted(matcher: GlossaryMatcher) -> None:
    req = matcher.required_terms(matcher.match("record of rights", "en"), "en-ur")
    assert check_terms("جمع بندی پیش کی گئی", req, "en-ur").ok


def test_ur_en_check_is_case_insensitive_with_plurals(matcher: GlossaryMatcher) -> None:
    req = matcher.required_terms(matcher.match("گواہ اور ملزم", "ur"), "ur-en")
    res = check_terms("The Witnesses and the ACCUSED appeared.", req, "ur-en")
    assert res.ok


def test_no_required_terms() -> None:
    res = check_terms("anything", [], "en-ur")
    assert res.ok and res.terms_required == []
