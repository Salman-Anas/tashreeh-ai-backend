from app.services.glossary import GlossaryMatcher, contains_term, english_term_pattern


def _found(matcher: GlossaryMatcher, text: str, lang: str) -> list[str]:
    return [text[m.span_start : m.span_end] for m in matcher.match(text, lang)]


# -- English -------------------------------------------------------------------
def test_english_case_insensitive_whole_word(matcher: GlossaryMatcher) -> None:
    assert _found(matcher, "BAIL was refused; the Bail application failed.", "en") == ["BAIL", "Bail"]
    assert _found(matcher, "The bailiff arrived.", "en") == []


def test_english_longest_match_first(matcher: GlossaryMatcher) -> None:
    matches = matcher.match("Produce the record of rights and fundamental rights.", "en")
    assert [(m.term_en) for m in matches] == ["record of rights", "rights"]


def test_english_plurals(matcher: GlossaryMatcher) -> None:
    assert _found(matcher, "Two witnesses and one witness appeared.", "en") == ["witnesses", "witness"]
    assert _found(matcher, "Both parties agreed; the party consented.", "en") == ["parties", "party"]
    assert _found(matcher, "Several judgments were cited.", "en") == ["judgments"]
    assert _found(matcher, "The summonses were issued.", "en") == ["summonses"]


def test_english_multiword_tolerates_whitespace_and_hyphens(matcher: GlossaryMatcher) -> None:
    assert _found(matcher, "the Pakistan\n Penal  Code", "en") == ["Pakistan\n Penal  Code"]
    assert english_term_pattern("record of rights").search("record-of-rights")


def test_english_spans(matcher: GlossaryMatcher) -> None:
    text = "The High Court granted bail."
    m = matcher.match(text, "en")
    assert [(x.span_start, x.span_end) for x in m] == [(4, 14), (23, 27)]


def test_same_source_term_is_deduplicated_to_lowest_id(matcher: GlossaryMatcher) -> None:
    m = matcher.match("record of rights", "en")
    assert len(m) == 1 and m[0].term_ur == "فرد ملکیت"
    req = matcher.required_terms(m, "en-ur")
    assert req[0].target == "فرد ملکیت"
    assert req[0].alternatives == ["جمع بندی"]


def test_required_terms_unique_and_directional(matcher: GlossaryMatcher) -> None:
    m = matcher.match("bail, bail and more bail for the accused", "en")
    assert len(m) == 4
    req = matcher.required_terms(m, "en-ur")
    assert [(r.source, r.target) for r in req] == [("bail", "ضمانت"), ("accused", "ملزم")]
    ur = matcher.match("ملزم کی ضمانت", "ur")
    req_ur = matcher.required_terms(ur, "ur-en")
    assert [(r.source, r.target) for r in req_ur] == [("ملزم", "accused"), ("ضمانت", "bail")]


# -- Urdu ----------------------------------------------------------------------
def test_urdu_basic_and_spans(matcher: GlossaryMatcher) -> None:
    text = "عدالت عالیہ نے ملزم کی ضمانت منظور کی"
    assert _found(matcher, text, "ur") == ["عدالت عالیہ", "ملزم", "ضمانت"]


def test_urdu_longest_match_first(matcher: GlossaryMatcher) -> None:
    text = "مجموعہ تعزیرات پاکستان کی دفعہ"
    m = matcher.match(text, "ur")
    assert [x.term_en for x in m] == ["Pakistan Penal Code"]


def test_urdu_word_boundaries(matcher: GlossaryMatcher) -> None:
    # "ملزمہ" is not a plural/oblique ending we accept; "بےضمانتی" is a different word.
    assert _found(matcher, "بےضمانتی", "ur") == []


def test_urdu_plural_and_oblique_forms(matcher: GlossaryMatcher) -> None:
    assert _found(matcher, "ملزمان حاضر تھے", "ur") == ["ملزمان"]
    assert _found(matcher, "گواہوں نے بیان دیا", "ur") == ["گواہوں"]
    assert _found(matcher, "فیصلے کے خلاف", "ur") == ["فیصلے"]
    assert _found(matcher, "فیصلوں میں", "ur") == ["فیصلوں"]


def test_urdu_matches_through_arabic_letters_and_diacritics(matcher: GlossaryMatcher) -> None:
    text = "عدالتِ عاليه نے مُلزِم كو"  # Arabic yeh/heh/kaf + diacritics
    found = _found(matcher, text, "ur")
    assert found == ["عدالتِ عاليه", "مُلزِم"]


def test_contains_term_helpers() -> None:
    assert contains_term("The witnesses were examined.", "witness", "en")
    assert not contains_term("Witnessing it", "witness", "en")
    assert contains_term("مجموعۂ تعزیرات پاکستان", "مجموعہ تعزیرات پاکستان", "ur")


def test_empty_glossary_and_text() -> None:
    assert GlossaryMatcher([]).match("anything", "en") == []
    assert GlossaryMatcher([]).match("", "ur") == []
