from app.services.glossary import GlossaryMatcher
from eval.benchmark import corpus_chrf, sentence_chrf, term_counts


def test_chrf_normalizes_urdu_equally() -> None:
    ref = "عدالت عالیہ نے ملزم کی ضمانت منظور کر لی۔"
    assert sentence_chrf(ref, ref, "en-ur") == 100.0
    # Arabic letter forms + diacritics must not be penalised.
    assert sentence_chrf("عدالتِ عاليه نے ملزم كی ضمانت منظور کر لی۔", ref, "en-ur") == 100.0
    assert sentence_chrf("", ref, "en-ur") is None
    assert corpus_chrf(["The witness appeared."], ["The witness appeared."], "ur-en") == 100.0


def test_term_counts(matcher: GlossaryMatcher) -> None:
    req = matcher.required_terms(matcher.match("The accused was granted bail.", "en"), "en-ur")
    assert term_counts("ملزم کی ضمانت", req, "en-ur") == (2, 2)
    assert term_counts("مشتبہ کی رہائی", req, "en-ur") == (0, 2)
