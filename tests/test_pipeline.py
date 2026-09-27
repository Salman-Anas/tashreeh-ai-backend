from __future__ import annotations

import asyncio

import pytest

from app.models import Direction, LLMTermUsed, LLMTranslation, TMExample
from app.services.glossary import GlossaryMatcher
from app.services.pipeline import TranslationPipeline, build_prompt, split_paragraphs


class FakeLLM:
    """Returns queued responses in order and records every prompt."""

    def __init__(self, responses: list[str] | None = None, by_source: dict[str, str] | None = None) -> None:
        self.responses = list(responses or [])
        self.by_source = by_source or {}
        self.calls: list[str] = []

    async def generate_structured(self, *, system_instruction, contents, schema, temperature=None):  # type: ignore[no-untyped-def]
        self.calls.append(contents)
        await asyncio.sleep(0)
        for src, out in self.by_source.items():
            if src in contents.split("SOURCE TEXT")[1].split(">>>")[0]:
                return LLMTranslation(translation=out, terms_used=[], notes=[])
        text = self.responses.pop(0)
        return LLMTranslation(translation=text, terms_used=[LLMTermUsed(source="x", target="y")], notes=["n"])


class FakeTM:
    def __init__(self, examples: list[TMExample] | None = None) -> None:
        self.examples = examples or []
        self.calls: list[tuple[str, Direction]] = []

    async def retrieve(self, text: str, direction: Direction) -> list[TMExample]:
        self.calls.append((text, direction))
        return self.examples


EXAMPLE = TMExample(id=7, text_en="The accused was granted bail.", text_ur="ملزم کی ضمانت منظور ہوئی۔", similarity=0.91)


@pytest.mark.asyncio
async def test_happy_path_uses_terms_and_examples(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(["عدالت عالیہ نے ملزم کی ضمانت منظور کر لی۔"])
    tm = FakeTM([EXAMPLE])
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=tm)

    res = await p.translate("The High Court granted bail to the accused.", "en-ur")

    assert res.translation == "عدالت عالیہ نے ملزم کی ضمانت منظور کر لی۔"
    assert [t.source for t in res.terms_required] == ["High Court", "bail", "accused"]
    assert res.terms_missing == []
    assert not res.retried
    assert res.tm_examples_used == [EXAMPLE]
    assert res.notes == ["n"]
    # Prompt carries required terms and examples.
    prompt = llm.calls[0]
    assert "- bail → ضمانت" in prompt and "- High Court → عدالت عالیہ" in prompt
    assert "ملزم کی ضمانت منظور ہوئی۔" in prompt
    # Highlights in both source and output.
    assert len(res.glossary_matches) == 3
    assert {m.term_en for m in res.output_matches} == {"High Court", "bail", "accused"}


@pytest.mark.asyncio
async def test_missing_term_triggers_one_retry_that_fixes_it(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(["مشتبہ کی ضمانت منظور ہوئی۔", "ملزم کی ضمانت منظور ہوئی۔"])
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=FakeTM())

    res = await p.translate("The accused was granted bail.", "en-ur")

    assert len(llm.calls) == 2
    assert "MISSING REQUIRED TERMS" in llm.calls[1] and "- accused → ملزم" in llm.calls[1]
    assert "- bail → ضمانت" not in llm.calls[1]  # only the missing term is listed
    assert res.retried
    assert res.terms_missing == []
    assert res.translation == "ملزم کی ضمانت منظور ہوئی۔"


@pytest.mark.asyncio
async def test_still_missing_after_retry_is_reported(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(["مشتبہ کی ضمانت۔", "مشتبہ کی ضمانت منظور۔"])
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=FakeTM())

    res = await p.translate("The accused was granted bail.", "en-ur")

    assert len(llm.calls) == 2  # retried once, not more
    assert [t.source for t in res.terms_missing] == ["accused"]
    assert [t.source for t in res.terms_found] == ["bail"]


@pytest.mark.asyncio
async def test_worse_retry_is_discarded(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(["مشتبہ کی ضمانت۔", "کچھ اور۔"])
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=FakeTM())
    res = await p.translate("The accused was granted bail.", "en-ur")
    assert res.translation == "مشتبہ کی ضمانت۔"


@pytest.mark.asyncio
async def test_no_glossary_terms_means_no_retry(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(["کچھ بھی"])
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=FakeTM())
    res = await p.translate("Nothing legal here.", "en-ur")
    assert len(llm.calls) == 1 and res.terms_required == [] and "(none)" in llm.calls[0]


@pytest.mark.asyncio
async def test_paragraphs_translated_separately_and_reassembled_in_order(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(by_source={"First": "پہلا", "Second": "دوسرا", "Third": "تیسرا"})
    tm = FakeTM()
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=tm, concurrency=2)

    res = await p.translate("First para.\n\nSecond para.\n\n\nThird para.", "en-ur")

    assert res.translation == "پہلا\n\nدوسرا\n\nتیسرا"
    assert len(llm.calls) == 3 and len(tm.calls) == 3
    assert len(res.segments) == 3


@pytest.mark.asyncio
async def test_ur_en_direction(matcher: GlossaryMatcher) -> None:
    llm = FakeLLM(["The witness appeared in the High Court."])
    p = TranslationPipeline(llm=llm, glossary=matcher, tm=FakeTM([EXAMPLE]))
    res = await p.translate("گواہ عدالت عالیہ میں پیش ہوا۔", "ur-en")
    assert res.terms_missing == []
    assert "DIRECTION: Urdu → English" in llm.calls[0]
    assert "- گواہ → witness" in llm.calls[0]
    # Examples are shown source-first for the direction.
    assert "[1] Urdu: ملزم کی ضمانت منظور ہوئی۔\n    English: The accused was granted bail." in llm.calls[0]


def test_split_paragraphs() -> None:
    assert split_paragraphs("  a\nb \n\n  c\n \n\nd  ") == ["a\nb", "c", "d"]
    assert split_paragraphs("") == []


def test_build_prompt_contains_source() -> None:
    prompt = build_prompt("Section 302 PPC", "en-ur", [], [])
    assert "SOURCE TEXT (English):\n<<<\nSection 302 PPC\n>>>" in prompt
