from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.main import app
from app.models import LLMExplanations, LLMTermExplanation, LLMTranslation
from app.routers import public_api
from app.services.api_keys import MemoryKeyStore, set_key_store
from app.services.explain import TermExplainer
from app.services.glossary import GlossaryMatcher
from app.services.pipeline import TranslationPipeline
from app.services.usage import record_generation
from tests.test_pipeline import FakeTM

SESSION = "session-explain-1"


class FakeExplainLLM:
    """Explains every term_id in the prompt; can translate too. Records token usage."""

    def __init__(self, skip: set[int] | None = None) -> None:
        self.calls: list[str] = []
        self.skip = skip or set()

    async def generate_structured(self, *, system_instruction, contents, schema, temperature=None):  # type: ignore[no-untyped-def]
        self.calls.append(contents)
        record_generation(SimpleNamespace(prompt_token_count=300, candidates_token_count=120, thoughts_token_count=0))
        if schema is LLMTranslation:
            return LLMTranslation(translation="عدالت عالیہ نے ملزم کی ضمانت منظور کر لی۔", terms_used=[], notes=[])
        ids = [int(i) for i in re.findall(r"term_id (\d+)", contents)]
        return LLMExplanations(
            explanations=[
                LLMTermExplanation(
                    term_id=i,
                    simple_en=f"Plain meaning of term {i}.",
                    simple_ur="آسان مطلب",
                    example_en=f"Example for term {i}.",
                    example_ur="مثال",
                )
                for i in ids
                if i not in self.skip
            ]
        )


# -- service -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explains_in_order_and_caches(matcher: GlossaryMatcher) -> None:
    llm = FakeExplainLLM()
    ex = TermExplainer(llm)
    terms = matcher.by_ids([2, 1])  # accused, bail

    first, missing = await ex.explain(terms)
    assert [e.term_en for e in first] == ["accused", "bail"] and missing == []
    assert first[0].simple_ur == "آسان مطلب" and not first[0].cached
    assert "category: criminal" in llm.calls[0]

    second, _ = await ex.explain(terms)
    assert len(llm.calls) == 1  # served from cache, no second model call
    assert all(e.cached for e in second)

    # Only the uncached term goes to the model.
    await ex.explain(matcher.by_ids([1, 3]))
    assert len(llm.calls) == 2 and "term_id 3" in llm.calls[1] and "term_id 1" not in llm.calls[1]


@pytest.mark.asyncio
async def test_missing_explanations_are_reported(matcher: GlossaryMatcher) -> None:
    ex = TermExplainer(FakeExplainLLM(skip={2}))
    got, missing = await ex.explain(matcher.by_ids([1, 2]))
    assert [e.term_id for e in got] == [1] and missing == [2]


def test_glossary_lookup_helpers(matcher: GlossaryMatcher) -> None:
    assert [t.term_en for t in matcher.by_ids([3, 999, 1, 3])] == ["witness", "bail"]
    assert matcher.find_by_text("  BAIL ").term_en == "bail"  # type: ignore[union-attr]
    assert matcher.find_by_text("عدالتِ عاليه").term_en == "High Court"  # type: ignore[union-attr]
    assert matcher.find_by_text("pizza") is None


# -- API -------------------------------------------------------------------------
@pytest.fixture
def client(matcher: GlossaryMatcher, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    llm = FakeExplainLLM()
    set_key_store(MemoryKeyStore())
    monkeypatch.setattr(deps, "_pipeline", TranslationPipeline(llm=llm, glossary=matcher, tm=FakeTM()))  # type: ignore[arg-type]
    monkeypatch.setattr(deps, "_explainer", TermExplainer(llm))
    monkeypatch.setattr(deps, "rate_limiter", deps.RateLimiter(100))
    monkeypatch.setattr(public_api, "key_rate_limiter", deps.RateLimiter(100))
    with TestClient(app) as c:
        deps.glossary.set_terms(matcher.terms)
        yield c
    set_key_store(None)


def api_key(client: TestClient) -> str:
    return client.post("/api/developer/keys", json={"name": "t", "session_id": SESSION}).json()["key"]


def test_site_explain_endpoint(client: TestClient) -> None:
    r = client.post("/api/explain", json={"term_ids": [1, 2, 999]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["term_en"] for e in body["explanations"]] == ["bail", "accused"]
    assert body["missing"] == [999]
    assert body["disclaimer_en"] and body["disclaimer_ur"]
    assert client.post("/api/explain", json={"term_ids": [999]}).status_code == 404
    assert client.post("/api/explain", json={"term_ids": []}).status_code == 422


def test_v1_translate_with_explain_bills_both_calls(client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {api_key(client)}"}
    body = {"text": "The High Court granted bail to the accused.", "direction": "en-ur"}

    plain = client.post("/v1/translate", json=body, headers=headers).json()
    assert plain["explanations"] is None and plain["usage"]["total_tokens"] == 420

    r = client.post("/v1/translate", json={**body, "explain": True}, headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert {e["term_en"] for e in data["explanations"]} == {"High Court", "bail", "accused"}
    assert data["usage"]["total_tokens"] == 840  # translation + explanation call
    assert data["usage"]["price_usd"] > data["usage"]["gemini_cost_usd"] > 0


def test_v1_explain_endpoint(client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {api_key(client)}"}
    r = client.post("/v1/explain", json={"terms": ["bail", "ملزم", "bail", "pizza"]}, headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert [e["term_en"] for e in data["explanations"]] == ["bail", "accused"]
    assert data["not_found"] == ["pizza"]
    assert data["usage"]["total_tokens"] == 420

    # Second time it is cached: no tokens, nothing billed.
    again = client.post("/v1/explain", json={"terms": ["bail"]}, headers=headers).json()
    assert again["explanations"][0]["cached"] is True and again["usage"]["total_tokens"] == 0 and again["usage"]["price_usd"] == 0

    r = client.post("/v1/explain", json={"terms": ["pizza"]}, headers=headers)
    assert r.status_code == 404
    assert client.post("/v1/explain", json={"terms": ["bail"]}).status_code == 401

    items = client.get("/api/developer/usage", params={"session_id": SESSION}).json()["items"]
    assert {i["endpoint"] for i in items} == {"/v1/explain"}
