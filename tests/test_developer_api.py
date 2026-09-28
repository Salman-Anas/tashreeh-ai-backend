from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.config import Settings
from app.main import app
from app.models import LLMTranslation
from app.routers import public_api
from app.services.api_keys import MemoryKeyStore, set_key_store
from app.services.glossary import GlossaryMatcher
from app.services.pipeline import TranslationPipeline
from app.services.pricing import compute_cost, price_sheet
from app.services.usage import UsageMeter, record_generation
from tests.test_pipeline import FakeTM

SESSION = "session-abc-123"


class MeteredLLM:
    async def generate_structured(self, **_: object) -> LLMTranslation:
        record_generation(SimpleNamespace(prompt_token_count=1000, candidates_token_count=200, thoughts_token_count=100))
        return LLMTranslation(translation="عدالت عالیہ نے ملزم کی ضمانت منظور کر لی۔", terms_used=[], notes=[])


@pytest.fixture
def client(matcher: GlossaryMatcher, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    store = MemoryKeyStore()
    set_key_store(store)
    monkeypatch.setattr(deps, "_pipeline", TranslationPipeline(llm=MeteredLLM(), glossary=matcher, tm=FakeTM()))  # type: ignore[arg-type]
    monkeypatch.setattr(public_api, "key_rate_limiter", deps.RateLimiter(100))
    with TestClient(app) as c:
        yield c
    set_key_store(None)


def create_key(client: TestClient, name: str = "My app") -> dict:  # type: ignore[type-arg]
    r = client.post("/api/developer/keys", json={"name": name, "session_id": SESSION})
    assert r.status_code == 201, r.text
    return r.json()


# -- pricing -------------------------------------------------------------------
def test_price_sheet_and_cost() -> None:
    s = Settings(gemini_model="gemini-3.8-flash", api_markup=1.25, _env_file=None)  # type: ignore[call-arg]
    sheet = price_sheet(s, today=date(2026, 9, 27))
    assert (sheet.input_per_m, sheet.output_per_m, sheet.embed_per_m) == (0.75, 3.75, 0.20)
    assert price_sheet(s, today=date(2027, 1, 1)).input_per_m == 1.50
    cost = compute_cost(UsageMeter(input_tokens=1_000_000, output_tokens=1_000_000, embedding_tokens=1_000_000), sheet)
    assert cost.gemini_cost_usd == pytest.approx(4.70)
    assert cost.price_usd == pytest.approx(5.875)


def test_price_overrides() -> None:
    s = Settings(price_input_per_m=1.0, price_output_per_m=2.0, price_embed_per_m=0.5, api_markup=2, _env_file=None)  # type: ignore[call-arg]
    sheet = price_sheet(s)
    assert (sheet.input_per_m, sheet.output_per_m, sheet.embed_per_m, sheet.charge(1.0)) == (1.0, 2.0, 0.5, 2.0)


def test_pricing_endpoint(client: TestClient) -> None:
    body = client.get("/api/developer/pricing").json()
    assert body["markup"] == 1.25 and body["client"]["input"] == pytest.approx(body["gemini"]["input"] * 1.25)


# -- keys ----------------------------------------------------------------------
def test_create_list_and_revoke_key(client: TestClient) -> None:
    created = create_key(client)
    assert created["key"].startswith("tsh_live_") and created["key"].startswith(created["key_prefix"])
    listed = client.get("/api/developer/keys", params={"session_id": SESSION}).json()
    assert [k["id"] for k in listed["items"]] == [created["id"]]
    assert "key" not in listed["items"][0]  # secret is never listed
    assert listed["persistent"] is False
    # Other sessions can't see or revoke it.
    assert client.get("/api/developer/keys", params={"session_id": "someone-else"}).json()["items"] == []
    assert client.delete(f"/api/developer/keys/{created['id']}", params={"session_id": "someone-else"}).status_code == 404
    assert client.delete(f"/api/developer/keys/{created['id']}", params={"session_id": SESSION}).status_code == 204


# -- public API ----------------------------------------------------------------
def test_v1_requires_valid_key(client: TestClient) -> None:
    body = {"text": "The accused was granted bail.", "direction": "en-ur"}
    r = client.post("/v1/translate", json=body)
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key"
    r = client.post("/v1/translate", json=body, headers={"Authorization": "Bearer tsh_live_notarealkey1234567890"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


def test_v1_translate_meters_usage_and_cost(client: TestClient) -> None:
    key = create_key(client)
    r = client.post(
        "/v1/translate",
        json={"text": "The High Court granted bail to the accused.", "direction": "en-ur"},
        headers={"Authorization": f"Bearer {key['key']}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["translation"].startswith("عدالت عالیہ")
    assert body["terms_missing"] == 0 and {t["source"] for t in body["terms"]} == {"High Court", "bail", "accused"}
    src = "The High Court granted bail to the accused."
    assert [src[h["start"]:h["end"]] for h in body["source_highlights"]] == ["High Court", "bail", "accused"]
    assert {body["translation"][h["start"]:h["end"]] for h in body["output_highlights"]} == {"عدالت عالیہ", "ملزم", "ضمانت"}
    assert body["references"] == []
    u = body["usage"]
    assert (u["input_tokens"], u["output_tokens"], u["total_tokens"]) == (1000, 300, 1300)
    expected = (1000 * 0.75 + 300 * 3.75) / 1e6
    assert u["gemini_cost_usd"] == pytest.approx(expected, abs=1e-6)
    assert u["price_usd"] == pytest.approx(expected * 1.25, abs=2e-6) and u["price_usd"] > u["gemini_cost_usd"]

    # X-API-Key header works too, and /v1/usage reports the totals.
    usage = client.get("/v1/usage", headers={"X-API-Key": key["key"]}).json()
    assert usage["usage"]["calls"] == 1 and usage["usage"]["total_tokens"] == 1300

    # The dashboard sees the call.
    dash = client.get("/api/developer/usage", params={"session_id": SESSION}).json()
    assert dash["summary"]["calls"] == 1
    assert dash["items"][0]["key_name"] == "My app" and dash["items"][0]["status_code"] == 200


def test_v1_failed_calls_are_recorded_but_not_billed(client: TestClient) -> None:
    key = create_key(client)
    r = client.post("/v1/translate", json={"text": "", "direction": "en-ur"}, headers={"Authorization": f"Bearer {key['key']}"})
    assert r.status_code == 422
    item = client.get("/api/developer/usage", params={"session_id": SESSION}).json()["items"][0]
    assert item["status_code"] == 422 and item["error_code"] == "empty_input" and item["billed_usd"] == 0


def test_revoked_key_is_rejected(client: TestClient) -> None:
    key = create_key(client)
    headers = {"Authorization": f"Bearer {key['key']}"}
    assert client.get("/v1/usage", headers=headers).status_code == 200
    client.delete(f"/api/developer/keys/{key['id']}", params={"session_id": SESSION})
    assert client.get("/v1/usage", headers=headers).status_code == 401


def test_cors_open_for_v1_only(client: TestClient) -> None:
    pre = {"Origin": "https://another-app.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"}
    r = client.options("/v1/translate", headers=pre)
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    r = client.options("/api/translate", headers=pre)
    assert r.status_code == 400
