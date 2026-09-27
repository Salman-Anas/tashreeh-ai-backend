from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.config import Settings
from app.models import LLMTranslation
from app.services.gemini_client import GeminiClient, GeminiError, GeminiRateLimitError
from app.services.usage import track_usage


def api_error(code: int, message: str = "boom") -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


class FakeModels:
    def __init__(self, name: str, behaviour: list[object]) -> None:
        self.name = name
        self.behaviour = behaviour
        self.calls = 0

    async def generate_content(self, **_: object) -> object:
        self.calls += 1
        outcome = self.behaviour.pop(0) if self.behaviour else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            parsed=LLMTranslation(translation=f"from {self.name}", terms_used=[], notes=[]),
            text=None,
            usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=40, thoughts_token_count=10),
        )


def make_client(behaviours: list[list[object]]) -> tuple[GeminiClient, list[FakeModels]]:
    keys = [f"key-{i}" for i in range(len(behaviours))]
    settings = Settings(gemini_api_key=keys[0], gemini_api_keys=",".join(keys[1:]), _env_file=None)  # type: ignore[call-arg]
    client = GeminiClient(settings)
    models = [FakeModels(k, b) for k, b in zip(keys, behaviours)]
    client._clients = {i: SimpleNamespace(aio=SimpleNamespace(models=m)) for i, m in enumerate(models)}  # type: ignore[assignment]
    return client, models


async def gen(client: GeminiClient) -> str:
    out = await client.generate_structured(system_instruction="s", contents="c", schema=LLMTranslation)
    return out.translation


def test_keys_are_collected_and_deduplicated() -> None:
    s = Settings(gemini_api_key="a", gemini_api_key_2="b", gemini_api_keys=" c , a ,", _env_file=None)  # type: ignore[call-arg]
    assert s.gemini_keys == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_rate_limited_key_fails_over_to_second_key() -> None:
    client, models = make_client([[api_error(429, "quota")], []])
    assert await gen(client) == "from key-1"
    # The second key is now active, and the first is skipped while cooling down.
    assert await gen(client) == "from key-1"
    assert models[0].calls == 1 and models[1].calls == 2


@pytest.mark.asyncio
async def test_invalid_key_fails_over() -> None:
    client, _ = make_client([[api_error(400, "API key not valid. Please pass a valid API key.")], []])
    assert await gen(client) == "from key-1"
    client, _ = make_client([[api_error(403, "permission denied")], []])
    assert await gen(client) == "from key-1"


@pytest.mark.asyncio
async def test_all_keys_rate_limited_raises_rate_limit_error() -> None:
    client, _ = make_client([[api_error(429)], [api_error(429)]])
    with pytest.raises(GeminiRateLimitError):
        await gen(client)


@pytest.mark.asyncio
async def test_non_key_errors_do_not_fail_over() -> None:
    client, models = make_client([[api_error(404, "model not found")], []])
    with pytest.raises(GeminiError, match="model not found|not found"):
        await gen(client)
    assert models[1].calls == 0


@pytest.mark.asyncio
async def test_transient_error_is_retried_on_same_key() -> None:
    client, models = make_client([[api_error(503)], []])
    assert await gen(client) == "from key-0"
    assert models[0].calls == 2 and models[1].calls == 0


@pytest.mark.asyncio
async def test_usage_is_metered() -> None:
    client, _ = make_client([[]])
    with track_usage() as meter:
        await gen(client)
        await gen(client)
    assert (meter.input_tokens, meter.output_tokens, meter.thinking_tokens, meter.gemini_calls) == (200, 100, 20, 2)
    assert meter.total_tokens == 300
    # Outside a tracking context nothing breaks.
    await gen(client)
