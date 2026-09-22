"""The OpenAI and Gemini backends: how a neutral conversation reaches them.

No network. These translations are where a wrong shape becomes a rejected
request, and the Gemini one is the fiddliest of the four — its schema is a
stricter subset than anyone else's.
"""

from __future__ import annotations

import json

import pytest

from browser_agent.gemini import GeminiLLM, _declaration, _schema
from browser_agent.llm import (
    LLMError,
    Reply,
    ToolCall,
    ToolResult,
    assistant,
    build,
    resolve,
    tool_results,
    user,
)
from browser_agent.openai import OpenAILLM, _arguments
from browser_agent.tools import TOOLS


def conversation() -> list[dict]:
    reply = Reply(
        text="Clicking the cart.",
        tool_calls=[ToolCall("call_a", "click", {"ref": 4})],
        raw=None,
    )
    return [
        user("Task: open the cart"),
        assistant(reply),
        tool_results([ToolResult("call_a", "click", "Clicked [4].")]),
    ]


def shot_conversation() -> list[dict]:
    reply = Reply("", [ToolCall("call_s", "screenshot", {})], raw=None)
    return [
        user("Task: look at it"),
        assistant(reply),
        tool_results([ToolResult("call_s", "screenshot", "Taken.", ("BASE64", "image/jpeg"))]),
    ]


# --- choosing a provider ------------------------------------------------------


def test_every_provider_can_be_named() -> None:
    from browser_agent.gemini import GeminiLLM as G
    from browser_agent.llm import AnthropicLLM
    from browser_agent.ollama import OllamaLLM

    assert isinstance(build("anthropic"), AnthropicLLM)
    assert isinstance(build("openai"), OpenAILLM)
    assert isinstance(build("gemini"), G)
    assert isinstance(build("ollama"), OllamaLLM)


def test_auto_prefers_gemini_because_its_free_tier_is(monkeypatch) -> None:
    monkeypatch.setattr("browser_agent.config.GEMINI_API_KEY", "g")
    monkeypatch.setattr("browser_agent.config.API_KEY", "a")
    monkeypatch.setattr("browser_agent.config.OPENAI_API_KEY", "o")
    assert resolve("auto") == "gemini"


def test_auto_walks_down_to_whatever_has_a_key(monkeypatch) -> None:
    monkeypatch.setattr("browser_agent.config.GEMINI_API_KEY", "")
    monkeypatch.setattr("browser_agent.config.API_KEY", "a")
    monkeypatch.setattr("browser_agent.config.OPENAI_API_KEY", "")
    assert resolve("auto") == "anthropic"

    monkeypatch.setattr("browser_agent.config.API_KEY", "")
    monkeypatch.setattr("browser_agent.config.OPENAI_API_KEY", "o")
    assert resolve("auto") == "openai"


def test_auto_falls_back_to_the_local_model(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(f"browser_agent.config.{name}", "")
    assert resolve("auto") == "ollama"


def test_a_key_handed_in_counts_as_having_one(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(f"browser_agent.config.{name}", "")
    assert build("openai", api_key="sk-given").api_key == "sk-given"


def test_a_handed_in_key_never_touches_the_environment(monkeypatch) -> None:
    # The web UI serves several people at once; one person's key must not
    # become process-wide state.
    monkeypatch.setattr("browser_agent.config.OPENAI_API_KEY", "from-config")
    assert build("openai", api_key="from-caller").api_key == "from-caller"
    assert build("openai").api_key == "from-config"


def test_an_unknown_provider_lists_the_real_ones() -> None:
    with pytest.raises(LLMError) as exc:
        build("skynet")
    for name in ("anthropic", "openai", "gemini", "ollama"):
        assert name in str(exc.value)


# --- OpenAI -------------------------------------------------------------------


def test_openai_renders_the_system_prompt_first() -> None:
    rendered = OpenAILLM()._render("be helpful", conversation())
    assert rendered[0] == {"role": "system", "content": "be helpful"}


def test_openai_serialises_tool_arguments_as_json() -> None:
    rendered = OpenAILLM()._render("sys", conversation())

    turn = rendered[2]
    assert turn["role"] == "assistant"
    call = turn["tool_calls"][0]
    assert call["id"] == "call_a"
    assert call["type"] == "function"
    # Arguments go over the wire as a string, both directions.
    assert json.loads(call["function"]["arguments"]) == {"ref": 4}


def test_openai_answers_each_call_by_id() -> None:
    rendered = OpenAILLM()._render("sys", conversation())
    assert rendered[-1] == {"role": "tool", "tool_call_id": "call_a", "content": "Clicked [4]."}


def test_openai_puts_a_screenshot_after_every_tool_result() -> None:
    rendered = OpenAILLM()._render("sys", shot_conversation())

    # The tool message comes first, so no call is left without its answer.
    assert rendered[-2]["role"] == "tool"
    image = rendered[-1]
    assert image["role"] == "user"
    assert image["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,BASE64")


def test_an_assistant_turn_with_no_text_sends_null_not_empty() -> None:
    rendered = OpenAILLM()._render("sys", [assistant(Reply("", [ToolCall("a", "click", {})]))])
    assert rendered[1]["content"] is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [('{"ref": 4}', {"ref": 4}), ({"ref": 4}, {"ref": 4}), ("nonsense", {}), (None, {})],
)
def test_openai_arguments_are_coerced(raw, expected: dict) -> None:
    assert _arguments(raw) == expected


async def test_openai_without_a_key_says_so() -> None:
    backend = OpenAILLM(api_key="")
    backend.api_key = ""
    with pytest.raises(LLMError) as exc:
        await backend.complete(system="s", tools=[], messages=[user("hi")])
    assert "OPENAI_API_KEY" in str(exc.value)


# --- Gemini schema ------------------------------------------------------------


def test_unsupported_schema_keys_are_dropped() -> None:
    cleaned = _schema(
        {
            "type": "object",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"ref": {"type": "integer", "description": "the number"}},
            "required": ["ref"],
        }
    )

    # additionalProperties is the one that gets a request rejected.
    assert "additionalProperties" not in cleaned
    assert "$schema" not in cleaned
    assert cleaned["required"] == ["ref"]
    assert cleaned["properties"]["ref"] == {"type": "integer", "description": "the number"}


def test_nested_schemas_are_cleaned_too() -> None:
    cleaned = _schema(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": False},
                }
            },
        }
    )
    assert "additionalProperties" not in cleaned["properties"]["items"]["items"]


def test_enums_survive_because_they_are_supported() -> None:
    cleaned = _schema({"type": "string", "enum": ["down", "up"]})
    assert cleaned["enum"] == ["down", "up"]


def test_a_tool_with_no_arguments_declares_no_parameters() -> None:
    # An empty properties object is not the same as no parameters, and the
    # empty one is what gets rejected.
    declaration = _declaration(
        {
            "name": "go_back",
            "description": "Go back.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    )
    assert "parameters" not in declaration
    assert declaration["name"] == "go_back"


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["name"])
def test_every_real_tool_survives_translation(tool: dict) -> None:
    declaration = _declaration(tool)
    assert declaration["name"] == tool["name"]
    assert declaration["description"]

    blob = json.dumps(declaration)
    assert "additionalProperties" not in blob

    if tool["input_schema"].get("properties"):
        assert set(declaration["parameters"]["properties"]) == set(
            tool["input_schema"]["properties"]
        )


# --- Gemini conversation ------------------------------------------------------


def test_gemini_calls_the_assistant_model() -> None:
    contents = GeminiLLM()._render(conversation())
    assert [c["role"] for c in contents] == ["user", "model", "user"]


def test_gemini_renders_a_tool_call_as_a_function_call_part() -> None:
    contents = GeminiLLM()._render(conversation())
    parts = contents[1]["parts"]
    assert parts[0] == {"text": "Clicking the cart."}
    assert parts[1] == {"functionCall": {"name": "click", "args": {"ref": 4}}}


def test_gemini_sends_a_result_as_a_function_response_in_a_user_turn() -> None:
    contents = GeminiLLM()._render(conversation())
    assert contents[2]["parts"][0] == {
        "functionResponse": {"name": "click", "response": {"result": "Clicked [4]."}}
    }


def test_gemini_attaches_a_screenshot_beside_its_response() -> None:
    contents = GeminiLLM()._render(shot_conversation())
    parts = contents[-1]["parts"]
    assert "functionResponse" in parts[0]
    assert parts[1] == {"inlineData": {"mimeType": "image/jpeg", "data": "BASE64"}}


def test_an_empty_model_turn_still_has_a_part() -> None:
    # A turn with no parts at all is not one Gemini will accept.
    contents = GeminiLLM()._render([assistant(Reply(""))])
    assert contents[0]["parts"] == [{"text": ""}]


async def test_gemini_without_a_key_says_so() -> None:
    backend = GeminiLLM(api_key="")
    backend.api_key = ""
    with pytest.raises(LLMError) as exc:
        await backend.complete(system="s", tools=[], messages=[user("hi")])
    assert "GEMINI_API_KEY" in str(exc.value)


# --- Gemini thought signatures ------------------------------------------------


def test_a_model_turn_is_replayed_exactly_as_it_arrived() -> None:
    # Gemini attaches a thoughtSignature to every functionCall and rejects the
    # next request if it comes back without one. Rebuilding the part by hand
    # loses it, so the turn has to be replayed verbatim.
    raw = [
        {"text": "Searching."},
        {
            "functionCall": {"name": "click", "args": {"ref": 4}},
            "thoughtSignature": "SIGNATURE",
        },
    ]
    messages = [assistant(Reply("Searching.", [ToolCall("call_a", "click", {"ref": 4})], raw))]

    contents = GeminiLLM()._render(messages)

    assert contents[0]["parts"] == raw
    assert contents[0]["parts"][1]["thoughtSignature"] == "SIGNATURE"


def test_a_turn_without_raw_parts_is_still_rebuilt() -> None:
    # Synthetic turns — a cancelled call, a replayed transcript — have no raw.
    messages = [assistant(Reply("hi", [ToolCall("a", "click", {"ref": 1})], raw=None))]
    parts = GeminiLLM()._render(messages)[0]["parts"]
    assert parts == [{"text": "hi"}, {"functionCall": {"name": "click", "args": {"ref": 1}}}]


def test_the_reply_carries_the_raw_parts_forward() -> None:
    # Whatever complete() returns as raw is what _render sends back, so the two
    # have to agree on the shape.
    raw = [{"functionCall": {"name": "click", "args": {}}, "thoughtSignature": "S"}]
    assert GeminiLLM()._render([assistant(Reply("", [], raw))])[0]["parts"] == raw


# --- retrying -----------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def post(self, url, **kwargs):
        self.calls += 1
        return self.responses.pop(0) if self.responses else FakeResponse(200)


async def test_a_good_response_is_not_retried() -> None:
    from browser_agent.llm import post_with_retry

    client = FakeClient(FakeResponse(200))
    response = await post_with_retry(client, "https://x/y", json={}, headers={})

    assert response.status_code == 200
    assert client.calls == 1


async def test_a_refusal_is_not_retried() -> None:
    from browser_agent.llm import post_with_retry

    # 400 means "no", not "not now" — retrying it just wastes the run's clock.
    client = FakeClient(FakeResponse(400))
    response = await post_with_retry(client, "https://x/y", json={}, headers={})

    assert response.status_code == 400
    assert client.calls == 1


async def test_high_demand_is_retried_and_can_succeed() -> None:
    from browser_agent.llm import post_with_retry

    client = FakeClient(FakeResponse(503), FakeResponse(200))
    response = await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)

    assert response.status_code == 200
    assert client.calls == 2


async def test_retrying_gives_up_and_hands_back_the_last_failure() -> None:
    from browser_agent.llm import post_with_retry

    client = FakeClient(FakeResponse(503), FakeResponse(503), FakeResponse(503))
    response = await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)

    assert response.status_code == 503
    assert client.calls == 3


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_every_transient_status_is_retried(status: int) -> None:
    from browser_agent.llm import post_with_retry

    client = FakeClient(FakeResponse(status), FakeResponse(200))
    await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)
    assert client.calls == 2


def test_the_servers_own_retry_advice_is_used_but_capped() -> None:
    from browser_agent.llm import _retry_after

    assert _retry_after(FakeResponse(429, {"retry-after": "7"}), 2.0) == 7.0
    # A server asking for an hour must not stall the whole run.
    assert _retry_after(FakeResponse(429, {"retry-after": "3600"}), 2.0) == 30.0
    assert _retry_after(FakeResponse(429, {"retry-after": "soon"}), 2.0) == 2.0
    assert _retry_after(FakeResponse(429), 2.0) == 2.0


# --- running out of free capacity ---------------------------------------------


def test_the_suggested_model_is_never_the_one_that_just_failed() -> None:
    # The first version of this message suggested gemini-3.1-flash-lite, which
    # stopped making sense the day that became the default: it told you to
    # switch to the model that had just run out of capacity.
    from browser_agent.gemini import _BY_CONTENTION, _try_instead

    for model in _BY_CONTENTION:
        assert _try_instead(model) != model
        assert _try_instead(model) in _BY_CONTENTION


def test_an_unknown_model_still_gets_a_suggestion() -> None:
    from browser_agent.gemini import _try_instead

    assert _try_instead("gemini-9-something") == "gemini-3.1-flash-lite"


def test_the_default_model_is_the_least_contended_one() -> None:
    from browser_agent import config
    from browser_agent.gemini import _BY_CONTENTION

    assert _BY_CONTENTION[0] == config.GEMINI_MODEL
