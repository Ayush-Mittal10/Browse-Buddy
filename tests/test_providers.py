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
