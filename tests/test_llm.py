"""The provider layer: how a neutral conversation is rendered for each backend.

No network. What matters here is the translation, since that is where a wrong
shape turns into a rejected request at the worst possible moment.
"""

from __future__ import annotations

import pytest

from browser_agent.llm import (
    AnthropicLLM,
    LLMError,
    Reply,
    ToolCall,
    ToolResult,
    _result_block,
    assistant,
    build,
    tool_results,
    user,
)
from browser_agent.ollama import OllamaLLM, _arguments

TOOLS = [
    {
        "name": "click",
        "description": "Click an element.",
        "input_schema": {
            "type": "object",
            "properties": {"ref": {"type": "integer"}},
            "required": ["ref"],
        },
    }
]


def conversation() -> list[dict]:
    reply = Reply(
        text="Clicking the cart.",
        tool_calls=[ToolCall("call_a", "click", {"ref": 4})],
        raw=[
            {"type": "text", "text": "Clicking the cart."},
            {"type": "tool_use", "id": "call_a", "name": "click", "input": {"ref": 4}},
        ],
    )
    return [
        user("Task: open the cart"),
        assistant(reply),
        tool_results([ToolResult("call_a", "click", "Clicked [4].")]),
    ]


# --- picking a backend --------------------------------------------------------


def test_named_providers(monkeypatch) -> None:
    assert isinstance(build("anthropic"), AnthropicLLM)
    assert isinstance(build("ollama"), OllamaLLM)


def test_auto_prefers_a_hosted_model_when_a_key_is_present(monkeypatch) -> None:
    # Every key has to be controlled, not just one: a real key in the
    # developer's own .env would otherwise decide the answer.
    monkeypatch.setattr("browser_agent.config.GEMINI_API_KEY", "")
    monkeypatch.setattr("browser_agent.config.OPENAI_API_KEY", "")
    monkeypatch.setattr("browser_agent.config.API_KEY", "sk-ant-something")
    assert isinstance(build("auto"), AnthropicLLM)


def test_auto_falls_back_to_the_local_model(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        monkeypatch.setattr(f"browser_agent.config.{name}", "")
    assert isinstance(build("auto"), OllamaLLM)


def test_an_unknown_provider_says_what_there_is() -> None:
    with pytest.raises(LLMError) as exc:
        build("gpt-9")
    assert "anthropic" in str(exc.value)
    assert "ollama" in str(exc.value)


def test_a_model_name_overrides_the_default() -> None:
    assert build("anthropic", "claude-sonnet-5").name == "claude-sonnet-5"
    assert build("ollama", "qwen3:14b").name == "qwen3:14b"


# --- Anthropic rendering ------------------------------------------------------


def test_anthropic_replays_the_assistant_turn_verbatim() -> None:
    rendered = AnthropicLLM()._render(conversation())

    assert rendered[0] == {"role": "user", "content": "Task: open the cart"}
    # The raw blocks go back exactly as they came — a thinking block that loses
    # its signature gets the next request rejected.
    assert rendered[1]["role"] == "assistant"
    assert rendered[1]["content"][1]["id"] == "call_a"


def test_anthropic_sends_tool_results_as_a_user_turn() -> None:
    rendered = AnthropicLLM()._render(conversation())

    assert rendered[2]["role"] == "user"
    assert rendered[2]["content"][0]["type"] == "tool_result"
    assert rendered[2]["content"][0]["tool_use_id"] == "call_a"


def test_a_plain_result_is_a_string() -> None:
    block = _result_block(ToolResult("a", "click", "Clicked [4]."))
    assert block == {"type": "tool_result", "tool_use_id": "a", "content": "Clicked [4]."}


def test_an_image_rides_inside_its_result() -> None:
    block = _result_block(ToolResult("a", "screenshot", "Taken.", ("BASE64", "image/jpeg")))

    parts = block["content"]
    assert parts[0] == {"type": "text", "text": "Taken."}
    assert parts[1]["type"] == "image"
    assert parts[1]["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "BASE64",
    }


def test_anthropic_declares_it_can_see() -> None:
    assert AnthropicLLM().supports_images is True


# --- Ollama rendering ---------------------------------------------------------


def test_ollama_puts_the_system_prompt_first() -> None:
    rendered = OllamaLLM()._render("be helpful", conversation())
    assert rendered[0] == {"role": "system", "content": "be helpful"}


def test_ollama_rebuilds_the_assistant_turn_from_the_neutral_form() -> None:
    rendered = OllamaLLM()._render("sys", conversation())

    turn = rendered[2]
    assert turn["role"] == "assistant"
    assert turn["content"] == "Clicking the cart."
    assert turn["tool_calls"] == [{"function": {"name": "click", "arguments": {"ref": 4}}}]


def test_ollama_sends_one_tool_message_per_result() -> None:
    messages = conversation()
    messages[-1] = tool_results(
        [ToolResult("a", "click", "Clicked."), ToolResult("b", "scroll", "Scrolled.")]
    )
    rendered = OllamaLLM()._render("sys", messages)

    tail = rendered[-2:]
    assert [m["role"] for m in tail] == ["tool", "tool"]
    assert [m["tool_name"] for m in tail] == ["click", "scroll"]
    assert [m["content"] for m in tail] == ["Clicked.", "Scrolled."]


def test_an_assistant_turn_without_calls_has_no_tool_calls_key() -> None:
    rendered = OllamaLLM()._render("sys", [assistant(Reply("just talking"))])
    assert "tool_calls" not in rendered[1]


def test_ollama_declares_it_cannot_see() -> None:
    assert OllamaLLM().supports_images is False


def test_ollama_defaults_come_from_config() -> None:
    from browser_agent import config

    backend = OllamaLLM()
    assert backend.name == config.OLLAMA_MODEL
    assert backend.num_ctx == config.OLLAMA_NUM_CTX
    # Thinking off by default: measured 27.7s against 1.7s for the same call.
    assert backend.think is False


def test_a_trailing_slash_on_the_host_is_dropped() -> None:
    assert OllamaLLM(host="http://localhost:11434/").host == "http://localhost:11434"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"ref": 4}, {"ref": 4}),
        ('{"ref": 4}', {"ref": 4}),
        ("not json", {}),
        ("[1, 2]", {}),
        (None, {}),
        (42, {}),
    ],
)
def test_arguments_are_coerced_to_a_dict(raw, expected: dict) -> None:
    assert _arguments(raw) == expected


# --- talking to a server that is not there ------------------------------------


async def test_an_unreachable_ollama_explains_itself() -> None:
    backend = OllamaLLM(model="qwen3:8b", host="http://127.0.0.1:1")
    with pytest.raises(LLMError) as exc:
        await backend.complete(system="s", tools=TOOLS, messages=[user("hi")])

    message = str(exc.value)
    assert "Could not reach Ollama" in message
    assert "qwen3:8b" in message
    await backend.close()
