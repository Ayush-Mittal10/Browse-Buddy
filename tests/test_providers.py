"""The OpenAI and Gemini backends: how a neutral conversation reaches them.

No network. These translations are where a wrong shape becomes a rejected
request, and the Gemini one is the fiddliest of the four — its schema is a
stricter subset than anyone else's.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from browser_agent import config
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
    backend.keys = []   # api_key is now derived from the list
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
    """Answers with each given response in turn. An exception is raised instead,
    which is how a connection that times out or drops arrives."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def post(self, url, **kwargs):
        self.calls += 1
        answer = self.responses.pop(0) if self.responses else FakeResponse(200)
        if isinstance(answer, BaseException):
            raise answer
        return answer


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


async def test_a_read_timeout_is_retried_and_can_succeed() -> None:
    # Seen on a real run: three steps in, Gemini took longer than the timeout
    # once and the whole task ended on it. A slow answer from a free tier under
    # load is the most ordinary transient failure there is, and it used to be
    # the only kind that escaped this loop entirely.
    from browser_agent.llm import post_with_retry

    client = FakeClient(httpx.ReadTimeout("timed out"), FakeResponse(200))
    response = await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)

    assert response.status_code == 200
    assert client.calls == 2


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("too slow"),
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("dropped"),
    ],
)
async def test_every_transport_failure_is_retried(error: Exception) -> None:
    from browser_agent.llm import post_with_retry

    client = FakeClient(error, FakeResponse(200))
    await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)
    assert client.calls == 2


async def test_a_connection_that_never_works_is_raised_not_swallowed() -> None:
    # The caller turns this into "Could not reach <provider>". Returning None
    # instead would surface as an AttributeError somewhere less obvious.
    from browser_agent.llm import post_with_retry

    client = FakeClient(*[httpx.ReadTimeout("timed out")] * 3)
    with pytest.raises(httpx.ReadTimeout):
        await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)

    assert client.calls == 3


async def test_a_bad_request_is_not_retried_even_after_a_timeout() -> None:
    # The retry must not turn a definite "no" into three of them.
    from browser_agent.llm import post_with_retry

    client = FakeClient(httpx.ReadTimeout("timed out"), FakeResponse(400))
    response = await post_with_retry(client, "https://x/y", json={}, headers={}, delay=0)

    assert response.status_code == 400
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


def test_the_default_model_is_somewhere_in_the_suggestions() -> None:
    from browser_agent import config
    from browser_agent.gemini import _BY_CONTENTION

    # The list is ordered by how contended each model is, and the default is
    # chosen for how well it runs a task — so the default is not necessarily
    # first, but suggesting a model nobody can reach would be useless.
    assert config.GEMINI_MODEL in _BY_CONTENTION


# --- more than one key --------------------------------------------------------


class _Client:
    """Answers with scripted status codes and records which key was used.

    A stream rather than a post, because that is what the Gemini backend calls
    now — the key rotation has to keep working over the streamed transport.
    """

    def __init__(self, *codes: int):
        self.codes = list(codes)
        self.keys_used: list[str] = []

    def stream(self, method, url, json=None, headers=None, **kwargs):
        self.keys_used.append(headers["x-goog-api-key"])
        code = self.codes.pop(0) if self.codes else 200
        ok = sse({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
        lines = ok if code < 400 else []
        return _StreamCtx(FakeStream(code, lines, b'{"error": {"message": "no"}}'))


class _StreamCtx:
    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *exc):
        return False


def _with_keys(*keys: str, codes: tuple[int, ...] = ()) -> GeminiLLM:
    llm = GeminiLLM()
    llm.keys = list(keys)
    llm._client = _Client(*codes)
    return llm


async def test_a_rate_limited_key_falls_through_to_the_next() -> None:
    # The free tier is limited per project, so a task of any length runs into
    # that before it runs into anything else.
    llm = _with_keys("one", "two", codes=(429, 429, 429, 200))

    reply = await llm.complete(system="s", tools=[], messages=[user("hi")])

    assert reply.text == "hi"
    assert "two" in llm._client.keys_used


async def test_the_key_that_worked_is_used_next_time() -> None:
    # Starting from the top every call would burn the exhausted key again.
    llm = _with_keys("one", "two", codes=(429, 429, 429, 200))
    await llm.complete(system="s", tools=[], messages=[user("hi")])
    assert llm.api_key == "two"


async def test_every_key_limited_says_so() -> None:
    llm = _with_keys("one", "two", codes=(429,) * 8)
    with pytest.raises(LLMError) as exc:
        await llm.complete(system="s", tools=[], messages=[user("hi")])
    assert "All of them are limited" in str(exc.value)


async def test_one_key_is_told_how_to_have_more() -> None:
    llm = _with_keys("only", codes=(429, 429, 429))
    with pytest.raises(LLMError) as exc:
        await llm.complete(system="s", tools=[], messages=[user("hi")])
    assert "GEMINI_API_KEYS" in str(exc.value)


async def test_a_working_key_is_not_rotated_away_from() -> None:
    llm = _with_keys("one", "two", codes=(200,))
    await llm.complete(system="s", tools=[], messages=[user("hi")])
    assert llm.api_key == "one"
    assert llm._client.keys_used == ["one"]


def test_a_caller_supplied_key_is_used_alone() -> None:
    # A visitor's own key must not fall through to the host's.
    assert GeminiLLM(api_key="theirs").keys == ["theirs"]


async def test_a_stalled_connection_is_explained_rather_than_named() -> None:
    # httpx raises a timeout carrying an empty message, so the obvious rendering
    # reaches the user as "Could not reach Gemini: ReadTimeout: " — a class name
    # and a colon with nothing after it. This is what a real run showed.
    from browser_agent.llm import unreachable

    message = unreachable("Gemini", httpx.ReadTimeout(""))

    assert "ReadTimeout" not in message
    assert "stopped sending" in message
    assert str(config.HTTP_STREAM_GAP_S) in message


async def test_a_stream_that_ran_long_is_a_different_story() -> None:
    # Going quiet and taking too long overall are different failures, and which
    # one it was is the part worth telling someone.
    from browser_agent.llm import unreachable

    message = unreachable("Gemini", TimeoutError())

    assert "still answering" in message
    assert str(config.HTTP_TIMEOUT_S) in message
    assert message != unreachable("Gemini", httpx.ReadTimeout(""))


async def test_other_connection_failures_keep_their_own_words() -> None:
    from browser_agent.llm import unreachable

    assert "refused" in unreachable("OpenAI", httpx.ConnectError("refused"))
    # An exception with nothing to say still names itself rather than trailing off.
    assert "ConnectError" in unreachable("OpenAI", httpx.ConnectError(""))


# --- streaming ----------------------------------------------------------------
#
# The plain generateContent endpoint sends nothing until the whole answer
# exists: measured, the first byte and the last byte of a ~900 byte reply
# arrive in the same instant, five times out of five. So its "read timeout" was
# really a cap on how long the model was allowed to think, and a slow answer
# looked exactly like a dead connection. Streamed, the first chunk lands in
# 1.6-6.7s and the rest follow ~100ms apart.


def sse(*events) -> list[str]:
    """The wire as Gemini writes it: `data: {json}` lines separated by blanks."""
    lines = []
    for event in events:
        lines += [f"data: {json.dumps(event)}", ""]
    return lines


class FakeStream:
    def __init__(self, status_code: int, lines: list[str], body: bytes = b""):
        self.status_code = status_code
        self._lines = lines
        self._body = body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class FakeStreamClient:
    """Answers each stream() with the next scripted outcome; an exception is
    raised instead, which is how a stalled connection arrives."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def stream(self, method, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else FakeStream(200, [])

        class _Ctx:
            async def __aenter__(self):
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


async def stream(client, **kw):
    from browser_agent.llm import stream_with_retry

    return await stream_with_retry(
        client, "https://x/y", json={}, headers={}, gap_s=1, total_s=5, delay=0, **kw
    )


async def test_the_events_of_a_stream_are_collected() -> None:
    client = FakeStreamClient(FakeStream(200, sse({"a": 1}, {"b": 2})))
    result = await stream(client)

    assert result.status_code == 200
    assert result.events == [{"a": 1}, {"b": 2}]


async def test_a_stalled_stream_is_retried() -> None:
    # The failure this whole change is about: the connection goes quiet.
    client = FakeStreamClient(httpx.ReadTimeout("quiet"), FakeStream(200, sse({"a": 1})))
    result = await stream(client)

    assert result.events == [{"a": 1}]
    assert client.calls == 2


async def test_a_stream_that_never_starts_is_raised() -> None:
    client = FakeStreamClient(*[httpx.ReadTimeout("quiet")] * 3)
    with pytest.raises(httpx.ReadTimeout):
        await stream(client)
    assert client.calls == 3


async def test_an_error_body_is_read_rather_than_streamed() -> None:
    # Errors are small and come back whole, and the caller needs the text to
    # say what was wrong rather than just the number.
    client = FakeStreamClient(FakeStream(400, [], b'{"error": {"message": "bad key"}}'))
    result = await stream(client)

    assert result.status_code == 400
    assert result.json()["error"]["message"] == "bad key"


async def test_a_transient_status_is_retried_on_a_stream_too() -> None:
    client = FakeStreamClient(FakeStream(503, []), FakeStream(200, sse({"a": 1})))
    result = await stream(client)

    assert result.status_code == 200
    assert client.calls == 2


async def test_every_event_is_offered_as_it_arrives() -> None:
    # What lets a caller show progress instead of a frozen last action.
    seen = []
    client = FakeStreamClient(FakeStream(200, sse({"a": 1}, {"b": 2})))
    await stream(client, on_event=seen.append)

    assert seen == [{"a": 1}, {"b": 2}]


async def test_a_stream_that_never_finishes_is_cut_off() -> None:
    from browser_agent.llm import stream_with_retry

    class Endless:
        status_code = 200

        async def aiter_lines(self):
            while True:
                await asyncio.sleep(0.01)
                yield "data: {}"

        async def aread(self):
            return b""

    client = FakeStreamClient(Endless(), Endless(), Endless())
    with pytest.raises(TimeoutError):
        await stream_with_retry(
            client, "https://x/y", json={}, headers={},
            gap_s=5, total_s=0.05, delay=0,
        )


# --- rebuilding one turn from its events --------------------------------------


def test_text_split_across_events_is_joined() -> None:
    from browser_agent.gemini import _collect

    parts, _ = _collect(
        sse_events := [
            {"candidates": [{"content": {"parts": [{"text": "Trains are"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " fast."}]}}]},
        ]
    )
    assert sse_events  # the shape really seen on the wire
    assert parts == [{"text": "Trains are fast."}]


def test_a_function_call_keeps_its_thought_signature() -> None:
    # Replaying a functionCall without its signature is a 400 on the next
    # request, so this is the difference between a working conversation and a
    # dead one. It must not be merged into neighbouring text either.
    from browser_agent.gemini import _collect

    parts, _ = _collect(
        [
            {"candidates": [{"content": {"parts": [{"text": "Clicking."}]}}]},
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {"name": "click", "args": {"ref": 1}},
                                    "thoughtSignature": "SIG",
                                }
                            ]
                        }
                    }
                ]
            },
        ]
    )

    assert parts[0] == {"text": "Clicking."}
    assert parts[1]["thoughtSignature"] == "SIG"
    assert parts[1]["functionCall"]["args"] == {"ref": 1}


def test_the_empty_trailing_part_is_dropped_but_a_signed_one_is_not() -> None:
    # The last event carries an empty text part to hang finishReason on. A part
    # that is empty *and* signed is a different thing: dropping it loses the
    # signature the next request has to replay.
    from browser_agent.gemini import _collect

    parts, _ = _collect(
        [
            {"candidates": [{"content": {"parts": [{"text": "Done."}]}}]},
            {"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"}]},
        ]
    )
    assert parts == [{"text": "Done."}]

    signed, _ = _collect(
        [{"candidates": [{"content": {"parts": [{"text": "", "thoughtSignature": "S"}]}}]}]
    )
    assert signed == [{"text": "", "thoughtSignature": "S"}]


def test_a_refused_prompt_is_noticed_across_events() -> None:
    from browser_agent.gemini import _collect

    _, blocked = _collect([{"promptFeedback": {"blockReason": "SAFETY"}}])
    assert blocked == "SAFETY"


def test_events_without_a_candidate_are_skipped() -> None:
    from browser_agent.gemini import _collect

    parts, blocked = _collect([{"usageMetadata": {"totalTokenCount": 10}}, {"candidates": []}])
    assert parts == []
    assert blocked == ""
