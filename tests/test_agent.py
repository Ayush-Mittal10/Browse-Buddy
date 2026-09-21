"""The agent loop, with a scripted stand-in for the API.

The browser is real; the model is not. Each test hands the agent the replies it
should receive, so what is under test is the loop — how a reply becomes tool
calls, how results become the next request, and how a run ends.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field

import pytest

from browser_agent.agent import (
    FINISHED,
    IN_PROGRESS,
    WAITING,
    BrowserAgent,
    _close_dangling_tool_calls,
    _extract_text,
    _prune_screenshots,
    _serialize_content,
)

PAGE = """
<!doctype html>
<title>Shop</title>
<h1>Shop</h1>
<a href="/cart">Cart</a>
<input name="q" placeholder="Search">
"""


# --- a stand-in for the model -------------------------------------------------


@dataclass
class Text:
    text: str
    type: str = "text"

    def model_dump(self, **kw):
        return {"type": "text", "text": self.text}


@dataclass
class ToolUse:
    name: str
    input: dict = field(default_factory=dict)
    id: str = "toolu_1"
    type: str = "tool_use"

    def model_dump(self, **kw):
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class Thinking:
    thinking: str = ""
    signature: str = "sig"
    type: str = "thinking"

    def model_dump(self, **kw):
        return {"type": "thinking", "thinking": self.thinking, "signature": self.signature}


@dataclass
class Reply:
    content: list


class FakeAPI:
    """Serves scripted replies and records every request it was sent."""

    def __init__(self, *replies, delay: float = 0.0):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.delay = delay
        self.messages = self

    async def create(self, **kwargs):
        # A copy, not the live list: the agent goes on mutating `messages`, so a
        # reference here would only ever show the end state of the run.
        self.requests.append(copy.deepcopy(kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.replies:
            # A script that ran out means the loop went somewhere unexpected.
            return Reply([Text("(no more scripted replies)")])
        return self.replies.pop(0)


def agent_with(session, *replies, **kwargs) -> BrowserAgent:
    agent = BrowserAgent(**kwargs)
    agent._session = session
    agent._client = FakeAPI(*replies)
    return agent


# --- how a run ends -----------------------------------------------------------


async def test_a_finished_task_reports_and_closes_the_browser(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply([Text("Opening the shop…"), ToolUse("navigate", {"url": url})]),
        Reply([ToolUse("finish_task", {"report": "The cart is empty."}, id="toolu_2")]),
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == FINISHED
    assert outcome.text == "The cart is empty."
    assert outcome.url == url
    assert agent._session is None  # closed on finishing


async def test_text_without_a_tool_call_waits_for_the_user(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply([Text("Which size do you want?")]))

    outcome = await agent.run("order a shirt")

    assert outcome.state == WAITING
    assert outcome.text == "Which size do you want?"
    assert agent._session is session  # still open for the answer


async def test_running_out_of_steps_asks_for_a_progress_line(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply([ToolUse("navigate", {"url": url})]),
        Reply([ToolUse("get_page", {}, id="toolu_2")]),
        Reply([Text("So far I have opened the shop; next I will find the cart.")]),
        max_steps=2,
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == IN_PROGRESS
    assert "opened the shop" in outcome.text
    assert agent._session is session  # left open to carry on


async def test_running_out_of_time_also_reports_progress(session, serve) -> None:
    await serve(PAGE)
    agent = BrowserAgent(timeout_s=0.05)
    agent._session = session
    agent._client = FakeAPI(
        Reply([ToolUse("get_page", {})]),
        Reply([Text("Still looking.")]),
        delay=0.2,
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == IN_PROGRESS
    assert agent._session is session


async def test_stopping_with_nothing_to_say_is_turned_into_a_message(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(
        session,
        Reply([]),  # no text, no tools
        Reply([Text("I have not started yet; I will search the shop next.")]),
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == WAITING
    assert "search the shop" in outcome.text


async def test_a_wrap_up_that_fails_still_returns_something_usable(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply([]))

    async def boom(**kwargs):
        raise RuntimeError("API down")

    original = agent._client.create
    calls = {"n": 0}

    async def create(**kwargs):
        calls["n"] += 1
        return await (boom(**kwargs) if calls["n"] > 1 else original(**kwargs))

    agent._client.create = create
    outcome = await agent.run("check the cart")

    assert outcome.state == WAITING
    assert "Still working on it" in outcome.text


# --- what goes on the wire ----------------------------------------------------


async def test_the_task_and_the_first_snapshot_go_out_together(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(session, Reply([Text("done looking")]))

    await agent.run("find the cart", start_url=url, context="the user is in a hurry")

    first = agent._client.requests[0]["messages"][0]["content"]
    assert "Task: find the cart" in first
    assert "the user is in a hurry" in first
    assert "Page: Shop" in first  # pre-navigated, so no step is wasted on it


async def test_no_start_url_means_no_navigation(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply([Text("where should I look?")]))

    await agent.run("find the cart")

    first = agent._client.requests[0]["messages"][0]["content"]
    assert "(none — decide where to go)" in first
    assert "Page:" not in first


async def test_a_follow_up_carries_the_history_and_a_fresh_snapshot(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply([Text("Which size?")]),
        Reply([Text("Ordered.")]),
    )

    await agent.run("order a shirt", start_url=url)
    await agent.run("medium")

    second = agent._client.requests[-1]["messages"]
    assert len(second) > 1  # the first turn is still there
    assert "The user says: medium" in second[-1]["content"]
    assert "Page: Shop" in second[-1]["content"]
    assert agent.turns == 2


async def test_tool_results_come_back_paired_with_their_calls(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply([ToolUse("navigate", {"url": url}, id="toolu_abc")]),
        Reply([Text("done")]),
    )

    await agent.run("open the shop")

    results = agent._client.requests[-1]["messages"][-1]["content"]
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "toolu_abc"
    assert "Opened" in results[0]["content"]


async def test_parallel_tool_calls_are_answered_in_one_message(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply(
            [
                ToolUse("navigate", {"url": url}, id="a"),
                ToolUse("get_page", {}, id="b"),
            ]
        ),
        Reply([Text("done")]),
    )

    await agent.run("open the shop")

    last = agent._client.requests[-1]["messages"][-1]
    assert last["role"] == "user"
    assert [block["tool_use_id"] for block in last["content"]] == ["a", "b"]


async def test_the_system_prompt_is_cached_and_carries_the_time(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply([Text("ok")]))

    await agent.run("do a thing")

    system = agent._client.requests[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "Current date and time:" in system[0]["text"]
    assert agent._client.requests[0]["tools"]


async def test_an_unknown_tool_is_answered_rather_than_raised(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(
        session,
        Reply([ToolUse("teleport", {"to": "mars"})]),
        Reply([Text("that did not work")]),
    )

    outcome = await agent.run("go to mars")

    results = agent._client.requests[-1]["messages"][-1]["content"]
    assert "no tool called 'teleport'" in results[0]["content"]
    assert outcome.state == WAITING


# --- screenshots --------------------------------------------------------------


async def test_a_screenshot_rides_inside_its_tool_result(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply([ToolUse("navigate", {"url": url}, id="a")]),
        Reply([ToolUse("screenshot", {}, id="b")]),
        Reply([Text("looks fine")]),
    )

    await agent.run("check the layout")

    result = agent._client.requests[-1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "b"
    parts = result["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image"
    assert parts[1]["source"]["media_type"] == "image/jpeg"
    assert parts[1]["source"]["data"]


async def test_old_screenshots_are_pruned_from_the_history(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply([ToolUse("navigate", {"url": url}, id="a")]),
        Reply([ToolUse("screenshot", {}, id="b")]),
        Reply([ToolUse("screenshot", {}, id="c")]),
        Reply([ToolUse("screenshot", {}, id="d")]),
        Reply([Text("done")]),
    )

    await agent.run("watch the page")

    images = [
        part
        for message in agent.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list)
        for part in block["content"]
        if part.get("type") == "image"
    ]
    assert len(images) == 2  # config.MAX_SCREENSHOTS

    notes = [m for m in agent.messages if "was removed to save space" in str(m)]
    assert notes


# --- callbacks ----------------------------------------------------------------


async def test_progress_callbacks_fire(session, serve) -> None:
    url = await serve(PAGE)
    said: list[str] = []
    did: list[tuple[str, dict]] = []

    agent = agent_with(
        session,
        Reply([Text("Opening the shop…"), ToolUse("navigate", {"url": url})]),
        Reply([Text("All done.")]),
        on_text=said.append,
        on_action=lambda name, args: did.append((name, args)),
    )

    await agent.run("open the shop")

    assert said == ["Opening the shop…", "All done."]
    assert did == [("navigate", {"url": url})]


# --- message helpers ----------------------------------------------------------


def test_serialize_keeps_thinking_blocks_intact() -> None:
    blocks = _serialize_content([Thinking("weighing options", "abc123"), Text("hello")])

    assert blocks[0] == {"type": "thinking", "thinking": "weighing options", "signature": "abc123"}
    assert blocks[1] == {"type": "text", "text": "hello"}


def test_extract_text_joins_every_text_block() -> None:
    assert _extract_text([Text("one"), ToolUse("click"), Text("two")]) == "one\ntwo"
    assert _extract_text([ToolUse("click")]) == ""


def test_prune_keeps_the_most_recent_images() -> None:
    def shot(n: int) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": str(n),
                    "content": [
                        {"type": "text", "text": "Screenshot taken"},
                        {"type": "image", "source": {"type": "base64", "data": str(n)}},
                    ],
                }
            ],
        }

    messages = [shot(1), shot(2), shot(3)]
    _prune_screenshots(messages, keep=1)

    assert isinstance(messages[0]["content"][0]["content"], str)
    assert isinstance(messages[1]["content"][0]["content"], str)
    assert isinstance(messages[2]["content"][0]["content"], list)
    # The pairing survives: every tool_use still has its tool_result.
    assert all(m["content"][0]["tool_use_id"] for m in messages)


def test_prune_with_keep_zero_removes_every_image() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "a",
                    "content": [{"type": "image", "source": {}}],
                }
            ],
        }
    ]
    _prune_screenshots(messages, keep=0)
    assert isinstance(messages[0]["content"][0]["content"], str)


def test_dangling_tool_calls_are_answered() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a", "name": "click", "input": {}},
                {"type": "tool_use", "id": "b", "name": "scroll", "input": {}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
    ]

    _close_dangling_tool_calls(messages, "Cancelled — out of time.")

    assert messages[-1]["content"][0]["tool_use_id"] == "b"
    assert "Cancelled" in messages[-1]["content"][0]["content"]


def test_nothing_is_added_when_every_call_was_answered() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "click", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
    ]
    before = len(messages)
    _close_dangling_tool_calls(messages, "note")
    assert len(messages) == before


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "text only"}],
        [{"role": "assistant", "content": [{"type": "text", "text": "no tools"}]}],
    ],
)
def test_closing_dangling_calls_copes_with_any_history(messages: list) -> None:
    _close_dangling_tool_calls(messages, "note")  # must not raise


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_names_no_product() -> None:
    from browser_agent.prompts import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    for word in ("robin", "browserbase", "langchain", "playwright", "anthropic", "claude"):
        assert word not in lowered
