"""The agent loop, with a scripted stand-in for the model.

The browser is real; the model is not. Each test hands the agent the replies it
should receive, so what is under test is the loop — how a reply becomes tool
calls, how results become the next request, and how a run ends.
"""

from __future__ import annotations

import asyncio
import copy

import pytest

from browser_agent.agent import (
    FINISHED,
    IN_PROGRESS,
    STOPPED,
    WAITING,
    BrowserAgent,
    _close_dangling_tool_calls,
    _prune_screenshots,
)
from browser_agent.llm import LLMError, Reply, ToolCall, ToolResult, tool_results
from browser_agent.llm import user as llm_user

PAGE = """
<!doctype html>
<title>Shop</title>
<h1>Shop</h1>
<a href="/cart">Cart</a>
<input name="q" placeholder="Search">
"""


class FakeLLM:
    """Serves scripted replies and records the conversation it was sent."""

    def __init__(self, *replies, delay: float = 0.0, supports_images: bool = True):
        self.name = "fake-model"
        self.supports_images = supports_images
        self.replies = list(replies)
        self.sent: list[list[dict]] = []
        self.systems: list[str] = []
        self.tools_seen: list[list[dict]] = []
        self.delay = delay
        self.closed = False
        self.fail_after: int | None = None

    async def complete(self, *, system, tools, messages):
        # A copy, not the live list: the agent keeps appending to it.
        self.sent.append(copy.deepcopy(messages))
        self.systems.append(system)
        self.tools_seen.append(tools)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_after is not None and len(self.sent) > self.fail_after:
            raise LLMError("the model fell over")
        if not self.replies:
            # A script that ran out means the loop went somewhere unexpected.
            return Reply("(no more scripted replies)")
        return self.replies.pop(0)

    async def close(self):
        self.closed = True


def call(name: str, args: dict | None = None, id: str = "call_1") -> ToolCall:
    return ToolCall(id, name, args or {})


def agent_with(session, *replies, **kwargs) -> BrowserAgent:
    backend = kwargs.pop("backend", None) or FakeLLM(*replies)
    agent = BrowserAgent(backend=backend, **kwargs)
    agent._session = session
    return agent


# --- how a run ends -----------------------------------------------------------


async def test_a_finished_task_reports_but_keeps_the_browser(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("Opening the shop…", [call("navigate", {"url": url})]),
        Reply("", [call("finish_task", {"report": "The cart is empty."}, "call_2")]),
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == FINISHED
    assert outcome.text == "The cart is empty."
    assert outcome.url == url
    # The browser stays: a finished task is not a finished conversation.
    assert agent._session is session


async def test_text_without_a_tool_call_waits_for_the_user(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply("Which size do you want?"))

    outcome = await agent.run("order a shirt")

    assert outcome.state == WAITING
    assert outcome.text == "Which size do you want?"
    assert agent._session is session


async def test_running_out_of_steps_asks_for_a_progress_line(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url})]),
        Reply("", [call("get_page", {}, "call_2")]),
        Reply("So far I have opened the shop; next I will find the cart."),
        max_steps=2,
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == IN_PROGRESS
    assert "opened the shop" in outcome.text
    assert agent._session is session


async def test_running_out_of_time_also_reports_progress(session, serve) -> None:
    await serve(PAGE)
    backend = FakeLLM(Reply("", [call("get_page")]), Reply("Still looking."), delay=0.2)
    agent = agent_with(session, backend=backend, timeout_s=0.05)

    outcome = await agent.run("check the cart")

    assert outcome.state == IN_PROGRESS
    assert agent._session is session


async def test_stopping_with_nothing_to_say_is_turned_into_a_message(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(
        session,
        Reply(""),
        Reply("I have not started yet; I will search the shop next."),
    )

    outcome = await agent.run("check the cart")

    assert outcome.state == WAITING
    assert "search the shop" in outcome.text


async def test_a_wrap_up_that_fails_still_returns_something_usable(session, serve) -> None:
    await serve(PAGE)
    backend = FakeLLM(Reply(""))
    backend.fail_after = 1
    agent = agent_with(session, backend=backend)

    outcome = await agent.run("check the cart")

    assert outcome.state == WAITING
    assert "Still working on it" in outcome.text


async def test_a_model_that_cannot_be_reached_is_reported_not_raised(session, serve) -> None:
    await serve(PAGE)
    backend = FakeLLM()
    backend.fail_after = 0
    agent = agent_with(session, backend=backend)

    outcome = await agent.run("check the cart")

    assert outcome.state == WAITING
    assert "fell over" in outcome.text


# --- what goes on the wire ----------------------------------------------------


async def test_the_task_and_the_first_snapshot_go_out_together(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(session, Reply("done looking"))

    await agent.run("find the cart", start_url=url, context="the user is in a hurry")

    first = agent.llm.sent[0][0]
    assert first["role"] == "user"
    assert "Task: find the cart" in first["text"]
    assert "the user is in a hurry" in first["text"]
    assert "Page: Shop" in first["text"]  # pre-navigated, so no step is wasted


async def test_no_start_url_means_no_navigation(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply("where should I look?"))

    await agent.run("find the cart")

    first = agent.llm.sent[0][0]["text"]
    assert "(none — decide where to go)" in first
    assert "Page:" not in first


async def test_a_follow_up_carries_the_history_and_a_fresh_snapshot(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(session, Reply("Which size?"), Reply("Ordered."))

    await agent.run("order a shirt", start_url=url)
    await agent.run("medium")

    latest = agent.llm.sent[-1]
    assert len(latest) > 1  # the first turn is still there
    assert "The user says: medium" in latest[-1]["text"]
    assert "Page: Shop" in latest[-1]["text"]
    assert agent.turns == 2


async def test_tool_results_come_back_paired_with_their_calls(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "call_abc")]),
        Reply("done"),
    )

    await agent.run("open the shop")

    results = agent.llm.sent[-1][-1]
    assert results["role"] == "tool"
    assert results["results"][0].id == "call_abc"
    assert "Opened" in results["results"][0].text


async def test_parallel_tool_calls_are_answered_in_one_batch(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "a"), call("get_page", {}, "b")]),
        Reply("done"),
    )

    await agent.run("open the shop")

    batch = agent.llm.sent[-1][-1]
    assert [r.id for r in batch["results"]] == ["a", "b"]


async def test_the_system_prompt_carries_the_time(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply("ok"))

    await agent.run("do a thing")

    assert "Current date and time:" in agent.llm.systems[0]


async def test_an_unknown_tool_is_answered_rather_than_raised(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("teleport", {"to": "mars"})]),
        Reply("that did not work"),
    )

    outcome = await agent.run("go to mars")

    results = agent.llm.sent[-1][-1]["results"]
    assert "no tool called 'teleport'" in results[0].text
    assert outcome.state == WAITING


# --- what the model is offered ------------------------------------------------


async def test_a_model_that_can_see_is_offered_the_camera(session) -> None:
    agent = agent_with(session, backend=FakeLLM(supports_images=True))
    assert "screenshot" in {tool["name"] for tool in agent.tools}


async def test_a_model_that_cannot_see_is_not(session) -> None:
    agent = agent_with(session, backend=FakeLLM(supports_images=False))
    names = {tool["name"] for tool in agent.tools}
    assert "screenshot" not in names
    assert {"navigate", "click", "type_text", "finish_task"} <= names


async def test_the_model_is_closed_with_the_agent(session) -> None:
    agent = agent_with(session, Reply("ok"))
    await agent.close()
    assert agent.llm.closed


# --- screenshots --------------------------------------------------------------


async def test_a_screenshot_is_attached_to_its_result(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "a")]),
        Reply("", [call("screenshot", {}, "b")]),
        Reply("looks fine"),
    )

    await agent.run("check the layout")

    shot = agent.llm.sent[-1][-1]["results"][0]
    assert shot.id == "b"
    assert shot.image is not None
    assert shot.image[1] == "image/jpeg"


async def test_old_screenshots_lose_their_image(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "a")]),
        Reply("", [call("screenshot", {}, "b")]),
        Reply("", [call("screenshot", {}, "c")]),
        Reply("", [call("screenshot", {}, "d")]),
        Reply("done"),
    )

    await agent.run("watch the page")

    results = [r for m in agent.messages if m["role"] == "tool" for r in m["results"]]
    assert len([r for r in results if r.image is not None]) == 2  # config.MAX_SCREENSHOTS
    assert any("was removed to save space" in r.text for r in results)


def test_prune_keeps_the_most_recent_images() -> None:
    messages = [
        tool_results(
            [ToolResult(str(n), "screenshot", "Screenshot taken", (str(n), "image/jpeg"))]
        )
        for n in range(3)
    ]

    _prune_screenshots(messages, keep=1)

    assert messages[0]["results"][0].image is None
    assert messages[1]["results"][0].image is None
    assert messages[2]["results"][0].image == ("2", "image/jpeg")
    # The pairing survives: every result still answers its call.
    assert [m["results"][0].id for m in messages] == ["0", "1", "2"]


def test_prune_with_keep_zero_removes_every_image() -> None:
    messages = [tool_results([ToolResult("a", "screenshot", "shot", ("x", "image/jpeg"))])]
    _prune_screenshots(messages, keep=0)
    assert messages[0]["results"][0].image is None


# --- dangling calls -----------------------------------------------------------


def test_dangling_tool_calls_are_answered() -> None:
    messages = [
        {"role": "user", "text": "go"},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [call("click", {}, "a"), call("scroll", {}, "b")],
            "raw": None,
        },
        tool_results([ToolResult("a", "click", "ok")]),
    ]

    _close_dangling_tool_calls(messages, "Cancelled — out of time.")

    assert messages[-1]["results"][0].id == "b"
    assert "Cancelled" in messages[-1]["results"][0].text


def test_nothing_is_added_when_every_call_was_answered() -> None:
    messages = [
        {"role": "assistant", "text": "", "tool_calls": [call("click", {}, "a")], "raw": None},
        tool_results([ToolResult("a", "click", "ok")]),
    ]
    before = len(messages)
    _close_dangling_tool_calls(messages, "note")
    assert len(messages) == before


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "text": "text only"}],
        [{"role": "assistant", "text": "no tools", "tool_calls": [], "raw": None}],
    ],
)
def test_closing_dangling_calls_copes_with_any_history(messages: list) -> None:
    _close_dangling_tool_calls(messages, "note")  # must not raise


# --- callbacks ----------------------------------------------------------------


async def test_progress_callbacks_fire(session, serve) -> None:
    url = await serve(PAGE)
    said: list[str] = []
    did: list[tuple[str, dict]] = []

    agent = agent_with(
        session,
        Reply("Opening the shop…", [call("navigate", {"url": url})]),
        Reply("All done."),
        on_text=said.append,
        on_action=lambda name, args: did.append((name, args)),
    )

    await agent.run("open the shop")

    assert said == ["Opening the shop…", "All done."]
    assert did == [("navigate", {"url": url})]


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_names_no_product() -> None:
    from browser_agent.prompts import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    for word in ("robin", "browserbase", "langchain", "playwright", "anthropic", "claude"):
        assert word not in lowered


def test_the_prompt_offers_search_engines_to_fall_through() -> None:
    # This started as "Bing and DuckDuckGo work; Google usually blocks", became
    # a single hardcoded engine once Google was measured blocking, and stayed
    # that way after the fix that unblocked it. Measured since: all three return
    # ~50-90 links from a cold browser. Whichever is having a bad day, there
    # must be somewhere to fall through to.
    from browser_agent.prompts import SYSTEM_PROMPT

    for engine in ("google.com/search?q=", "bing.com/search?q=", "duckduckgo.com/?q="):
        assert engine in SYSTEM_PROMPT
    assert "move to the next" in SYSTEM_PROMPT


# --- getting stuck ------------------------------------------------------------


async def test_repeating_the_same_action_earns_a_nudge(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "a")]),
        Reply("", [call("scroll", {"direction": "down"}, "b")]),
        Reply("", [call("scroll", {"direction": "down"}, "c")]),
        Reply("", [call("scroll", {"direction": "down"}, "d")]),
        Reply("Right, I will use what is already here."),
    )

    await agent.run("read the page")

    texts = [r.text for m in agent.messages if m["role"] == "tool" for r in m["results"]]
    assert "exact action 3 times in a row" not in texts[1]  # first scroll: no nudge
    assert "exact action 3 times in a row" in texts[3]  # third in a row: nudged


async def test_alternating_actions_are_not_nudged(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "a")]),
        Reply("", [call("scroll", {"direction": "down"}, "b")]),
        Reply("", [call("get_page", {}, "c")]),
        Reply("", [call("scroll", {"direction": "down"}, "d")]),
        Reply("done"),
    )

    await agent.run("read the page")

    texts = [r.text for m in agent.messages if m["role"] == "tool" for r in m["results"]]
    assert not any("exact action" in t for t in texts)


async def test_the_same_tool_with_different_arguments_is_not_repetition(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": url}, "a")]),
        Reply("", [call("scroll", {"direction": "down"}, "b")]),
        Reply("", [call("scroll", {"direction": "up"}, "c")]),
        Reply("", [call("scroll", {"direction": "down"}, "d")]),
        Reply("done"),
    )

    await agent.run("look around")

    texts = [r.text for m in agent.messages if m["role"] == "tool" for r in m["results"]]
    assert not any("exact action" in t for t in texts)


# --- what survives a finished task --------------------------------------------


async def test_the_conversation_outlives_a_finished_task(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("finish_task", {"report": "The cart is empty."})]),
        Reply("There were two items yesterday."),
    )

    await agent.run("check the cart")
    before = len(agent.messages)
    await agent.run("what about yesterday?")

    # The follow-up is asked with everything that came before it, or the user
    # is repeating themselves to something that just answered them.
    sent = agent.llm.sent[-1]
    assert len(sent) > before
    assert any("check the cart" in (m.get("text") or "") for m in sent)
    assert agent.turns == 2


async def test_a_follow_up_after_finishing_uses_the_same_browser(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(
        session,
        Reply("", [call("finish_task", {"report": "done"})]),
        Reply("still here"),
    )

    await agent.run("look at the shop", start_url=url)
    await agent.run("and the cart?")

    assert agent._session is session
    # The page is still the one the task ended on, so the model is told so.
    assert "browser is where you left it" in agent.llm.sent[-1][-1]["text"]


async def test_a_browser_that_went_away_is_admitted_to(session, serve) -> None:
    url = await serve(PAGE)
    agent = agent_with(session, Reply("ok"), Reply("ok again"))
    await agent.run("look at the shop", start_url=url)

    # Simulate the browser dying between turns.
    agent._session = None
    try:
        await agent.run("carry on")
        text = agent.llm.sent[-1][-1]["text"]
        # Telling the model the page is where it left it would be a lie, and it
        # would act on elements that no longer exist.
        assert "reopened on a blank page" in text
        assert "browser is where you left it" not in text
    finally:
        await agent.close()


async def test_the_same_dead_url_is_noticed_even_when_spread_out(session, serve) -> None:
    url = await serve(PAGE)
    dead = "https://this-host-does-not-resolve.invalid/"
    agent = agent_with(
        session,
        Reply("", [call("navigate", {"url": dead}, "a")]),
        Reply("", [call("navigate", {"url": url}, "b")]),
        Reply("", [call("navigate", {"url": dead}, "c")]),
        Reply("", [call("get_page", {}, "d")]),
        Reply("", [call("navigate", {"url": dead}, "e")]),
        Reply("giving up"),
    )

    await agent.run("open the dead site")

    texts = [r.text for m in agent.messages if m["role"] == "tool" for r in m["results"]]
    # Three attempts at the same URL, never three in a row — the old check only
    # looked at the last few actions and so never saw it.
    assert "exact action 3 times" in texts[-1]


# --- stopping -----------------------------------------------------------------


class SlowLLM(FakeLLM):
    """Takes its time, so a stop has something to land in the middle of."""

    def __init__(self, *replies, **kwargs):
        super().__init__(*replies, **kwargs)
        self.started = asyncio.Event()

    async def complete(self, *, system, tools, messages):
        self.started.set()
        await asyncio.sleep(30)
        return await super().complete(system=system, tools=tools, messages=messages)


async def test_stopping_pauses_rather_than_cancels(session, serve) -> None:
    await serve(PAGE)
    backend = SlowLLM(Reply("never reached"))
    agent = agent_with(session, backend=backend)

    async def press_stop():
        await backend.started.wait()
        agent.stop()

    asyncio.create_task(press_stop())
    outcome = await agent.run("do something long")

    # Its own state, not IN_PROGRESS: being told "out of budget" when you
    # pressed stop is simply untrue.
    assert outcome.state == STOPPED
    assert "Stopped" in outcome.text
    assert "continue" in outcome.text
    # The page is exactly where it was, which is the point of a pause.
    assert agent._session is session


async def test_stopping_lands_during_a_model_call_not_after_it(session, serve) -> None:
    await serve(PAGE)
    backend = SlowLLM(Reply("never reached"))
    agent = agent_with(session, backend=backend)

    async def press_stop():
        await backend.started.wait()
        agent.stop()

    asyncio.create_task(press_stop())
    started = asyncio.get_running_loop().time()
    await agent.run("do something long")
    took = asyncio.get_running_loop().time() - started

    # The backend sleeps 30s. Waiting for the step to finish would make the
    # button useless on exactly the slow runs people want to stop.
    assert took < 5, f"stop took {took:.1f}s to land"


async def test_a_stopped_run_leaves_a_history_the_model_will_accept(session, serve) -> None:
    url = await serve(PAGE)
    backend = SlowLLM(Reply("never reached"))
    agent = BrowserAgent(backend=backend)
    agent._session = session

    # One completed step, so there is a tool call on the record, then a stop
    # during the next model call.
    agent.messages.append(llm_user("earlier"))

    async def press_stop():
        await backend.started.wait()
        agent.stop()

    asyncio.create_task(press_stop())
    await agent.run("go", start_url=url)

    for message in agent.messages:
        if message["role"] == "assistant":
            answered = {
                r.id
                for m in agent.messages
                if m["role"] == "tool"
                for r in m["results"]
            }
            for c in message.get("tool_calls") or []:
                assert c.id in answered, "a call with no result is rejected on the next request"


async def test_continuing_after_a_stop_keeps_the_conversation(session, serve) -> None:
    await serve(PAGE)
    backend = SlowLLM(Reply("never reached"))
    agent = agent_with(session, backend=backend)

    async def press_stop():
        await backend.started.wait()
        agent.stop()

    asyncio.create_task(press_stop())
    await agent.run("find the cart")

    # A fresh backend for the second turn, answering immediately.
    agent.llm = FakeLLM(Reply("Carrying on."))
    outcome = await agent.run("continue")

    assert outcome.state == WAITING
    assert any("find the cart" in (m.get("text") or "") for m in agent.llm.sent[-1])


async def test_a_stop_does_not_carry_into_the_next_turn(session, serve) -> None:
    await serve(PAGE)
    agent = agent_with(session, Reply("first"), Reply("second"))

    agent.stop()          # pressed while nothing was running
    outcome = await agent.run("go")

    # Cleared at the start of a run, or the stop would kill the turn after the
    # one it was meant for.
    assert outcome.state == WAITING
    assert outcome.text == "first"
