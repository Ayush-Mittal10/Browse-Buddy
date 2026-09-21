"""The tool schemas the model is offered, and the dispatcher behind them."""

from __future__ import annotations

import json

import pytest

from browser_agent.session import BrowserSession
from browser_agent.tools import _DISPATCH, TOOL_NAMES, TOOLS, execute_tool

PAGE = """
<!doctype html>
<title>Tools</title>
<h1>Heading</h1>
<a href="/next">Next page</a>
<input name="q" placeholder="Query">
"""

EXPECTED = {
    "navigate",
    "click",
    "type_text",
    "select_option",
    "press_key",
    "scroll",
    "go_back",
    "get_page",
    "read_text",
    "screenshot",
    "wait",
    "switch_tab",
    "finish_task",
}


# --- the schemas --------------------------------------------------------------


def test_every_tool_is_present() -> None:
    assert TOOL_NAMES == EXPECTED


def test_names_are_unique_and_order_is_stable() -> None:
    names = [tool["name"] for tool in TOOLS]
    assert len(names) == len(set(names))
    # The tool block is the head of the cached prompt prefix; reordering it
    # silently costs a cache hit on every request.
    assert names[0] == "navigate"
    assert names[-1] == "finish_task"


def test_schemas_are_json_serialisable() -> None:
    assert json.loads(json.dumps(TOOLS)) == TOOLS


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["name"])
def test_schema_shape(tool: dict) -> None:
    assert set(tool) == {"name", "description", "input_schema"}
    assert tool["description"].strip()

    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    for key in schema.get("required", []):
        assert key in schema["properties"], f"{key} is required but not defined"

    for name, prop in schema["properties"].items():
        assert "type" in prop, f"{name} has no type"
        assert prop["type"] in {"string", "integer", "number", "boolean"}


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["name"])
def test_every_schema_has_a_dispatcher(tool: dict) -> None:
    assert tool["name"] in _DISPATCH


def test_the_dispatcher_has_nothing_the_model_cannot_see() -> None:
    assert set(_DISPATCH) == TOOL_NAMES


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["name"])
def test_descriptions_do_not_leak_the_implementation(tool: dict) -> None:
    text = (tool["description"] + json.dumps(tool["input_schema"])).lower()
    for leak in ("playwright", "chromium", "locator", "selector", "css", "dom"):
        assert leak not in text


# --- dispatch -----------------------------------------------------------------


async def test_no_session_is_reported_not_raised() -> None:
    out = await execute_tool("get_page", {})
    assert "No browser session is active" in out


async def test_an_unknown_tool_lists_the_real_ones() -> None:
    async with BrowserSession(headless=True):
        out = await execute_tool("teleport", {"to": "mars"})
    assert "no tool called 'teleport'" in out
    assert "navigate" in out


async def test_the_session_is_found_through_the_context_var(serve, session) -> None:
    url = await serve(PAGE)
    # Publish the already-started session the way the context manager would.
    from browser_agent.session import _current

    token = _current.set(session)
    try:
        assert "Opened" in await execute_tool("navigate", {"url": url})
        assert "Page: Tools" in await execute_tool("get_page", {})
    finally:
        _current.reset(token)


async def test_an_explicit_session_overrides_the_context_var(session, serve) -> None:
    url = await serve(PAGE)
    out = await execute_tool("navigate", {"url": url}, session)
    assert "Opened" in out


async def test_each_tool_round_trips(session, serve) -> None:
    url = await serve(PAGE)
    await execute_tool("navigate", {"url": url}, session)

    assert "Page: Tools" in await execute_tool("get_page", {}, session)
    assert "Scrolled down" in await execute_tool("scroll", {"direction": "down"}, session)
    assert "Waited" in await execute_tool("wait", {"seconds": 0.5}, session)
    assert "Screenshot taken" in await execute_tool("screenshot", {}, session)
    assert "Text 0-" in await execute_tool("read_text", {}, session)
    assert "Pressed Escape" in await execute_tool("press_key", {"key": "Escape"}, session)
    assert "Clicked [1]" in await execute_tool("click", {"ref": 1}, session)


async def test_type_text_goes_through_with_its_flag(session, serve) -> None:
    url = await serve(PAGE)
    await execute_tool("navigate", {"url": url}, session)

    out = await execute_tool(
        "type_text", {"ref": 2, "text": "hello", "press_enter": True}, session
    )
    assert "pressed Enter" in out
    assert await session.page.input_value("[name=q]") == "hello"


async def test_missing_arguments_fall_back_instead_of_crashing(session, serve) -> None:
    await execute_tool("navigate", {"url": await serve(PAGE)}, session)

    # Every one of these is a malformed call; none may raise.
    assert isinstance(await execute_tool("navigate", {}, session), str)
    assert isinstance(await execute_tool("click", {}, session), str)
    assert isinstance(await execute_tool("scroll", {}, session), str)
    assert isinstance(await execute_tool("wait", {}, session), str)
    assert isinstance(await execute_tool("read_text", {}, session), str)
    assert isinstance(await execute_tool("switch_tab", {}, session), str)


async def test_args_may_be_omitted_entirely(session, serve) -> None:
    await execute_tool("navigate", {"url": await serve(PAGE)}, session)
    assert "Page: Tools" in await execute_tool("get_page", session=session)


async def test_a_tool_that_blows_up_comes_back_as_a_string(session, monkeypatch) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("the wheels came off")

    monkeypatch.setattr(session, "get_page", boom)
    out = await execute_tool("get_page", {}, session)

    assert "The get_page tool failed" in out
    assert "the wheels came off" in out


# --- finish_task --------------------------------------------------------------


async def test_finish_task_records_the_report(session) -> None:
    assert session.finished_report is None

    out = await execute_tool("finish_task", {"report": "  Found three cafés.  "}, session)

    assert "Task marked finished" in out
    assert session.finished_report == "Found three cafés."


async def test_finish_task_without_a_report_still_finishes(session) -> None:
    await execute_tool("finish_task", {}, session)
    assert session.finished_report == ""


async def test_finish_task_does_not_close_the_browser_itself(session) -> None:
    await execute_tool("finish_task", {"report": "done"}, session)
    # The loop closes the browser after the turn; the tool only raises the flag,
    # or the snapshot in this same result would be taken against a dead page.
    assert session.page is not None
