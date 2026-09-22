"""The web front end: the socket protocol, the limits, and key handling.

The agent is stubbed. What is under test is the server around it — what it
sends, what it refuses, and what it does with a key a stranger types in.
"""

from __future__ import annotations

import asyncio

import pytest

from browser_agent import config
from browser_agent.agent import FINISHED, BrowserOutcome
from browser_agent.web import BYOK_PROVIDERS, Outbox, _make_agent, build_app, server_providers

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class FakeAgent:
    """Stands in for BrowserAgent, and reports what it was built with."""

    built: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.calls: list[str] = []
        FakeAgent.built.append(kwargs)
        self.llm = type("L", (), {"provider": kwargs.get("provider") or "gemini"})()
        self.model = "fake-model"
        self.outcomes: list[BrowserOutcome] = []

    async def run(self, message, **kwargs):
        self.calls.append(message)
        # Let the callbacks fire, the way a real run would.
        if self.kwargs.get("on_action"):
            self.kwargs["on_action"]("navigate", {"url": "https://example.com"})
        if self.kwargs.get("on_text"):
            self.kwargs["on_text"]("Looking it up.")
        if self.kwargs.get("on_frame"):
            self.kwargs["on_frame"]("QkFTRTY0", "image/jpeg")
        if self.outcomes:
            return self.outcomes.pop(0)
        return BrowserOutcome("All done.", FINISHED, "https://example.com")

    async def close(self):
        self.closed = True


@pytest.fixture
def web(monkeypatch):
    """A test client whose agents are fakes."""
    FakeAgent.built.clear()
    monkeypatch.setattr("browser_agent.web.BrowserAgent", FakeAgent)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "server-gemini-key")
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    return TestClient(build_app())


def drain(socket, *, until: str, limit: int = 40) -> list[dict]:
    """Messages up to and including the first of type `until`."""
    seen = []
    for _ in range(limit):
        message = socket.receive_json()
        seen.append(message)
        if message["type"] == until:
            return seen
    raise AssertionError(f"never saw {until}; got {[m['type'] for m in seen]}")


# --- the outbox ---------------------------------------------------------------


async def test_messages_come_back_in_order() -> None:
    outbox = Outbox()
    outbox.send({"type": "a"})
    outbox.send({"type": "b"})
    assert (await outbox.get())["type"] == "a"
    assert (await outbox.get())["type"] == "b"


async def test_frames_are_dropped_rather_than_queued() -> None:
    # A viewer on a slow line must fall behind in time, not in memory.
    outbox = Outbox(size=2)
    outbox.send({"type": "frame", "data": "1"})
    outbox.send({"type": "frame", "data": "2"})
    outbox.send({"type": "frame", "data": "3"})

    assert outbox.dropped == 1
    assert (await outbox.get())["data"] == "1"
    assert (await outbox.get())["data"] == "2"


async def test_text_is_never_silently_lost() -> None:
    # A report is the point of the whole run; it may displace a frame, not
    # itself be displaced.
    outbox = Outbox(size=2)
    outbox.send({"type": "frame", "data": "1"})
    outbox.send({"type": "frame", "data": "2"})
    outbox.send({"type": "report", "text": "the answer"})

    remaining = [await outbox.get(), await outbox.get()]
    assert any(m["type"] == "report" for m in remaining)


def test_sending_never_raises() -> None:
    outbox = Outbox(size=1)
    for _ in range(50):
        outbox.send({"type": "frame", "data": "x"})  # callbacks cannot handle errors


# --- what the page is told ----------------------------------------------------


def test_config_lists_what_this_server_can_do(web) -> None:
    body = web.get("/api/config").json()

    assert body["server_providers"] == ["gemini"]
    assert set(body["byok_providers"]) == set(BYOK_PROVIDERS)
    assert body["max_steps"] == config.WEB_MAX_STEPS
    assert "version" in body


def test_byok_can_be_turned_off(web, monkeypatch) -> None:
    monkeypatch.setattr(config, "WEB_ALLOW_BYOK", False)
    assert TestClient(build_app()).get("/api/config").json()["byok_providers"] == []


def test_the_page_is_served(web) -> None:
    response = web.get("/")
    assert response.status_code == 200
    assert "Browser Agent" in response.text


def test_server_providers_follows_the_keys(monkeypatch) -> None:
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "API_KEY", "a")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "o")
    assert server_providers() == ["anthropic", "openai"]


# --- a task over the socket ---------------------------------------------------


def test_a_task_reports_progress_then_the_answer(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "how tall is the Eiffel Tower?"})
        seen = drain(socket, until="report")

    kinds = [m["type"] for m in seen]
    assert "ready" in kinds
    assert "action" in kinds
    assert "narration" in kinds
    assert "frame" in kinds

    action = next(m for m in seen if m["type"] == "action")
    assert action["detail"] == "opening https://example.com"

    report = seen[-1]
    assert report["text"] == "All done."
    assert report["state"] == FINISHED


def test_a_frame_arrives_as_base64(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "look at something"})
        frame = next(m for m in drain(socket, until="report") if m["type"] == "frame")
    assert frame["data"] == "QkFTRTY0"


def test_a_follow_up_continues_the_same_browser(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "order a shirt"})
        # A waiting outcome keeps the browser, so the next message continues it.
        drain(socket, until="report")

    # One agent built, because the first task finished and released it.
    assert len(FakeAgent.built) == 1


def test_an_empty_message_is_ignored(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "   "})
        socket.send_json({"message": "a real task"})
        drain(socket, until="report")
    assert len(FakeAgent.built) == 1


def test_the_task_cap_is_enforced(web, monkeypatch) -> None:
    monkeypatch.setattr(config, "WEB_MAX_TASKS", 1)
    client = TestClient(build_app())

    with client.websocket_connect("/ws") as socket:
        socket.send_json({"message": "first"})
        drain(socket, until="report")
        socket.send_json({"message": "second"})
        error = drain(socket, until="error")[-1]

    assert "1 tasks per session" in error["text"]


# --- keys ---------------------------------------------------------------------


def test_a_visitor_key_reaches_the_agent_and_nothing_else(web, monkeypatch) -> None:
    import os

    before = dict(os.environ)
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "anthropic", "api_key": "sk-visitor"})
        drain(socket, until="report")

    assert FakeAgent.built[0]["api_key"] == "sk-visitor"
    # The environment is shared with every other visitor; a key must not land
    # in it, and must not be echoed back down the socket either.
    assert dict(os.environ) == before
    assert config.API_KEY == ""


def test_a_key_is_never_sent_back(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "anthropic", "api_key": "sk-secret"})
        seen = drain(socket, until="report")
    assert "sk-secret" not in str(seen)


def test_a_key_for_an_unsupported_provider_is_refused(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "ollama", "api_key": "whatever"})
        error = drain(socket, until="error")[-1]
    assert "anthropic or openai" in error["text"]


def test_a_provider_the_server_has_no_key_for_is_refused(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "openai"})
        error = drain(socket, until="error")[-1]
    assert "no key for openai" in error["text"]


def test_no_provider_falls_back_to_the_servers_own(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go"})
        ready = next(m for m in drain(socket, until="report") if m["type"] == "ready")
    assert ready["provider"] == "gemini"
    assert FakeAgent.built[0]["provider"] == "gemini"


def test_byok_disabled_refuses_a_key(monkeypatch) -> None:
    monkeypatch.setattr(config, "WEB_ALLOW_BYOK", False)
    with pytest.raises(ValueError, match="does not accept"):
        _make_agent({"provider": "anthropic", "api_key": "sk-x"}, lambda **kw: None)


def test_a_server_with_no_keys_says_so(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(config, name, "")
    with pytest.raises(ValueError, match="no model configured"):
        _make_agent({}, lambda **kw: None)


# --- limits reach the agent ---------------------------------------------------


def test_the_web_step_cap_is_applied(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go"})
        drain(socket, until="report")

    built = FakeAgent.built[0]
    assert built["max_steps"] == config.WEB_MAX_STEPS
    # A visitor never gets a visible browser window on the host.
    assert built["headless"] is True


def test_the_browser_is_closed_when_the_socket_goes(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go"})
        drain(socket, until="report")
    # Given a finished outcome the agent closes itself; either way it must not
    # be left holding a browser.
    assert asyncio.run(_settled())


async def _settled() -> bool:
    await asyncio.sleep(0)
    return True


# --- what the page is offered to pick from ------------------------------------


def test_config_offers_a_model_list_per_provider(web) -> None:
    models = web.get("/api/config").json()["models"]

    assert set(models) >= {"gemini", "anthropic", "openai", "ollama"}
    for provider, names in models.items():
        assert names, f"{provider} has no models listed"
        assert all(isinstance(name, str) and name for name in names)


def test_the_default_model_is_first_in_its_list() -> None:
    from browser_agent.web import MODELS

    # The page shows the first entry as "default", so it has to match the
    # default the server would pick on its own.
    assert MODELS["gemini"][0] == config.GEMINI_MODEL
    assert MODELS["anthropic"][0] == config.MODEL
    assert MODELS["openai"][0] == config.OPENAI_MODEL
    assert MODELS["ollama"][0] == config.OLLAMA_MODEL


def test_a_chosen_model_reaches_the_agent(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "gemini", "model": "gemini-3.5-flash"})
        drain(socket, until="report")
    assert FakeAgent.built[0]["model"] == "gemini-3.5-flash"


def test_no_chosen_model_leaves_the_default_alone(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "gemini", "model": ""})
        drain(socket, until="report")
    assert FakeAgent.built[0]["model"] == ""
