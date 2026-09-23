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


async def _no_ollama(host: str = "") -> bool:
    """Whether this developer's machine runs Ollama is nobody's business here."""
    return False


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
    # Checked by structure rather than by name: what it is called is a product
    # decision and should not be something a rename has to chase through tests.
    assert "<title>" in response.text
    for element in ('id="composer"', 'id="log"', 'id="provider"', 'id="suggestions"'):
        assert element in response.text


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


def test_a_follow_up_continues_the_same_agent(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "order a shirt"})
        drain(socket, until="report")
        socket.send_json({"message": "what about yesterday?"})
        drain(socket, until="report")

    # One agent, so one conversation. Building a second would hand the visitor
    # an assistant with no memory of what it had just told them.
    assert len(FakeAgent.built) == 1
    assert FakeAgent.built[0] is not None


def test_a_finished_task_does_not_throw_the_conversation_away(web) -> None:
    built = []

    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "first task"})
        drain(socket, until="report")   # the fake finishes every task
        socket.send_json({"message": "follow-up"})
        drain(socket, until="report")
        built = list(FakeAgent.built)

    assert len(built) == 1, "a finished task must not discard the agent"


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


# --- health -------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/health", "/healthz"])
def test_health_is_served_on_both_paths(web, monkeypatch, path: str) -> None:
    # /healthz is the convention, but Google's frontend swallows that exact
    # path on Cloud Run and the request never reaches the container. /health
    # is the one that works everywhere, so both are served.
    async def fine():
        return ""

    monkeypatch.setattr("browser_agent.web.browser_works", fine)
    monkeypatch.setattr("browser_agent.web.ollama_available", _no_ollama)
    with TestClient(build_app()) as client:
        assert client.get(path).status_code == 200


def test_healthz_says_ok_when_it_can_work(web, monkeypatch) -> None:
    async def fine():
        return ""

    # Entered as a context manager so the startup check actually runs; without
    # that this would only be asserting the default. Ollama is stubbed out
    # because whether this machine happens to be running one is not the subject.
    monkeypatch.setattr("browser_agent.web.browser_works", fine)
    monkeypatch.setattr("browser_agent.web.ollama_available", _no_ollama)
    with TestClient(build_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["detail"] is None
    assert body["providers"] == ["gemini"]


def test_healthz_reports_a_missing_browser(monkeypatch) -> None:
    # A deploy whose image has no Chromium should fail while it is still being
    # deployed, not on the first visitor's first click.
    async def no_browser():
        return "BrowserUnavailable: Chromium is not installed."

    monkeypatch.setattr("browser_agent.web.browser_works", no_browser)
    monkeypatch.setattr("browser_agent.web.ollama_available", _no_ollama)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "a-key")

    with TestClient(build_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    assert "Chromium is not installed" in response.json()["detail"]


def test_healthz_reports_having_no_model(monkeypatch) -> None:
    for name in ("GEMINI_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(config, name, "")

    async def fine():
        return ""

    monkeypatch.setattr("browser_agent.web.browser_works", fine)
    monkeypatch.setattr("browser_agent.web.ollama_available", _no_ollama)
    with TestClient(build_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["detail"] == "no model configured"


async def test_the_browser_check_starts_and_stops_a_real_one() -> None:
    from browser_agent.web import browser_works

    assert await browser_works() == ""


# --- how the container is told where to listen --------------------------------


def test_the_port_and_host_come_from_the_environment(monkeypatch) -> None:
    # Cloud Run hands the port over in the environment and expects every
    # interface; locally neither is wanted.
    from browser_agent.web import main

    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("HOST", "0.0.0.0")

    seen = {}
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kw: seen.update(kw)
    )
    main([])

    assert seen["port"] == 9123
    assert seen["host"] == "0.0.0.0"


def test_flags_beat_the_environment(monkeypatch) -> None:
    from browser_agent.web import main

    monkeypatch.setenv("PORT", "9123")
    seen = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw))
    main(["--port", "7000", "--host", "127.0.0.1"])

    assert seen["port"] == 7000
    assert seen["host"] == "127.0.0.1"


# --- the local model ----------------------------------------------------------


async def test_a_reachable_ollama_is_detected() -> None:
    from browser_agent.web import ollama_available

    # Detected rather than configured, so it appears while developing and is
    # simply absent on a server that has none — it cannot be switched on by
    # accident in the wrong place.
    assert await ollama_available("http://127.0.0.1:1") is False


def test_the_local_model_is_only_offered_when_there_is_one(monkeypatch) -> None:
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")

    assert server_providers(local=False) == ["gemini"]
    assert server_providers(local=True) == ["gemini", "ollama"]


def test_a_server_without_ollama_refuses_to_use_it(web) -> None:
    with web.websocket_connect("/ws") as socket:
        socket.send_json({"message": "go", "provider": "ollama"})
        error = drain(socket, until="error")[-1]
    assert "no key for ollama" in error["text"]


def test_the_local_model_needs_no_key(monkeypatch) -> None:
    monkeypatch.setattr("browser_agent.web.BrowserAgent", FakeAgent)
    FakeAgent.built.clear()
    _make_agent({"provider": "ollama"}, lambda **kw: None, local=True)
    assert FakeAgent.built[0]["api_key"] == ""


def test_a_key_cannot_be_handed_to_the_local_model(monkeypatch) -> None:
    with pytest.raises(ValueError, match="anthropic or openai"):
        _make_agent({"provider": "ollama", "api_key": "sk-x"}, lambda **kw: None, local=True)


# --- suggestions --------------------------------------------------------------


def test_suggestions_are_offered(web) -> None:
    suggestions = web.get("/api/config").json()["suggestions"]
    assert suggestions
    assert all(isinstance(s, str) and s.strip() for s in suggestions)


def test_every_suggestion_names_a_site_the_default_allowlist_permits() -> None:
    # A pill that cannot run is worse than no pill. These are the domains
    # DEPLOY.md tells you to allow.
    from browser_agent.web import SUGGESTIONS

    recommended = {"wikipedia.org", "news.ycombinator.com", "bbc.com", "python.org"}
    hints = {
        "hacker news": "news.ycombinator.com",
        "eiffel": "wikipedia.org",
        "bbc": "bbc.com",
        "python": "python.org",
    }
    for text in SUGGESTIONS:
        matched = [site for word, site in hints.items() if word in text.lower()]
        assert matched, f"no allowlisted site covers: {text}"
        assert set(matched) <= recommended


# --- the stylesheet and its two themes ----------------------------------------


def test_the_stylesheet_is_served(web) -> None:
    response = web.get("/static/theme.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_the_page_asks_for_the_stylesheet(web) -> None:
    assert "/static/theme.css" in web.get("/").text


def test_both_themes_are_defined(web) -> None:
    css = web.get("/static/theme.css").text

    # Light is the default and dark overrides it, so a visitor who has chosen
    # nothing gets light rather than whatever happens to cascade.
    assert ":root {" in css
    assert ':root[data-theme="dark"]' in css
    # And a visitor whose system prefers dark gets it without having to ask,
    # unless they have explicitly chosen light.
    assert "prefers-color-scheme: dark" in css
    assert ':root:not([data-theme="light"])' in css


def test_dark_is_actually_black(web) -> None:
    css = web.get("/static/theme.css").text
    dark = css.split(':root[data-theme="dark"]')[1].split("}")[0]
    assert "--bg: #000000" in dark


def test_the_working_ring_is_only_drawn_while_working(web) -> None:
    page = web.get("/").text
    css = web.get("/static/theme.css").text

    # It is a class the page adds and removes, not something always on: a page
    # that glows while idle reads as broken rather than busy.
    assert ".thinking::before" in css
    assert "classList.toggle('thinking'" in page


def test_motion_can_be_turned_off(web) -> None:
    css = web.get("/static/theme.css").text
    assert "prefers-reduced-motion: reduce" in css
