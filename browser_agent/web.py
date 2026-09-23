"""The agent as a web page: chat on one side, the live browser on the other.

One WebSocket per visitor, one browser per WebSocket. The agent already reports
everything through callbacks, so this file is mostly plumbing: narration and
actions become chat lines, frames become an image, and the socket closing takes
the browser down with it.

Two things it has to get right beyond that.

Frames are dropped, not queued. A viewer on a slow connection must fall behind
in time rather than in memory, and a stale frame is worth nothing anyway — so
the outbox is bounded and the oldest frame goes when it fills.

A key a visitor types is used for their session and never stored, never logged
and never sent back. It goes into the agent as an argument; it does not touch
the environment, because the environment is shared with everyone else here.
"""

# Deliberately no `from __future__ import annotations`: FastAPI resolves a
# route's type hints from module globals, and WebSocket is imported inside
# build_app so that the CLI never needs FastAPI. Deferred annotations would
# leave it unresolvable, and the socket parameter would be read as a query
# string instead — which fails at connect time, not import time.
import asyncio
import contextlib
import logging
import os
from pathlib import Path

from browser_agent import __version__, config
from browser_agent.agent import BrowserAgent, BrowserUnavailable
from browser_agent.cli import describe_action

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

# Deep enough to ride out a hiccup, shallow enough that a stalled viewer is
# noticed rather than accumulated.
OUTBOX_SIZE = 32

# Providers a visitor may bring a key for. Gemini is deliberately absent: it
# runs on the host's key so that someone who has no key at all can still try
# this, which is the entire point of putting it on the web.
BYOK_PROVIDERS = ("anthropic", "openai")

# Models the page offers per provider, first one being the default. A free-text
# box here just produces typos and a 404 from the provider; this is the set that
# is known to work with tool calling.
MODELS = {
    "gemini": [
        # The bigger Flash models answer 503 on the free tier most of the time.
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
    ],
    "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
    "openai": ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini", "gpt-4o-mini"],
    "ollama": ["qwen3:8b", "qwen3:14b", "qwen3:4b"],
}


# Tasks worth offering a visitor who has not thought of one. They stay inside
# the default allowlist below, and they are sites that do not fight automation —
# a demo's first impression should not be a CAPTCHA.
SUGGESTIONS = [
    "What is the top story on Hacker News right now?",
    "How tall is the Eiffel Tower?",
    "What is the top headline on BBC News?",
    "Find the Python release notes for the newest version",
]


async def ollama_available(host: str = "") -> bool:
    """Whether a local Ollama is answering.

    Detected rather than configured, so the local model appears while
    developing and simply is not there on a server that has no Ollama —
    it cannot be switched on by accident in the wrong place.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(f"{(host or config.OLLAMA_HOST).rstrip('/')}/api/version")
        return response.status_code == 200
    except Exception:
        return False


def server_providers(*, local: bool = False) -> list[str]:
    """Providers this server can run without the visitor supplying anything."""
    available = []
    if config.GEMINI_API_KEY:
        available.append("gemini")
    if config.API_KEY:
        available.append("anthropic")
    if config.OPENAI_API_KEY:
        available.append("openai")
    if local:
        available.append("ollama")
    return available


class Outbox:
    """Messages waiting to go down the socket, with frames that may be dropped."""

    def __init__(self, size: int = OUTBOX_SIZE):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=size)
        self.dropped = 0

    def send(self, message: dict) -> None:
        """Queue a message. Never blocks and never raises — callers are callbacks."""
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            # Drop a frame to make room; a late picture is worth nothing. Text
            # is not droppable, so it waits for the next round instead.
            if message.get("type") == "frame":
                self.dropped += 1
                return
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(message)

    async def get(self) -> dict:
        return await self._queue.get()


async def _pump(websocket, outbox: Outbox) -> None:
    """Drain the outbox into the socket until cancelled."""
    while True:
        message = await outbox.get()
        await websocket.send_json(message)


async def browser_works() -> str:
    """Launch a browser once and throw it away. Returns "" or what went wrong.

    Run at startup so a deploy with no Chromium in the image fails while it is
    still being deployed, rather than on the first visitor's first click.
    """
    from browser_agent.session import BrowserSession

    session = BrowserSession(headless=True)
    try:
        await session.start()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        await session.close()
    return ""


def build_app():
    """The FastAPI app. Imported here so the CLI never needs FastAPI installed."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    @asynccontextmanager
    async def lifespan(app):
        app.state.ollama = await ollama_available()
        if app.state.ollama:
            logger.info("A local Ollama is answering; offering it as a provider")
        app.state.browser_error = await browser_works()
        if app.state.browser_error:
            logger.error("No usable browser: %s", app.state.browser_error)
        else:
            logger.info("Browser checked and working")
        yield

    app = FastAPI(title="Browse Buddy", version=__version__, lifespan=lifespan)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # One browser per visitor is the real cost here, so the cap is on browsers.
    browsers = asyncio.Semaphore(config.WEB_MAX_SESSIONS)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    # Two paths for one check. /healthz is the convention and works locally and
    # under Docker, but Google's frontend swallows that exact path on Cloud Run
    # — the request never reaches the container and the caller gets Google's own
    # 404 page instead. /health is the one that works everywhere.
    @app.get("/health")
    @app.get("/healthz")
    async def health():
        """503 when this instance cannot do its job, so it is taken out of
        rotation instead of accepting visitors it will only disappoint."""
        problem = getattr(app.state, "browser_error", "")
        providers = server_providers(local=getattr(app.state, "ollama", False))
        if not providers:
            problem = problem or "no model configured"
        return JSONResponse(
            {
                "status": "error" if problem else "ok",
                "detail": problem or None,
                "version": __version__,
                "providers": providers,
            },
            status_code=503 if problem else 200,
        )

    @app.get("/api/config")
    async def configuration():
        return JSONResponse(
            {
                "version": __version__,
                "server_providers": server_providers(
                    local=getattr(app.state, "ollama", False)
                ),
                "byok_providers": list(BYOK_PROVIDERS) if config.WEB_ALLOW_BYOK else [],
                "max_steps": config.WEB_MAX_STEPS,
                "max_tasks": config.WEB_MAX_TASKS,
                "models": MODELS,
                "suggestions": SUGGESTIONS,
                "allowed_domains": list(config.ALLOWED_DOMAINS),
            }
        )

    @app.websocket("/ws")
    async def socket(websocket: WebSocket):
        await websocket.accept()
        outbox = Outbox()
        pump = asyncio.create_task(_pump(websocket, outbox))
        agent = None
        held = False
        tasks_run = 0

        def say(**message) -> None:
            outbox.send(message)

        try:
            while True:
                try:
                    request = await asyncio.wait_for(
                        websocket.receive_json(), timeout=config.WEB_IDLE_TIMEOUT_S
                    )
                except TimeoutError:
                    say(type="error", text="Closed after a few minutes of quiet.")
                    await asyncio.sleep(0.2)
                    break

                message = (request.get("message") or "").strip()
                if not message:
                    continue

                if tasks_run >= config.WEB_MAX_TASKS:
                    say(
                        type="error",
                        text=f"This demo allows {config.WEB_MAX_TASKS} tasks per session. "
                        "Reload to start a new one.",
                    )
                    continue

                if agent is None:
                    if not held:
                        if browsers.locked():
                            say(type="status", state="queued")
                        await browsers.acquire()
                        held = True
                    try:
                        agent = _make_agent(
                            request, say, local=getattr(app.state, "ollama", False)
                        )
                    except ValueError as e:
                        say(type="error", text=str(e))
                        browsers.release()
                        held = False
                        continue
                    say(type="ready", provider=agent.llm.provider, model=agent.model)

                tasks_run += 1
                say(type="status", state="running")
                try:
                    outcome = await agent.run(message)
                except BrowserUnavailable as e:
                    say(type="error", text=str(e))
                    break
                except Exception as e:
                    logger.exception("Task failed")
                    say(type="error", text=f"{type(e).__name__}: {e}")
                    continue

                say(type="report", text=outcome.text, state=outcome.state, url=outcome.url)
                say(type="status", state="idle")
                # The agent is deliberately kept, finished or not. It holds the
                # conversation, so discarding it here is what made a follow-up
                # arrive with no memory of what was just asked or answered — the
                # visitor is still in the same chat and expects it to know.

        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Socket failed")
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
            if agent is not None:
                await agent.close()
            if held:
                browsers.release()
            with contextlib.suppress(Exception):
                await websocket.close()

    return app


def _make_agent(request: dict, say, *, local: bool = False) -> BrowserAgent:
    """An agent for this visitor, on whichever model they are entitled to."""
    provider = (request.get("provider") or "").strip().lower()
    api_key = (request.get("api_key") or "").strip()
    model = (request.get("model") or "").strip()

    if api_key:
        if not config.WEB_ALLOW_BYOK:
            raise ValueError("This server does not accept visitor API keys.")
        if provider not in BYOK_PROVIDERS:
            allowed = " or ".join(BYOK_PROVIDERS)
            raise ValueError(f"A key can only be given for {allowed}.")
    elif provider and provider not in server_providers(local=local):
        raise ValueError(
            f"This server has no key for {provider}. Provide one, or pick another provider."
        )
    elif not provider:
        available = server_providers(local=local)
        if not available:
            raise ValueError("This server has no model configured.")
        provider = available[0]

    agent = BrowserAgent(
        headless=True,
        provider=provider,
        model=model,
        api_key=api_key,
        max_steps=config.WEB_MAX_STEPS,
        on_text=lambda text: say(type="narration", text=text),
        on_action=lambda name, args: say(
            type="action", name=name, detail=describe_action(name, args)
        ),
        on_frame=lambda data, mime: say(type="frame", data=data),
    )
    return agent


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="browse-buddy-web",
        description="Serve the browser agent as a web page.",
    )
    # Cloud Run and friends hand the port over in the environment and expect
    # the process to listen on every interface; locally, neither is wanted.
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        print("The web UI needs its extras. Run: pip install -e '.[web]'")
        return 2

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
