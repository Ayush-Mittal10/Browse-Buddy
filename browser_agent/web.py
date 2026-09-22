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
from pathlib import Path

from browser_agent import __version__, config
from browser_agent.agent import FINISHED, BrowserAgent, BrowserUnavailable
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


def server_providers() -> list[str]:
    """Providers this server can run without the visitor supplying anything."""
    available = []
    if config.GEMINI_API_KEY:
        available.append("gemini")
    if config.API_KEY:
        available.append("anthropic")
    if config.OPENAI_API_KEY:
        available.append("openai")
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


def build_app():
    """The FastAPI app. Imported here so the CLI never needs FastAPI installed."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="browser-agent", version=__version__)

    # One browser per visitor is the real cost here, so the cap is on browsers.
    browsers = asyncio.Semaphore(config.WEB_MAX_SESSIONS)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    async def configuration():
        return JSONResponse(
            {
                "version": __version__,
                "server_providers": server_providers(),
                "byok_providers": list(BYOK_PROVIDERS) if config.WEB_ALLOW_BYOK else [],
                "max_steps": config.WEB_MAX_STEPS,
                "max_tasks": config.WEB_MAX_TASKS,
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
                        agent = _make_agent(request, say)
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
                if outcome.state == FINISHED:
                    # The agent closed its browser; the next task opens a fresh
                    # one, so let go of the slot in the meantime.
                    await agent.close()
                    agent = None
                    if held:
                        browsers.release()
                        held = False

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


def _make_agent(request: dict, say) -> BrowserAgent:
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
    elif provider and provider not in server_providers():
        raise ValueError(
            f"This server has no key for {provider}. Provide one, or pick another provider."
        )
    elif not provider:
        available = server_providers()
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
        prog="browser-agent-web",
        description="Serve the browser agent as a web page.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
