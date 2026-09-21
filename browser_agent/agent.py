"""The agent loop: one browser and one memory across the turns of a task.

A plain tool loop over the Anthropic Messages API. Each step the model reads the
page as text, picks an action, and gets the resulting page back — so the history
grows by a whole snapshot per step. Two things keep that affordable: a cache
breakpoint on the system prompt, so every step re-reads the prefix at cached
rates, and screenshot pruning, so old images stop riding along.

One agent owns one browser. ``run()`` can be called again and again: the first
call opens the browser, later calls continue in the same conversation with the
page exactly where it was left. That is what makes a pause for the user — an
OTP, a choice, a go-ahead — just the next call, and what lets a turn that runs
out of budget carry on instead of starting over.

Each run is bounded by a step cap and a wall clock. Hitting either loses
nothing: the model is asked for a progress line and the browser stays open.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from browser_agent import config
from browser_agent.prompts import FOLLOW_UP_TEMPLATE, SYSTEM_PROMPT, TASK_TEMPLATE
from browser_agent.session import BrowserSession, BrowserUnavailable, use_session
from browser_agent.tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)

__all__ = [
    "FINISHED",
    "IN_PROGRESS",
    "WAITING",
    "BrowserAgent",
    "BrowserOutcome",
    "BrowserUnavailable",
]

# How a run ended. FINISHED closes the browser; the other two leave it open
# with its memory, and the next run continues from there.
FINISHED = "finished"        # finish_task was called: `text` is the report
WAITING = "waiting"          # the model replied in text (a question, a check-in)
IN_PROGRESS = "in_progress"  # the run's budget ran out mid-task


class BrowserOutcome(NamedTuple):
    text: str
    state: str = FINISHED
    url: str = ""


# Sentinels from _run_loop.
_NO_REPORT = object()  # the model stopped with empty text
_FINISHED = object()   # finish_task was called
_OUT_OF_TIME = object()

# The wrap-up call gets a short leash — it must not double the run.
_SUMMARY_TIMEOUT_S = 40

_PRUNED_NOTE = "[An earlier screenshot was removed to save space — take a new one if you need it.]"


# ── message helpers ──────────────────────────────────────────────────────────


def _serialize_content(blocks) -> list[dict]:
    """The API's response blocks as plain dicts we can send back and store.

    Every block is kept, including thinking blocks and their signatures: the
    assistant turn has to go back exactly as it came, or the next request is
    rejected for a history that does not match what the model produced.
    """
    out = []
    for block in blocks:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(exclude_none=True))
        else:
            out.append(dict(block))
    return out


def _extract_text(blocks) -> str:
    parts = []
    for block in blocks:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "text":
            parts.append(block["text"] if isinstance(block, dict) else block.text)
    return "\n".join(p for p in parts if p).strip()


def _tool_uses(blocks) -> list:
    return [b for b in blocks if getattr(b, "type", None) == "tool_use"]


def _screenshot_result(tool_use_id: str, status: str, shot: tuple[str, str]) -> dict:
    """A tool_result carrying the image itself, which is where Anthropic wants
    it — an image in its own user message would break the tool_result pairing."""
    data, mime = shot
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {"type": "text", "text": status},
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
        ],
    }


def _prune_screenshots(messages: list[dict], keep: int = config.MAX_SCREENSHOTS) -> None:
    """Replace all but the last `keep` screenshots with a one-line note, in place.

    Only the tool_result's content changes, so every tool_use keeps its answer.
    The edit does invalidate the cached prefix from that point, which is the
    price of not re-sending a stale image on every step from here to the end.
    """
    shots = [
        block
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
        and isinstance(block.get("content"), list)
        and any(part.get("type") == "image" for part in block["content"])
    ]
    for block in (shots[:-keep] if keep > 0 else shots):
        block["content"] = _PRUNED_NOTE


def _close_dangling_tool_calls(messages: list[dict], note: str) -> None:
    """Answer every tool_use that never got a tool_result.

    The API rejects a history with an unanswered tool_use, so a loop cut off
    mid-batch could not even be asked for a summary without this.
    """
    last = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if last is None or not isinstance(last.get("content"), list):
        return
    pending = [b["id"] for b in last["content"] if b.get("type") == "tool_use"]
    if not pending:
        return
    answered = {
        block.get("tool_use_id")
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    }
    missing = [i for i in pending if i not in answered]
    if missing:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": i, "content": note} for i in missing
                ],
            }
        )


# ── the agent ────────────────────────────────────────────────────────────────


class BrowserAgent:
    """One browser, one conversation, many runs.

    ``on_text`` receives the model's narration line for each step and ``on_action``
    the tool it is about to run — both optional, and both there so a caller can
    show progress without the agent knowing anything about a terminal.
    """

    def __init__(
        self,
        *,
        headless: bool | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        timeout_s: int | None = None,
        on_text: Callable[[str], None] | None = None,
        on_action: Callable[[str, dict], None] | None = None,
    ):
        self.headless = config.HEADLESS if headless is None else headless
        self.model = model or config.MODEL
        self.max_steps = config.MAX_STEPS if max_steps is None else max_steps
        self.timeout_s = config.TIMEOUT_S if timeout_s is None else timeout_s
        self.on_text = on_text
        self.on_action = on_action

        self.messages: list[dict] = []
        self.task = ""
        self.turns = 0

        self._session: BrowserSession | None = None
        self._client = None

    async def __aenter__(self) -> BrowserAgent:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- plumbing -----------------------------------------------------------

    def _anthropic(self):
        """Built on first use so that importing the package needs no API key."""
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=config.require_api_key())
        return self._client

    def _system(self) -> list[dict]:
        now = datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")
        return [
            {
                "type": "text",
                "text": SYSTEM_PROMPT.format(current_datetime=now),
                # Tools render before the system prompt, so one breakpoint here
                # caches both — and they are re-sent on every step of every run.
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def _ask(self, messages: list[dict]) -> Any:
        return await self._anthropic().messages.create(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            system=self._system(),
            tools=TOOLS,
            messages=messages,
        )

    # -- the loop -----------------------------------------------------------

    async def _run_loop(self, session: BrowserSession):
        """One run's tool loop.

        Returns the model's text (it spoke to the user), _FINISHED, _NO_REPORT
        (it stopped with nothing to say), or None (the step cap).
        """
        for step in range(self.max_steps):
            response = await self._ask(self.messages)
            self.messages.append(
                {"role": "assistant", "content": _serialize_content(response.content)}
            )

            text = _extract_text(response.content)
            if text and self.on_text:
                self.on_text(text)

            calls = _tool_uses(response.content)
            if not calls:
                logger.info("Replied after %d step(s)", step + 1)
                # A tool-bound model can end a turn with empty text. That is not
                # a message to the user; let the wrap-up ask for one.
                return text or _NO_REPORT

            results = []
            for call in calls:
                args = call.input if isinstance(call.input, dict) else {}
                if self.on_action:
                    self.on_action(call.name, args)
                status = await execute_tool(call.name, args, session)
                shot = session.take_screenshot() if call.name == "screenshot" else None
                if shot:
                    results.append(_screenshot_result(call.id, status, shot))
                else:
                    results.append(
                        {"type": "tool_result", "tool_use_id": call.id, "content": status}
                    )

            self.messages.append({"role": "user", "content": results})
            _prune_screenshots(self.messages)

            # finish_task ends the run here, after the whole batch has been
            # answered, so the history stays well-formed.
            if session.finished_report is not None:
                logger.info("Finished after %d step(s)", step + 1)
                return _FINISHED
        return None

    async def _wrap_up(self, reason: str) -> str:
        """The run is over with nothing said to the user: ask for one line.

        The tools stay on the request even though nothing will be executed —
        dropping them would change the cached prefix and re-read the whole
        history uncached, to save nothing.
        """
        _close_dangling_tool_calls(self.messages, f"Cancelled — {reason}.")
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"[System: {reason}. This turn is over; tool calls will not be executed "
                    "now. The browser stays open and you continue on the next turn. Reply in "
                    "text only, one or two sentences: what you have done so far and what you "
                    "will do next. If you need anything from the user, ask for it now.]"
                ),
            }
        )
        try:
            response = await asyncio.wait_for(
                self._ask(self.messages), timeout=_SUMMARY_TIMEOUT_S
            )
            self.messages.append(
                {"role": "assistant", "content": _serialize_content(response.content)}
            )
            text = _extract_text(response.content)
            if text:
                return text
            logger.warning("Wrap-up returned no text")
        except Exception as e:
            logger.warning("Wrap-up failed: %s", e)
        return (
            "Still working on it — the browser is open where I left off; "
            "say 'continue' to carry on."
        )

    # -- public API ---------------------------------------------------------

    async def run(self, message: str, *, start_url: str = "", context: str = "") -> BrowserOutcome:
        """Run one turn.

        ``message`` is the task on the first call and the user's reply or next
        instruction after that. The outcome says whether the browser closed
        (FINISHED) or is still open with its memory (WAITING / IN_PROGRESS).

        Raises BrowserUnavailable when no browser could be started; every other
        failure comes back as text.
        """
        message = (message or "").strip()
        # "First" is about the conversation, not the browser: an agent handed a
        # session it did not open still has to introduce the task.
        first = not self.messages

        if self._session is None:
            self._session = BrowserSession(headless=self.headless)
            await self._session.start()

        if first:
            self.task = message
            extra = f" start_url={start_url}" if start_url else ""
            logger.info("Started: %r%s", message[:120], extra)
        else:
            logger.info("Continuing (turn %d): %r", self.turns + 1, message[:120])

        session = self._session
        session.finished_report = None
        self.turns += 1

        with use_session(session):
            if first:
                human = TASK_TEMPLATE.format(
                    task=message,
                    start_url=start_url.strip() or "(none — decide where to go)",
                    context=context.strip() or "(none)",
                )
                if start_url.strip():
                    # Save the model a step: open the page it was pointed at and
                    # hand it the first snapshot along with the task.
                    human += "\n" + await session.navigate(start_url)
            else:
                context_line = (
                    f"Context from the conversation: {context.strip()}\n" if context.strip() else ""
                )
                human = FOLLOW_UP_TEMPLATE.format(
                    message=message or "continue", context_line=context_line
                )
                human += "\n" + await session.get_page()
            self.messages.append({"role": "user", "content": human})

            try:
                result = await asyncio.wait_for(self._run_loop(session), timeout=self.timeout_s)
            except TimeoutError:
                result = _OUT_OF_TIME

            url = session.current_url()

            if result is _FINISHED:
                report = session.finished_report or "Done."
                await self.close()
                return BrowserOutcome(report, FINISHED, url)

            if isinstance(result, str):
                return BrowserOutcome(result, WAITING, url)

            if result is _NO_REPORT:
                text = await self._wrap_up("you stopped without saying anything to the user")
                return BrowserOutcome(text, WAITING, url)

            reason = (
                "the step budget for this turn is used up"
                if result is None
                else "the time budget for this turn is used up"
            )
            logger.warning("%s", reason.capitalize())
            text = await self._wrap_up(reason)
            return BrowserOutcome(text, IN_PROGRESS, url)

    async def close(self) -> None:
        """Close the browser. The conversation is kept, so it can be inspected."""
        if self._session is not None:
            await self._session.close()
            self._session = None
