"""The agent loop: one browser and one memory across the turns of a task.

A plain tool loop. Each step the model reads the page as text, picks an action,
and gets the resulting page back — so the history grows by a whole snapshot per
step. Two things keep that affordable, and both depend on the history staying
append-only: the prompt prefix is cached (billed at a discount by a hosted
model, not recomputed at all by a local one), and old screenshots lose their
image so a stale JPEG stops riding along.

Which model runs this is llm.py's business, not the loop's.

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
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import NamedTuple
from zoneinfo import ZoneInfo

from browser_agent import config, llm
from browser_agent.llm import LLM, LLMError, ToolResult
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

# How many identical actions in a row before the loop says something. Seen live
# on a local model: eight consecutive scrolls down a Wikipedia article, each one
# returning a page it had already been given, until the clock ran out. The
# prompt tells it not to; a model small enough to loop is a model small enough
# to forget that, so the nudge goes where it cannot be missed — in the result.
_REPEAT_LIMIT = 3

_STUCK_NOTE = (
    "[You have now done this exact action {n} times in a row and the page is not changing. "
    "It is not working. Do something else: use what is already on the page, try a different "
    "element, or say what you have found.]"
)

_PRUNED_NOTE = "[An earlier screenshot was removed to save space — take a new one if you need it.]"


# ── history ──────────────────────────────────────────────────────────────────


def _prune_screenshots(messages: list[dict], keep: int = config.MAX_SCREENSHOTS) -> None:
    """Drop the image from all but the last `keep` screenshots, in place.

    Only the result's text and image change, so every tool call keeps its
    answer. It does cost a cache hit from that point on — the prefix is no
    longer what it was — which is worth it against re-sending a stale JPEG on
    every remaining step. Nothing else in the loop edits history, because that
    append-only property is what keeps a local model fast.
    """
    shots = [
        result
        for message in messages
        if message["role"] == "tool"
        for result in message["results"]
        if result.image is not None
    ]
    for result in shots[:-keep] if keep > 0 else shots:
        result.image = None
        result.text = _PRUNED_NOTE


def _close_dangling_tool_calls(messages: list[dict], note: str) -> None:
    """Answer every tool call that never got a result.

    A history with an unanswered call is rejected, so a loop cut off mid-batch
    could not even be asked for a summary without this.
    """
    last = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    if last is None or not last.get("tool_calls"):
        return
    answered = {
        result.id
        for message in messages
        if message["role"] == "tool"
        for result in message["results"]
    }
    missing = [c for c in last["tool_calls"] if c.id not in answered]
    if missing:
        messages.append(llm.tool_results([ToolResult(c.id, c.name, note) for c in missing]))


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
        provider: str = "",
        api_key: str = "",
        max_steps: int | None = None,
        timeout_s: int | None = None,
        backend: LLM | None = None,
        on_text: Callable[[str], None] | None = None,
        on_action: Callable[[str, dict], None] | None = None,
    ):
        self.headless = config.HEADLESS if headless is None else headless
        self.llm = backend or llm.build(provider, model or "", api_key)
        self.max_steps = config.MAX_STEPS if max_steps is None else max_steps
        self.timeout_s = config.TIMEOUT_S if timeout_s is None else timeout_s
        self.on_text = on_text
        self.on_action = on_action

        # A model that cannot see is not offered the camera.
        self.tools = [
            tool
            for tool in TOOLS
            if tool["name"] != "screenshot" or self.llm.supports_images
        ]

        self.messages: list[dict] = []
        self.task = ""
        self.turns = 0

        self._session: BrowserSession | None = None

    @property
    def model(self) -> str:
        return self.llm.name

    async def __aenter__(self) -> BrowserAgent:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # -- plumbing -----------------------------------------------------------

    def _system(self) -> str:
        now = datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")
        return SYSTEM_PROMPT.format(current_datetime=now)

    async def _ask(self):
        return await self.llm.complete(
            system=self._system(), tools=self.tools, messages=self.messages
        )

    # -- the loop -----------------------------------------------------------

    async def _run_loop(self, session: BrowserSession):
        """One run's tool loop.

        Returns the model's text (it spoke to the user), _FINISHED, _NO_REPORT
        (it stopped with nothing to say), or None (the step cap).
        """
        recent: list[str] = []
        for step in range(self.max_steps):
            reply = await self._ask()
            self.messages.append(llm.assistant(reply))

            if reply.text and self.on_text:
                self.on_text(reply.text)

            if not reply.tool_calls:
                logger.info("Replied after %d step(s)", step + 1)
                # A tool-bound model can end a turn with empty text. That is not
                # a message to the user; let the wrap-up ask for one.
                return reply.text or _NO_REPORT

            results = []
            for call in reply.tool_calls:
                if self.on_action:
                    self.on_action(call.name, call.input)
                status = await execute_tool(call.name, call.input, session)
                recent.append(f"{call.name}:{json.dumps(call.input, sort_keys=True, default=str)}")
                if len(recent) >= _REPEAT_LIMIT and len(set(recent[-_REPEAT_LIMIT:])) == 1:
                    logger.warning("Repeating %s; nudging", call.name)
                    status += "\n\n" + _STUCK_NOTE.format(n=_REPEAT_LIMIT)
                shot = session.take_screenshot() if call.name == "screenshot" else None
                results.append(ToolResult(call.id, call.name, status, shot))

            self.messages.append(llm.tool_results(results))
            if any(r.image for r in results):
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
        dropping them would change the prefix and re-read the whole history
        from cold, to save nothing.
        """
        _close_dangling_tool_calls(self.messages, f"Cancelled — {reason}.")
        self.messages.append(
            llm.user(
                f"[System: {reason}. This turn is over; tool calls will not be executed "
                "now. The browser stays open and you continue on the next turn. Reply in "
                "text only, one or two sentences: what you have done so far and what you "
                "will do next. If you need anything from the user, ask for it now.]"
            )
        )
        try:
            reply = await asyncio.wait_for(self._ask(), timeout=config.WRAP_UP_TIMEOUT_S)
            self.messages.append(llm.assistant(reply))
            if reply.text:
                return reply.text
            logger.warning("Wrap-up returned no text")
        except Exception as e:
            logger.warning("Wrap-up failed: %s: %s", type(e).__name__, e)
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
                    f"Context from the conversation: {context.strip()}\n"
                    if context.strip()
                    else ""
                )
                human = FOLLOW_UP_TEMPLATE.format(
                    message=message or "continue", context_line=context_line
                )
                human += "\n" + await session.get_page()
            self.messages.append(llm.user(human))

            try:
                result = await asyncio.wait_for(self._run_loop(session), timeout=self.timeout_s)
            except TimeoutError:
                result = _OUT_OF_TIME
            except LLMError as e:
                logger.error("The model call failed: %s", e)
                return BrowserOutcome(str(e), WAITING, session.current_url())

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
        """Close the browser and the model client. The conversation is kept."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        await self.llm.close()
