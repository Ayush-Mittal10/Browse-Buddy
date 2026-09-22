"""The command line: one-shot for a single errand, interactive for a conversation.

    browse-buddy "find the cheapest flight Delhi to Mumbai on 5 Oct"
    browse-buddy --headed

Printing is deliberately kept here rather than in the agent. The agent reports
progress through two callbacks and knows nothing about a terminal, which is what
lets the same class sit behind a web UI later.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from browser_agent import __version__, config, llm
from browser_agent.agent import FINISHED, IN_PROGRESS, BrowserAgent, BrowserUnavailable

_QUIT = {"quit", "exit", "q", ":q"}


# ── output ───────────────────────────────────────────────────────────────────


class Printer:
    """Everything the user sees. Colour only when a terminal is attached."""

    def __init__(self, stream=None, colour: bool | None = None):
        self.stream = stream or sys.stdout
        if colour is None:
            colour = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.colour = colour

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def write(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def banner(self, model: str, headless: bool) -> None:
        where = "headless" if headless else "headed"
        self.write(self._paint(f"Browse Buddy {__version__}", "1"))
        self.write(self._paint(f"{model} · {where}", "2"))
        self.write(self._paint('Type a task, or "quit" to exit.', "2"))
        self.write()

    def narration(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self.write(self._paint(f"  {line.strip()}", "2"))

    def action(self, name: str, args: dict) -> None:
        self.write(self._paint(f"  → {describe_action(name, args)}", "36"))

    def report(self, text: str, state: str) -> None:
        self.write()
        self.write(text)
        if state == IN_PROGRESS:
            self.write()
            self.write(self._paint('(Out of budget — say "continue" to carry on.)', "2"))

    def note(self, text: str) -> None:
        self.write(self._paint(text, "2"))

    def error(self, text: str) -> None:
        print(self._paint(text, "31"), file=sys.stderr, flush=True)


def describe_action(name: str, args: dict) -> str:
    """One short line naming what is about to happen, in the user's terms."""
    ref = args.get("ref")
    if name == "navigate":
        return f"opening {args.get('url', '')}"
    if name == "click":
        return f"clicking [{ref}]"
    if name == "type_text":
        text = str(args.get("text") or "")
        shown = text[:30] + ("…" if len(text) > 30 else "")
        return f'typing "{shown}" into [{ref}]'
    if name == "select_option":
        return f"selecting {args.get('option', '')!r} in [{ref}]"
    if name == "press_key":
        return f"pressing {args.get('key', '')}"
    if name == "scroll":
        return f"scrolling {args.get('direction', 'down')}"
    if name == "read_text":
        return f"reading [{ref}]" if ref else "reading the page"
    if name == "wait":
        return f"waiting {args.get('seconds', 2)}s"
    if name == "switch_tab":
        return f"switching to tab {args.get('index', '')}"
    if name == "finish_task":
        return "wrapping up"
    return name.replace("_", " ")


# ── arguments ────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="browse-buddy",
        description="Give a browser a task in plain English and watch it get done.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="",
        help="What to do. Omit it to start an interactive session.",
    )
    parser.add_argument("--url", default="", help="Page to start from.")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window instead of running it hidden.",
    )
    parser.add_argument(
        "--provider",
        default="",
        choices=[*llm.PROVIDERS, "auto"],
        help="Which model to drive (default: auto — the first one you have a key for).",
    )
    parser.add_argument("--model", default="", help="Model name to use.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help=f"Actions allowed per turn (default: {config.MAX_STEPS}).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show internal logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only show the final answer.")
    parser.add_argument("--version", action="version", version=f"browse-buddy {__version__}")
    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """The printed output is the interface; logging is for diagnosis.

    So the agent's own INFO chatter stays off unless it is asked for, rather
    than interleaving with the progress lines the user is actually reading.
    """
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("browser_agent").setLevel(level)


# ── running ──────────────────────────────────────────────────────────────────


async def _ainput(prompt: str) -> str:
    """input() off the event loop, so the browser keeps breathing while we wait."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def one_turn(agent: BrowserAgent, out: Printer, message: str, url: str = "") -> str:
    outcome = await agent.run(message, start_url=url)
    out.report(outcome.text, outcome.state)
    return outcome.state


async def interactive(agent: BrowserAgent, out: Printer, first: str = "", url: str = "") -> int:
    out.banner(agent.model, agent.headless)
    message, start_url = first, url

    while True:
        if not message:
            try:
                message = (await _ainput("> ")).strip()
            except (EOFError, KeyboardInterrupt):
                out.write()
                return 0
            if not message:
                continue
            if message.lower() in _QUIT:
                return 0

        state = await one_turn(agent, out, message, start_url)
        message, start_url = "", ""
        out.write()
        if state == FINISHED:
            out.note("The browser has closed. Type another task, or \"quit\".")


async def _run(args: argparse.Namespace) -> int:
    out = Printer()
    agent = BrowserAgent(
        headless=not args.headed,
        provider=args.provider,
        model=args.model or None,
        max_steps=args.max_steps,
        on_text=None if args.quiet else out.narration,
        on_action=None if args.quiet else out.action,
    )
    try:
        if args.task:
            await one_turn(agent, out, args.task, args.url)
            return 0
        return await interactive(agent, out, url=args.url)
    finally:
        await agent.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.quiet)
    out = Printer()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        out.write()
        return 130
    except config.ConfigError as e:
        out.error(str(e))
        return 2
    except BrowserUnavailable as e:
        out.error(str(e))
        return 3


if __name__ == "__main__":
    sys.exit(main())
