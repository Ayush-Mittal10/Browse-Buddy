"""Argument parsing, the progress lines, and the two run modes.

The agent is stubbed throughout — what is under test is the terminal front end,
not the loop it drives.
"""

from __future__ import annotations

import io

import pytest

from browser_agent import config
from browser_agent.agent import FINISHED, IN_PROGRESS, WAITING, BrowserOutcome
from browser_agent.cli import Printer, build_parser, describe_action, interactive, main, one_turn


class FakeAgent:
    """Answers each run() with the next scripted outcome."""

    def __init__(self, *outcomes, model: str = "test-model", headless: bool = True):
        self.outcomes = list(outcomes)
        self.model = model
        self.headless = headless
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def run(self, message, *, start_url="", context=""):
        self.calls.append((message, start_url))
        if self.outcomes:
            return self.outcomes.pop(0)
        return BrowserOutcome("done", FINISHED, "")

    async def close(self):
        self.closed = True


def printer() -> tuple[Printer, io.StringIO]:
    buf = io.StringIO()
    return Printer(stream=buf, colour=False), buf


# --- arguments ----------------------------------------------------------------


def test_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.task == ""
    assert args.url == ""
    assert args.headed is False
    assert args.model == ""
    assert args.max_steps is None


def test_a_task_can_be_given_positionally() -> None:
    args = build_parser().parse_args(["find a cafe"])
    assert args.task == "find a cafe"


def test_every_flag_parses() -> None:
    args = build_parser().parse_args(
        [
            "do a thing",
            "--url", "https://example.com",
            "--headed",
            "--model", "x",
            "--max-steps", "5",
        ]
    )
    assert args.url == "https://example.com"
    assert args.headed is True
    assert args.model == "x"
    assert args.max_steps == 5


@pytest.mark.parametrize("flag", ["-v", "--verbose", "-q", "--quiet"])
def test_verbosity_flags_parse(flag: str) -> None:
    build_parser().parse_args([flag])


def test_version_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_help_mentions_the_defaults_it_promises() -> None:
    text = build_parser().format_help()
    assert str(config.MAX_STEPS) in text
    # The model default depends on the provider, so the help names the
    # providers rather than a model that may not apply.
    for provider in ("anthropic", "ollama", "auto"):
        assert provider in text


def test_the_provider_flag_parses() -> None:
    assert build_parser().parse_args(["--provider", "ollama"]).provider == "ollama"
    assert build_parser().parse_args([]).provider == ""


def test_the_provider_reaches_the_agent(monkeypatch) -> None:
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return FakeAgent(BrowserOutcome("Done.", FINISHED, ""))

    monkeypatch.setattr("browser_agent.cli.BrowserAgent", build)
    main(["task", "--provider", "ollama"])

    assert seen["provider"] == "ollama"


# --- progress lines -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("navigate", {"url": "https://example.com"}, "opening https://example.com"),
        ("click", {"ref": 4}, "clicking [4]"),
        ("select_option", {"ref": 2, "option": "Japan"}, "selecting 'Japan' in [2]"),
        ("press_key", {"key": "Enter"}, "pressing Enter"),
        ("scroll", {"direction": "up"}, "scrolling up"),
        ("scroll", {}, "scrolling down"),
        ("read_text", {}, "reading the page"),
        ("read_text", {"ref": 7}, "reading [7]"),
        ("wait", {"seconds": 3}, "waiting 3s"),
        ("switch_tab", {"index": 2}, "switching to tab 2"),
        ("finish_task", {"report": "..."}, "wrapping up"),
        ("get_page", {}, "get page"),
    ],
)
def test_actions_are_described_in_plain_words(name: str, args: dict, expected: str) -> None:
    assert describe_action(name, args) == expected


def test_typed_text_is_shown_but_truncated() -> None:
    assert describe_action("type_text", {"ref": 1, "text": "hello"}) == 'typing "hello" into [1]'
    long = describe_action("type_text", {"ref": 1, "text": "x" * 80})
    assert "…" in long
    assert "x" * 80 not in long


def test_narration_is_indented_and_blank_lines_dropped() -> None:
    out, buf = printer()
    out.narration("Searching…\n\n  Found it.")
    assert buf.getvalue() == "  Searching…\n  Found it.\n"


def test_colour_is_off_for_a_plain_stream() -> None:
    out, buf = printer()
    out.narration("hello")
    assert "\033[" not in buf.getvalue()


def test_colour_is_on_for_a_terminal() -> None:
    buf = io.StringIO()
    buf.isatty = lambda: True  # type: ignore[method-assign]
    Printer(stream=buf).narration("hello")
    assert "\033[" in buf.getvalue()


def test_an_out_of_budget_report_says_how_to_continue() -> None:
    out, buf = printer()
    out.report("Got halfway.", IN_PROGRESS)
    assert "Got halfway." in buf.getvalue()
    assert "continue" in buf.getvalue()


def test_a_finished_report_has_no_continuation_hint() -> None:
    out, buf = printer()
    out.report("All done.", FINISHED)
    assert "continue" not in buf.getvalue()


# --- one-shot -----------------------------------------------------------------


async def test_one_shot_runs_the_task_and_prints_the_report() -> None:
    out, buf = printer()
    agent = FakeAgent(BrowserOutcome("Three cafés nearby.", FINISHED, "https://x"))

    state = await one_turn(agent, out, "find cafes", "https://maps.example")

    assert state == FINISHED
    assert agent.calls == [("find cafes", "https://maps.example")]
    assert "Three cafés nearby." in buf.getvalue()


def test_main_runs_a_one_shot_task(monkeypatch) -> None:
    # Sync on purpose: main() owns the event loop, so it cannot be awaited.
    agent = FakeAgent(BrowserOutcome("Done.", FINISHED, ""))
    monkeypatch.setattr("browser_agent.cli.BrowserAgent", lambda **kw: agent)

    assert main(["do a thing"]) == 0
    assert agent.calls == [("do a thing", "")]
    assert agent.closed


def test_flags_reach_the_agent(monkeypatch) -> None:
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return FakeAgent(BrowserOutcome("Done.", FINISHED, ""))

    monkeypatch.setattr("browser_agent.cli.BrowserAgent", build)
    main(["task", "--headed", "--model", "some-model", "--max-steps", "7"])

    assert seen["headless"] is False
    assert seen["model"] == "some-model"
    assert seen["max_steps"] == 7
    assert seen["on_text"] is not None


def test_quiet_silences_the_progress_lines(monkeypatch) -> None:
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return FakeAgent(BrowserOutcome("Done.", FINISHED, ""))

    monkeypatch.setattr("browser_agent.cli.BrowserAgent", build)
    main(["task", "--quiet"])

    assert seen["on_text"] is None
    assert seen["on_action"] is None


# --- interactive --------------------------------------------------------------


async def test_interactive_runs_turns_until_quit(monkeypatch) -> None:
    out, buf = printer()
    agent = FakeAgent(
        BrowserOutcome("Which size?", WAITING, ""),
        BrowserOutcome("Ordered.", FINISHED, ""),
    )
    replies = iter(["order a shirt", "medium", "quit"])
    monkeypatch.setattr("browser_agent.cli._ainput", lambda prompt: _immediately(next(replies)))

    assert await interactive(agent, out) == 0
    assert agent.calls == [("order a shirt", ""), ("medium", "")]
    assert "Which size?" in buf.getvalue()
    assert "Ordered." in buf.getvalue()


async def test_a_first_task_skips_the_first_prompt(monkeypatch) -> None:
    out, _ = printer()
    agent = FakeAgent(BrowserOutcome("Done.", FINISHED, ""))
    monkeypatch.setattr("browser_agent.cli._ainput", lambda prompt: _immediately("quit"))

    await interactive(agent, out, first="go", url="https://example.com")

    assert agent.calls == [("go", "https://example.com")]


async def test_blank_lines_are_ignored(monkeypatch) -> None:
    out, _ = printer()
    agent = FakeAgent()
    replies = iter(["", "   ", "quit"])
    monkeypatch.setattr("browser_agent.cli._ainput", lambda prompt: _immediately(next(replies)))

    assert await interactive(agent, out) == 0
    assert agent.calls == []


@pytest.mark.parametrize("word", ["quit", "exit", "q", "QUIT"])
async def test_the_quit_words(monkeypatch, word: str) -> None:
    out, _ = printer()
    agent = FakeAgent()
    monkeypatch.setattr("browser_agent.cli._ainput", lambda prompt: _immediately(word))

    assert await interactive(agent, out) == 0


@pytest.mark.parametrize("raised", [EOFError, KeyboardInterrupt])
async def test_ctrl_c_and_ctrl_d_leave_quietly(monkeypatch, raised) -> None:
    out, _ = printer()
    agent = FakeAgent()

    async def boom(prompt):
        raise raised()

    monkeypatch.setattr("browser_agent.cli._ainput", boom)
    assert await interactive(agent, out) == 0


async def test_a_closed_browser_is_announced(monkeypatch) -> None:
    out, buf = printer()
    agent = FakeAgent(BrowserOutcome("All done.", FINISHED, ""))
    replies = iter(["do it", "quit"])
    monkeypatch.setattr("browser_agent.cli._ainput", lambda prompt: _immediately(next(replies)))

    await interactive(agent, out)
    assert "browser has closed" in buf.getvalue()


# --- failure modes ------------------------------------------------------------


def test_a_missing_api_key_is_explained_not_traced(monkeypatch, capsys) -> None:
    def build(**kwargs):
        raise config.ConfigError("ANTHROPIC_API_KEY is not set. Export it…")

    monkeypatch.setattr("browser_agent.cli.BrowserAgent", build)

    assert main(["task"]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_a_missing_browser_is_explained_not_traced(monkeypatch, capsys) -> None:
    from browser_agent.agent import BrowserUnavailable

    class Failing(FakeAgent):
        async def run(self, *a, **kw):
            raise BrowserUnavailable("Chromium is not installed. Run: playwright install chromium")

    monkeypatch.setattr("browser_agent.cli.BrowserAgent", lambda **kw: Failing())

    assert main(["task"]) == 3
    assert "playwright install chromium" in capsys.readouterr().err


def test_an_interrupted_run_exits_with_the_usual_code(monkeypatch) -> None:
    class Interrupted(FakeAgent):
        async def run(self, *a, **kw):
            raise KeyboardInterrupt

    monkeypatch.setattr("browser_agent.cli.BrowserAgent", lambda **kw: Interrupted())
    assert main(["task"]) == 130


async def _immediately(value):
    return value
