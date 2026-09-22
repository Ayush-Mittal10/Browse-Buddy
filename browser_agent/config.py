"""Every knob the agent has, read from the environment once at import time.

Defaults are chosen so that the only thing you *must* set is
``ANTHROPIC_API_KEY``. Everything else has a value that works.

Naming: all overrides are prefixed ``BROWSER_AGENT_`` except ``ANTHROPIC_API_KEY``,
which the Anthropic SDK reads under that name anyway.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def load_dotenv(start: Path | None = None) -> Path | None:
    """Read the nearest .env into the environment. Returns the file it used.

    Enough of the format to be useful and no more: KEY=value a line, blank
    lines and # comments skipped, optional surrounding quotes, an optional
    `export` prefix. Not worth a dependency.

    A variable already set in the environment always wins, so an explicit
    `export` on the command line beats the file — which is the behaviour people
    expect when they override something for one run.
    """
    directory = (start or Path.cwd()).resolve()
    for folder in [directory, *directory.parents]:
        path = folder / ".env"
        if path.is_file():
            break
    else:
        return None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
    return path


# Read before anything below looks at the environment, so a .env can set any of
# it. Import-time, because every value here is read once at import too.
DOTENV_PATH = load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        # A typo in an env var shouldn't take the agent down.
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


# --- LLM ---------------------------------------------------------------------

# The key is deliberately not required at import time — importing the package
# should never fail. It is checked when an agent actually needs to talk to the
# API; see require_api_key() below.
API_KEY = _env_str("ANTHROPIC_API_KEY", "")

MODEL = _env_str("BROWSER_AGENT_MODEL", "claude-opus-5")

# Each turn is a short bit of reasoning plus a tool call, not an essay, but this
# ceiling also has to cover the model's thinking — which is on by default on the
# current models. It is here to stop a runaway response, not to shape the answer.
MAX_TOKENS = _env_int("BROWSER_AGENT_MAX_TOKENS", 8192)


# --- Agent loop --------------------------------------------------------------

# How many model turns one task gets before we stop and report progress.
MAX_STEPS = _env_int("BROWSER_AGENT_MAX_STEPS", 40)

# Wall-clock budget for a single task, in seconds.
TIMEOUT_S = _env_int("BROWSER_AGENT_TIMEOUT_S", 240)

# Screenshots are expensive in tokens. Only the most recent ones stay in the
# conversation; older ones are replaced with a note.
MAX_SCREENSHOTS = _env_int("BROWSER_AGENT_MAX_SCREENSHOTS", 2)


# --- Browser -----------------------------------------------------------------

# Headless by default; flip this (or pass --headed) to watch the agent work.
HEADLESS = _env_bool("BROWSER_AGENT_HEADLESS", True)

VIEWPORT_WIDTH = _env_int("BROWSER_AGENT_VIEWPORT_WIDTH", 1280)
VIEWPORT_HEIGHT = _env_int("BROWSER_AGENT_VIEWPORT_HEIGHT", 800)
VIEWPORT = {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}

LOCALE = _env_str("BROWSER_AGENT_LOCALE", "en-US")
TIMEZONE = _env_str("BROWSER_AGENT_TIMEZONE", "Asia/Kolkata")

# Page load. Generous, because a slow page is normal and a failed navigation is
# something the model should see rather than something that kills the run.
NAV_TIMEOUT_MS = _env_int("BROWSER_AGENT_NAV_TIMEOUT_MS", 25_000)

# A single click/type/select. Short, because a miss here usually means a stale
# element reference and the fix is to take a fresh look at the page.
ACTION_TIMEOUT_MS = _env_int("BROWSER_AGENT_ACTION_TIMEOUT_MS", 8_000)

# Breathing room after an action so single-page apps finish painting before we
# read the DOM back.
SETTLE_MS = _env_int("BROWSER_AGENT_SETTLE_MS", 700)

# A hung page.evaluate must not hang the agent.
SNAPSHOT_TIMEOUT_S = _env_int("BROWSER_AGENT_SNAPSHOT_TIMEOUT_S", 12)

# Longest a single explicit wait() can pause for.
MAX_WAIT_S = _env_int("BROWSER_AGENT_MAX_WAIT_S", 10)

# Low enough to keep the base64 payload small, high enough to read UI text.
SCREENSHOT_JPEG_QUALITY = _env_int("BROWSER_AGENT_SCREENSHOT_QUALITY", 55)


# --- Storage -----------------------------------------------------------------

STORE_DIR = Path(
    _env_str("BROWSER_AGENT_STORE_DIR", str(Path.home() / ".browser-agent"))
).expanduser()

CONVERSATIONS_DIR = STORE_DIR / "conversations"


class ConfigError(RuntimeError):
    """Something the user has to fix before the agent can run."""


def require_api_key() -> str:
    """Return the Anthropic API key, or explain how to set it."""
    key = API_KEY or (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set. Export it, or put it in a .env file:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return key
