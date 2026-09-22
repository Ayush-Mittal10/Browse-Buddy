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

# "anthropic", "openai", "gemini", "ollama", or "auto" — the first hosted
# provider with a key, falling back to the local model, so this runs out of the
# box whatever you happen to have.
PROVIDER = _env_str("BROWSER_AGENT_PROVIDER", "auto").lower()

# The key is deliberately not required at import time — importing the package
# should never fail. It is checked when an agent actually needs to talk to the
# API; see require_api_key() below.
API_KEY = _env_str("ANTHROPIC_API_KEY", "")

MODEL = _env_str("BROWSER_AGENT_MODEL", "claude-opus-5")

# OpenAI.
OPENAI_API_KEY = _env_str("OPENAI_API_KEY", "")
OPENAI_MODEL = _env_str("BROWSER_AGENT_OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_BASE_URL = _env_str("BROWSER_AGENT_OPENAI_BASE_URL", "https://api.openai.com/v1")

# Gemini. GOOGLE_API_KEY is what Google's own tooling sets, so accept either.
GEMINI_API_KEY = _env_str("GEMINI_API_KEY", "") or _env_str("GOOGLE_API_KEY", "")
# Flash-lite rather than the bigger Flash models on purpose: measured on the
# free tier, gemini-3.8-flash, gemini-3.5-flash and gemini-flash-latest all
# answered 503 "high demand" while flash-lite answered in 1.2s. A default
# that is usually unavailable is not a default.
GEMINI_MODEL = _env_str("BROWSER_AGENT_GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_BASE_URL = _env_str(
    "BROWSER_AGENT_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)

# How long to wait on a hosted provider.
HTTP_TIMEOUT_S = _env_int("BROWSER_AGENT_HTTP_TIMEOUT_S", 120)

# Local models, via Ollama.
OLLAMA_HOST = _env_str("BROWSER_AGENT_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = _env_str("BROWSER_AGENT_OLLAMA_MODEL", "qwen3:8b")
# Big enough for a dozen steps of page snapshots. Every token of it is KV cache
# held in memory, so raising this costs RAM whether or not it gets used.
OLLAMA_NUM_CTX = _env_int("BROWSER_AGENT_OLLAMA_NUM_CTX", 32768)
# Off by default. Measured on qwen3:8b: 27.7s a step with thinking against 1.7s
# without, for the same correct tool call. Choosing one element off a page is
# not a reasoning problem.
OLLAMA_THINK = _env_bool("BROWSER_AGENT_OLLAMA_THINK", False)
# A cold prompt on a long history can take minutes before the cache is warm.
OLLAMA_TIMEOUT_S = _env_int("BROWSER_AGENT_OLLAMA_TIMEOUT_S", 600)

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


# --- Serving this to other people --------------------------------------------

def _env_list(name: str) -> tuple[str, ...]:
    raw = (os.getenv(name) or "").replace(",", " ").split()
    return tuple(item.strip().lower().lstrip(".") for item in raw if item.strip())


# Empty means any public website. Set it to a handful of hosts before putting
# this somewhere strangers can type into it: an agent that will visit anything
# is an open proxy wearing a nice hat.
ALLOWED_DOMAINS = _env_list("BROWSER_AGENT_ALLOWED_DOMAINS")

# How many browsers may run at once. Each one is a Chromium, so this is really
# a statement about the box it is on.
WEB_MAX_SESSIONS = _env_int("BROWSER_AGENT_WEB_MAX_SESSIONS", 2)
# Tighter than the CLI on purpose: a visitor's runaway task spends the host's
# tokens, not their own.
WEB_MAX_STEPS = _env_int("BROWSER_AGENT_WEB_MAX_STEPS", 15)
WEB_MAX_TASKS = _env_int("BROWSER_AGENT_WEB_MAX_TASKS", 6)
# A forgotten tab holds a browser open, so hang up on one that goes quiet.
WEB_IDLE_TIMEOUT_S = _env_int("BROWSER_AGENT_WEB_IDLE_TIMEOUT_S", 300)
# Whether visitors may supply their own key. Their key, their spend.
WEB_ALLOW_BYOK = _env_bool("BROWSER_AGENT_WEB_ALLOW_BYOK", True)


# --- The live view -----------------------------------------------------------

# Frames streamed out of the DevTools protocol, for a viewer watching the run.
# Measured at ~10fps and ~67 KB/s with these values, from a headless browser.
LIVE_QUALITY = _env_int("BROWSER_AGENT_LIVE_QUALITY", 50)
LIVE_MAX_WIDTH = _env_int("BROWSER_AGENT_LIVE_MAX_WIDTH", 1024)
LIVE_MAX_HEIGHT = _env_int("BROWSER_AGENT_LIVE_MAX_HEIGHT", 640)
# 1 means every frame Chromium paints. Raise it to thin the stream out.
LIVE_EVERY_NTH = _env_int("BROWSER_AGENT_LIVE_EVERY_NTH", 1)


# --- How much page to show the model -----------------------------------------

# A big model copes with a hundred-odd numbered elements; a small one does not.
# Measured on qwen3:8b: given 120 elements it clicked the wrong one, given 40 it
# clicked the right one. So the local defaults are deliberately tighter — less
# page per step, but the right element.
_ANY_HOSTED_KEY = bool(API_KEY or OPENAI_API_KEY or GEMINI_API_KEY)
_LOCAL = PROVIDER == "ollama" or (PROVIDER == "auto" and not _ANY_HOSTED_KEY)

MAX_ELEMENTS = _env_int("BROWSER_AGENT_MAX_ELEMENTS", 40 if _LOCAL else 120)
MAX_TEXT_CHARS = _env_int("BROWSER_AGENT_MAX_TEXT_CHARS", 1500 if _LOCAL else 3500)

# Asking for the progress line when a turn runs out. It must not double the
# run, but a local model re-reading a long history from cold is slow, and a
# wrap-up that times out loses the report the budget was spent earning.
WRAP_UP_TIMEOUT_S = _env_int("BROWSER_AGENT_WRAP_UP_TIMEOUT_S", 180 if _LOCAL else 40)


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
