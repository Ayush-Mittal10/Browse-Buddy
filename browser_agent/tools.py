"""The browser as tools: what the model is offered, and what running one does.

Two halves. ``TOOLS`` is the JSON the Anthropic API sees — the descriptions are
the entire interface, so they say when to reach for a tool and what comes back,
and say nothing about Playwright. ``execute_tool`` maps a call onto a
BrowserSession method.

The session is found through a ContextVar, never passed by the model, so a tool
call cannot be aimed at another task's browser. And nothing here raises: a tool
result is a string, including when the thing failed, because an exception
escaping into the agent loop ends the turn while a sentence lets the model try
something else.
"""

from __future__ import annotations

import logging

from browser_agent.session import BrowserSession, current_session

logger = logging.getLogger(__name__)

_REF = {
    "type": "integer",
    "description": "The [number] of the element, from the latest snapshot.",
}

# The order is part of the cached prompt prefix — keep it stable.
TOOLS: list[dict] = [
    {
        "name": "navigate",
        "description": (
            "Open a web page. Use a full URL (https://...) — a site's homepage, a search "
            "engine results URL, or a page you know. Returns a snapshot of the loaded "
            "page: its title, URL, numbered interactive elements and visible text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to open."},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "click",
        "description": (
            "Click an element by its [number] from the latest snapshot — a link, button, "
            "tab, checkbox, menu item or dropdown trigger. Returns a fresh snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ref": _REF},
            "required": ["ref"],
            "additionalProperties": False,
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type into a text field, search box or editor by its [number]. Replaces what "
            "was there. Set press_enter=true to submit (search boxes, single-field forms). "
            "Works for passwords and codes too — the value is never echoed back. Returns "
            "a fresh snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": _REF,
                "text": {"type": "string", "description": "The text to type."},
                "press_enter": {
                    "type": "boolean",
                    "description": "Press Enter after typing, to submit.",
                },
            },
            "required": ["ref", "text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "select_option",
        "description": (
            "Choose an option in a dropdown (<select>) by its [number]; `option` is the "
            "visible option text as shown in the snapshot. Returns a fresh snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": _REF,
                "option": {
                    "type": "string",
                    "description": "The option's visible text, as listed in the snapshot.",
                },
            },
            "required": ["ref", "option"],
            "additionalProperties": False,
        },
    },
    {
        "name": "press_key",
        "description": (
            "Press a keyboard key on the page: Enter, Escape (close a popup), Tab, "
            "ArrowDown/ArrowUp (move in a list), PageDown, End. Returns a fresh snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key name, e.g. Enter."},
            },
            "required": ["key"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scroll",
        "description": (
            'Scroll the page (or the list under the centre of the screen) "down" or "up" '
            'by a number of screens. Use it to reach elements marked "~" or to read more '
            "of a long page. Returns a fresh snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["down", "up"]},
                "pages": {
                    "type": "number",
                    "description": "How many screens to scroll. Default 1.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "go_back",
        "description": "Go back to the previous page in this tab. Returns a fresh snapshot.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_page",
        "description": (
            "Re-read the current page without acting: title, URL, numbered elements, "
            "visible text. Use after waiting, or when the numbers may have changed."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_text",
        "description": (
            "Read the text of the whole page, or of one element by [number], in chunks. "
            "Use it for long articles, result lists or tables the snapshot truncated. "
            "Pass `start` from the previous chunk's end to continue reading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "integer",
                    "description": "Read one element instead of the whole page.",
                },
                "start": {
                    "type": "integer",
                    "description": "Character offset to read from. Default 0.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "screenshot",
        "description": (
            "Take a picture of what is currently on screen. Use it only when the text "
            "snapshot is not enough — maps, images, charts, a layout question, or when "
            "the page looks broken. The image arrives right after this result."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "wait",
        "description": (
            "Pause for the page to finish loading or updating (0.5–10 seconds), then "
            "return a fresh snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "How long to wait. Default 2."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "switch_tab",
        "description": (
            'Switch to another open tab by its number from the "Tabs:" line of the '
            "snapshot. Returns a snapshot of that tab."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "The tab number, starting at 1."},
            },
            "required": ["index"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish_task",
        "description": (
            "Call this when the task is complete and nothing more is expected on this "
            "site. `report` is your final message to the user: plain text, findings "
            "first, the key facts with the site each came from, exact phone numbers, "
            "prices, times and addresses, and anything that still needs them. The browser "
            "stays open and the conversation continues, so a follow-up costs nothing. To "
            "ask the user something before you can go on, reply in text instead of "
            "calling this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "Your final message to the user."},
            },
            "required": ["report"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)

FINISHED_NOTE = "Task marked finished."


async def _finish_task(session: BrowserSession, report: str) -> str:
    session.finished_report = (report or "").strip()
    return FINISHED_NOTE


# One entry per tool, each pulling its own arguments out of the model's input.
# Arguments are read defensively: a missing or misspelled key becomes a default
# the session can explain, never a KeyError in the middle of a turn.
_DISPATCH = {
    "navigate": lambda s, a: s.navigate(a.get("url") or ""),
    "click": lambda s, a: s.click(a.get("ref")),
    "type_text": lambda s, a: s.type_text(
        a.get("ref"), a.get("text") or "", bool(a.get("press_enter"))
    ),
    "select_option": lambda s, a: s.select_option(a.get("ref"), a.get("option") or ""),
    "press_key": lambda s, a: s.press_key(a.get("key") or ""),
    "scroll": lambda s, a: s.scroll(a.get("direction") or "down", a.get("pages") or 1.0),
    "go_back": lambda s, a: s.go_back(),
    "get_page": lambda s, a: s.get_page(),
    "read_text": lambda s, a: s.read_text(a.get("ref") or "", a.get("start") or 0),
    "screenshot": lambda s, a: s.screenshot(),
    "wait": lambda s, a: s.wait(a.get("seconds") or 2.0),
    "switch_tab": lambda s, a: s.switch_tab(a.get("index")),
    "finish_task": lambda s, a: _finish_task(s, a.get("report") or ""),
}


async def execute_tool(
    name: str,
    args: dict | None = None,
    session: BrowserSession | None = None,
) -> str:
    """Run one tool call and return its result as a string.

    Never raises. The agent loop feeds whatever comes back to the model as a
    tool result, so a failure has to arrive as something the model can read and
    act on rather than as an exception that ends the turn.
    """
    session = session or current_session()
    if session is None:
        return "No browser session is active. The task cannot continue."

    action = _DISPATCH.get(name)
    if action is None:
        available = ", ".join(sorted(TOOL_NAMES))
        return f"There is no tool called {name!r}. Available tools: {available}."

    try:
        return await action(session, args or {})
    except Exception as e:
        logger.warning("Tool %s failed: %s: %s", name, type(e).__name__, e)
        return f"The {name} tool failed: {type(e).__name__}: {e}"
