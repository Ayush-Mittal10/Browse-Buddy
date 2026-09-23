"""Gemini, over the generateContent API.

The one with a usable free tier, which makes it the sensible default for a demo
somebody else is clicking on.

Its shapes differ from everyone else's in four ways that matter, and each is
handled below: turns are "user" and "model" rather than user and assistant, a
tool result is a part inside a user turn rather than a role of its own, the
function schema is an OpenAPI subset that rejects keys the other providers
accept — `additionalProperties` among them — and a function call carries a
thought signature that has to come back with it on the next request.
"""

from __future__ import annotations

import logging
import uuid

from browser_agent import config
from browser_agent.llm import LLMError, Reply, ToolCall, stream_with_retry, unreachable

logger = logging.getLogger(__name__)

# What Gemini's Schema actually accepts. Anything else is dropped rather than
# sent and rejected — our tool schemas are written for a stricter validator.
_SCHEMA_KEYS = {
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "minItems",
    "maxItems",
    "anyOf",
}


def _schema(node):
    """A tool's input schema, with everything Gemini does not understand removed."""
    if isinstance(node, list):
        return [_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned = {}
    for key, value in node.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: _schema(prop) for name, prop in value.items()}
        elif key in ("items", "anyOf"):
            cleaned[key] = _schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _declaration(tool: dict) -> dict:
    """One function declaration. A tool with no arguments declares none.

    An empty properties object is not the same thing as no parameters, and the
    empty version is the one that gets rejected.
    """
    declaration = {"name": tool["name"], "description": tool["description"]}
    parameters = _schema(tool["input_schema"])
    if parameters.get("properties"):
        declaration["parameters"] = parameters
    return declaration


# Free-tier capacity moves around between models, so when one has none another
# usually does. Smallest first, because the small ones are the least contended.
_BY_CONTENTION = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
)


def _collect(events: list[dict]) -> tuple[list[dict], str]:
    """One turn's parts, rebuilt from the events they arrived in.

    Text is split across events and has to be joined back up; a functionCall
    arrives whole, carrying the thoughtSignature that must be replayed with it
    on the next request. So only *pure* text parts merge into each other — a
    part with a signature attached is not a text fragment and has to survive as
    itself, which is the difference between a working conversation and a 400.
    """
    parts: list[dict] = []
    blocked = ""
    for event in events:
        blocked = blocked or (event.get("promptFeedback") or {}).get("blockReason") or ""
        candidates = event.get("candidates") or []
        if not candidates:
            continue
        for part in (candidates[0].get("content") or {}).get("parts") or []:
            plain = set(part) == {"text"}
            # The last event carries an empty text part to hang finishReason on.
            if plain and not part["text"]:
                continue
            if plain and parts and set(parts[-1]) == {"text"}:
                parts[-1] = {"text": parts[-1]["text"] + part["text"]}
            else:
                parts.append(part)
    return parts, blocked


def _try_instead(current: str) -> str:
    """A model worth suggesting when `current` has no capacity — never itself."""
    for name in _BY_CONTENTION:
        if name != current:
            return name
    return ""


class GeminiLLM:
    provider = "gemini"
    supports_images = True
    on_progress = None

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        self.name = model or config.GEMINI_MODEL
        # One key, or several to fall through. The free tier is rate limited per
        # project and a task of any length hits that before anything else, so a
        # second key is the difference between finishing and stopping halfway.
        self.keys = [api_key] if api_key else [
            k for k in (list(config.GEMINI_API_KEYS) or [config.GEMINI_API_KEY]) if k
        ]
        self.base_url = (base_url or config.GEMINI_BASE_URL).rstrip("/")
        # Where to start next time. Kept so a key that has said no is not asked
        # again on every subsequent call.
        self._key = 0
        self._client = None

    @property
    def api_key(self) -> str:
        return self.keys[self._key] if self.keys else ""

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT_S)
        return self._client

    def _render(self, messages: list[dict]) -> list[dict]:
        contents: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": message["text"]}]})
            elif role == "assistant":
                # Replay the model's own parts when we have them. A functionCall
                # comes back carrying a thoughtSignature, and sending it again
                # without one is a 400 — the same contract Anthropic has for
                # thinking blocks. Rebuilding the part by hand loses it.
                parts = message.get("raw") or []
                if not parts:
                    if message.get("text"):
                        parts.append({"text": message["text"]})
                    for call in message.get("tool_calls") or []:
                        parts.append({"functionCall": {"name": call.name, "args": call.input}})
                # A turn with no parts at all is not one Gemini will accept.
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:
                parts = []
                for result in message["results"]:
                    parts.append(
                        {
                            "functionResponse": {
                                "name": result.name,
                                "response": {"result": result.text},
                            }
                        }
                    )
                    if result.image:
                        data, mime = result.image
                        parts.append({"inlineData": {"mimeType": mime, "data": data}})
                contents.append({"role": "user", "parts": parts})
        return contents

    async def complete(self, *, system: str, tools: list[dict], messages: list[dict]) -> Reply:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": self._render(messages),
            "tools": [{"functionDeclarations": [_declaration(t) for t in tools]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": config.MAX_TOKENS},
        }
        # Streamed, because the plain endpoint sends nothing at all until the
        # whole answer exists — measured, first byte and last byte in the same
        # instant — which made every slow reply look exactly like a dead
        # connection and cost the run a 60s wait to find out otherwise.
        url = f"{self.base_url}/models/{self.name}:streamGenerateContent?alt=sse"
        # Each key gets one go. stream_with_retry has already waited out the
        # transient case by the time one answers 429, so a key that still says
        # no here is out of quota rather than momentarily busy.
        for attempt in range(len(self.keys)):
            try:
                response = await stream_with_retry(
                    self._http(),
                    url,
                    json=body,
                    headers={"x-goog-api-key": self.api_key},
                    gap_s=config.HTTP_STREAM_GAP_S,
                    total_s=config.HTTP_TIMEOUT_S,
                    on_progress=self.on_progress,
                )
            except Exception as e:
                raise LLMError(unreachable("Gemini", e)) from e
            if response.status_code != 429 or attempt == len(self.keys) - 1:
                break
            self._key = (self._key + 1) % len(self.keys)
            logger.info("Gemini key %d is rate limited; trying the next one", attempt + 1)

        if response.status_code == 429:
            spare = (
                " All of them are limited right now."
                if len(self.keys) > 1
                else " Set GEMINI_API_KEYS with more than one key to fall through to another."
            )
            raise LLMError(
                "Gemini's free tier is rate limited and this key has hit it. "
                "Wait a minute, or use a different provider." + spare
            )
        if response.status_code == 503:
            instead = _try_instead(self.name)
            raise LLMError(
                f"Gemini has no free capacity for {self.name} right now. "
                + (f"Try again in a moment, or switch to {instead}." if instead
                   else "Try again in a moment.")
            )
        if response.status_code >= 400:
            raise LLMError(f"Gemini returned {response.status_code}: {_detail(response)}")

        parts, blocked = _collect(response.events)
        if blocked:
            raise LLMError(f"Gemini declined the request ({blocked}).")
        if not parts:
            return Reply()

        text, calls = [], []
        for part in parts:
            if "text" in part and part["text"]:
                text.append(part["text"])
            function_call = part.get("functionCall")
            if function_call:
                # Gemini hands out no call ids, and the loop pairs results to
                # calls by id, so make one here.
                calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=function_call.get("name") or "",
                        input=function_call.get("args") or {},
                    )
                )
        # raw is the turn as Gemini rendered it, thought signatures included.
        return Reply(text="\n".join(text).strip(), tool_calls=calls, raw=parts)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _detail(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error or body)[:300]
