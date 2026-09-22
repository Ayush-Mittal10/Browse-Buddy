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
from browser_agent.llm import LLMError, Reply, ToolCall, post_with_retry

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


class GeminiLLM:
    supports_images = True

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        self.name = model or config.GEMINI_MODEL
        self.api_key = api_key or config.GEMINI_API_KEY
        self.base_url = (base_url or config.GEMINI_BASE_URL).rstrip("/")
        self._client = None

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
        url = f"{self.base_url}/models/{self.name}:generateContent"
        try:
            response = await post_with_retry(
                self._http(), url, json=body, headers={"x-goog-api-key": self.api_key}
            )
        except Exception as e:
            raise LLMError(f"Could not reach Gemini: {type(e).__name__}: {e}") from e

        if response.status_code == 429:
            raise LLMError(
                "Gemini's free tier is rate limited and this key has hit it. "
                "Wait a minute, or use a different provider."
            )
        if response.status_code == 503:
            raise LLMError(
                f"Gemini has no free capacity for {self.name} right now. "
                "A smaller model (gemini-3.1-flash-lite) is usually available."
            )
        if response.status_code >= 400:
            raise LLMError(f"Gemini returned {response.status_code}: {_detail(response)}")

        try:
            payload = response.json()
        except ValueError as e:
            raise LLMError(f"Gemini sent a reply we could not read: {e}") from e

        candidates = payload.get("candidates") or []
        if not candidates:
            # A blocked prompt comes back as a 200 with no candidate at all.
            blocked = (payload.get("promptFeedback") or {}).get("blockReason")
            if blocked:
                raise LLMError(f"Gemini declined the request ({blocked}).")
            return Reply()

        parts = (candidates[0].get("content") or {}).get("parts") or []
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
