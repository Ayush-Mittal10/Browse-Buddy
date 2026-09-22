"""OpenAI, over the chat completions API.

Raw HTTP rather than the SDK, for the same reason the Ollama backend is: this
project has two dependencies and adding a third for forty lines of JSON is a bad
trade. The wire format is stable and public.

The model is configurable and worth configuring. Tool calling on the newest
reasoning models is steering towards the responses API, so the default here is a
model that does tools over chat completions without ceremony.
"""

from __future__ import annotations

import json
import logging

from browser_agent import config
from browser_agent.llm import LLMError, Reply, ToolCall, post_with_retry

logger = logging.getLogger(__name__)


class OpenAILLM:
    """Talks to any OpenAI-compatible /chat/completions endpoint."""

    provider = "openai"
    supports_images = True

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        self.name = model or config.OPENAI_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT_S)
        return self._client

    def _render(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for message in messages:
            role = message["role"]
            if role == "user":
                out.append({"role": "user", "content": message["text"]})
            elif role == "assistant":
                turn: dict = {"role": "assistant", "content": message.get("text") or None}
                calls = message.get("tool_calls") or []
                if calls:
                    turn["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.input)},
                        }
                        for c in calls
                    ]
                out.append(turn)
            else:
                images = []
                for result in message["results"]:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.id,
                            "content": result.text,
                        }
                    )
                    if result.image:
                        images.append(result.image)
                # A tool message cannot carry an image, so a screenshot follows
                # as its own user turn — after every tool result in the batch,
                # so no call is left sitting without its answer.
                for data, mime in images:
                    out.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Screenshot of the current page:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{data}"},
                                },
                            ],
                        }
                    )
        return out

    async def complete(self, *, system: str, tools: list[dict], messages: list[dict]) -> Reply:
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        body = {
            "model": self.name,
            "messages": self._render(system, messages),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ],
        }
        try:
            response = await post_with_retry(
                self._http(),
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except Exception as e:
            raise LLMError(f"Could not reach OpenAI: {type(e).__name__}: {e}") from e

        if response.status_code >= 400:
            raise LLMError(f"OpenAI returned {response.status_code}: {_detail(response)}")

        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"OpenAI sent a reply we could not read: {e}") from e

        calls = [
            ToolCall(
                id=call.get("id") or "",
                name=(call.get("function") or {}).get("name") or "",
                input=_arguments((call.get("function") or {}).get("arguments")),
            )
            for call in message.get("tool_calls") or []
        ]
        return Reply(text=(message.get("content") or "").strip(), tool_calls=calls, raw=None)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _arguments(raw) -> dict:
    """Tool arguments arrive as a JSON string. Never match on it as text."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _detail(response) -> str:
    """The provider's own explanation, which is usually the useful part."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error or body)[:300]
