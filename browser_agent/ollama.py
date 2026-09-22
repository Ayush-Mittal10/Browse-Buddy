"""A model running on this machine, through Ollama.

Three things had to be measured rather than assumed, and they shape this file:

* Thinking costs everything and buys nothing here. Qwen3 with thinking on took
  27.7s for the step it got right in 1.7s with thinking off — same tool call.
  Picking one action off a page is not a reasoning problem.
* The history must stay append-only. Ollama reuses the KV cache for an
  unchanged prefix, which is the difference between ~2s a step and a cold
  30k-token prompt that takes four minutes. Editing an earlier message throws
  that away.
* Small models need a smaller page. Handed 120 numbered elements, Qwen3-8B
  picked the wrong one; handed 40, it picked right. The snapshot limits for a
  local model are deliberately tighter than for a hosted one.
"""

from __future__ import annotations

import json
import logging
import uuid

from browser_agent import config
from browser_agent.llm import LLMError, Reply, ToolCall

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Talks to a local Ollama server over its chat API."""

    provider = "ollama"

    # Text-only models cannot read a screenshot. The agent checks this and
    # stops offering the tool rather than letting the model call something
    # that can only disappoint it.
    supports_images = False

    def __init__(self, model: str = "", host: str = "", num_ctx: int | None = None):
        self.name = model or config.OLLAMA_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.num_ctx = num_ctx or config.OLLAMA_NUM_CTX
        self.think = config.OLLAMA_THINK
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT_S)
        return self._client

    def _render(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for message in messages:
            role = message["role"]
            if role == "user":
                out.append({"role": "user", "content": message["text"]})
            elif role == "assistant":
                turn: dict = {"role": "assistant", "content": message.get("text") or ""}
                calls = message.get("tool_calls") or []
                if calls:
                    turn["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.input}} for c in calls
                    ]
                out.append(turn)
            else:
                # One tool message per result, which is the shape Ollama expects.
                for result in message["results"]:
                    out.append(
                        {"role": "tool", "tool_name": result.name, "content": result.text}
                    )
        return out

    async def complete(self, *, system: str, tools: list[dict], messages: list[dict]) -> Reply:
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
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        try:
            response = await self._http().post(f"{self.host}/api/chat", json=body)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            raise LLMError(
                f"Could not reach Ollama at {self.host} ({type(e).__name__}: {e}). "
                f"Is it running, and is {self.name!r} pulled?"
            ) from e

        message = payload.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            # Ollama does not hand out call ids, but the loop pairs results to
            # calls by id, so make one here.
            calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=function.get("name") or "",
                    input=_arguments(function.get("arguments")),
                )
            )
        return Reply(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            raw=None,  # nothing provider-specific to replay
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _arguments(raw) -> dict:
    """Arguments as a dict, whether they arrived as one or as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
