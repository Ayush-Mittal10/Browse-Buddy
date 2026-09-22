"""The model, behind an interface, so the loop does not know whose it is.

The agent keeps its conversation in a small neutral format — a user turn, an
assistant turn with its tool calls, a batch of tool results — and a backend
translates that into whatever shape its provider wants on the way out. Adding a
provider means writing one class here, not touching the loop.

Backends also declare what they can do. A local text model cannot look at a
screenshot, so the agent drops that tool rather than offering something that
will fail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from browser_agent import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model could not be reached or refused the request."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    id: str
    name: str
    text: str
    image: tuple[str, str] | None = None  # (base64, mime)


@dataclass
class Reply:
    """One assistant turn, as the loop sees it."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The provider's own rendering of this turn, replayed verbatim on the next
    # request. Anthropic needs it: a thinking block must go back with its
    # signature intact or the following request is rejected.
    raw: Any = None


# --- the neutral conversation -------------------------------------------------
#
# {"role": "user",      "text": str}
# {"role": "assistant", "text": str, "tool_calls": [ToolCall], "raw": Any}
# {"role": "tool",      "results": [ToolResult]}


def user(text: str) -> dict:
    return {"role": "user", "text": text}


def assistant(reply: Reply) -> dict:
    return {
        "role": "assistant",
        "text": reply.text,
        "tool_calls": reply.tool_calls,
        "raw": reply.raw,
    }


def tool_results(results: list[ToolResult]) -> dict:
    return {"role": "tool", "results": results}


class LLM(Protocol):
    name: str
    supports_images: bool

    async def complete(self, *, system: str, tools: list[dict], messages: list[dict]) -> Reply: ...

    async def close(self) -> None: ...


# ── Anthropic ────────────────────────────────────────────────────────────────


class AnthropicLLM:
    """The hosted path. Caches the prompt prefix and can read screenshots."""

    supports_images = True

    def __init__(self, model: str = "", api_key: str = "", max_tokens: int | None = None):
        self.name = model or config.MODEL
        self.api_key = api_key
        self.max_tokens = max_tokens or config.MAX_TOKENS
        self._client = None

    def _api(self):
        if self._client is None:
            import anthropic

            key = self.api_key or config.require_api_key()
            self._client = anthropic.AsyncAnthropic(api_key=key)
        return self._client

    def _render(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "user":
                out.append({"role": "user", "content": message["text"]})
            elif role == "assistant":
                out.append({"role": "assistant", "content": message["raw"]})
            else:
                blocks = [_result_block(r) for r in message["results"]]
                out.append({"role": "user", "content": blocks})
        return out

    async def complete(self, *, system: str, tools: list[dict], messages: list[dict]) -> Reply:
        try:
            response = await self._api().messages.create(
                model=self.name,
                max_tokens=self.max_tokens,
                # Tools render before the system prompt, so one breakpoint here
                # caches both — and they go out on every step of every run.
                system=[
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
                tools=tools,
                messages=self._render(messages),
            )
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e

        text, calls, raw = [], [], []
        for block in response.content:
            dump = getattr(block, "model_dump", None)
            raw.append(dump(exclude_none=True) if dump else block)
            if block.type == "text":
                text.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCall(block.id, block.name, args))
        return Reply("\n".join(t for t in text if t).strip(), calls, raw)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _result_block(result: ToolResult) -> dict:
    """A tool_result, with the image inside it when there is one.

    The image has to sit in the result rather than in a message of its own, or
    it lands between a tool_use and its answer and the pairing breaks.
    """
    if result.image is None:
        return {"type": "tool_result", "tool_use_id": result.id, "content": result.text}
    data, mime = result.image
    return {
        "type": "tool_result",
        "tool_use_id": result.id,
        "content": [
            {"type": "text", "text": result.text},
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
        ],
    }


# ── choosing one ─────────────────────────────────────────────────────────────


PROVIDERS = ("anthropic", "openai", "gemini", "ollama")


def resolve(provider: str = "", *, keys: dict[str, str] | None = None) -> str:
    """Which provider to actually use.

    "auto" takes the first hosted one there is a key for and falls back to the
    local model, so this runs whatever the person happens to have. The order is
    deliberate: Gemini first, because its free tier is the one a stranger can
    use without paying for it.
    """
    provider = (provider or config.PROVIDER).strip().lower()
    if provider != "auto":
        return provider
    keys = keys or {}
    for name, configured in (
        ("gemini", config.GEMINI_API_KEY),
        ("anthropic", config.API_KEY),
        ("openai", config.OPENAI_API_KEY),
    ):
        if keys.get(name) or configured:
            return name
    return "ollama"


def build(provider: str = "", model: str = "", api_key: str = "") -> LLM:
    """The backend to run this task on.

    `api_key` is passed in rather than read from the environment so that a
    caller serving several people at once — a web front end where each brings
    their own key — never has to put one in a process-wide variable.
    """
    provider = resolve(provider, keys={provider: api_key} if api_key else None)

    if provider == "anthropic":
        return AnthropicLLM(model, api_key=api_key)
    if provider == "openai":
        from browser_agent.openai import OpenAILLM

        return OpenAILLM(model, api_key=api_key)
    if provider == "gemini":
        from browser_agent.gemini import GeminiLLM

        return GeminiLLM(model, api_key=api_key)
    if provider == "ollama":
        from browser_agent.ollama import OllamaLLM

        return OllamaLLM(model)
    raise LLMError(f"Unknown provider {provider!r}. Use one of: {', '.join(PROVIDERS)}.")
