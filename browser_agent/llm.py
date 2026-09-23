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

import asyncio
import logging
from dataclasses import dataclass, field
from json import loads as _json_loads
from typing import Any, Protocol

import httpx

from browser_agent import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model could not be reached or refused the request."""


# Worth trying again: rate limits and the provider being briefly out of
# capacity. Seen constantly on Gemini's free tier, where the popular models
# answer 503 "high demand" and then work fine a few seconds later.
RETRYABLE = frozenset({429, 500, 502, 503, 504})


async def post_with_retry(client, url, *, json, headers, attempts: int = 3, delay: float = 2.0):
    """POST, retrying the statuses that mean "not now" rather than "no".

    A connection that times out or drops counts as "not now" too. It used to
    escape this loop as an exception and end the whole task on the first
    occurrence, which is the wrong end of the scale: a read timeout on a free
    tier under load is the most ordinary transient failure there is, and the
    task had already spent several steps by the time it happened.
    """
    response = None
    host = url.split("/")[2]
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            response = await client.post(url, json=json, headers=headers)
        except httpx.TransportError as e:
            # Out of attempts: let it go up as the error the caller reports.
            if last:
                raise
            wait = delay * (2**attempt)
            logger.info(
                "%s from %s; retrying in %.0fs", type(e).__name__, host, wait
            )
            await asyncio.sleep(wait)
            continue
        if response.status_code not in RETRYABLE or last:
            return response
        wait = _retry_after(response, delay * (2**attempt))
        logger.info("HTTP %s from %s; retrying in %.0fs", response.status_code, host, wait)
        await asyncio.sleep(wait)
    return response


@dataclass
class Streamed:
    """A server-sent-event response, collected.

    Shaped like the bits of an httpx response the callers already use, so the
    error paths do not have to care which transport produced them.
    """

    status_code: int
    events: list[dict] = field(default_factory=list)
    text: str = ""  # the body, read only when the status says it is an error

    def json(self):
        return _json_loads(self.text)


async def stream_with_retry(
    client,
    url,
    *,
    json,
    headers,
    gap_s: float,
    total_s: float,
    on_event=None,
    attempts: int = 3,
    delay: float = 2.0,
) -> Streamed:
    """POST expecting SSE, collecting the events, with the same retry policy.

    Streaming is what makes the read timeout mean what it says. On the
    non-streaming endpoint the server computes the whole answer before sending
    a byte — measured, first byte and last byte arrive in the same instant — so
    a "read timeout" was really a cap on how long the model was allowed to
    think, and a slow answer was indistinguishable from a dead connection.
    Here `gap_s` bounds the silence *between* chunks and `total_s` bounds the
    whole response, which are two different failures and worth telling apart.

    A failure part-way through discards what arrived and re-sends: the reply is
    only useful whole, and half a tool call is not worth keeping.
    """
    host = url.split("/")[2]
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            result = await _stream_once(
                client, url, json=json, headers=headers, gap_s=gap_s,
                total_s=total_s, on_event=on_event,
            )
        except (httpx.TransportError, TimeoutError) as e:
            if last:
                raise
            wait = delay * (2**attempt)
            logger.info("%s from %s; retrying in %.0fs", type(e).__name__, host, wait)
            await asyncio.sleep(wait)
            continue
        if result.status_code not in RETRYABLE or last:
            return result
        wait = delay * (2**attempt)
        logger.info("HTTP %s from %s; retrying in %.0fs", result.status_code, host, wait)
        await asyncio.sleep(wait)
    return result


async def _stream_once(client, url, *, json, headers, gap_s, total_s, on_event) -> Streamed:
    """One attempt. `gap_s` is httpx's read timeout, which on a stream is the
    wait for the *next* chunk; `total_s` bounds the response as a whole."""
    timeout = httpx.Timeout(gap_s, connect=min(gap_s, 15.0))
    async with asyncio.timeout(total_s):
        async with client.stream(
            "POST", url, json=json, headers=headers, timeout=timeout
        ) as response:
            if response.status_code >= 400:
                # Errors are small and are not streamed; read the body so the
                # caller can say what was actually wrong.
                body = await response.aread()
                return Streamed(response.status_code, [], body.decode("utf-8", "replace"))
            events = []
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = _json_loads(payload)
                except ValueError:
                    continue
                events.append(event)
                if on_event is not None:
                    on_event(event)
            return Streamed(response.status_code, events)


def unreachable(provider: str, e: BaseException) -> str:
    """Why a request never got an answer, in words rather than a class name.

    httpx raises a timeout with an empty message, so the obvious rendering comes
    out as "ReadTimeout: " and tells the reader nothing about which half of the
    request was slow or what they might do about it.
    """
    # Two different stalls, and which one it was is the useful part. httpx's
    # timeout means the connection went quiet; asyncio's means it kept dribbling
    # but never finished.
    if isinstance(e, httpx.TimeoutException):
        return (
            f"{provider} stopped sending for {config.HTTP_STREAM_GAP_S}s, and did the same "
            "on the retries. It is usually load on the provider rather than anything about "
            "the request; a smaller model normally answers when a larger one will not."
        )
    if isinstance(e, TimeoutError):
        return (
            f"{provider} was still answering after {config.HTTP_TIMEOUT_S}s and had to be "
            "cut off, on every attempt. A shorter task or a smaller model is the way out."
        )
    detail = str(e) or type(e).__name__
    return f"Could not reach {provider}: {detail}"


def _retry_after(response, fallback: float) -> float:
    """The server's own advice when it gives any, capped so a run can't stall."""
    raw = response.headers.get("retry-after", "")
    try:
        return min(float(raw), 30.0)
    except (TypeError, ValueError):
        return fallback


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
    provider: str
    supports_images: bool

    async def complete(self, *, system: str, tools: list[dict], messages: list[dict]) -> Reply: ...

    async def close(self) -> None: ...


# ── Anthropic ────────────────────────────────────────────────────────────────


class AnthropicLLM:
    """The hosted path. Caches the prompt prefix and can read screenshots."""

    provider = "anthropic"
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
