# browser-agent

An LLM-driven browser automation agent. Describe a task in plain English and it
drives a real Chromium locally through Playwright — navigating, clicking,
typing, reading and screenshotting — until the task is done.

The model never sees raw HTML. After every action it gets a compact snapshot of
the page: the interactive elements numbered `[1]`, `[2]`, `[3]`… plus the
visible text. It acts by referring to those numbers.

It runs against a hosted model or one on your own machine.

## Install

```bash
pip install -e .
playwright install chromium
```

## Run it

```bash
browser-agent "what is the top story on bbc.com/news right now?"
browser-agent --headed                 # interactive, watch the browser work
browser-agent --url https://en.wikipedia.org "how tall is the Eiffel Tower?"
```

In interactive mode the browser stays open between turns, so answering a
question it asked continues on the same page instead of starting over.

## Or run it as a web page

```bash
pip install -e ".[web]"
browser-agent-web
```

Chat on the left, the live browser on the right. The browser runs on the
server and is streamed to the page as JPEG frames straight out of Chromium's
DevTools protocol — headless, so no display stack is needed to host it.

Suggestion pills sit above the input for anyone who has not thought of a task.

Visitors can bring their own Anthropic or OpenAI key; a key typed into the page
is used for that session only and is never stored, logged or sent back. A local
Ollama is offered as a provider when one is answering, which means it is there
while you develop and simply absent on a server that has none — it cannot be
exposed by accident. There
are caps for anything public: concurrent browsers, steps per task, tasks per
session, an idle timeout, and `BROWSER_AGENT_ALLOWED_DOMAINS` to limit which
sites it may visit at all.

## Choosing a model

Four backends. Set a key for any of them and it gets used; set none and it falls
back to a model on your own machine.

| Provider | Key | Notes |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` | has a free tier — no card |
| `anthropic` | `ANTHROPIC_API_KEY` | strongest on long tasks |
| `openai` | `OPENAI_API_KEY` | any OpenAI-compatible endpoint |
| `ollama` | — | runs locally, free, no key |

```bash
browser-agent --provider gemini "how tall is the Eiffel Tower?"

ollama pull qwen3:8b
browser-agent --provider ollama "how tall is the Eiffel Tower?"
```

`--provider auto` (the default) takes the first one you have a key for, Gemini
first, and drops to the local model when there is none.

A local 8B model runs this at roughly two seconds a step on Apple silicon,
because the conversation only ever grows at the end and the KV cache is reused.
It is meaningfully worse than a frontier model at long or fiddly tasks, and it
cannot look at screenshots, so that tool is not offered to it. The page it is
shown is deliberately smaller too — a small model handed 120 numbered elements
picks the wrong one.

## In a container

```bash
docker build -t browser-agent .
docker run -p 8080:8080 -e GEMINI_API_KEY=... browser-agent
```

`/healthz` answers 503 if the image has no working browser, so a bad build
fails at deploy rather than on someone's first click. See
[DEPLOY.md](DEPLOY.md) for Cloud Run.

## Configure

```bash
cp .env.example .env
```

Every option and its default is in [`browser_agent/config.py`](browser_agent/config.py).

## How it works

```
snapshot the page  →  model picks one action  →  do it  →  snapshot again
```

Element numbers are written onto the DOM and change with every snapshot, so a
stale reference fails safely ("take a fresh look") instead of clicking whatever
happens to be at that number now. Password, OTP and card fields are filled when
the task calls for it but never echoed back into the conversation.

Each turn is bounded by a step cap and a wall clock. Running out of either
doesn't lose anything: the agent reports where it got to and the browser stays
open for the next turn.

## Limitations

- The browser runs on this machine, in this process, and closes with it.
- No CAPTCHA solving. If a site blocks it, it says so.
- Conversation history lives in memory only.

## License

MIT
