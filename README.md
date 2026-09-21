# browser-agent

An LLM-driven browser automation agent. Describe a task in plain English and it
drives a real Chromium locally through Playwright — navigating, clicking,
typing, reading and screenshotting — until the task is done.

The model never sees raw HTML. After every action it gets a compact snapshot of
the page: the interactive elements numbered `[1]`, `[2]`, `[3]`… plus the
visible text. It acts by referring to those numbers.

**Status:** early. Nothing works yet beyond configuration.

## Install

```bash
pip install -e .
playwright install chromium
```

## Configure

```bash
cp .env.example .env   # then set ANTHROPIC_API_KEY
```

See [`browser_agent/config.py`](browser_agent/config.py) for every option and
its default.

## License

MIT
