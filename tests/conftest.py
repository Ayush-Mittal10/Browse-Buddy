"""Fixtures for the tests that need a real browser.

Pages are served by intercepting requests to a stand-in public host rather than
by running a local HTTP server: is_url_allowed refuses localhost and every
private address, so a real local server would be unreachable by design. This
way the tests exercise navigate() exactly as the agent uses it, and never touch
the network.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from browser_agent.session import BrowserSession, BrowserUnavailable

HOST = "https://example.test"


@pytest_asyncio.fixture
async def session():
    """A started BrowserSession, closed afterwards."""
    s = BrowserSession(headless=True)
    try:
        await s.start()
    except BrowserUnavailable as e:
        pytest.skip(f"browser unavailable: {e}")
    try:
        yield s
    finally:
        await s.close()


@pytest_asyncio.fixture
async def serve(session):
    """Serve HTML at {HOST}{path}. Returns the URL to navigate to.

    Call it more than once to add pages; later calls do not disturb earlier ones.
    """
    pages: dict[str, str] = {}

    async def handler(route):
        path = "/" + route.request.url.split(HOST + "/", 1)[-1].split("?", 1)[0].lstrip("/")
        body = pages.get(path)
        if body is None:
            await route.fulfill(status=404, content_type="text/html", body="<h1>Not found</h1>")
        else:
            await route.fulfill(status=200, content_type="text/html", body=body)

    await session._context.route(f"{HOST}/**", handler)

    async def _serve(html: str, path: str = "/") -> str:
        pages[path] = html
        return f"{HOST}{path}"

    return _serve
