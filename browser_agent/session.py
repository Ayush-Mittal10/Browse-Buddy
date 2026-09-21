"""One browser per task: a local Chromium driven with Playwright.

The browser runs in this process and dies with it. ``start()`` launches
Chromium, ``close()`` shuts it down; there is no remote session to keep alive,
reconnect to or bill for. Headless by default, or visible when you want to watch
the run.

Every action returns a STRING for the model: a one-line status followed by a
fresh snapshot of the page (see ``snapshot.py``), so the model always sees the
page as it is after acting and never has to ask for it. Failures come back as
strings too — a Playwright error must not take out the agent's turn.

Things that bite:

* Refs are attributes on the live DOM. A re-render drops them, so a stale
  ``click(12)`` is answered with "take a fresh look", not a wrong click.
* Popups become the current tab, which is what a person sees happen. The switch
  is noted in the next snapshot, and ``switch_tab`` goes back.
* The session is exposed through a ContextVar rather than passed around, so the
  tools the model calls never take a session argument, and two tasks running in
  the same event loop can never see each other's page.
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import logging
from urllib.parse import urlsplit

from browser_agent import config
from browser_agent.snapshot import SNAPSHOT_JS, format_snapshot

logger = logging.getLogger(__name__)


class BrowserUnavailable(Exception):
    """The browser could not be started. The message is safe to show the user."""


# The session the browser tools act on. A ContextVar so that two browser tasks
# sharing an event loop never see each other's page.
_current: contextvars.ContextVar[BrowserSession | None] = contextvars.ContextVar(
    "browser_session", default=None
)


def current_session() -> BrowserSession | None:
    """The session the tools should act on, or None outside a session block."""
    return _current.get()


# ── URL policy ───────────────────────────────────────────────────────────────

_BLOCKED_HOSTS = {"localhost", "metadata", "metadata.google.internal", "instance-data"}
_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".arpa")


def is_url_allowed(url: str) -> tuple[bool, str]:
    """Only public http(s) pages.

    Rejects other schemes (file:, javascript:, data:, chrome:), loopback,
    private and link-local addresses, and single-label or internal hostnames.
    The browser here runs on your machine with your network, so this is a real
    guard and not just a sanity check: without it a task could be talked into
    reading localhost or a cloud metadata endpoint.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return False, "That is not a valid URL."
    if parts.scheme not in ("http", "https"):
        return False, "Only http and https URLs can be opened."
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return False, "That URL has no host."
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        return False, "That address is not a public website."
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return False, "That address is not a public website."
    if ip is None and "." not in host:
        return False, "That address is not a public website."
    return True, ""


def _normalise_url(url: str) -> str:
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    return url


def _short_error(e: BaseException, limit: int = 240) -> str:
    text = " ".join(str(e).split())
    # Playwright puts a multi-line call log after the first sentence; the first
    # line is the only part that means anything to the model.
    text = text.split(" Call log:", 1)[0]
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── Session ──────────────────────────────────────────────────────────────────


class BrowserSession:
    """Browser lifecycle plus every action. Use as ``async with BrowserSession()``.

    Entering the block launches Chromium and installs the session in the
    ContextVar the tools read; leaving it closes the browser, even if the task
    was cancelled.
    """

    def __init__(self, *, headless: bool | None = None):
        self.headless = config.HEADLESS if headless is None else headless
        self.finished_report: str | None = None  # set by finish_task
        self.page = None
        self.pages: list = []
        self.notes: list[str] = []  # surfaced once, in the next snapshot
        self.last_screenshot: tuple[str, str] | None = None  # (base64, mime)
        self._pw = None
        self._browser = None
        self._context = None
        self._token = None

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> BrowserSession:
        await self.start()
        self._token = _current.set(self)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None
        # Shielded: a cancelled run (the user pressed Ctrl+C) must still get the
        # browser shut down, or Chromium is left running with nothing driving it.
        try:
            await asyncio.shield(asyncio.wait_for(self.close(), timeout=20))
        except Exception as e:
            logger.warning("Browser close failed: %s", e)

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise BrowserUnavailable(
                "Playwright is not installed. Run:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from e

        self._pw = await async_playwright().start()
        try:
            self._browser = await self._pw.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport=config.VIEWPORT,
                locale=config.LOCALE,
                timezone_id=config.TIMEZONE,
            )
            page = await self._context.new_page()
        except Exception as e:
            await self.close()
            # The common first-run failure by far, and Playwright's own message
            # buries the fix under a wall of text.
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                raise BrowserUnavailable(
                    "Chromium is not installed. Run: playwright install chromium"
                ) from e
            logger.error("Browser start failed: %s: %s", type(e).__name__, e)
            raise BrowserUnavailable("The browser could not be started.") from e

        self._context.set_default_timeout(config.ACTION_TIMEOUT_MS)
        self._context.set_default_navigation_timeout(config.NAV_TIMEOUT_MS)
        self._context.on("page", self._on_new_page)
        self._adopt_page(page)
        logger.info("Browser session started (headless=%s)", self.headless)

    async def close(self) -> None:
        browser, pw = self._browser, self._pw
        self._browser = self._pw = self._context = None
        self.page, self.pages = None, []
        if browser is not None:
            try:
                await browser.close()
            except Exception as e:
                logger.warning("browser.close failed: %s", e)
        if pw is not None:
            try:
                await pw.stop()
            except Exception as e:
                logger.warning("playwright.stop failed: %s", e)

    def current_url(self) -> str:
        try:
            return self.page.url if self.page is not None else ""
        except Exception:
            return ""

    # -- page bookkeeping ---------------------------------------------------

    def _adopt_page(self, page) -> None:
        if page not in self.pages:
            self.pages.append(page)
            page.on("dialog", self._on_dialog)
            page.on("close", lambda p=page: self._on_page_closed(p))
        self.page = page

    def _on_new_page(self, page) -> None:
        self._adopt_page(page)
        self.notes.append("A new tab opened and is now the current tab (switch_tab to go back).")

    def _on_page_closed(self, page) -> None:
        if page in self.pages:
            self.pages.remove(page)
        if self.page is page:
            self.page = self.pages[-1] if self.pages else None
            self.notes.append("The tab you were on closed; showing the previous tab.")

    def _on_dialog(self, dialog) -> None:
        # Accept confirms and alerts so the page keeps moving, and say what it
        # said — an unanswered dialog blocks every later action on the page.
        self.notes.append(
            f'The page showed a {dialog.type} dialog: "{dialog.message[:200]}" (accepted).'
        )
        asyncio.ensure_future(dialog.accept())

    def _require_page(self):
        if self.page is None:
            raise RuntimeError("No open tab. Use navigate to open a page.")
        return self.page

    async def _tab_titles(self) -> list[str]:
        titles = []
        for i, p in enumerate(self.pages, start=1):
            try:
                t = await p.title()
            except Exception:
                t = "(unavailable)"
            titles.append(f"[{i}] {t[:50]}" + (" (current)" if p is self.page else ""))
        return titles

    async def _settle(self) -> None:
        """Let a single-page app finish painting before we read the DOM back."""
        page = self.page
        if page is None:
            return
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        try:
            await page.wait_for_timeout(config.SETTLE_MS)
        except Exception:
            pass

    async def snapshot(self) -> str:
        page = self.page
        if page is None:
            return "No open tab. Use navigate to open a page."
        try:
            data = await asyncio.wait_for(
                page.evaluate(SNAPSHOT_JS), timeout=config.SNAPSHOT_TIMEOUT_S
            )
        except TimeoutError:
            return (
                "The page is not responding (still loading or frozen). "
                "Try wait, then get_page, or navigate elsewhere."
            )
        except Exception as e:
            return f"Could not read the page: {_short_error(e)}. Try wait, then get_page."
        tabs = await self._tab_titles() if len(self.pages) > 1 else None
        notes, self.notes = self.notes, []
        return format_snapshot(data, tabs=tabs, notes=notes)

    async def _after(self, status: str) -> str:
        """A status line plus the page as it now is — what every action returns."""
        await self._settle()
        return f"{status}\n\n{await self.snapshot()}"

    # -- actions (each returns a string for the model) -----------------------

    async def navigate(self, url: str) -> str:
        url = _normalise_url(url)
        ok, why = is_url_allowed(url)
        if not ok:
            return f"Cannot open {url!r}: {why}"
        page = self._require_page()
        try:
            resp = await page.goto(url, wait_until="domcontentloaded")
        except Exception as e:
            logger.warning("navigate failed: %s", _short_error(e))
            return await self._after(
                f"Could not open {url}: {_short_error(e)}. "
                "Try again once, then try a different page or a search engine."
            )
        status = f" (HTTP {resp.status})" if resp is not None and resp.status >= 400 else ""
        return await self._after(f"Opened {page.url}{status}.")

    async def get_page(self) -> str:
        return await self.snapshot()
