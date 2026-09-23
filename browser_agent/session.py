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
import base64
import contextlib
import contextvars
import ipaddress
import logging
import random
import time
from urllib.parse import urlsplit

from browser_agent import config
from browser_agent.live import Screencast
from browser_agent.snapshot import (
    FIELD_INFO_JS,
    READ_TEXT_CHARS,
    READ_TEXT_JS,
    SNAPSHOT_JS,
    format_snapshot,
    is_sensitive_field,
)

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


@contextlib.contextmanager
def use_session(session: BrowserSession):
    """Publish `session` as the current one for the duration of the block.

    The async-with form does this on entry, but an agent that keeps one browser
    across several runs starts and stops the session itself and still needs the
    tools to find it.
    """
    token = _current.set(session)
    try:
        yield
    finally:
        _current.reset(token)


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
    if config.ALLOWED_DOMAINS and not _in_allowlist(host):
        allowed = ", ".join(config.ALLOWED_DOMAINS)
        return False, f"This agent is limited to: {allowed}."
    return True, ""


def _in_allowlist(host: str) -> bool:
    """Whether `host` is an allowed domain or a subdomain of one.

    Matched on label boundaries, so allowing "example.com" does not also allow
    "notexample.com" — the check has to be about the domain, not the string.
    """
    return any(
        host == domain or host.endswith("." + domain) for domain in config.ALLOWED_DOMAINS
    )


def _normalise_url(url: str) -> str:
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    return url


def _launch_args() -> list[str]:
    """Flags Chromium is started with."""
    args = [
        # Without this, navigator.webdriver is true and a page reads it in one
        # line. Several large sites then refuse to serve us.
        "--disable-blink-features=AutomationControlled",
    ]
    if config.CONTAINER:
        # Chromium's sandbox needs privileges a container normally withholds,
        # and /dev/shm is small enough there that Chromium exhausts it and
        # crashes somewhere unrelated-looking.
        args += ["--no-sandbox", "--disable-dev-shm-usage"]
    return args


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

    def __init__(self, *, headless: bool | None = None, on_frame=None):
        self.headless = config.HEADLESS if headless is None else headless
        # A viewer somewhere else watching this run, or nobody.
        self.live = Screencast(on_frame) if on_frame else None
        self.finished_report: str | None = None  # set by finish_task
        self.page = None
        self.pages: list = []
        self.notes: list[str] = []  # surfaced once, in the next snapshot
        self._nav_failures: dict[str, int] = {}  # host -> times it would not load
        self._last_hit: dict[str, float] = {}    # host -> when we were last there
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
            self._browser = await self._pw.chromium.launch(
                headless=self.headless,
                args=_launch_args(),
            )
            self._context = await self._browser.new_context(
                viewport=config.VIEWPORT,
                locale=config.LOCALE,
                timezone_id=config.TIMEZONE,
                user_agent=config.USER_AGENT or await self._plausible_user_agent(),
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
        if self.live is not None:
            await self.live.follow(self._context, page)
        logger.info("Browser session started (headless=%s)", self.headless)

    async def _plausible_user_agent(self) -> str | None:
        """Chromium's own user agent, minus the word that gives the game away.

        Headless Chromium introduces itself as "HeadlessChrome/153.0.…", which
        a site reads in one line — and several large ones then refuse to serve
        it. This is not pretending to be a different browser: it is the same
        Chromium, the same version, just not volunteering that nobody is
        watching the window. Derived from the live browser rather than written
        out by hand, so it stays true across versions and platforms.
        """
        try:
            probe = await self._browser.new_context()
            try:
                page = await probe.new_page()
                agent = await page.evaluate("navigator.userAgent")
            finally:
                await probe.close()
        except Exception as e:
            logger.debug("Could not read the default user agent: %s", e)
            return None
        return agent.replace("HeadlessChrome", "Chrome") or None

    async def close(self) -> None:
        if self.live is not None:
            await self.live.stop()
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
        self._follow_live(page)

    def _follow_live(self, page) -> None:
        """Point the live view at the tab the agent is actually on.

        Scheduled rather than awaited because the places the current tab
        changes — a popup opening, a tab closing — are synchronous event
        handlers.
        """
        if self.live is not None and self._context is not None and page is not None:
            asyncio.ensure_future(self.live.follow(self._context, page))

    def _on_new_page(self, page) -> None:
        self._adopt_page(page)
        self.notes.append("A new tab opened and is now the current tab (switch_tab to go back).")

    def _on_page_closed(self, page) -> None:
        if page in self.pages:
            self.pages.remove(page)
        if self.page is page:
            self.page = self.pages[-1] if self.pages else None
            self.notes.append("The tab you were on closed; showing the previous tab.")
            self._follow_live(self.page)

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
        return format_snapshot(
            data,
            tabs=tabs,
            notes=notes,
            max_elements=config.MAX_ELEMENTS,
            max_text=config.MAX_TEXT_CHARS,
        )

    async def _after(self, status: str) -> str:
        """A status line plus the page as it now is — what every action returns."""
        await self._settle()
        return f"{status}\n\n{await self.snapshot()}"

    # -- element refs -------------------------------------------------------

    def _locator(self, ref) -> tuple:
        """(locator, error). The error is set when the ref is not a number."""
        try:
            n = int(str(ref).strip().strip("[]"))
        except (TypeError, ValueError):
            return None, (
                f"'{ref}' is not an element number. Use the [number] from the latest snapshot."
            )
        page = self._require_page()
        return page.locator(f'[data-agent-ref="{n}"]').first, ""

    async def _resolve(self, ref):
        """Turn [12] into a locator, or explain why it is stale.

        Refs live on the DOM and a re-render wipes them, so the failure mode
        here is "that number means nothing any more" — which must read as an
        instruction to look again, never as a click on whatever is at 12 now.
        """
        loc, err = self._locator(ref)
        if err:
            return None, err
        try:
            if await loc.count() == 0:
                return None, (
                    f"Element [{ref}] is no longer on the page (it re-rendered). "
                    "Call get_page and use the new numbers."
                )
        except Exception as e:
            return None, f"Could not find element [{ref}]: {_short_error(e)}"
        return loc, ""

    # -- actions (each returns a string for the model) -----------------------

    async def _pace(self, host: str) -> None:
        """Wait, if we were on this host a moment ago.

        Nothing here is trying to look human. It is about not arriving in
        bursts: the slow part of a run is the model, so in normal operation
        this waits for nothing at all, and it only bites when the agent starts
        hammering one host — which is exactly when it should.
        """
        gap = config.HOST_GAP_S
        if gap <= 0 or not host:
            return
        since = time.monotonic() - self._last_hit.get(host, 0.0)
        if since < gap:
            await asyncio.sleep(gap - since + random.uniform(0, gap / 3))
        self._last_hit[host] = time.monotonic()

    async def navigate(self, url: str) -> str:
        url = _normalise_url(url)
        ok, why = is_url_allowed(url)
        if not ok:
            return f"Cannot open {url!r}: {why}"
        page = self._require_page()
        await self._pace((urlsplit(url).hostname or "").lower())
        try:
            resp = await page.goto(url, wait_until="domcontentloaded")
        except Exception as e:
            logger.warning("navigate failed: %s", _short_error(e))
            host = (urlsplit(url).hostname or url).lower()
            failures = self._nav_failures[host] = self._nav_failures.get(host, 0) + 1
            advice = (
                # A site that has refused twice is not having a bad moment. Left
                # to "try again once" the model will keep coming back to it, or
                # wander off substituting other sites without saying so.
                f"{host} has now failed {failures} times and is not going to load in this "
                "session — the site is refusing us, not loading slowly. Do not try it again. "
                "If the task named this site specifically, tell the user it is unreachable "
                "and offer an alternative rather than quietly using a different one."
                if failures >= 2
                else "Try again once, then try a different page or a search engine."
            )
            return await self._after(f"Could not open {url}: {_short_error(e)}. {advice}")
        status = f" (HTTP {resp.status})" if resp is not None and resp.status >= 400 else ""
        return await self._after(f"Opened {page.url}{status}.")

    async def click(self, ref) -> str:
        loc, err = await self._resolve(ref)
        if err:
            return err
        try:
            await loc.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass
        try:
            await loc.click(timeout=config.ACTION_TIMEOUT_MS)
        except Exception as first:
            # Something is covering it — a sticky header, a cookie banner. A
            # forced click gets what a person scrolling past the overlay would.
            try:
                await loc.click(timeout=3000, force=True)
            except Exception:
                return await self._after(
                    f"Click on [{ref}] failed: {_short_error(first)}. "
                    "If a banner or dialog is in the way, close it first."
                )
        return await self._after(f"Clicked [{ref}].")

    async def type_text(self, ref, text: str, press_enter: bool = False) -> str:
        loc, err = await self._resolve(ref)
        if err:
            return err
        try:
            info = await loc.evaluate(FIELD_INFO_JS)
        except Exception as e:
            return f"Could not inspect [{ref}]: {_short_error(e)}"
        # A credential, code or payment field is filled like any other when the
        # task calls for it — signing in IS a task. What changes is the echo:
        # the value must not come back in the tool result, because that result
        # is kept in the conversation history, nor appear in a log line. The
        # snapshot already masks such values.
        sensitive = is_sensitive_field(info or {})
        if sensitive:
            logger.info("Typing into a sensitive field (ref %s)", ref)
        try:
            if info and info.get("fillable"):
                await loc.fill(text or "")
            else:
                await loc.click(timeout=config.ACTION_TIMEOUT_MS)
                await self._require_page().keyboard.type(text or "")
            if press_enter:
                await loc.press("Enter")
        except Exception as e:
            return await self._after(f"Typing into [{ref}] failed: {_short_error(e)}")
        if sensitive:
            shown = "••••"
        else:
            shown = (text or "")[:40] + ("…" if len(text or "") > 40 else "")
        return await self._after(
            f'Typed "{shown}" into [{ref}]' + (" and pressed Enter." if press_enter else ".")
        )

    async def select_option(self, ref, option: str) -> str:
        loc, err = await self._resolve(ref)
        if err:
            return err
        try:
            await loc.select_option(label=option)
        except Exception:
            try:
                await loc.select_option(value=option)
            except Exception as e:
                return (
                    f"Could not select {option!r} in [{ref}]: {_short_error(e)}. "
                    "If it is not a real dropdown, click it and pick the option from "
                    "the list instead."
                )
        return await self._after(f"Selected {option!r} in [{ref}].")

    async def press_key(self, key: str) -> str:
        page = self._require_page()
        try:
            await page.keyboard.press(key)
        except Exception as e:
            return (
                f"Could not press {key!r}: {_short_error(e)}. "
                "Use names like Enter, Escape, Tab, ArrowDown, PageDown."
            )
        return await self._after(f"Pressed {key}.")

    async def scroll(self, direction: str = "down", pages: float = 1.0) -> str:
        page = self._require_page()
        try:
            pages = max(0.1, min(float(pages or 1.0), 10.0))
        except (TypeError, ValueError):
            pages = 1.0
        sign = -1 if str(direction).lower().startswith("up") else 1
        dy = sign * int(pages * config.VIEWPORT_HEIGHT * 0.9)
        try:
            # A wheel event over the viewport centre scrolls whatever is under
            # it — the document, or an inner scroll pane that window.scrollBy
            # would sail straight past.
            await page.mouse.move(config.VIEWPORT_WIDTH / 2, config.VIEWPORT_HEIGHT / 2)
            await page.mouse.wheel(0, dy)
            await page.wait_for_timeout(300)
        except Exception as e:
            return f"Scroll failed: {_short_error(e)}"
        return f"Scrolled {'up' if sign < 0 else 'down'}.\n\n{await self.snapshot()}"

    async def go_back(self) -> str:
        page = self._require_page()
        try:
            resp = await page.go_back(wait_until="domcontentloaded")
        except Exception as e:
            return await self._after(f"Could not go back: {_short_error(e)}")
        if resp is None and page.url in ("about:blank", ""):
            return "There is no previous page in this tab."
        return await self._after("Went back.")

    async def get_page(self) -> str:
        return await self.snapshot()

    async def read_text(self, ref: str = "", start: int = 0) -> str:
        """A chunk of the page's text, for content too long for a snapshot."""
        page = self._require_page()
        ref_n = None
        if str(ref or "").strip():
            try:
                ref_n = int(str(ref).strip().strip("[]"))
            except ValueError:
                return f"'{ref}' is not an element number."
        try:
            start = max(0, int(start or 0))
        except (TypeError, ValueError):
            start = 0
        try:
            data = await asyncio.wait_for(
                page.evaluate(READ_TEXT_JS, [ref_n, start, READ_TEXT_CHARS]),
                timeout=config.SNAPSHOT_TIMEOUT_S,
            )
        except Exception as e:
            return f"Could not read text: {_short_error(e)}"
        if not data:
            return f"Element [{ref}] is no longer on the page. Call get_page for fresh numbers."
        total, chunk = int(data.get("total") or 0), data.get("chunk") or ""
        if not chunk:
            if start == 0:
                return "No text there."
            return f"Nothing past character {start} (total {total})."
        end = start + len(chunk)
        head = f"Text {start}-{end} of {total}"
        if end < total:
            head += f" (read_text with start={end} for more)"
        return head + ":\n" + chunk

    async def screenshot(self) -> str:
        page = self._require_page()
        try:
            raw = await page.screenshot(type="jpeg", quality=config.SCREENSHOT_JPEG_QUALITY)
        except Exception as e:
            return f"Screenshot failed: {_short_error(e)}"
        self.last_screenshot = (base64.b64encode(raw).decode("ascii"), "image/jpeg")
        return "Screenshot taken — it is attached below this result."

    def take_screenshot(self) -> tuple[str, str] | None:
        """Pop the last screenshot (base64, mime) for the agent loop to attach."""
        shot, self.last_screenshot = self.last_screenshot, None
        return shot

    async def wait(self, seconds: float = 2.0) -> str:
        page = self._require_page()
        try:
            seconds = max(0.5, min(float(seconds or 2.0), config.MAX_WAIT_S))
        except (TypeError, ValueError):
            seconds = 2.0
        await page.wait_for_timeout(int(seconds * 1000))
        return f"Waited {seconds:g}s.\n\n{await self.snapshot()}"

    async def switch_tab(self, index: int) -> str:
        try:
            i = int(index)
        except (TypeError, ValueError):
            return "Give the tab number from the Tabs line, e.g. 1."
        if not (1 <= i <= len(self.pages)):
            return f"There is no tab {i}. Open tabs: {len(self.pages)}."
        self.page = self.pages[i - 1]
        if self.live is not None:
            await self.live.follow(self._context, self.page)
        try:
            await self.page.bring_to_front()
        except Exception:
            pass
        return f"Switched to tab {i}.\n\n{await self.snapshot()}"
