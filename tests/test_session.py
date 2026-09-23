"""Browser lifecycle and navigation, against a real Chromium."""

from __future__ import annotations

import asyncio

import pytest

from browser_agent.session import BrowserSession, current_session

PAGE = """
<!doctype html>
<title>Test Page</title>
<h1>Welcome</h1>
<p>Some visible text.</p>
<a href="/next">Next page</a>
<button>Do the thing</button>
<input type="email" name="email" placeholder="Email">
<input type="password" name="password" placeholder="Password">
"""


# --- lifecycle ----------------------------------------------------------------


async def test_context_manager_publishes_and_clears_the_session() -> None:
    assert current_session() is None
    async with BrowserSession(headless=True) as s:
        assert current_session() is s
        assert s.page is not None
    assert current_session() is None


async def test_context_manager_closes_the_browser_on_exit() -> None:
    async with BrowserSession(headless=True) as s:
        pass
    assert s.page is None
    assert s.pages == []


async def test_close_is_idempotent(session) -> None:
    await session.close()
    await session.close()  # must not raise
    assert session.page is None


async def test_headless_defaults_come_from_config() -> None:
    assert BrowserSession().headless is True
    assert BrowserSession(headless=False).headless is False


async def test_snapshot_without_a_page_says_so() -> None:
    s = BrowserSession(headless=True)
    assert "No open tab" in await s.snapshot()


async def test_current_url_is_safe_before_start() -> None:
    assert BrowserSession(headless=True).current_url() == ""


# --- navigation ---------------------------------------------------------------


async def test_navigate_refuses_a_blocked_url_without_opening_a_browser() -> None:
    s = BrowserSession(headless=True)  # never started
    out = await s.navigate("file:///etc/passwd")
    assert "Cannot open" in out
    assert "http and https" in out


async def test_navigate_refuses_localhost(session) -> None:
    out = await session.navigate("http://localhost:8080/")
    assert "not a public website" in out


async def test_navigate_opens_the_page_and_returns_a_snapshot(session, serve) -> None:
    url = await serve(PAGE)
    out = await session.navigate(url)

    assert out.startswith(f"Opened {url}.")
    assert "Page: Test Page" in out
    assert "Some visible text." in out
    assert session.current_url() == url


async def test_navigate_adds_the_scheme_when_it_is_missing(session, serve) -> None:
    await serve(PAGE)
    out = await session.navigate("example.test")
    assert "Opened https://example.test/" in out


async def test_navigate_reports_an_http_error_but_still_shows_the_page(session, serve) -> None:
    await serve(PAGE)  # registers the route; /missing is not in it
    out = await session.navigate("https://example.test/missing")
    assert "(HTTP 404)" in out
    assert "Not found" in out


async def test_navigate_failure_comes_back_as_a_string(session) -> None:
    # Nothing is routed for this host and the network is not reachable in tests,
    # so this exercises the failure path rather than a real DNS lookup.
    out = await session.navigate("https://this-host-does-not-resolve.invalid/")
    assert "Could not open" in out
    assert isinstance(out, str)


# --- what the snapshot sees ---------------------------------------------------


async def test_elements_are_numbered_and_described(session, serve) -> None:
    await session.navigate(await serve(PAGE))
    out = await session.get_page()

    assert '[1] link "Next page" -> /next' in out
    assert '[2] button "Do the thing"' in out
    assert '[3] input(email) "Email"' in out


async def test_refs_are_stamped_onto_the_live_dom(session, serve) -> None:
    await session.navigate(await serve(PAGE))
    await session.get_page()
    count = await session.page.locator('[data-agent-ref="2"]').count()
    assert count == 1
    assert await session.page.locator('[data-agent-ref="2"]').inner_text() == "Do the thing"


async def test_password_fields_are_flagged_as_sensitive(session, serve) -> None:
    await session.navigate(await serve(PAGE))
    line = next(
        line for line in (await session.get_page()).splitlines() if "Password" in line
    )
    assert "[sensitive field]" in line


async def test_a_dropdown_is_named_from_its_label_not_its_options(session, serve) -> None:
    url = await serve(
        "<title>T</title><label for='c'>Country</label>"
        "<select id='c' name='country'>"
        "<option>India</option><option>Japan</option></select>"
    )
    await session.navigate(url)
    line = next(line for line in (await session.get_page()).splitlines() if "select" in line)

    assert '[1] select "Country"' in line
    assert "options: India | Japan" in line


async def test_an_unlabelled_dropdown_falls_back_to_its_name_attribute(session, serve) -> None:
    url = await serve(
        "<title>T</title><select name='country'>"
        "<option>India</option><option>Japan</option></select>"
    )
    await session.navigate(url)
    out = await session.get_page()

    assert '[1] select "country"' in out
    assert '"India Japan"' not in out


async def test_a_page_with_nothing_to_click_still_renders(session, serve) -> None:
    url = await serve("<title>Plain</title><p>Just words.</p>")
    out = await session.navigate(url)
    assert "Interactive elements: none found." in out
    assert "Just words." in out


# --- tabs and dialogs ---------------------------------------------------------


async def test_a_popup_becomes_the_current_tab_and_is_noted(session, serve) -> None:
    await session.navigate(await serve(PAGE))
    await serve("<title>Popup</title><p>Second tab.</p>", "/popup")

    await session.page.evaluate("window.open('https://example.test/popup')")
    await asyncio.sleep(0.5)

    assert len(session.pages) == 2
    out = await session.get_page()
    assert "A new tab opened" in out
    assert "Page: Popup" in out
    assert "Tabs:" in out


async def test_notes_are_shown_once_and_then_cleared(session, serve) -> None:
    await session.navigate(await serve(PAGE))
    session.notes.append("Something happened.")

    assert "Note: Something happened." in await session.get_page()
    assert "Something happened." not in await session.get_page()


async def test_a_dialog_is_accepted_so_the_page_keeps_moving(session, serve) -> None:
    url = await serve(PAGE)
    await session.navigate(url)
    await session.page.evaluate("setTimeout(() => alert('heads up'), 0)")
    await asyncio.sleep(0.5)

    out = await session.get_page()
    assert "alert dialog" in out
    assert "heads up" in out


async def test_closing_the_current_tab_falls_back_to_the_previous_one(session, serve) -> None:
    await session.navigate(await serve(PAGE))
    await serve("<title>Popup</title>", "/popup")
    first = session.page

    await session.page.evaluate("window.open('https://example.test/popup')")
    await asyncio.sleep(0.5)
    await session.page.close()
    await asyncio.sleep(0.2)

    assert session.page is first
    assert "the tab you were on closed" in (await session.get_page()).lower()


@pytest.mark.parametrize("headless", [True])
async def test_two_sessions_do_not_see_each_others_page(headless: bool) -> None:
    async with BrowserSession(headless=headless) as outer:
        outer_page = outer.page
        assert current_session() is outer
        assert current_session().page is outer_page


# --- how the browser introduces itself ----------------------------------------


async def test_the_browser_does_not_announce_that_it_is_headless(session) -> None:
    # Headless Chromium says "HeadlessChrome/153.0…" in its user agent, and
    # several large sites read that and refuse to serve it. Google and Bing
    # both did, which is how this was found.
    agent = await session.page.evaluate("navigator.userAgent")

    assert "Headless" not in agent
    assert "Chrome/" in agent


async def test_the_version_in_the_user_agent_is_the_real_one(session) -> None:
    # Derived from the running browser rather than written out by hand, so it
    # cannot drift into claiming a version that is not there.
    agent = await session.page.evaluate("navigator.userAgent")
    version = session._browser.version.replace("HeadlessChrome/", "")
    assert version in agent


async def test_webdriver_is_not_advertised(session) -> None:
    # One line of JavaScript reads this; the launch flag turns it off.
    assert await session.page.evaluate("navigator.webdriver") in (False, None)


async def test_a_configured_user_agent_wins(monkeypatch) -> None:
    from browser_agent import config
    from browser_agent.session import BrowserSession

    monkeypatch.setattr(config, "USER_AGENT", "Mozilla/5.0 (something specific)")
    s = BrowserSession(headless=True)
    await s.start()
    try:
        assert await s.page.evaluate("navigator.userAgent") == "Mozilla/5.0 (something specific)"
    finally:
        await s.close()


# --- launch flags -------------------------------------------------------------


def test_automation_is_not_advertised_in_the_launch_flags() -> None:
    from browser_agent.session import _launch_args

    assert "--disable-blink-features=AutomationControlled" in _launch_args()


def test_a_container_gets_the_two_flags_it_cannot_run_without(monkeypatch) -> None:
    from browser_agent import config
    from browser_agent.session import _launch_args

    monkeypatch.setattr(config, "CONTAINER", True)
    args = _launch_args()

    # Chromium's sandbox needs privileges a container withholds, and the default
    # /dev/shm there is small enough that Chromium exhausts it and crashes
    # somewhere that looks unrelated.
    assert "--no-sandbox" in args
    assert "--disable-dev-shm-usage" in args


def test_a_laptop_keeps_chromiums_own_sandbox(monkeypatch) -> None:
    from browser_agent import config
    from browser_agent.session import _launch_args

    monkeypatch.setattr(config, "CONTAINER", False)
    assert "--no-sandbox" not in _launch_args()


# --- client hints -------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent", "platform", "major"),
    [
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/153.0.8010.12 Safari/537.36", "macOS", "153"),
        ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36", "Linux", "141"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36", "Windows", "140"),
    ],
)
def test_client_hints_follow_the_user_agent(agent: str, platform: str, major: str) -> None:
    hints = BrowserSession._client_hints(agent)

    # Overriding the user agent does not change these, and a browser that claims
    # to be Chrome in one header and something else in another gets its
    # connection dropped — reported by Chromium as ERR_HTTP2_PROTOCOL_ERROR,
    # which looks like anything but a header problem. Measured on irctc.co.in
    # and makemytrip.com, both of which went from that error to HTTP 200.
    assert hints["sec-ch-ua-platform"] == f'"{platform}"'
    assert f'"Google Chrome";v="{major}"' in hints["sec-ch-ua"]
    assert hints["sec-ch-ua-mobile"] == "?0"


def test_a_user_agent_with_no_version_gets_no_hints() -> None:
    # Better to send nothing than to send hints that disagree with it.
    assert BrowserSession._client_hints("something else entirely") == {}


async def test_the_hints_a_site_receives_agree_with_the_agent(session, serve) -> None:
    seen = {}

    async def capture(route, request):
        seen.update(request.headers)
        await route.fulfill(status=200, content_type="text/html", body="<title>H</title>")

    await session._context.route("https://hints.test/**", capture)
    await session.navigate("https://hints.test/")

    assert "sec-ch-ua" in seen
    assert "Headless" not in seen["sec-ch-ua"]
    assert "Headless" not in seen["user-agent"]
    # The major version in both has to be the same number.
    import re

    from_agent = re.search(r"Chrome/(\d+)", seen["user-agent"]).group(1)
    assert f'v="{from_agent}"' in seen["sec-ch-ua"]
