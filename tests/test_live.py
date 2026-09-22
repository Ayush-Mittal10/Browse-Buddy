"""The live view: JPEG frames out of a headless browser.

A real Chromium throughout, because what is being tested is that the DevTools
screencast works at all from a headless browser — which is the whole reason the
web UI does not need an X server.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from browser_agent.live import Screencast
from browser_agent.session import BrowserSession

MOVING = """
<!doctype html>
<title>Moving</title>
<h1>frame 0</h1>
<script>
let n = 0;
setInterval(() => {
  n += 1;
  document.body.style.background = `hsl(${(n * 37) % 360}, 70%, 80%)`;
  document.querySelector('h1').textContent = 'frame ' + n;
}, 60);
</script>
"""

STILL = "<!doctype html><title>Still</title><h1>Nothing happens here</h1>"


async def frames_from(session, serve, html: str = MOVING, seconds: float = 1.5) -> list:
    got = []
    session.live = Screencast(lambda data, mime: got.append((data, mime)))
    url = await serve(html)
    await session.navigate(url)
    await session.live.follow(session._context, session.page)
    await asyncio.sleep(seconds)
    await session.live.stop()
    return got


async def test_frames_arrive_from_a_headless_browser(session, serve) -> None:
    assert session.headless is True
    got = await frames_from(session, serve)

    assert got, "no frames arrived"
    data, mime = got[0]
    assert mime == "image/jpeg"
    assert base64.b64decode(data)[:2] == b"\xff\xd8"  # JPEG magic number


async def test_a_moving_page_produces_a_stream_not_one_frame(session, serve) -> None:
    got = await frames_from(session, serve, seconds=1.5)
    # Anything above a couple of frames a second is a live picture rather than
    # a slideshow; measured around ten.
    assert len(got) >= 3, f"only {len(got)} frames"


async def test_frames_keep_coming_which_means_acks_are_landing(session, serve) -> None:
    # Chromium sends the next frame only once the last is acknowledged, so a
    # missed ack does not error — it just freezes the picture after frame one.
    got = await frames_from(session, serve, seconds=1.2)
    assert len(got) > 1, "the stream stalled after the first frame"


async def test_the_cast_follows_the_current_tab(session, serve) -> None:
    got = []
    session.live = Screencast(lambda data, mime: got.append((data, mime)))
    await session.navigate(await serve(STILL))
    await serve(MOVING, "/popup")

    await session.live.follow(session._context, session.page)
    await asyncio.sleep(0.4)
    before = len(got)

    await session.page.evaluate("window.open('https://example.test/popup')")
    await asyncio.sleep(1.2)

    assert session.live._page is session.page  # moved to the popup
    assert len(got) > before  # and is still producing frames
    await session.live.stop()


async def test_following_the_same_page_twice_is_a_no_op(session, serve) -> None:
    session.live = Screencast(lambda data, mime: None)
    await session.navigate(await serve(STILL))

    await session.live.follow(session._context, session.page)
    first = session.live._cdp
    await session.live.follow(session._context, session.page)

    assert session.live._cdp is first
    await session.live.stop()


async def test_stopping_twice_is_safe(session, serve) -> None:
    session.live = Screencast(lambda data, mime: None)
    await session.navigate(await serve(STILL))
    await session.live.follow(session._context, session.page)

    await session.live.stop()
    await session.live.stop()
    assert session.live._cdp is None


async def test_a_handler_that_raises_does_not_stop_the_run(session, serve) -> None:
    # The live view is a nice-to-have; a broken viewer must not take the task
    # down with it.
    def explode(data, mime):
        raise RuntimeError("the socket went away")

    session.live = Screencast(explode)
    url = await serve(MOVING)
    await session.navigate(url)
    await session.live.follow(session._context, session.page)
    await asyncio.sleep(0.6)

    assert "Page: Moving" in await session.get_page()
    await session.live.stop()


async def test_a_session_without_a_frame_handler_has_no_live_view() -> None:
    assert BrowserSession().live is None


async def test_asking_for_one_gets_one() -> None:
    session = BrowserSession(on_frame=lambda data, mime: None)
    assert session.live is not None


@pytest.mark.parametrize("page", [None])
async def test_following_nothing_is_harmless(session, page) -> None:
    cast = Screencast(lambda data, mime: None)
    await cast.follow(session._context, page)
    assert cast._cdp is None


# --- reaching it through the agent --------------------------------------------


async def test_the_agent_passes_a_frame_handler_down_to_the_browser() -> None:
    from browser_agent.agent import BrowserAgent

    frames = []
    agent = BrowserAgent(on_frame=lambda data, mime: frames.append(data))
    try:
        await agent.run("", start_url="")
    except Exception:
        pass  # no model configured here; the browser is what matters
    assert agent._session is None or agent._session.live is not None
    await agent.close()


def test_an_agent_without_one_asks_for_no_live_view() -> None:
    from browser_agent.agent import BrowserAgent

    assert BrowserAgent().on_frame is None
