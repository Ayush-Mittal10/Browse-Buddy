"""Watching the browser work, without a browser window.

Chromium's DevTools protocol will stream the page as JPEG frames, and it does
that from a headless browser — no X server, no VNC, no display of any kind. So a
viewer somewhere else can see the page move while the browser itself stays
headless on the machine doing the work.

Measured on an M5: ~10 frames a second at ~67 KB/s with the defaults below.
Cheap enough to push down a WebSocket and forget about.

Frames follow the current tab. When the agent opens a popup or switches tabs the
cast moves with it, because the point is to show what the agent is looking at.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable

from browser_agent import config

logger = logging.getLogger(__name__)

# One frame, as (base64 jpeg, mime).
FrameHandler = Callable[[str, str], None]


class Screencast:
    """A JPEG stream from whichever page is current."""

    def __init__(
        self,
        on_frame: FrameHandler,
        *,
        quality: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
        every_nth: int | None = None,
    ):
        self.on_frame = on_frame
        self.quality = quality or config.LIVE_QUALITY
        self.max_width = max_width or config.LIVE_MAX_WIDTH
        self.max_height = max_height or config.LIVE_MAX_HEIGHT
        self.every_nth = every_nth or config.LIVE_EVERY_NTH
        self.frames = 0
        self._page = None
        self._cdp = None

    async def follow(self, context, page) -> None:
        """Cast `page` instead of whatever was being cast before."""
        if page is None or page is self._page:
            return
        await self.stop()
        try:
            cdp = await context.new_cdp_session(page)
            cdp.on("Page.screencastFrame", self._frame)
            await cdp.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": self.quality,
                    "maxWidth": self.max_width,
                    "maxHeight": self.max_height,
                    "everyNthFrame": self.every_nth,
                },
            )
        except Exception as e:
            # A live view is a nice-to-have. Losing it must never take the run
            # with it, so this is logged and shrugged off.
            logger.warning("Could not start the live view: %s", e)
            return
        self._page, self._cdp = page, cdp
        logger.debug("Live view following %s", getattr(page, "url", ""))

    def _frame(self, params: dict) -> None:
        self.frames += 1
        try:
            self.on_frame(params["data"], "image/jpeg")
        except Exception as e:
            logger.warning("Live view handler raised: %s", e)
        # Chromium sends the next frame only once this one is acknowledged, so
        # a missed ack silently freezes the picture rather than erroring.
        session = self._cdp
        if session is not None:
            asyncio.ensure_future(_ack(session, params.get("sessionId")))

    async def stop(self) -> None:
        cdp, self._cdp, self._page = self._cdp, None, None
        if cdp is None:
            return
        with contextlib.suppress(Exception):
            await cdp.send("Page.stopScreencast")
        with contextlib.suppress(Exception):
            await cdp.detach()


async def _ack(session, session_id) -> None:
    if session_id is None:
        return
    with contextlib.suppress(Exception):
        # The page closing mid-frame is ordinary, not an error.
        await session.send("Page.screencastFrameAck", {"sessionId": session_id})
