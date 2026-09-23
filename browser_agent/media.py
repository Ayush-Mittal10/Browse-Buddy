"""Recognising when the agent has landed on something meant to be watched.

The browser runs on the server, and a server has no speakers. Headless Chromium
has no audio output device at all, so a video there plays silently into nothing
— and streaming that audio back would mean a virtual sound card, an encoder and
a second transport, to deliver a worse copy of something the viewer's own
browser can play perfectly.

So the agent finds the video and hands it over. What plays, plays where the
person is: with sound, at full quality, under their control, and still going
after the task and even the connection have ended.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class Media(NamedTuple):
    provider: str
    id: str
    url: str       # what to open for the viewer
    embed: str     # what to put in an iframe, when embedding is wanted


# Ordered, and each one anchored to the host so that a path containing
# "youtube.com" on some other site cannot match.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("youtube", re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/watch\?(?:.*&)?v=([\w-]{6,})")),
    ("youtube", re.compile(r"^https?://youtu\.be/([\w-]{6,})")),
    ("youtube", re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/shorts/([\w-]{6,})")),
    ("youtube", re.compile(r"^https?://(?:www\.)?youtube\.com/embed/([\w-]{6,})")),
    ("vimeo", re.compile(r"^https?://(?:www\.)?vimeo\.com/(\d{6,})")),
    ("spotify", re.compile(r"^https?://open\.spotify\.com/(?:intl-\w+/)?(track|album|playlist|episode)/(\w+)")),
]


def detect(url: str) -> Media | None:
    """The media at `url`, or None if it is an ordinary page."""
    url = (url or "").strip()
    if not url:
        return None

    for provider, pattern in _PATTERNS:
        match = pattern.match(url)
        if not match:
            continue
        if provider == "spotify":
            kind, ident = match.group(1), match.group(2)
            return Media(
                provider,
                ident,
                f"https://open.spotify.com/{kind}/{ident}",
                f"https://open.spotify.com/embed/{kind}/{ident}",
            )
        ident = match.group(1)
        if provider == "youtube":
            return Media(
                provider,
                ident,
                f"https://www.youtube.com/watch?v={ident}",
                # enablejsapi is deliberately absent: nothing here drives the
                # player, and the fewer things the embed can be asked to do the
                # better it behaves inside someone else's page.
                f"https://www.youtube.com/embed/{ident}?autoplay=1",
            )
        return Media(
            provider,
            ident,
            f"https://vimeo.com/{ident}",
            f"https://player.vimeo.com/video/{ident}?autoplay=1",
        )
    return None
