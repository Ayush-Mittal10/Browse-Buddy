"""Recognising a page the viewer should be watching rather than reading."""

from __future__ import annotations

import pytest

from browser_agent.media import detect


@pytest.mark.parametrize(
    ("url", "ident"),
    [
        ("https://www.youtube.com/watch?v=EGN6A0Hrn48", "EGN6A0Hrn48"),
        ("https://youtube.com/watch?v=EGN6A0Hrn48", "EGN6A0Hrn48"),
        ("https://m.youtube.com/watch?v=EGN6A0Hrn48", "EGN6A0Hrn48"),
        # The id is rarely the first parameter in a real URL.
        ("https://www.youtube.com/watch?list=PL1&v=jfKfPfyJRdk&t=42", "jfKfPfyJRdk"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/abc123XYZ", "abc123XYZ"),
        ("https://www.youtube.com/embed/abc123XYZ", "abc123XYZ"),
    ],
)
def test_youtube_in_its_various_shapes(url: str, ident: str) -> None:
    found = detect(url)
    assert found is not None
    assert found.provider == "youtube"
    assert found.id == ident
    assert found.url == f"https://www.youtube.com/watch?v={ident}"
    assert found.embed.startswith(f"https://www.youtube.com/embed/{ident}")


def test_vimeo() -> None:
    found = detect("https://vimeo.com/123456789")
    assert (found.provider, found.id) == ("vimeo", "123456789")
    assert "player.vimeo.com" in found.embed


@pytest.mark.parametrize("kind", ["track", "album", "playlist", "episode"])
def test_spotify_keeps_the_kind_it_found(kind: str) -> None:
    found = detect(f"https://open.spotify.com/{kind}/4cOdK2wGLETKBW3PvgPWqT")
    assert found.provider == "spotify"
    # An album embedded as a track would play the wrong thing entirely.
    assert f"/embed/{kind}/" in found.embed


def test_spotify_with_a_locale_segment() -> None:
    assert detect("https://open.spotify.com/intl-de/track/4cOdK2wGLETKBW3PvgPWqT") is not None


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://news.ycombinator.com/",
        "https://en.wikipedia.org/wiki/YouTube",
        "https://www.youtube.com/",           # the home page is not a video
        "https://www.youtube.com/results?search_query=x",
        "about:blank",
    ],
)
def test_ordinary_pages_are_not_media(url: str) -> None:
    assert detect(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.test/youtube.com/watch?v=abc123",
        "https://youtube.com.evil.test/watch?v=abc123",
        "https://notyoutube.com/watch?v=abc123",
    ],
)
def test_a_lookalike_host_is_not_youtube(url: str) -> None:
    # The pattern is anchored to the host, so a path or a suffix that merely
    # contains the name cannot get a link opened in someone's browser.
    assert detect(url) is None


def test_none_and_whitespace_are_safe() -> None:
    assert detect(None) is None
    assert detect("   ") is None
