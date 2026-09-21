"""The guard on what the agent is allowed to open, and the two string helpers
around it. All pure — no browser needed.
"""

from __future__ import annotations

import pytest

from browser_agent.session import _normalise_url, _short_error, is_url_allowed


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "https://example.com/path?q=1#frag",
        "https://sub.domain.example.co.uk/",
        "https://example.com:8443/",
        "https://8.8.8.8/",  # public IP
    ],
)
def test_public_pages_are_allowed(url: str) -> None:
    allowed, why = is_url_allowed(url)
    assert allowed is True, why
    assert why == ""


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<h1>hi</h1>",
        "chrome://settings",
        "about:blank",
        "ftp://example.com/file",
    ],
)
def test_non_http_schemes_are_refused(url: str) -> None:
    allowed, why = is_url_allowed(url)
    assert allowed is False
    assert "http and https" in why


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://metadata.google.internal/",
        "http://instance-data/",
        "http://printer.local/",
        "http://api.internal/",
        "http://0.0.0.0/",
        "http://intranet/",  # single-label host
    ],
)
def test_private_and_internal_addresses_are_refused(url: str) -> None:
    allowed, why = is_url_allowed(url)
    assert allowed is False
    assert why == "That address is not a public website."


def test_trailing_dot_does_not_sneak_past_the_host_check() -> None:
    assert is_url_allowed("http://localhost./")[0] is False


def test_uppercase_host_does_not_sneak_past() -> None:
    assert is_url_allowed("http://LOCALHOST/")[0] is False


def test_url_without_a_host_is_refused() -> None:
    allowed, why = is_url_allowed("https://")
    assert allowed is False
    assert "no host" in why


@pytest.mark.parametrize("url", ["", "   ", None])
def test_empty_input_is_refused(url: str | None) -> None:
    assert is_url_allowed(url)[0] is False


# --- helpers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("example.com", "https://example.com"),
        ("  example.com  ", "https://example.com"),
        ("https://example.com", "https://example.com"),
        ("http://example.com", "http://example.com"),
        ("", ""),
    ],
)
def test_normalise_url_assumes_https(given: str, expected: str) -> None:
    assert _normalise_url(given) == expected


def test_short_error_drops_playwrights_call_log() -> None:
    err = Exception(
        "Timeout 8000ms exceeded. Call log:\n  - waiting for locator('[data-agent-ref=\"3\"]')"
    )
    assert _short_error(err) == "Timeout 8000ms exceeded."


def test_short_error_collapses_whitespace_and_truncates() -> None:
    assert _short_error(Exception("a\n\n  b\tc")) == "a b c"
    long = _short_error(Exception("x" * 500), limit=40)
    assert len(long) == 40
    assert long.endswith("…")
