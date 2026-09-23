"""Telling a bot check apart from the page that was wanted.

Reproduced from a real run: every Google search from this browser lands on
google.com/sorry/index, whether the query is typed into the box or opened as a
/search URL, and whatever the query is. The page that comes back is short and
ordinary-looking, so without this the agent treats it as a normal page and
tries to interact with it.
"""

from __future__ import annotations

import pytest

from browser_agent.session import blocked_advice, looks_blocked

# An article that discusses bot checks says all the same words, at length.
ARTICLE = (
    "A CAPTCHA is a challenge-response test used to determine whether the user is human. "
    "The phrase 'not a robot' appears on reCAPTCHA checkboxes. "
) + "Historical background, criticism, and accessibility concerns. " * 120


@pytest.mark.parametrize(
    ("url", "text"),
    [
        ("https://www.google.com/sorry/index?continue=x", ""),
        ("https://www.google.com/search?q=x", "Our systems have detected unusual traffic"),
        ("https://x.test/", "Checking your browser before accessing"),
        ("https://x.test/cdn-cgi/challenge-platform/", ""),
        ("https://x.test/", "Please verify you are human"),
        ("https://x.test/", "enable JavaScript and cookies to continue"),
    ],
)
def test_a_bot_check_is_recognised(url: str, text: str) -> None:
    assert looks_blocked(url, text) is True


@pytest.mark.parametrize(
    ("url", "text"),
    [
        ("https://en.wikipedia.org/wiki/CAPTCHA", ARTICLE),
        ("https://news.ycombinator.com/", "Hacker News new past comments ask show jobs"),
        ("https://duckduckgo.com/html/?q=irctc", "IRCTC Next Generation eTicketing System"),
        ("", ""),
    ],
)
def test_an_ordinary_page_is_not(url: str, text: str) -> None:
    # Calling a real page a block sends the agent away from somewhere it could
    # have used, which is the worse mistake of the two.
    assert looks_blocked(url, text) is False


def test_a_long_page_is_judged_by_its_url_only() -> None:
    # The phrases are common enough in prose that they only mean anything on a
    # page too short to be prose.
    assert looks_blocked("https://x.test/article", "unusual traffic " + "words " * 800) is False
    assert looks_blocked("https://x.test/sorry/", "unusual traffic " + "words " * 800) is True


def test_the_advice_says_not_to_solve_it() -> None:
    advice = blocked_advice("https://x.test/")
    assert "Do not attempt to solve it" in advice
    # Reloading is the thing a model reaches for first, and it never works.
    assert "reloading, waiting or trying another URL" in advice


def test_google_gets_told_it_is_hopeless_specifically() -> None:
    advice = blocked_advice("https://www.google.com/sorry/index")

    # Measured: the home page loads, every search ends here, typing into the box
    # is no different from opening a /search URL.
    assert "Google blocks automated searching entirely" in advice
    # The /html endpoint answers 403 now; the plain query form works and
    # returns more results. Measured, not assumed.
    assert "duckduckgo.com/?q=" in advice


def test_other_sites_get_general_advice() -> None:
    advice = blocked_advice("https://somewhere.test/challenge")
    assert "Go to a different site" in advice
    assert "tell the user it is blocking" in advice
