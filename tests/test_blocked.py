"""Telling a bot check apart from the page that was wanted.

The page a bot check returns is short and ordinary-looking, so without this the
agent treats it as a normal page and tries to interact with it.

Google was once the example here: every search landed on google.com/sorry/index.
That turned out to be two separate things, and both have since been fixed. The
headers were one — we sent a Sec-CH-UA saying HeadlessChrome alongside a user
agent saying Chrome, and sites that read both refused the mismatch. The other is
that Google challenges a browser holding none of its cookies and *sets those
cookies on the challenge page*, so the second request succeeds where the first
did not. Measured after both fixes: four out of four cold searches return
results. The retry is in `BrowserSession._retry_past_interstitial`.
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
    # By the time the model sees this the retry has already happened, so
    # reloading really is spent — which was not true when the advice was written.
    assert "reloading again will not change it" in advice
    assert "tell the user it is blocking" in advice


def test_the_advice_offers_the_other_engines() -> None:
    # It used to say "Google blocks automated searching entirely" and send the
    # agent to one specific engine. Naming a single winner is what turned a
    # fallback chain into a hardcoded choice; all three are named now, and one
    # being blocked is explicitly not evidence about the others.
    for url in ("https://www.google.com/sorry/index", "https://somewhere.test/challenge"):
        advice = blocked_advice(url)
        assert "blocks automated searching entirely" not in advice
        assert "google.com/search?q=" in advice
        assert "bing.com/search?q=" in advice
        assert "duckduckgo.com/?q=" in advice


def test_the_advice_does_not_depend_on_the_url() -> None:
    # Nothing is host-specific any more, and the argument is optional.
    assert blocked_advice("https://www.google.com/sorry/") == blocked_advice()
