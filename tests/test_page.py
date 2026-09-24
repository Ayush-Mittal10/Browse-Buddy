"""The page's own JavaScript, run in a real browser.

The markdown renderer is the one piece of front-end logic with enough edge
cases to be worth testing directly, and it is the piece that turns text from a
web page the agent read into markup — so its escaping is a security property,
not a nicety.

The page is loaded from the app's own response, so what is tested is what is
actually served.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from browser_agent.web import build_app  # noqa: E402


@pytest.fixture
async def rendered(session):
    """A function that renders markdown the way the page does."""
    html = TestClient(build_app()).get("/").text
    page = await session._context.new_page()
    await page.set_content(html)

    async def render(text: str) -> dict:
        return await page.evaluate(
            """(text) => {
                const el = document.createElement('div');
                el.innerHTML = markdown(text);
                return {
                  links: [...el.querySelectorAll('a')].map(a => a.getAttribute('href')),
                  rels: [...el.querySelectorAll('a')].map(a => a.rel),
                  text: el.textContent,
                  html: el.innerHTML,
                  code: [...el.querySelectorAll('code')].map(c => c.textContent),
                  starts: [...el.querySelectorAll('ol')].map(o => o.getAttribute('start')),
                  imgs: el.querySelectorAll('img').length,
                  scripts: el.querySelectorAll('script').length,
                };
            }""",
            text,
        )

    yield render
    await page.close()


# --- links --------------------------------------------------------------------


async def test_a_bare_url_becomes_a_link(rendered) -> None:
    # How a model actually writes one: "you can watch it at <url>".
    out = await rendered("Watch it at https://www.youtube.com/watch?v=EGN6A0Hrn48 now.")
    assert out["links"] == ["https://www.youtube.com/watch?v=EGN6A0Hrn48"]


async def test_a_full_stop_is_not_part_of_the_address(rendered) -> None:
    out = await rendered("See https://example.com/page.")
    assert out["links"] == ["https://example.com/page"]
    assert out["text"].endswith(".")


async def test_a_wrapping_bracket_is_not_part_of_the_address(rendered) -> None:
    out = await rendered("(see https://example.com/a)")
    assert out["links"] == ["https://example.com/a"]


async def test_but_a_bracket_the_url_opened_itself_is(rendered) -> None:
    # Half of Wikipedia is written this way.
    out = await rendered("https://en.wikipedia.org/wiki/Nice_(France)")
    assert out["links"] == ["https://en.wikipedia.org/wiki/Nice_(France)"]


async def test_a_query_string_survives(rendered) -> None:
    out = await rendered("https://youtube.com/watch?v=a1&t=30s ok")
    assert out["links"] == ["https://youtube.com/watch?v=a1&t=30s"]


async def test_a_markdown_link_still_works(rendered) -> None:
    out = await rendered("Read [the docs](https://example.com/docs) first.")
    assert out["links"] == ["https://example.com/docs"]
    assert "the docs" in out["text"]


async def test_a_markdown_link_is_not_linked_twice(rendered) -> None:
    # The bare-URL pass must not reach inside an anchor the markdown pass built.
    out = await rendered("[Docs](https://example.com/d) and https://example.com/e")
    assert out["links"] == ["https://example.com/d", "https://example.com/e"]


async def test_a_url_inside_code_is_left_alone(rendered) -> None:
    out = await rendered("run `curl https://example.com` please")
    assert out["links"] == []
    assert out["code"] == ["curl https://example.com"]


@pytest.mark.parametrize(
    "text",
    ["ftp://example.com", "just example.com", "javascript:alert(1)", "mailto:a@b.test"],
)
async def test_only_http_becomes_a_link(rendered, text: str) -> None:
    assert (await rendered(text))["links"] == []


async def test_every_link_is_safe_to_click(rendered) -> None:
    out = await rendered("https://a.test/b and [c](https://c.test/d)")
    assert out["rels"] == ["noopener noreferrer", "noopener noreferrer"]


# --- escaping -----------------------------------------------------------------


async def test_markup_in_the_text_stays_text(rendered) -> None:
    # The agent reads untrusted pages for a living, so a report is not a safe
    # string: whatever a page said can end up here.
    out = await rendered('x <img src=y onerror="alert(1)"> https://a.test/b')
    assert out["imgs"] == 0
    assert out["scripts"] == 0
    assert "<img" in out["text"]


async def test_a_script_tag_is_not_a_script_tag(rendered) -> None:
    out = await rendered("<script>alert(1)</script>")
    assert out["scripts"] == 0
    assert "alert(1)" in out["text"]


async def test_an_anchor_in_the_text_cannot_disguise_itself(rendered) -> None:
    out = await rendered('<a href="https://evil.test">click</a>')

    # The tag is shown as characters, not obeyed.
    assert "<a href=" in out["text"]
    # The address inside it is still made clickable, which is the feature —
    # but the label is the address itself. A page the agent read cannot get a
    # link into the chat whose words say one thing and whose href says another.
    assert out["links"] == ["https://evil.test"]
    assert "https://evil.test" in out["text"]


async def test_a_linked_address_always_shows_itself(rendered) -> None:
    out = await rendered("go to https://real.test/page now")
    assert out["links"] == ["https://real.test/page"]
    assert "https://real.test/page" in out["text"]


# --- the rest of the markdown -------------------------------------------------


async def test_bold_and_lists_still_render(rendered) -> None:
    out = await rendered("**Zolo:** ₹5,200\n\n* one\n* two\n\n1. first\n2. second")
    assert "<strong>Zolo:</strong>" in out["html"]
    assert out["html"].count("<li>") == 4
    assert "<ul>" in out["html"] and "<ol start=" in out["html"]


async def test_a_numbered_list_keeps_counting_past_its_bullets(rendered) -> None:
    """The shape every "here are your options" report comes back in.

    Details hang off each entry as bullets, which end the numbered list. What
    the reader must still see is 1, 2, 3 — not three entries all called 1.
    """
    out = await rendered(
        "1. Hasdeo Express\n- Departure: 06:30\n"
        "2. Wainganga Express\n- Departure: 08:05\n"
        "3. Chhattisgarh Express\n- Departure: 11:33"
    )
    assert out["starts"] == ["1", "2", "3"]
