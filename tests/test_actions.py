"""The browser actions, against a real Chromium.

Two things every one of these checks, beyond doing the thing: the result is a
string the model can read, and a failure is reported rather than raised.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from browser_agent import config

FORM = """
<!doctype html>
<title>Form</title>
<h1>Sign in</h1>
<input type="email" name="email" placeholder="Email">
<input type="password" name="password" placeholder="Password">
<select name="country">
  <option value="in">India</option>
  <option value="jp">Japan</option>
</select>
<button onclick="document.getElementById('out').textContent = 'clicked'">Submit</button>
<div id="out"></div>
"""

TALL = """
<!doctype html>
<title>Tall</title>
<div style="height: 4000px">
  <p style="margin-top: 2500px">Down here at the bottom.</p>
</div>
<a href="/next">Bottom link</a>
"""


async def refs(session) -> dict[str, int]:
    """Map each element's name to its current ref, from a fresh snapshot."""
    out = {}
    for line in (await session.get_page()).splitlines():
        line = line.lstrip("~")
        if line.startswith("[") and '"' in line:
            ref = int(line[1 : line.index("]")])
            out[line.split('"')[1]] = ref
    return out


# --- resolving refs -----------------------------------------------------------


async def test_a_stale_ref_asks_for_a_fresh_look_instead_of_clicking(session, serve) -> None:
    await session.navigate(await serve(FORM))
    await session.get_page()
    out = await session.click(999)

    assert "no longer on the page" in out
    assert "get_page" in out


async def test_a_ref_that_is_not_a_number_is_explained(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.click("the submit button")

    assert "is not an element number" in out


async def test_bracketed_refs_are_accepted(session, serve) -> None:
    await session.navigate(await serve(FORM))
    ref = (await refs(session))["Submit"]
    assert "Clicked" in await session.click(f"[{ref}]")


async def test_refs_go_stale_when_the_page_re_renders(session, serve) -> None:
    await session.navigate(await serve(FORM))
    ref = (await refs(session))["Submit"]
    await session.page.evaluate("document.body.innerHTML = '<p>gone</p>'")

    out = await session.click(ref)
    assert "no longer on the page" in out


# --- click --------------------------------------------------------------------


async def test_click_acts_and_returns_the_new_page(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.click((await refs(session))["Submit"])

    assert out.startswith("Clicked [")
    assert "clicked" in out  # the div the handler wrote into


async def test_click_falls_back_to_force_when_something_covers_the_target(
    session, serve
) -> None:
    url = await serve(
        FORM + "<div style='position:fixed;inset:0;background:red;z-index:99'>cover</div>"
    )
    await session.navigate(url)
    out = await session.click((await refs(session))["Submit"])

    assert out.startswith("Clicked [")


# --- type_text ----------------------------------------------------------------


async def test_type_text_fills_a_field_and_echoes_what_it_typed(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.type_text((await refs(session))["Email"], "me@example.com")

    assert 'Typed "me@example.com"' in out
    assert await session.page.input_value("[name=email]") == "me@example.com"


async def test_a_password_is_filled_but_never_echoed(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.type_text((await refs(session))["Password"], "hunter2")

    assert "hunter2" not in out
    assert "••••" in out
    # Still actually typed — masking is on the output, not the input.
    assert await session.page.input_value("[name=password]") == "hunter2"


async def test_long_values_are_truncated_in_the_echo(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.type_text((await refs(session))["Email"], "x" * 100)

    assert "…" in out
    assert "x" * 100 not in out


async def test_type_text_can_press_enter(session, serve) -> None:
    url = await serve(
        "<title>Search</title><form onsubmit=\"document.title='submitted';return false\">"
        "<input name='q' placeholder='Query'></form>"
    )
    await session.navigate(url)
    out = await session.type_text((await refs(session))["Query"], "hello", press_enter=True)

    assert "pressed Enter" in out
    assert await session.page.title() == "submitted"


async def test_typing_into_something_unfillable_is_reported(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.type_text((await refs(session))["Submit"], "text")

    assert isinstance(out, str)
    assert "Typed" in out or "failed" in out


# --- select_option ------------------------------------------------------------


async def test_select_option_by_visible_label(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.select_option((await refs(session))["country"], "Japan")

    assert "Selected 'Japan'" in out
    assert await session.page.input_value("[name=country]") == "jp"


async def test_select_option_falls_back_to_the_value(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.select_option((await refs(session))["country"], "jp")

    assert "Selected 'jp'" in out
    assert await session.page.input_value("[name=country]") == "jp"


async def test_selecting_something_that_is_not_there_explains_the_alternative(
    session, serve
) -> None:
    await session.navigate(await serve(FORM))
    out = await session.select_option((await refs(session))["country"], "Atlantis")

    assert "Could not select 'Atlantis'" in out
    assert "click it and pick" in out


# --- press_key ----------------------------------------------------------------


async def test_press_key(session, serve) -> None:
    url = await serve(
        "<title>Keys</title><p id='out'>none</p>"
        "<script>document.addEventListener('keydown', e => "
        "document.getElementById('out').textContent = e.key)</script>"
    )
    await session.navigate(url)
    out = await session.press_key("ArrowDown")

    assert out.startswith("Pressed ArrowDown.")
    assert "ArrowDown" in out


async def test_an_unknown_key_name_suggests_real_ones(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.press_key("NotAKey")

    assert "Could not press" in out
    assert "Enter, Escape, Tab" in out


# --- scroll -------------------------------------------------------------------


async def test_scroll_moves_down_and_back_up(session, serve) -> None:
    await session.navigate(await serve(TALL))
    assert "at top" in await session.get_page()

    out = await session.scroll("down", 3)
    assert out.startswith("Scrolled down.")
    assert await session.page.evaluate("window.scrollY") > 0

    out = await session.scroll("up", 10)
    assert out.startswith("Scrolled up.")
    assert await session.page.evaluate("window.scrollY") == 0


async def test_scroll_clamps_a_silly_page_count(session, serve) -> None:
    await session.navigate(await serve(TALL))
    assert "Scrolled down." in await session.scroll("down", 10_000)


async def test_scroll_defaults_to_down_on_a_junk_direction(session, serve) -> None:
    await session.navigate(await serve(TALL))
    assert (await session.scroll("sideways", 1)).startswith("Scrolled down.")


async def test_scroll_survives_a_non_numeric_page_count(session, serve) -> None:
    await session.navigate(await serve(TALL))
    assert (await session.scroll("down", "two")).startswith("Scrolled down.")


# --- go_back ------------------------------------------------------------------


async def test_go_back_returns_to_the_previous_page(session, serve) -> None:
    first = await serve(FORM)
    second = await serve("<title>Second</title><p>Second page.</p>", "/second")
    await session.navigate(first)
    await session.navigate(second)

    out = await session.go_back()
    assert out.startswith("Went back.")
    assert "Page: Form" in out


async def test_go_back_with_no_history_says_so(session) -> None:
    out = await session.go_back()
    assert "no previous page" in out


# --- read_text ----------------------------------------------------------------


async def test_read_text_returns_the_whole_body_when_it_is_short(session, serve) -> None:
    await session.navigate(await serve("<title>T</title><p>Short body text.</p>"))
    out = await session.read_text()

    assert "Short body text." in out
    assert out.startswith("Text 0-")


async def test_read_text_chunks_long_content_and_points_at_the_next_chunk(
    session, serve
) -> None:
    body = "word " * 4000
    await session.navigate(await serve(f"<title>Long</title><p>{body}</p>"))
    first = await session.read_text()

    assert "read_text with start=6000 for more" in first

    second = await session.read_text(start=6000)
    assert "Text 6000-" in second


async def test_read_text_past_the_end_says_so(session, serve) -> None:
    await session.navigate(await serve("<title>T</title><p>Short.</p>"))
    assert "Nothing past character" in await session.read_text(start=99_999)


async def test_read_text_can_target_one_element(session, serve) -> None:
    url = await serve(
        "<title>T</title><p>Ignore this.</p>"
        "<article><p>The part that matters.</p></article>"
        "<a href='/x'>link</a>"
    )
    await session.navigate(url)
    await session.get_page()
    # The article is not interactive, so give read_text a ref that is.
    out = await session.read_text(ref=(await refs(session))["link"])
    assert "link" in out
    assert "Ignore this." not in out


async def test_read_text_rejects_a_non_numeric_ref(session, serve) -> None:
    await session.navigate(await serve(FORM))
    assert "is not an element number" in await session.read_text(ref="the article")


async def test_read_text_on_a_vanished_element_asks_for_fresh_numbers(session, serve) -> None:
    await session.navigate(await serve(FORM))
    assert "no longer on the page" in await session.read_text(ref=999)


# --- screenshot ---------------------------------------------------------------


async def test_screenshot_is_stored_for_the_loop_to_attach(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.screenshot()

    assert "Screenshot taken" in out
    data, mime = session.last_screenshot
    assert mime == "image/jpeg"
    assert base64.b64decode(data)[:2] == b"\xff\xd8"  # JPEG magic number


async def test_take_screenshot_pops_it_exactly_once(session, serve) -> None:
    await session.navigate(await serve(FORM))
    await session.screenshot()

    assert session.take_screenshot() is not None
    assert session.take_screenshot() is None


# --- wait ---------------------------------------------------------------------


async def test_wait_pauses_and_returns_the_page(session, serve) -> None:
    await session.navigate(await serve(FORM))
    out = await session.wait(0.5)

    assert out.startswith("Waited 0.5s.")
    assert "Page: Form" in out


async def test_wait_is_clamped_at_both_ends(session, serve) -> None:
    await session.navigate(await serve(FORM))
    assert (await session.wait(0.01)).startswith("Waited 0.5s.")

    out = await session.wait(9999)
    assert out.startswith(f"Waited {config.MAX_WAIT_S}s.")


# --- switch_tab ---------------------------------------------------------------


async def test_switch_tab_goes_back_to_the_first_tab(session, serve) -> None:
    await session.navigate(await serve(FORM))
    await serve("<title>Popup</title><p>Second tab.</p>", "/popup")
    await session.page.evaluate("window.open('https://example.test/popup')")
    await asyncio.sleep(0.5)

    assert "Page: Popup" in await session.get_page()

    out = await session.switch_tab(1)
    assert out.startswith("Switched to tab 1.")
    assert "Page: Form" in out


@pytest.mark.parametrize("index", [0, 2, 99])
async def test_switching_to_a_tab_that_is_not_open(session, serve, index: int) -> None:
    await session.navigate(await serve(FORM))
    out = await session.switch_tab(index)

    assert f"There is no tab {index}" in out
    assert "Open tabs: 1" in out


async def test_switch_tab_rejects_a_non_numeric_index(session, serve) -> None:
    await session.navigate(await serve(FORM))
    assert "Give the tab number" in await session.switch_tab("the second one")
