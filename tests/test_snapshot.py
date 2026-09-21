"""The half of the snapshot that runs in Python: the sensitivity check and the
formatter. The JS half needs a real browser and is exercised in session tests.
"""

from __future__ import annotations

import pytest

from browser_agent.snapshot import (
    MAX_ELEMENTS,
    _scroll_line,
    format_snapshot,
    is_sensitive_field,
)


def element(ref: int, **overrides: object) -> dict:
    """A snapshot element with everything SNAPSHOT_JS would set."""
    el = {
        "ref": ref,
        "tag": "button",
        "role": "",
        "type": "",
        "name": f"Button {ref}",
        "value": "",
        "state": [],
        "href": "",
        "options": [],
        "inView": True,
        "id": "",
        "fieldName": "",
        "placeholder": "",
        "ariaLabel": "",
        "autocomplete": "",
        "label": "",
    }
    el.update(overrides)
    return el


def page(**overrides: object) -> dict:
    data = {
        "url": "https://example.com/",
        "title": "Example Domain",
        "scrollY": 0,
        "scrollHeight": 800,
        "viewportHeight": 800,
        "elements": [],
        "text": "Example Domain",
        "textLength": 14,
    }
    data.update(overrides)
    return data


# --- sensitivity --------------------------------------------------------------


@pytest.mark.parametrize(
    "info",
    [
        {"type": "password"},
        {"autocomplete": "cc-number"},
        {"autocomplete": "one-time-code"},
        {"autocomplete": "current-password"},
        {"fieldName": "card_number"},
        {"fieldName": "cardNumber"},
        {"placeholder": "Card Number"},
        {"ariaLabel": "CVV"},
        {"label": "Enter OTP"},
        {"placeholder": "Enter your ATM pin"},
        {"label": "Aadhaar number"},
        {"fieldName": "pan_card"},
        {"label": "PAN"},
        {"placeholder": "IFSC"},
        {"label": "Passport number"},
        {"fieldName": "expiry"},
    ],
)
def test_sensitive_fields_are_flagged(info: dict) -> None:
    assert is_sensitive_field(info) is True


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"type": "text", "fieldName": "email"},
        {"placeholder": "Search"},
        {"fieldName": "pincode"},  # every Indian address form has one
        {"placeholder": "Pin code"},
        {"label": "Pan-fried noodles"},  # a menu item, not the tax id
        {"autocomplete": "email"},
        {"fieldName": "username"},
    ],
)
def test_ordinary_fields_are_not_flagged(info: dict) -> None:
    assert is_sensitive_field(info) is False


# --- scroll line --------------------------------------------------------------


def test_scroll_line_short_page() -> None:
    assert "whole page fits" in _scroll_line(page())


def test_scroll_line_top_middle_bottom() -> None:
    tall = {"scrollHeight": 5000, "viewportHeight": 800}
    assert "at top" in _scroll_line({**tall, "scrollY": 0})
    assert "at bottom" in _scroll_line({**tall, "scrollY": 4200})
    assert _scroll_line({**tall, "scrollY": 2100}) == "Scroll: 50% down"


# --- formatting ---------------------------------------------------------------


def test_missing_data_tells_the_model_what_to_do() -> None:
    out = format_snapshot(None)
    assert "no content" in out
    assert "get_page" in out


def test_header_carries_title_and_url() -> None:
    out = format_snapshot(page())
    assert "Page: Example Domain" in out
    assert "URL: https://example.com/" in out


def test_untitled_page_still_renders() -> None:
    assert "Page: (untitled)" in format_snapshot(page(title=""))


def test_element_line_shape() -> None:
    el = element(
        3,
        tag="a",
        name="Docs",
        href="/docs",
        state=["disabled"],
    )
    line = next(
        line for line in format_snapshot(page(elements=[el])).splitlines() if line.startswith("[3]")
    )
    assert line == '[3] link "Docs" -> /docs  [disabled]'


def test_input_kind_includes_its_type_and_value() -> None:
    el = element(1, tag="input", type="email", name="Email", value="a@b.com")
    assert '[1] input(email) "Email" = "a@b.com"' in format_snapshot(page(elements=[el]))


def test_select_lists_its_options() -> None:
    el = element(1, tag="select", name="Country", options=["India", "Japan"])
    assert "options: India | Japan" in format_snapshot(page(elements=[el]))


def test_role_wins_over_tag() -> None:
    el = element(1, tag="div", role="checkbox", name="Agree", state=["unchecked"])
    assert '[1] checkbox "Agree"  [unchecked]' in format_snapshot(page(elements=[el]))


def test_sensitive_element_is_marked_in_the_listing() -> None:
    el = element(1, tag="input", type="password", name="Password")
    assert "[sensitive field]" in format_snapshot(page(elements=[el]))


def test_offscreen_elements_are_marked_and_explained() -> None:
    out = format_snapshot(page(elements=[element(1), element(2, inView=False)]))
    assert "~[2]" in out
    assert "outside the viewport" in out


def test_no_tilde_legend_when_everything_is_in_view() -> None:
    out = format_snapshot(page(elements=[element(1), element(2)]))
    assert "outside the viewport" not in out
    assert "Interactive elements (2 of 2):" in out


def test_empty_page_says_so() -> None:
    out = format_snapshot(page(text="", textLength=0))
    assert "Interactive elements: none found." in out
    assert "Visible text: none." in out


def test_element_cap_prefers_the_viewport_and_counts_the_rest() -> None:
    offscreen = [element(i, inView=False) for i in range(1, 121)]
    onscreen = [element(i) for i in range(121, 131)]
    out = format_snapshot(page(elements=offscreen + onscreen))

    assert f"Interactive elements ({MAX_ELEMENTS} of 130" in out
    # All ten in-viewport elements survive the cap...
    for ref in range(121, 131):
        assert f"[{ref}] " in out
    # ...and the leftovers are counted, not silently dropped.
    assert "+10 more not listed" in out


def test_elements_stay_in_document_order_after_capping() -> None:
    elements = [element(i, inView=i > 100) for i in range(1, 131)]
    listed = [
        int(line.lstrip("~").split("]")[0][1:])
        for line in format_snapshot(page(elements=elements)).splitlines()
        if line.startswith(("[", "~["))
    ]
    assert listed == sorted(listed)


def test_long_text_is_truncated_with_a_pointer_to_read_text() -> None:
    body = "x" * 9000
    out = format_snapshot(page(text=body, textLength=len(body)))
    assert "read_text for more" in out
    assert "first 3500 of 9000 chars" in out
    assert out.rstrip().endswith("…")


def test_tabs_only_shown_when_there_is_more_than_one() -> None:
    assert "Tabs:" not in format_snapshot(page(), tabs=["Only tab"])
    assert "Tabs: One; Two" in format_snapshot(page(), tabs=["One", "Two"])


def test_notes_are_surfaced() -> None:
    out = format_snapshot(page(), notes=["A dialog was dismissed."])
    assert "Note: A dialog was dismissed." in out
