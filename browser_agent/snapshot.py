"""What the model sees when it looks at a page.

A page is handed over as TEXT, not as a DOM: the interactive elements it can act
on, each with a number, plus the visible text. ``SNAPSHOT_JS`` writes those
numbers onto the elements as a ``data-agent-ref`` attribute and the click/type
tools read them back, so "[12]" in the snapshot and ``click(12)`` name the same
node without a CSS selector ever crossing the model boundary.

Two rules keep this workable on a real site:

* Bounded. A snapshot is re-sent on every step of the agent loop, so it is
  capped in both elements and text. When a page has more, elements inside the
  viewport are kept first and the rest are counted; the text is truncated with a
  hint to use ``read_text`` for the remainder.
* Sensitive fields are flagged here, in Python, from the raw attributes the JS
  returns — one regex, one definition — and ``type_text`` uses the same
  predicate. A password, card, OTP or ID field shows up as ``[sensitive field]``
  and can be filled like any other; what the flag changes is that its VALUE is
  never echoed back, since tool results are kept in the conversation history.
"""

from __future__ import annotations

import re

MAX_ELEMENTS = 120      # numbered elements per snapshot
MAX_TEXT_CHARS = 3500   # visible-text digest per snapshot
READ_TEXT_CHARS = 6000  # per read_text call
_JS_TEXT_CAP = 20000    # what the JS ships back at most; Python trims further

# Returns the page as JSON-serialisable data. A plain arrow function, so
# page.evaluate() runs it as a function expression taking no arguments.
SNAPSHOT_JS = r"""
() => {
  const INTERACTIVE = 'a[href], button, input, select, textarea, summary, ' +
    '[role="button"], [role="link"], [role="tab"], [role="menuitem"], [role="menuitemcheckbox"], ' +
    '[role="checkbox"], [role="radio"], [role="combobox"], [role="textbox"], [role="searchbox"], ' +
    '[role="option"], [role="switch"], [role="slider"], [contenteditable="true"], [onclick], ' +
    '[tabindex]:not([tabindex="-1"])';
  document.querySelectorAll('[data-agent-ref]').forEach(e => e.removeAttribute('data-agent-ref'));
  const vw = window.innerWidth, vh = window.innerHeight;
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const visible = el => {
    if (el.closest('[aria-hidden="true"]')) return false;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || parseFloat(st.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width >= 2 && r.height >= 2;
  };
  const labelOf = el => {
    if (el.labels && el.labels.length) return clean(Array.from(el.labels).map(l => l.innerText).join(' '));
    const by = el.getAttribute('aria-labelledby');
    if (by) return clean(by.split(/\s+/).map(id => document.getElementById(id)).filter(Boolean).map(e => e.innerText).join(' '));
    return '';
  };
  const nameOf = el => {
    const aria = clean(el.getAttribute('aria-label')); if (aria) return aria;
    const lab = labelOf(el); if (lab) return lab;
    const ph = clean(el.getAttribute('placeholder')); if (ph) return ph;
    // A <select>'s innerText is every option run together, which says nothing
    // the options list below it doesn't already say. Fall through to its name.
    if (el.tagName.toLowerCase() !== 'select') {
      const tx = clean(el.innerText || el.textContent); if (tx) return tx;
    }
    const own = clean(el.getAttribute('title') || el.getAttribute('alt') || el.getAttribute('value') || el.getAttribute('name') || '');
    if (own) return own;
    // Icon-only controls: an <img alt>, an <svg><title>, or a labelled child.
    const child = el.querySelector('img[alt], svg title, [aria-label]');
    if (!child) return '';
    return clean(child.getAttribute('alt') || child.getAttribute('aria-label') || child.textContent || '');
  };
  const FORM_TAGS = ['input', 'select', 'textarea', 'button', 'a', 'summary'];
  const all = Array.from(document.querySelectorAll(INTERACTIVE)).filter(visible);
  const inList = new Set(all);
  const items = [];
  let ref = 0;
  for (const el of all) {
    // A link wrapping a button (or a button wrapping a span[onclick]) is one
    // control, not two: skip the inner one when its name matches an ancestor's.
    let p = el.parentElement, dup = false;
    while (p) { if (inList.has(p) && nameOf(p) === nameOf(el)) { dup = true; break; } p = p.parentElement; }
    if (dup) continue;
    // A nameless div[onclick] or [tabindex] is noise the model cannot act on by
    // description; a nameless real control (icon button, bare input) stays.
    const nm = nameOf(el);
    if (!nm && !FORM_TAGS.includes(el.tagName.toLowerCase()) && !el.getAttribute('role') && !el.isContentEditable) continue;
    ref += 1;
    el.setAttribute('data-agent-ref', String(ref));
    const r = el.getBoundingClientRect();
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') || '';
    const type = tag === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : '';
    let value = '';
    if (tag === 'select') value = el.selectedOptions && el.selectedOptions[0] ? clean(el.selectedOptions[0].text) : '';
    else if (tag === 'input' || tag === 'textarea') value = type === 'password' ? (el.value ? '••••' : '') : clean(el.value).slice(0, 60);
    else if (el.isContentEditable) value = clean(el.innerText).slice(0, 60);
    const state = [];
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') state.push('disabled');
    if (type === 'checkbox' || type === 'radio' || role === 'checkbox' || role === 'radio' || role === 'switch' || role === 'menuitemcheckbox')
      state.push((el.checked || el.getAttribute('aria-checked') === 'true') ? 'checked' : 'unchecked');
    const exp = el.getAttribute('aria-expanded'); if (exp) state.push(exp === 'true' ? 'expanded' : 'collapsed');
    if (el.getAttribute('aria-selected') === 'true') state.push('selected');
    if (el.required) state.push('required');
    let href = '';
    if (tag === 'a') {
      try { const u = new URL(el.href, location.href); href = (u.host === location.host ? u.pathname : u.host + u.pathname).slice(0, 60); } catch (e) {}
    }
    items.push({
      ref, tag, role, type,
      name: nm.slice(0, 80),
      value, state, href,
      options: tag === 'select' ? Array.from(el.options).slice(0, 12).map(o => clean(o.text)).filter(Boolean) : [],
      inView: r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw,
      // Raw attributes for the sensitivity check, which happens in Python.
      id: el.id || '', fieldName: el.getAttribute('name') || '', placeholder: el.getAttribute('placeholder') || '',
      ariaLabel: el.getAttribute('aria-label') || '', autocomplete: el.getAttribute('autocomplete') || '', label: labelOf(el),
    });
  }
  const text = clean(document.body ? document.body.innerText : '');
  return {
    url: location.href, title: clean(document.title),
    scrollY: Math.round(window.scrollY), scrollHeight: Math.round(document.documentElement.scrollHeight), viewportHeight: vh,
    elements: items,
    text: text.slice(0, __TEXT_CAP__), textLength: text.length,
  };
}
""".replace("__TEXT_CAP__", str(_JS_TEXT_CAP))

# Runs on ONE element (locator.evaluate). The same raw attributes as above, plus
# whether Playwright's fill() will work on it.
FIELD_INFO_JS = r"""
el => {
  const tag = el.tagName.toLowerCase();
  const type = tag === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : '';
  const labels = el.labels ? Array.from(el.labels).map(l => l.innerText).join(' ') : '';
  const fillable = (tag === 'input' && !['checkbox','radio','button','submit','reset','file','image','range','color'].includes(type))
    || tag === 'textarea' || el.isContentEditable;
  return {
    tag, type, fillable,
    id: el.id || '', fieldName: el.getAttribute('name') || '', placeholder: el.getAttribute('placeholder') || '',
    ariaLabel: el.getAttribute('aria-label') || '', autocomplete: el.getAttribute('autocomplete') || '',
    label: (labels || '').replace(/\s+/g, ' ').trim(),
  };
}
"""

# innerText of one element (by ref) or of the whole body, sliced.
# Args: [ref, start, limit].
READ_TEXT_JS = r"""
([ref, start, limit]) => {
  const el = ref ? document.querySelector('[data-agent-ref="' + ref + '"]') : document.body;
  if (!el) return null;
  const text = (el.innerText || el.textContent || '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  return { total: text.length, chunk: text.slice(start, start + limit) };
}
"""

# Fields whose value must never be echoed back. Matched against
# name/id/placeholder/label/aria-label, with space, underscore or dash as the
# separator, so "card_number", "card-number" and "cardNumber" all match.
# Word-bounded so "pin" does not catch "pincode" — a postal field on every
# Indian address form — while "ATM pin" and "UPI PIN" still do. PAN (the tax id)
# is caught next to card/number, or as the bare upper-case word, so that
# "Pan-fried" does not lock a restaurant menu field.
_SEP = r"[\s_-]*"
_SENSITIVE_WORDS = re.compile(
    rf"(card{_SEP}(number|no\b|num)|cvv|cvc|\botp\b|one{_SEP}time|passcode|password|passwd|"
    rf"\bpin\b(?!{_SEP}code)|expir|exp\.?{_SEP}date|aadhaar|aadhar|\bpan{_SEP}(card|number|no\b|num)|"
    rf"\bssn\b|security{_SEP}code|account{_SEP}(number|no\b|num)|\bifsc\b|routing{_SEP}(number|no\b)|"
    rf"passport{_SEP}(number|no\b|num))",
    re.IGNORECASE,
)
_SENSITIVE_UPPER = re.compile(r"\bPAN\b")  # case-sensitive: the tax id, not the cookware
_SENSITIVE_AUTOCOMPLETE = ("cc-", "one-time-code", "current-password", "new-password")


def is_sensitive_field(info: dict) -> bool:
    """Whether a form field holds a credential, a payment detail or an ID.

    Works off raw attributes so that the snapshot (many elements) and type_text
    (one element) share a single definition. This is a flag, not a gate: such
    fields are filled when the task calls for it, but their value is masked
    everywhere it would otherwise be echoed — tool results, logs, and the
    snapshot's value column.
    """
    if (info.get("type") or "").lower() == "password":
        return True
    autocomplete = (info.get("autocomplete") or "").lower()
    if any(marker in autocomplete for marker in _SENSITIVE_AUTOCOMPLETE):
        return True
    haystack = " ".join(
        str(info.get(k) or "")
        for k in ("fieldName", "id", "placeholder", "ariaLabel", "label", "name")
    )
    return bool(_SENSITIVE_WORDS.search(haystack) or _SENSITIVE_UPPER.search(haystack))


def _kind(el: dict) -> str:
    role, tag, typ = el.get("role") or "", el.get("tag") or "", el.get("type") or ""
    if role:
        return role
    if tag == "a":
        return "link"
    if tag == "input":
        return f"input({typ})"
    return tag


def _element_line(el: dict) -> str:
    prefix = "" if el.get("inView", True) else "~"
    line = f'{prefix}[{el["ref"]}] {_kind(el)} "{el.get("name") or ""}"'
    if el.get("value"):
        line += f' = "{el["value"]}"'
    if el.get("href"):
        line += f" -> {el['href']}"
    if el.get("options"):
        line += "  options: " + " | ".join(el["options"])
    if el.get("state"):
        line += "  [" + ", ".join(el["state"]) + "]"
    if is_sensitive_field(el):
        line += "  [sensitive field]"
    return line


def _pick_elements(elements: list[dict], cap: int) -> list[dict]:
    """All of them when they fit; otherwise everything in the viewport first,
    topped up in document order, and presented in document order."""
    if len(elements) <= cap:
        return elements
    chosen = [el for el in elements if el.get("inView")][:cap]
    seen = {el["ref"] for el in chosen}
    for el in elements:
        if len(chosen) >= cap:
            break
        if el["ref"] not in seen:
            chosen.append(el)
            seen.add(el["ref"])
    return sorted(chosen, key=lambda el: el["ref"])


def _scroll_line(data: dict) -> str:
    y = int(data.get("scrollY") or 0)
    total = int(data.get("scrollHeight") or 0)
    vh = int(data.get("viewportHeight") or 0)
    if total <= vh + 2:
        return "Scroll: whole page fits on screen"
    if y <= 2:
        return f"Scroll: at top (page is {total}px tall, {vh}px visible)"
    if y + vh >= total - 2:
        return "Scroll: at bottom"
    return f"Scroll: {round(100 * y / max(total - vh, 1))}% down"


def format_snapshot(
    data: dict | None,
    *,
    tabs: list[str] | None = None,
    notes: list[str] | None = None,
    max_elements: int = MAX_ELEMENTS,
    max_text: int = MAX_TEXT_CHARS,
) -> str:
    """Render SNAPSHOT_JS output as the block of text the model reads."""
    if not data:
        return (
            "The page returned no content (blank, still loading, or not an HTML page). "
            "Try wait then get_page."
        )
    lines = [f"Page: {data.get('title') or '(untitled)'}", f"URL: {data.get('url') or ''}"]
    if tabs and len(tabs) > 1:
        lines.append("Tabs: " + "; ".join(tabs))
    for note in notes or []:
        lines.append(f"Note: {note}")
    lines.append(_scroll_line(data))

    elements = data.get("elements") or []
    shown = _pick_elements(elements, max_elements)
    lines.append("")
    if not elements:
        lines.append("Interactive elements: none found.")
    else:
        head = f"Interactive elements ({len(shown)} of {len(elements)}"
        head += (
            '; "~" = outside the viewport, scroll to reach it):'
            if any(not e.get("inView", True) for e in shown)
            else "):"
        )
        lines.append(head)
        lines.extend(_element_line(el) for el in shown)
        if len(shown) < len(elements):
            lines.append(
                f"+{len(elements) - len(shown)} more not listed — scroll to bring them into view."
            )

    text = (data.get("text") or "").strip()
    total_len = int(data.get("textLength") or len(text))
    lines.append("")
    if not text:
        lines.append("Visible text: none.")
    elif total_len > max_text:
        lines.append(f"Visible text (first {max_text} of {total_len} chars; read_text for more):")
        lines.append(text[:max_text] + "…")
    else:
        lines.append("Visible text:")
        lines.append(text)
    return "\n".join(lines)
