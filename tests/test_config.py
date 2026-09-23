"""Configuration: the .env reader and the env-var coercions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from browser_agent.config import (
    ConfigError,
    _env_bool,
    _env_int,
    _env_str,
    load_dotenv,
    require_api_key,
)


@pytest.fixture
def env(monkeypatch):
    """A clean environment that is restored afterwards."""
    for key in list(os.environ):
        if key.startswith("BROWSER_AGENT_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


# --- .env ---------------------------------------------------------------------


def test_values_are_read_into_the_environment(tmp_path: Path, env) -> None:
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-test\nBROWSER_AGENT_MODEL=some-model\n"
    )

    assert load_dotenv(tmp_path) == tmp_path / ".env"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert os.environ["BROWSER_AGENT_MODEL"] == "some-model"


def test_the_file_is_looked_for_up_the_tree(tmp_path: Path, env) -> None:
    (tmp_path / ".env").write_text("BROWSER_AGENT_MODEL=from-the-root\n")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)

    assert load_dotenv(deep) == tmp_path / ".env"
    assert os.environ["BROWSER_AGENT_MODEL"] == "from-the-root"


def test_an_existing_variable_is_not_overwritten(tmp_path: Path, env) -> None:
    env.setenv("BROWSER_AGENT_MODEL", "from-the-shell")
    (tmp_path / ".env").write_text("BROWSER_AGENT_MODEL=from-the-file\n")

    load_dotenv(tmp_path)
    # An export for one run has to beat the file, or overriding is impossible.
    assert os.environ["BROWSER_AGENT_MODEL"] == "from-the-shell"


def test_comments_blanks_quotes_and_export_are_handled(tmp_path: Path, env) -> None:
    (tmp_path / ".env").write_text(
        "\n"
        "# a comment\n"
        "  \n"
        'BROWSER_AGENT_MODEL="quoted-model"\n'
        "export BROWSER_AGENT_TIMEZONE='Asia/Kolkata'\n"
        "  BROWSER_AGENT_LOCALE = en-GB  \n"
        "NOT_A_PAIR\n"
    )

    load_dotenv(tmp_path)

    assert os.environ["BROWSER_AGENT_MODEL"] == "quoted-model"
    assert os.environ["BROWSER_AGENT_TIMEZONE"] == "Asia/Kolkata"
    assert os.environ["BROWSER_AGENT_LOCALE"] == "en-GB"
    assert "NOT_A_PAIR" not in os.environ


def test_a_value_containing_equals_survives(tmp_path: Path, env) -> None:
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-a=b=c\n")
    load_dotenv(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-a=b=c"


def test_no_file_anywhere_is_not_an_error(tmp_path: Path, env) -> None:
    assert load_dotenv(tmp_path / "nothing" / "here") is None


def test_an_unreadable_file_is_not_an_error(tmp_path: Path, env) -> None:
    path = tmp_path / ".env"
    path.write_text("BROWSER_AGENT_MODEL=x\n")
    path.chmod(0o000)
    try:
        assert load_dotenv(tmp_path) is None
    finally:
        path.chmod(0o600)


# --- coercions ----------------------------------------------------------------


def test_env_str_falls_back_on_blank(env) -> None:
    assert _env_str("BROWSER_AGENT_MODEL", "fallback") == "fallback"
    env.setenv("BROWSER_AGENT_MODEL", "   ")
    assert _env_str("BROWSER_AGENT_MODEL", "fallback") == "fallback"
    env.setenv("BROWSER_AGENT_MODEL", "  chosen  ")
    assert _env_str("BROWSER_AGENT_MODEL", "fallback") == "chosen"


def test_env_int_survives_a_typo(env) -> None:
    assert _env_int("BROWSER_AGENT_MAX_STEPS", 40) == 40
    env.setenv("BROWSER_AGENT_MAX_STEPS", "12")
    assert _env_int("BROWSER_AGENT_MAX_STEPS", 40) == 12
    env.setenv("BROWSER_AGENT_MAX_STEPS", "twelve")
    assert _env_int("BROWSER_AGENT_MAX_STEPS", 40) == 40


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_values(env, value: str) -> None:
    env.setenv("BROWSER_AGENT_HEADLESS", value)
    assert _env_bool("BROWSER_AGENT_HEADLESS", False) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_falsy_values(env, value: str) -> None:
    env.setenv("BROWSER_AGENT_HEADLESS", value)
    assert _env_bool("BROWSER_AGENT_HEADLESS", True) is False


def test_an_unrecognised_bool_keeps_the_default(env) -> None:
    env.setenv("BROWSER_AGENT_HEADLESS", "maybe")
    assert _env_bool("BROWSER_AGENT_HEADLESS", True) is True


# --- the API key --------------------------------------------------------------


def test_the_key_is_read_at_call_time(env) -> None:
    env.setenv("ANTHROPIC_API_KEY", "sk-ant-set-later")
    assert require_api_key() == "sk-ant-set-later"


def test_a_missing_key_explains_how_to_set_it(env, monkeypatch) -> None:
    monkeypatch.setattr("browser_agent.config.API_KEY", "")
    with pytest.raises(ConfigError) as exc:
        require_api_key()
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


# --- lists in environment variables -------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "wikipedia.org,news.ycombinator.com,bbc.com",
        "wikipedia.org news.ycombinator.com bbc.com",
        " wikipedia.org , news.ycombinator.com ,bbc.com ",
        "WIKIPEDIA.ORG,News.YCombinator.com,BBC.com",
        ".wikipedia.org,.news.ycombinator.com,.bbc.com",
    ],
)
def test_a_domain_list_is_read_however_it_was_written(env, raw: str) -> None:
    # gcloud reads a comma as the separator between environment variables, so a
    # comma-separated value has to be given with spaces instead. Both work.
    from browser_agent.config import _env_list

    env.setenv("BROWSER_AGENT_ALLOWED_DOMAINS", raw)
    assert _env_list("BROWSER_AGENT_ALLOWED_DOMAINS") == (
        "wikipedia.org",
        "news.ycombinator.com",
        "bbc.com",
    )


def test_an_unset_list_is_empty(env) -> None:
    from browser_agent.config import _env_list

    assert _env_list("BROWSER_AGENT_ALLOWED_DOMAINS") == ()


# --- however the keys were written --------------------------------------------


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        # One key, the ordinary case.
        ({"GEMINI_API_KEY": "AAA"}, ("AAA",)),
        # Two in the SINGULAR name. This is the one that mattered: it used to be
        # sent as a single 107-character key and came back as a 401 about OAuth
        # credentials, which looks like anything but a typo.
        ({"GEMINI_API_KEY": "AAA,BBB"}, ("AAA", "BBB")),
        ({"GEMINI_API_KEYS": "AAA,BBB"}, ("AAA", "BBB")),
        ({"GEMINI_API_KEYS": "AAA BBB CCC"}, ("AAA", "BBB", "CCC")),
        ({"GEMINI_API_KEYS": " AAA , BBB "}, ("AAA", "BBB")),
        # Google's own tooling sets this name.
        ({"GOOGLE_API_KEY": "AAA,BBB"}, ("AAA", "BBB")),
        # The plural wins when both are given, rather than being merged.
        ({"GEMINI_API_KEYS": "AAA,BBB", "GEMINI_API_KEY": "ZZZ"}, ("AAA", "BBB")),
        ({}, ()),
    ],
)
def test_keys_are_read_the_same_way_whichever_name_holds_them(
    env, monkeypatch, tmp_path, variables: dict, expected: tuple
) -> None:
    import importlib

    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    import browser_agent.config as config

    # Somewhere with no .env above it. Patching load_dotenv does not work here:
    # reload re-executes the module, which redefines the function and then
    # calls it, so the developer's own keys walk straight back in.
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    try:
        assert config.GEMINI_API_KEYS == expected
        # The singular is the first of the list, so anything that only wants to
        # know whether Gemini is usable still gets a straight answer.
        assert config.GEMINI_API_KEY == (expected[0] if expected else "")
    finally:
        importlib.reload(config)


def test_keys_keep_their_case(env, monkeypatch, tmp_path) -> None:
    import importlib

    monkeypatch.setenv("GEMINI_API_KEYS", "AQ.MixedCaseKey")
    import browser_agent.config as config

    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    try:
        # An API key that has been lowercased is not an API key.
        assert config.GEMINI_API_KEYS == ("AQ.MixedCaseKey",)
    finally:
        importlib.reload(config)
