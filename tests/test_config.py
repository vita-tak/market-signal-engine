"""Tests for the only behaviour in config.py: environment flag parsing."""

import importlib

import pytest

import config


def test_env_flag_returns_default_when_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOME_FLAG", raising=False)

    assert config.env_flag("SOME_FLAG", default=True) is True
    assert config.env_flag("SOME_FLAG", default=False) is False


def test_env_flag_returns_default_when_variable_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOME_FLAG", "   ")

    assert config.env_flag("SOME_FLAG", default=True) is True


@pytest.mark.parametrize("raw", ["true", "TRUE", " True ", "1", "yes", "on"])
def test_env_flag_parses_true_values(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("SOME_FLAG", raw)

    assert config.env_flag("SOME_FLAG", default=False) is True


@pytest.mark.parametrize("raw", ["false", "FALSE", " False ", "0", "no", "off"])
def test_env_flag_parses_false_values(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("SOME_FLAG", raw)

    assert config.env_flag("SOME_FLAG", default=True) is False


def test_env_flag_rejects_unrecognised_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOME_FLAG", "fasle")

    with pytest.raises(ValueError) as excinfo:
        config.env_flag("SOME_FLAG", default=True)

    message = str(excinfo.value)
    assert "SOME_FLAG" in message
    assert "fasle" in message


def test_offline_flags_default_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with no flags set must never reach a real API."""
    monkeypatch.delenv("USE_FIXTURES", raising=False)
    monkeypatch.delenv("USE_LLM_STUB", raising=False)

    reloaded = importlib.reload(config)

    assert reloaded.USE_FIXTURES is True
    assert reloaded.USE_LLM_STUB is True
