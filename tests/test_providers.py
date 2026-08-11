"""
Tests for the LLM provider layer.

Availability and selection are the interesting logic here — the vendor SDKs are
not installed in CI, and these tests assert that this degrades cleanly rather
than raising at import.
"""

import pytest

import providers
from providers import (
    AnthropicProvider,
    CompletionResult,
    OfflineProvider,
    OpenAIProvider,
    available_providers,
    resolve_provider,
)


# -- Offline fallback -------------------------------------------------------


def test_offline_provider_is_always_available():
    assert OfflineProvider.is_available() is True


def test_offline_completion_explains_itself():
    result = OfflineProvider().complete("sys", [{"role": "user", "content": "hi"}])

    assert "No LLM backend is configured" in result.text
    assert result.ok is True


def test_offline_completion_echoes_the_request():
    result = OfflineProvider().complete(
        "sys", [{"role": "user", "content": "build a parser"}]
    )

    assert "build a parser" in result.text


def test_offline_completion_survives_an_empty_conversation():
    assert OfflineProvider().complete("sys", []).text


# -- Availability detection -------------------------------------------------


def test_anthropic_unavailable_without_sdk(monkeypatch):
    monkeypatch.setattr(providers, "_has_anthropic_profile", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert AnthropicProvider.is_available() is False


def test_openai_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert OpenAIProvider.is_available() is False


def test_available_providers_always_includes_offline():
    assert "offline" in available_providers()


# -- Missing SDKs degrade to an error result, never an exception -------------


def test_anthropic_without_sdk_returns_error_result():
    result = AnthropicProvider().complete("sys", [{"role": "user", "content": "hi"}])

    assert result.ok is False
    assert "anthropic" in result.error


def test_openai_without_sdk_returns_error_result():
    result = OpenAIProvider().complete("sys", [{"role": "user", "content": "hi"}])

    assert result.ok is False
    assert "openai" in result.error


# -- Provider resolution ----------------------------------------------------


def test_resolve_honours_an_explicit_name():
    assert resolve_provider("offline").name == "offline"


def test_resolve_is_case_insensitive():
    assert resolve_provider("OFFLINE").name == "offline"


def test_resolve_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_provider("gpt-9000")


def test_explicit_name_wins_over_availability():
    """A misconfigured backend must report its own error, not silently downgrade."""
    provider = resolve_provider("anthropic")

    assert provider.name == "anthropic"


def test_resolve_reads_the_environment(monkeypatch):
    monkeypatch.setenv("BRION_PROVIDER", "offline")

    assert resolve_provider().name == "offline"


def test_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("BRION_PROVIDER", "anthropic")

    assert resolve_provider("offline").name == "offline"


def test_resolve_falls_back_to_offline(monkeypatch):
    monkeypatch.delenv("BRION_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_has_anthropic_profile", lambda: False)

    assert resolve_provider().name == "offline"


def test_model_override_is_applied():
    assert resolve_provider("anthropic", model="claude-sonnet-5").model == "claude-sonnet-5"


def test_default_models_are_set():
    assert AnthropicProvider().model == "claude-opus-5"
    assert OpenAIProvider().model == "gpt-4o"


# -- Result shape -----------------------------------------------------------


def test_completion_result_defaults():
    result = CompletionResult(text="x", provider="p", model="m")

    assert result.tokens_used == 0
    assert result.refused is False
    assert result.ok is True
