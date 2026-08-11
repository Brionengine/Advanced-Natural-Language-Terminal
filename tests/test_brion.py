"""
Tests for the Brion NLP terminal core.

A stub provider stands in for every backend, so the suite exercises the real
conversation, trimming, and error-handling logic without an SDK, an API key, or
a network call.
"""

import pytest

from brion import Brion, strip_code_fence
from providers import CompletionResult, LLMProvider


class StubProvider(LLMProvider):
    """Records what it was asked and returns a scripted result."""

    name = "stub"
    default_model = "stub-model"

    def __init__(self, result=None, model=None):
        super().__init__(model=model)
        self.result = result or CompletionResult(
            text="print('hello')", provider="stub", model="stub-model", tokens_used=10
        )
        self.calls = []

    @classmethod
    def is_available(cls):
        return True

    def complete(self, system, messages, max_tokens=16000):
        self.calls.append(
            {"system": system, "messages": list(messages), "max_tokens": max_tokens}
        )
        return self.result


@pytest.fixture
def brion():
    agent = Brion(provider="offline")
    agent.provider = StubProvider()
    return agent


def make(**kwargs):
    base = {"text": "", "provider": "stub", "model": "stub-model"}
    base.update(kwargs)
    return CompletionResult(**base)


# -- Generation -------------------------------------------------------------


def test_generate_returns_provider_text(brion):
    assert brion.generate_code("write hello world") == "print('hello')"


def test_generate_appends_both_turns(brion):
    brion.generate_code("write hello world")

    assert [m["role"] for m in brion.conversation] == ["user", "assistant"]


def test_system_prompt_is_not_a_conversation_turn(brion):
    """It travels as a separate field, so trimming can never drop it."""
    brion.generate_code("hi")

    assert all(m["role"] != "system" for m in brion.conversation)
    assert brion.provider.calls[0]["system"] == Brion.DEFAULT_SYSTEM_PROMPT


def test_generate_tracks_tokens_and_counts(brion):
    brion.generate_code("hi")

    assert brion.total_tokens_used == 10
    assert brion.generation_count == 1


def test_generate_records_history_entry(brion):
    brion.generate_code("hi")

    entry = brion.command_history[0]
    assert entry["type"] == "generate"
    assert entry["failed"] is False


def test_conversation_accumulates_across_turns(brion):
    brion.generate_code("first")
    brion.generate_code("second")

    assert len(brion.conversation) == 4
    # The second call must carry the first exchange as context.
    assert len(brion.provider.calls[1]["messages"]) == 3


# -- Failure handling -------------------------------------------------------


def test_generation_error_returns_comment(brion):
    brion.provider = StubProvider(make(error="connection refused"))

    assert brion.generate_code("hi").startswith("# Generation error")


def test_failed_turn_is_removed_from_conversation(brion):
    """An unanswered user turn must not linger, or the next call sends two in a row."""
    brion.provider = StubProvider(make(error="boom"))

    brion.generate_code("hi")

    assert brion.conversation == []


def test_conversation_stays_valid_after_a_failure(brion):
    failing = StubProvider(make(error="boom"))
    brion.provider = failing
    brion.generate_code("first")

    brion.provider = StubProvider()
    brion.generate_code("second")

    roles = [m["role"] for m in brion.provider.calls[0]["messages"]]
    assert roles == ["user"]


def test_refusal_is_reported_and_counted(brion):
    brion.provider = StubProvider(make(refused=True))

    result = brion.generate_code("something disallowed")

    assert "declined" in result
    assert brion.refusal_count == 1


def test_refusal_does_not_count_as_a_generation(brion):
    brion.provider = StubProvider(make(refused=True))
    brion.generate_code("x")

    assert brion.generation_count == 0


def test_error_result_is_not_ok():
    assert make(error="x").ok is False


def test_refused_result_is_not_ok():
    assert make(refused=True).ok is False


def test_plain_result_is_ok():
    assert make(text="hi").ok is True


# -- Correction -------------------------------------------------------------


def test_correct_code_returns_corrected_source(brion):
    brion.provider = StubProvider(make(text="print('fixed')", tokens_used=5))

    assert brion.correct_code("print(", "SyntaxError") == "print('fixed')"


def test_correct_code_strips_markdown_fence(brion):
    brion.provider = StubProvider(make(text="```python\nprint('fixed')\n```"))

    assert brion.correct_code("print(", "SyntaxError") == "print('fixed')"


def test_correct_code_returns_original_on_failure(brion):
    """Callers write this straight back to a file — it must never be empty."""
    brion.provider = StubProvider(make(error="timeout"))

    assert brion.correct_code("original code", "SyntaxError") == "original code"


def test_correct_code_returns_original_on_refusal(brion):
    brion.provider = StubProvider(make(refused=True))

    assert brion.correct_code("original code", "err") == "original code"


def test_correction_does_not_pollute_the_conversation(brion):
    brion.generate_code("write something")
    before = len(brion.conversation)

    brion.correct_code("broken", "SyntaxError")

    assert len(brion.conversation) == before


def test_correction_uses_the_correction_system_prompt(brion):
    brion.correct_code("broken", "SyntaxError")

    assert brion.provider.calls[0]["system"] == Brion.CORRECTION_SYSTEM_PROMPT


# -- Code fence stripping ---------------------------------------------------


def test_strip_fence_leaves_unfenced_text_alone():
    assert strip_code_fence("print('x')") == "print('x')"


def test_strip_fence_removes_language_tag():
    assert strip_code_fence("```python\nprint('x')\n```") == "print('x')"


def test_strip_fence_handles_bare_fence():
    assert strip_code_fence("```\nprint('x')\n```") == "print('x')"


def test_strip_fence_keeps_last_line_when_fence_unclosed():
    """A truncated response must not lose its final line of code."""
    assert strip_code_fence("```python\nprint('x')") == "print('x')"


def test_strip_fence_preserves_interior_blank_lines():
    assert strip_code_fence("```\na\n\nb\n```") == "a\n\nb"


# -- History trimming -------------------------------------------------------


def test_history_is_bounded(brion):
    for i in range(60):
        brion.generate_code(f"message {i}")

    assert len(brion.conversation) <= Brion.MAX_HISTORY


def test_trimmed_history_never_starts_with_an_assistant_turn(brion):
    """A leading assistant turn is rejected by every backend."""
    for i in range(60):
        brion.generate_code(f"message {i}")

    assert brion.conversation[0]["role"] == "user"


def test_short_conversation_is_untouched(brion):
    brion.generate_code("only one")

    assert len(brion.conversation) == 2


def test_clear_conversation_empties_turns(brion):
    brion.generate_code("hi")
    brion.clear_conversation()

    assert brion.conversation == []


def test_clear_conversation_keeps_system_prompt(brion):
    brion.clear_conversation()

    assert brion.system_prompt == Brion.DEFAULT_SYSTEM_PROMPT


# -- Session stats ----------------------------------------------------------


def test_session_stats_report_provider_and_counts(brion):
    brion.generate_code("hi")
    stats = brion.get_session_stats()

    assert stats["provider"] == "stub"
    assert stats["generations"] == 1
    assert stats["total_tokens"] == 10


def test_session_stats_list_available_providers(brion):
    # The offline backend is always available, so this is never empty.
    assert "offline" in brion.get_session_stats()["available_providers"]


def test_custom_system_prompt_is_used():
    agent = Brion(provider="offline", system_prompt="Be terse.")
    agent.provider = StubProvider()
    agent.generate_code("hi")

    assert agent.provider.calls[0]["system"] == "Be terse."
