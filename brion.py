"""
Brion Quantum - Advanced NLP Terminal Core v2.1
Multi-provider LLM integration with conversation memory,
code generation, error correction, and command history.

Backends live in providers.py; this module owns conversation state, history
trimming, and the code-generation and correction workflows on top of them.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from providers import CompletionResult, LLMProvider, available_providers, resolve_provider

logger = logging.getLogger(__name__)


class Brion:
    """
    Brion NLP Terminal v2.1

    Advanced natural language code generation and correction engine.
    Features:
    - Multi-turn conversation with persistent memory
    - Pluggable LLM backends (Claude, OpenAI, offline)
    - Code generation with syntax awareness
    - Automatic error correction with retry
    - Command history and session management
    - Token usage tracking
    """

    VERSION = "2.1.0"
    MAX_HISTORY = 50
    DEFAULT_MAX_TOKENS = 16000

    DEFAULT_SYSTEM_PROMPT = (
        "You are Brion, an advanced AI coding assistant created by Brion Quantum. "
        "You specialize in Python, quantum computing, AI systems, and cybersecurity. "
        "Always provide clean, well-structured, production-ready code."
    )

    CORRECTION_SYSTEM_PROMPT = (
        "You are a code correction specialist. Return only corrected code."
    )

    def __init__(self, model: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 provider: Optional[str] = None):
        # The system prompt is held separately rather than as conversation[0]:
        # backends disagree on whether it is a message or a top-level field, and
        # keeping it out of the turn list means history trimming can never drop
        # or duplicate it.
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.conversation: List[Dict[str, str]] = []
        self.command_history: List[Dict[str, Any]] = []
        self.total_tokens_used = 0
        self.generation_count = 0
        self.correction_count = 0
        self.refusal_count = 0
        self._start_time = time.time()

        self.provider: LLMProvider = resolve_provider(provider, model)
        self.model = self.provider.model

    # -- Generation ---------------------------------------------------------

    def generate_code(self, user_input: str) -> str:
        """Generate code or a response from natural language input."""
        self.conversation.append({"role": "user", "content": user_input})
        self._trim_history()

        result = self.provider.complete(
            system=self.system_prompt,
            messages=self.conversation,
            max_tokens=self.DEFAULT_MAX_TOKENS,
        )

        if result.error:
            # Drop the user turn: leaving it in place would resend a turn the
            # model never answered, so the next call would look like two
            # consecutive user messages.
            self.conversation.pop()
            self._record("generate", user_input, result, failed=True)
            return f"# Generation error: {result.error}"

        if result.refused:
            self.conversation.pop()
            self.refusal_count += 1
            self._record("generate", user_input, result, failed=True)
            return "# The request was declined by the provider's safety policy."

        self.conversation.append({"role": "assistant", "content": result.text})
        self.total_tokens_used += result.tokens_used
        self.generation_count += 1
        self._record("generate", user_input, result)
        return result.text

    def correct_code(self, code: str, error_message: str) -> str:
        """
        Attempt to correct code that produced an error.

        Runs outside the conversation so a failed snippet does not pollute the
        session's memory, and returns the original code unchanged when the
        backend cannot help — callers write this straight back to a file.
        """
        prompt = (
            f"The following Python code contains an error:\n\n"
            f"```python\n{code}\n```\n\n"
            f"The error message is:\n{error_message}\n\n"
            f"Please provide ONLY the corrected Python code, no explanations."
        )

        result = self.provider.complete(
            system=self.CORRECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.DEFAULT_MAX_TOKENS,
        )

        if result.error or result.refused:
            if result.refused:
                self.refusal_count += 1
            logger.error("Correction unavailable: %s", result.error or "refused")
            self._record("correction", error_message[:200], result, failed=True)
            return code

        corrected = strip_code_fence(result.text)
        self.total_tokens_used += result.tokens_used
        self.correction_count += 1
        self._record("correction", error_message[:200], result)
        return corrected

    # -- Session management -------------------------------------------------

    def clear_conversation(self):
        """Clear conversation history. The system prompt is unaffected."""
        self.conversation = []

    def get_session_stats(self) -> Dict[str, Any]:
        """Return session statistics."""
        return {
            "version": self.VERSION,
            "provider": self.provider.name,
            "model": self.model,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "generations": self.generation_count,
            "corrections": self.correction_count,
            "refusals": self.refusal_count,
            "total_tokens": self.total_tokens_used,
            "conversation_length": len(self.conversation),
            "commands_in_history": len(self.command_history),
            "available_providers": available_providers(),
        }

    # -- Helpers ------------------------------------------------------------

    def _trim_history(self):
        """
        Bound the conversation to MAX_HISTORY turns, oldest first.

        Trimming must not leave an assistant message at the front: a leading
        assistant turn is rejected by every backend here, so the window is
        advanced to the next user turn when it lands mid-exchange.
        """
        if len(self.conversation) <= self.MAX_HISTORY:
            return

        window = self.conversation[-self.MAX_HISTORY:]
        first_user = next(
            (i for i, m in enumerate(window) if m.get("role") == "user"), None
        )
        self.conversation = window[first_user:] if first_user is not None else []

    def _record(self, kind: str, subject: str, result: CompletionResult,
                failed: bool = False):
        """Append one entry to the command history."""
        self.command_history.append({
            "type": kind,
            "input": subject,
            "output_length": len(result.text),
            "tokens": result.tokens_used,
            "provider": result.provider,
            "failed": failed,
            "timestamp": time.time(),
        })


def strip_code_fence(text: str) -> str:
    """
    Remove a surrounding markdown code fence, if present.

    Only strips when the text actually opens with a fence, and only removes the
    closing fence when one is there — a truncated response that opens a fence
    and never closes it would otherwise lose its final line of code.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.split("\n")
    lines = lines[1:]  # drop the opening fence (and any language tag)
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
