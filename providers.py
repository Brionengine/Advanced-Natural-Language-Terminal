"""
Brion Quantum - LLM Provider Layer
===================================
Backend-agnostic completion interface for the Brion NLP Terminal.

Each provider wraps one vendor SDK behind a single `complete()` call so the
terminal can switch backends without touching conversation handling. Providers
whose SDK is not installed report themselves unavailable rather than raising at
import time, which is what lets the terminal run — and be tested — on a machine
with no vendor SDK and no API key.

Developed by Brion Quantum AI Team
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """One completion, normalized across backends."""

    text: str
    provider: str
    model: str
    tokens_used: int = 0
    # True when the backend declined the request on policy grounds. This is a
    # successful call with no usable text, not an error — callers that treat any
    # non-exception result as content will surface an empty string otherwise.
    refused: bool = False
    error: Optional[str] = None
    raw_usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.refused


class LLMProvider(ABC):
    """A single chat-completion backend."""

    name: str = "base"
    default_model: str = ""

    def __init__(self, model: Optional[str] = None):
        self.model = model or self.default_model

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this backend can actually be called right now."""

    @abstractmethod
    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 16000) -> CompletionResult:
        """
        Run one completion.

        `messages` carries only user/assistant turns; the system prompt is
        passed separately because backends disagree on where it belongs.
        """


class AnthropicProvider(LLMProvider):
    """
    Claude backend via the official `anthropic` SDK.

    The system prompt is a top-level parameter here rather than a message, and
    `max_tokens` bounds thinking plus visible text together — Claude Opus 5
    thinks by default, so a limit sized only for the answer can truncate it.
    Sampling parameters are deliberately not sent: they are rejected outright
    on this model generation.
    """

    name = "anthropic"
    default_model = "claude-opus-5"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY")) or _has_anthropic_profile()

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 16000) -> CompletionResult:
        try:
            import anthropic
        except ImportError:
            return CompletionResult(
                text="", provider=self.name, model=self.model,
                error="anthropic SDK not installed (pip install anthropic)",
            )

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as text
            logger.error("Anthropic completion failed: %s", exc)
            return CompletionResult(
                text="", provider=self.name, model=self.model, error=str(exc)
            )

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        total = usage["input_tokens"] + usage["output_tokens"]

        # Check the stop reason before reading content: a declined request comes
        # back as a normal response whose content list is empty or partial.
        if getattr(response, "stop_reason", None) == "refusal":
            return CompletionResult(
                text="", provider=self.name, model=self.model,
                tokens_used=total, refused=True, raw_usage=usage,
            )

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return CompletionResult(
            text=text, provider=self.name, model=self.model,
            tokens_used=total, raw_usage=usage,
        )


class OpenAIProvider(LLMProvider):
    """
    OpenAI backend via the `openai` v1+ client.

    The module-level `openai.ChatCompletion.create()` call this replaces was
    removed in openai 1.0 and raises on every modern install.
    """

    name = "openai"
    default_model = "gpt-4o"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 16000) -> CompletionResult:
        try:
            from openai import OpenAI
        except ImportError:
            return CompletionResult(
                text="", provider=self.name, model=self.model,
                error="openai SDK not installed (pip install 'openai>=1.0')",
            )

        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}] + list(messages),
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as text
            logger.error("OpenAI completion failed: %s", exc)
            return CompletionResult(
                text="", provider=self.name, model=self.model, error=str(exc)
            )

        text = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
        }
        return CompletionResult(
            text=text, provider=self.name, model=self.model,
            tokens_used=getattr(usage_obj, "total_tokens", 0), raw_usage=usage,
        )


class OfflineProvider(LLMProvider):
    """
    Always-available fallback that generates nothing.

    Its job is to keep the terminal importable and runnable with no SDK and no
    API key: it echoes the request back with an explanation instead of raising,
    so the surrounding application can be exercised end to end offline.
    """

    name = "offline"
    default_model = "none"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def complete(self, system: str, messages: List[Dict[str, str]],
                 max_tokens: int = 16000) -> CompletionResult:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        text = (
            "# No LLM backend is configured, so nothing was generated.\n"
            "# Install a provider SDK and set its API key:\n"
            "#   pip install anthropic   -> export ANTHROPIC_API_KEY=...\n"
            "#   pip install openai      -> export OPENAI_API_KEY=...\n"
            f"# Received request: {last_user}"
        )
        return CompletionResult(text=text, provider=self.name, model=self.model)


# Preference order when no provider is named: first available wins.
PROVIDERS: List[type] = [AnthropicProvider, OpenAIProvider, OfflineProvider]

PROVIDERS_BY_NAME: Dict[str, type] = {p.name: p for p in PROVIDERS}


def _has_anthropic_profile() -> bool:
    """
    Whether an `ant auth login` profile exists on disk.

    An unset ANTHROPIC_API_KEY does not mean there are no credentials — the SDK
    also reads a stored OAuth profile, so checking only the env var would report
    a working install as unavailable.
    """
    config_dir = os.environ.get("ANTHROPIC_CONFIG_DIR") or os.path.expanduser(
        "~/.config/anthropic"
    )
    return os.path.isdir(os.path.join(config_dir, "credentials"))


def available_providers() -> List[str]:
    """Names of every backend that could serve a request right now."""
    return [p.name for p in PROVIDERS if p.is_available()]


def resolve_provider(name: Optional[str] = None,
                     model: Optional[str] = None) -> LLMProvider:
    """
    Pick a backend.

    An explicit `name` is honoured even when unavailable, so a misconfigured
    backend reports its own error rather than silently degrading to offline —
    a silent downgrade looks like the model answering badly, not like a missing
    API key. Otherwise the BRION_PROVIDER environment variable wins, then the
    first available backend in preference order.
    """
    requested = name or os.environ.get("BRION_PROVIDER")

    if requested:
        key = requested.strip().lower()
        if key not in PROVIDERS_BY_NAME:
            raise ValueError(
                f"Unknown provider {requested!r}. "
                f"Available: {', '.join(PROVIDERS_BY_NAME)}"
            )
        return PROVIDERS_BY_NAME[key](model=model)

    for provider_cls in PROVIDERS:
        if provider_cls.is_available():
            return provider_cls(model=model)

    return OfflineProvider(model=model)
