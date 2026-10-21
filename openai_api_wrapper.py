"""
Code correction through OpenAI, kept for callers of `correct_code_with_llm()`.

The original called `openai.Completion.create()`, removed in openai 1.0. It now
routes through Brion's OpenAI provider (openai>=1.0 client), and returns the
original code unchanged when the backend cannot help.
"""

from brion import Brion


def correct_code_with_llm(code, error_message):
    return Brion(provider="openai").correct_code(code, error_message)
