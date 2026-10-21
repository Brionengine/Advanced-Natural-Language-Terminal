"""
The original NLP engine entry point, now backed by nlp_engine.py.

`parse_natural_language()` keeps its signature but no longer calls
`openai.Completion.create()`, which was removed in openai 1.0. It goes through
the multi-provider engine instead (Claude, OpenAI, or the offline fallback).

Run this file directly to start the terminal.
"""

from nlp_engine import parse_natural_language  # noqa: F401 - re-exported

if __name__ == "__main__":
    import sys

    from start_terminal import main

    sys.exit(main())
