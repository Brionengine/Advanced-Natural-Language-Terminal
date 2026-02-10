"""
Brion Quantum - Advanced NLP Terminal Core v2.0
Multi-provider LLM integration with conversation memory,
code generation, error correction, and command history.
"""

import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class Brion:
    """
    Brion NLP Terminal v2.0

    Advanced natural language code generation and correction engine.
    Features:
    - Multi-turn conversation with persistent memory
    - Code generation with syntax awareness
    - Automatic error correction with retry
    - Command history and session management
    - Token usage tracking
    """

    VERSION = "2.0.22"
    DEFAULT_MODEL = "gpt-4o"
    MAX_HISTORY = 50

    def __init__(self, model: str = None, system_prompt: str = None):
        self.model = model or self.DEFAULT_MODEL
        self.conversation: List[Dict[str, str]] = []
        self.command_history: List[Dict[str, Any]] = []
        self.total_tokens_used = 0
        self.generation_count = 0
        self.correction_count = 0
        self._start_time = time.time()

        # Set system prompt
        default_system = (
            "You are Brion, an advanced AI coding assistant created by Brion Quantum. "
            "You specialize in Python, quantum computing, AI systems, and cybersecurity. "
            "Always provide clean, well-structured, production-ready code."
        )
        self.system_prompt = system_prompt or default_system
        self.conversation.append({"role": "system", "content": self.system_prompt})

    def generate_code(self, user_input: str) -> str:
        """Generate code or response from natural language input."""
        if not OPENAI_AVAILABLE:
            return f"# OpenAI not available. Input: {user_input}\n# Install: pip install openai"

        self.conversation.append({"role": "user", "content": user_input})

        # Trim history if too long
        if len(self.conversation) > self.MAX_HISTORY:
            self.conversation = [self.conversation[0]] + self.conversation[-self.MAX_HISTORY:]

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.conversation,
                temperature=0.7,
            )

            assistant_message = response.choices[0].message["content"]
            self.conversation.append({"role": "assistant", "content": assistant_message})

            # Track usage
            usage = response.get('usage', {})
            self.total_tokens_used += usage.get('total_tokens', 0)
            self.generation_count += 1

            self.command_history.append({
                'type': 'generate',
                'input': user_input,
                'output_length': len(assistant_message),
                'tokens': usage.get('total_tokens', 0),
                'timestamp': time.time(),
            })

            return assistant_message

        except Exception as e:
            error_msg = f"# Generation error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def correct_code(self, code: str, error_message: str) -> str:
        """Attempt to correct code that produced an error."""
        if not OPENAI_AVAILABLE:
            return f"# Cannot correct - OpenAI not available\n# Error: {error_message}"

        prompt = (
            f"The following Python code contains an error:\n\n"
            f"```python\n{code}\n```\n\n"
            f"The error message is:\n{error_message}\n\n"
            f"Please provide ONLY the corrected Python code, no explanations."
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code correction specialist. Return only corrected code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )

            corrected_code = response.choices[0].message["content"]
            self.correction_count += 1

            # Strip markdown code blocks if present
            if corrected_code.startswith("```"):
                lines = corrected_code.split('\n')
                corrected_code = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

            self.command_history.append({
                'type': 'correction',
                'error': error_message[:200],
                'timestamp': time.time(),
            })

            return corrected_code.strip()

        except Exception as e:
            logger.error(f"Correction error: {e}")
            return code  # Return original if correction fails

    def clear_conversation(self):
        """Clear conversation history, keeping system prompt."""
        self.conversation = [{"role": "system", "content": self.system_prompt}]

    def get_session_stats(self) -> Dict[str, Any]:
        """Return session statistics."""
        uptime = time.time() - self._start_time
        return {
            'version': self.VERSION,
            'model': self.model,
            'uptime_seconds': round(uptime, 2),
            'generations': self.generation_count,
            'corrections': self.correction_count,
            'total_tokens': self.total_tokens_used,
            'conversation_length': len(self.conversation),
            'commands_in_history': len(self.command_history),
        }
