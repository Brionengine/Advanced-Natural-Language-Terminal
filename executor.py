"""
Brion Quantum - Code Executor
==============================
Runs generated code in a child process with a timeout.

The original terminal ran generated code with `exec()` inside its own process,
so a stray `sys.exit()`, infinite loop, or crash took the terminal down with
it. A subprocess isolates the terminal from those failures.

It is NOT a security sandbox: the code runs as you, with your files and your
network. Read generated code before running it.

Developed by Brion Quantum AI Team
"""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_TIMEOUT = 60

INTERPRETERS = {
    "python": [sys.executable],
    "qiskit": [sys.executable],
    "javascript": ["node"],
    "bash": ["bash"],
}

EXTENSIONS = {"python": ".py", "qiskit": ".py", "javascript": ".js", "bash": ".sh"}


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: Optional[int]
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def error_text(self) -> str:
        """What to hand the correction loop when the run failed."""
        if self.timed_out:
            return "The program did not finish within the time limit."
        return self.stderr.strip() or f"Exited with status {self.returncode}"


def run_code(code: str, language: str = "python", timeout: int = DEFAULT_TIMEOUT,
             cwd: Optional[Path] = None, stdin: str = "") -> ExecutionResult:
    """Run `code` as a script in a child process and capture its output."""
    command: Optional[List[str]] = INTERPRETERS.get(language)
    if command is None:
        raise ValueError(f"No interpreter for {language!r}; runnable: {', '.join(INTERPRETERS)}")
    if shutil.which(command[0]) is None:
        raise FileNotFoundError(f"{command[0]} is not installed or not on PATH")

    with tempfile.TemporaryDirectory(prefix="brion-run-") as tmp:
        script = Path(tmp) / f"generated{EXTENSIONS[language]}"
        script.write_text(code)
        try:
            proc = subprocess.run(
                command + [str(script)], input=stdin, capture_output=True,
                text=True, timeout=timeout, cwd=cwd or tmp,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=_text(exc.stdout), stderr=_text(exc.stderr),
                returncode=None, timed_out=True,
            )
    return ExecutionResult(proc.stdout, proc.stderr, proc.returncode)


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value
