"""
Brion Quantum - Natural Language Programming Engine v3.0
=========================================================
The translation layer: English in, runnable code out.

`NLPEngine.translate()` turns a request into a `Translation` for a target
language, checks that the result at least parses, asks the model to repair it
when it does not, and works out which third-party packages the code needs —
the package-manager half of the job, so a translation arrives with the install
command that makes it runnable.

Quantum targets ("qiskit", "openqasm") produce circuits that quantum_bridge.py
can sample on a simulator or submit to Origin Quantum / IBM hardware.

Developed by Brion Quantum AI Team
"""

import ast
import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from brion import Brion, strip_code_fence

logger = logging.getLogger(__name__)


# -- Languages ----------------------------------------------------------------


@dataclass(frozen=True)
class LanguageSpec:
    """How to ask for, recognise, check and run one target language."""

    name: str
    display: str
    extension: str
    fence_tags: Tuple[str, ...]
    instructions: str
    package_manager: Optional[str] = None  # "pip" | "npm" | None
    quantum: bool = False


LANGUAGES: Dict[str, LanguageSpec] = {
    spec.name: spec for spec in [
        LanguageSpec(
            name="python", display="Python", extension=".py",
            fence_tags=("python", "py", "python3"), package_manager="pip",
            instructions=(
                "Write a complete, runnable Python 3 script. Put executable "
                "statements under `if __name__ == \"__main__\":`. Print results "
                "so running the script shows them."
            ),
        ),
        LanguageSpec(
            name="javascript", display="JavaScript", extension=".js",
            fence_tags=("javascript", "js", "node"), package_manager="npm",
            instructions=(
                "Write a complete, runnable Node.js script (CommonJS or ES "
                "modules). Log results with console.log."
            ),
        ),
        LanguageSpec(
            name="bash", display="Bash", extension=".sh",
            fence_tags=("bash", "sh", "shell", "zsh"),
            instructions=(
                "Write a complete Bash script starting with `#!/usr/bin/env bash` "
                "and `set -euo pipefail`. Never run destructive commands "
                "(rm -rf, mkfs, dd) unless the request explicitly asks for them."
            ),
        ),
        LanguageSpec(
            name="qiskit", display="Qiskit (Python)", extension=".py",
            fence_tags=("python", "py", "qiskit"), package_manager="pip",
            quantum=True,
            instructions=(
                "Write Python using Qiskit 2.x that builds the requested quantum "
                "circuit. Bind the finished circuit to a module-level variable "
                "named `circuit` (a qiskit.QuantumCircuit). Do NOT run, sample, "
                "transpile for, or submit to any backend, and do not import "
                "qiskit_aer or any provider: execution is handled separately. "
                "Use only standard gates (h, x, y, z, s, t, rx, ry, rz, cx, cz, "
                "swap, ccx, p, cp, u) so the circuit exports to OpenQASM 2."
            ),
        ),
        LanguageSpec(
            name="openqasm", display="OpenQASM 2.0", extension=".qasm",
            fence_tags=("qasm", "openqasm", "qasm2"), quantum=True,
            instructions=(
                "Write a single OpenQASM 2.0 program. Start with "
                "`OPENQASM 2.0;` and `include \"qelib1.inc\";`, declare qreg and "
                "creg, and end with measurements."
            ),
        ),
    ]
}

ALIASES: Dict[str, str] = {
    "py": "python", "python3": "python",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "sh": "bash", "shell": "bash",
    "quantum": "qiskit", "qasm": "openqasm",
}


def resolve_language(name: str) -> LanguageSpec:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in LANGUAGES:
        raise ValueError(
            f"Unknown language {name!r}. Available: {', '.join(LANGUAGES)}"
        )
    return LANGUAGES[key]


# -- Results ------------------------------------------------------------------


@dataclass
class Dependency:
    """One third-party package the generated code imports."""

    module: str           # the name the code imports
    package: str          # the name the package manager installs
    manager: str          # "pip" | "npm"
    installed: Optional[bool] = None  # None when it cannot be determined


@dataclass
class Translation:
    """One natural-language request rendered as code."""

    request: str
    language: str
    code: str
    syntax_ok: Optional[bool] = None   # None when no checker is available
    syntax_error: Optional[str] = None
    repairs: int = 0
    dependencies: List[Dependency] = field(default_factory=list)
    # Set when generation itself failed; `code` then holds the error comment.
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.syntax_ok is not False

    @property
    def missing(self) -> List[Dependency]:
        return [d for d in self.dependencies if d.installed is False]

    def install_commands(self) -> List[List[str]]:
        return install_commands(self.missing)


# -- Code extraction ----------------------------------------------------------

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(text: str, spec: LanguageSpec) -> str:
    """
    Pull the code out of a model reply.

    Models often wrap code in prose or return several fenced blocks; prefer a
    block tagged with the target language, then the first fenced block, then
    the whole reply when nothing is fenced.
    """
    blocks = [(tag.strip().lower(), body) for tag, body in _FENCE.findall(text)]
    if not blocks:
        return strip_code_fence(text)
    for tag, body in blocks:
        if tag in spec.fence_tags:
            return body.strip()
    return blocks[0][1].strip()


# -- Syntax checking ----------------------------------------------------------


def check_syntax(code: str, spec: LanguageSpec) -> Tuple[Optional[bool], Optional[str]]:
    """
    (ok, error) for `code` — parsing only, nothing is executed.

    Returns (None, None) when no checker exists for the language on this
    machine, so callers can tell "unchecked" apart from "valid".
    """
    if spec.extension == ".py":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return True, None

    if spec.name == "openqasm":
        try:
            from qiskit import qasm2
        except ImportError:
            return None, None
        try:
            qasm2.loads(code, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        except Exception as exc:  # noqa: BLE001 - qasm2 raises several types
            return False, f"QASM error: {exc}"
        return True, None

    checker = {"javascript": ["node", "--check"], "bash": ["bash", "-n"]}.get(spec.name)
    if checker is None or shutil.which(checker[0]) is None:
        return None, None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"snippet{spec.extension}"
        path.write_text(code)
        proc = subprocess.run(checker + [str(path)], capture_output=True,
                              text=True, timeout=30)
    if proc.returncode == 0:
        return True, None
    return False, (proc.stderr or proc.stdout).strip().replace(str(path), "<snippet>")


# -- Dependency resolution ----------------------------------------------------

# Import name -> installable package name, where the two differ.
PIP_NAMES: Dict[str, str] = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "google.generativeai": "google-generativeai",
    "jwt": "pyjwt",
    "magic": "python-magic",
    "PIL": "pillow",
    "pptx": "python-pptx",
    "qiskit_aer": "qiskit-aer",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
    "serial": "pyserial",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "telegram": "python-telegram-bot",
    "yaml": "pyyaml",
    "zmq": "pyzmq",
}

NODE_BUILTINS = frozenset({
    "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
    "crypto", "dgram", "diagnostics_channel", "dns", "events", "fs", "http",
    "http2", "https", "inspector", "module", "net", "os", "path", "perf_hooks",
    "process", "querystring", "readline", "repl", "stream", "string_decoder",
    "test", "timers", "tls", "tty", "url", "util", "v8", "vm", "wasi",
    "worker_threads", "zlib",
})

_JS_IMPORT = re.compile(
    r"""(?:require\(\s*|import\s*\(\s*|from\s+|import\s+)['"]([^'"]+)['"]"""
)


def _python_imports(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def _local_modules() -> set:
    """Modules shipped in this repository, which must never be pip-installed."""
    return {p.stem for p in Path(__file__).resolve().parent.glob("*.py")}


def resolve_dependencies(code: str, spec: LanguageSpec,
                         project_dir: Optional[Path] = None) -> List[Dependency]:
    """Third-party packages `code` imports, each marked installed or not."""
    if spec.package_manager == "pip":
        local = _local_modules()
        seen: Dict[str, Dependency] = {}
        for name in _python_imports(code):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in local or top == "__future__":
                continue
            package = PIP_NAMES.get(name) or PIP_NAMES.get(top) or top.replace("_", "-")
            if package in seen:
                continue
            try:
                installed = importlib.util.find_spec(top) is not None
            except (ImportError, ValueError):
                installed = False
            seen[package] = Dependency(top, package, "pip", installed)
        return list(seen.values())

    if spec.package_manager == "npm":
        root = project_dir or Path.cwd()
        seen = {}
        for spec_name in _JS_IMPORT.findall(code):
            if spec_name.startswith((".", "/", "node:")):
                continue
            parts = spec_name.split("/")
            package = "/".join(parts[:2]) if spec_name.startswith("@") else parts[0]
            if package in NODE_BUILTINS or package in seen:
                continue
            installed = (root / "node_modules" / package).is_dir()
            seen[package] = Dependency(package, package, "npm", installed)
        return list(seen.values())

    return []


def install_commands(deps: List[Dependency]) -> List[List[str]]:
    """The commands that would install `deps`, grouped by package manager."""
    commands: List[List[str]] = []
    pip = [d.package for d in deps if d.manager == "pip"]
    npm = [d.package for d in deps if d.manager == "npm"]
    if pip:
        commands.append([sys.executable, "-m", "pip", "install", *pip])
    if npm:
        commands.append(["npm", "install", *npm])
    return commands


def install(deps: List[Dependency],
            runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
            ) -> List[subprocess.CompletedProcess]:
    """
    Install `deps`. Never called implicitly — the terminal asks first.

    Package names come from model output, so a caller should show the command
    to a person before running it: a hallucinated name can resolve to an
    unrelated (or malicious) package on the public index.
    """
    return [runner(cmd, check=False) for cmd in install_commands(deps)]


# -- Engine -------------------------------------------------------------------


class NLPEngine:
    """
    Natural language -> code, with syntax repair and dependency resolution.

    Wraps a Brion session so every translation shares one conversation: a
    follow-up such as "now make it async" sees the code it is refining.
    """

    MAX_REPAIRS = 2

    def __init__(self, brion: Optional[Brion] = None, language: str = "python",
                 provider: Optional[str] = None, model: Optional[str] = None):
        self.brion = brion or Brion(provider=provider, model=model)
        self.language = resolve_language(language)
        self.translations: List[Translation] = []

    def system_prompt(self, spec: LanguageSpec) -> str:
        return (
            f"{self.brion.system_prompt}\n\n"
            f"You are the translation layer of a natural-language programming "
            f"terminal. Translate the user's request into {spec.display}. "
            f"{spec.instructions}\n"
            f"Reply with exactly one fenced ```{spec.fence_tags[0]} code block. "
            f"Brief comments inside the code are fine; no prose outside it."
        )

    def translate(self, request: str, language: Optional[str] = None,
                  repair: bool = True) -> Translation:
        spec = resolve_language(language) if language else self.language
        reply = self.brion.generate_code(request, system=self.system_prompt(spec))

        # Brion reports failures as a comment string; its history says which.
        if self.brion.command_history[-1]["failed"]:
            result = Translation(request, spec.name, reply, error=reply.lstrip("# "))
            self.translations.append(result)
            return result

        code = extract_code(reply, spec)
        ok, err = check_syntax(code, spec)
        repairs = 0
        while repair and ok is False and repairs < self.MAX_REPAIRS:
            repairs += 1
            fixed = extract_code(self.brion.correct_code(code, err or "", spec.display), spec)
            if fixed == code:
                break  # the backend could not help; don't spin
            code = fixed
            ok, err = check_syntax(code, spec)

        result = Translation(
            request=request, language=spec.name, code=code,
            syntax_ok=ok, syntax_error=err, repairs=repairs,
            dependencies=resolve_dependencies(code, spec),
        )
        self.translations.append(result)
        return result

    def revise(self, previous: Translation, code: str) -> Translation:
        """Record corrected code for `previous`, re-checking syntax and packages."""
        spec = resolve_language(previous.language)
        code = extract_code(code, spec)
        ok, err = check_syntax(code, spec)
        result = Translation(
            request=previous.request, language=spec.name, code=code,
            syntax_ok=ok, syntax_error=err, repairs=previous.repairs + 1,
            dependencies=resolve_dependencies(code, spec),
        )
        self.translations.append(result)
        return result

    @property
    def last(self) -> Optional[Translation]:
        return self.translations[-1] if self.translations else None


def parse_natural_language(input_text: str, language: str = "python") -> Optional[str]:
    """
    One-shot translation, kept for the original `NLP Engine.py` interface.

    Returns None on failure, as the original did.
    """
    result = NLPEngine(language=language).translate(input_text)
    return result.code if result.error is None else None
