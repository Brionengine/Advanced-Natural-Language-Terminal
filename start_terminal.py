"""
Brion Quantum - Advanced Natural Language Terminal v3.0
========================================================
Type what you want in English; get code back, check it, run it.

    python start_terminal.py                         interactive terminal
    python start_terminal.py --lang qiskit "make a 3-qubit GHZ state"
                                                     one-shot, prints the code

Generated code is never run or installed without you asking: /run executes the
last translation, /install installs its missing packages, and /qrun on real
quantum hardware asks again before spending QPU time.

Developed by Brion Quantum AI Team
"""

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import quantum_bridge
from executor import run_code
from nlp_engine import LANGUAGES, NLPEngine, Translation, install, resolve_language

HELP = """\
Type a request in plain English to translate it into code.

  /lang [name]            show or switch the target language
  /langs                  list target languages
  /run                    run the last translation (python, javascript, bash)
  /fix                    after a failed /run, send the error back for a fix
  /qrun [backend] [shots] sample the last quantum circuit
                          (default: simulator 1024; e.g. /qrun origin:WK_C180 2000)
  /backends               list quantum backends and their status
  /job <job_id>           check a quantum hardware job
  /install                install the last translation's missing packages
  /save <path>            write the last translation to a file
  /show                   print the last translation again
  /history                list this session's translations
  /stats                  session statistics
  /clear                  forget the conversation (keeps the language)
  /help                   this text
  /exit                   quit
"""


class Terminal:
    """The command loop, with I/O injected so it can be tested."""

    def __init__(self, engine: NLPEngine,
                 read: Callable[[str], str] = input,
                 write: Callable[[str], None] = print):
        self.engine = engine
        self.read = read
        self.write = write
        self.last_run_error: Optional[str] = None
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "lang": self.cmd_lang, "langs": self.cmd_langs, "run": self.cmd_run,
            "fix": self.cmd_fix, "qrun": self.cmd_qrun, "backends": self.cmd_backends,
            "job": self.cmd_job, "install": self.cmd_install, "save": self.cmd_save,
            "show": self.cmd_show, "history": self.cmd_history, "stats": self.cmd_stats,
            "clear": self.cmd_clear, "help": self.cmd_help,
        }

    # -- Loop ---------------------------------------------------------------

    def loop(self):
        self.write("Brion Natural Language Terminal v3.0 — type /help for commands.")
        self.write(f"Provider: {self.engine.brion.provider.name} "
                   f"({self.engine.brion.model}) | language: {self.engine.language.name}")
        while True:
            try:
                line = self.read(f"[{self.engine.language.name}] >> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.write("")
                break
            if line.lower() in ("exit", "quit", "/exit", "/quit"):
                break
            if line:
                self.handle(line)
        self.write("Goodbye!")

    def handle(self, line: str):
        if not line.startswith("/"):
            self.show(self.engine.translate(line))
            return
        try:
            name, *args = shlex.split(line[1:])
        except ValueError as exc:
            self.write(f"Could not parse command: {exc}")
            return
        command = self.commands.get(name.lower())
        if command is None:
            self.write(f"Unknown command /{name}. Type /help.")
            return
        try:
            command(args)
        except Exception as exc:  # noqa: BLE001 - one bad command must not end the session
            self.write(f"Error: {exc}")

    # -- Display ------------------------------------------------------------

    def show(self, t: Translation):
        if t.error:
            self.write(t.code)
            return
        self.write(f"\n--- {LANGUAGES[t.language].display} ---\n{t.code}\n---")
        if t.syntax_ok is False:
            self.write(f"Syntax check failed after {t.repairs} repair attempt(s): {t.syntax_error}")
        elif t.repairs:
            self.write(f"Syntax repaired automatically ({t.repairs} attempt(s)).")
        if t.dependencies:
            listed = ", ".join(
                f"{d.package}{'' if d.installed else ' (missing)'}" for d in t.dependencies
            )
            self.write(f"Packages: {listed}")
        if t.missing:
            self.write("Run /install to install the missing packages.")
        hint = "/qrun" if LANGUAGES[t.language].quantum else "/run"
        self.write(f"Type {hint} to execute it.")

    def _last(self) -> Translation:
        t = self.engine.last
        if t is None or t.error:
            raise RuntimeError("No translation yet — type a request first.")
        return t

    def _confirm(self, prompt: str) -> bool:
        return self.read(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")

    # -- Commands -----------------------------------------------------------

    def cmd_lang(self, args):
        if args:
            self.engine.language = resolve_language(args[0])
        self.write(f"Language: {self.engine.language.name}")

    def cmd_langs(self, args):
        for spec in LANGUAGES.values():
            tag = " (quantum)" if spec.quantum else ""
            self.write(f"  {spec.name:<11} {spec.display}{tag}")

    def cmd_run(self, args):
        t = self._last()
        if LANGUAGES[t.language].quantum:
            self.write("This is a quantum circuit — use /qrun.")
            return
        if t.missing:
            self.write("Missing packages: " + ", ".join(d.package for d in t.missing)
                       + ". Run /install first, or the script will likely fail.")
        result = run_code(t.code, t.language)
        if result.stdout:
            self.write(result.stdout.rstrip())
        if result.ok:
            self.last_run_error = None
            self.write("[exit 0]")
            return
        self.last_run_error = result.error_text
        self.write(result.stderr.rstrip() if result.stderr else result.error_text)
        self.write(f"[failed: {'timeout' if result.timed_out else result.returncode}] "
                   "Type /fix to have the error corrected.")

    def cmd_fix(self, args):
        t = self._last()
        if not self.last_run_error:
            self.write("Nothing to fix — /run something that fails first.")
            return
        spec = LANGUAGES[t.language]
        fixed = self.engine.brion.correct_code(t.code, self.last_run_error, spec.display)
        if fixed == t.code:
            self.write("The model could not produce a fix.")
            return
        self.last_run_error = None
        self.show(self.engine.revise(t, fixed))

    def cmd_qrun(self, args):
        t = self._last()
        if not LANGUAGES[t.language].quantum:
            self.write("This is not a quantum circuit — use /run, or /lang qiskit first.")
            return
        backend = args[0] if args else "simulator"
        shots = int(args[1]) if len(args) > 1 else 1024
        hardware = not backend.startswith(("simulator", "fake:"))
        if hardware and not self._confirm(
                f"Submit to {backend} with {shots} shots? This spends paid QPU time."):
            self.write("Cancelled.")
            return
        qasm = quantum_bridge.to_qasm_text(t.code, t.language)
        result = quantum_bridge.run(qasm, backend=backend, shots=shots,
                                    confirm_hardware=hardware)
        self.write(json.dumps(result, indent=2, default=str))

    def cmd_backends(self, args):
        self.write(json.dumps(quantum_bridge.backends(), indent=2, default=str))

    def cmd_job(self, args):
        if not args:
            self.write("Usage: /job <job_id>")
            return
        self.write(json.dumps(quantum_bridge.job_status(args[0]), indent=2, default=str))

    def cmd_install(self, args):
        t = self._last()
        if not t.missing:
            self.write("Nothing to install.")
            return
        for cmd in t.install_commands():
            self.write("  " + " ".join(shlex.quote(c) for c in cmd))
        self.write("Package names come from generated code; check them before installing.")
        if not self._confirm("Run the command(s) above?"):
            self.write("Cancelled.")
            return
        failed = [p for p in install(t.missing) if p.returncode != 0]
        self.write("Install failed." if failed else "Installed.")

    def cmd_save(self, args):
        t = self._last()
        if not args:
            self.write("Usage: /save <path>")
            return
        path = Path(args[0]).expanduser()
        if path.exists() and not self._confirm(f"{path} exists. Overwrite?"):
            self.write("Cancelled.")
            return
        path.write_text(t.code + "\n")
        self.write(f"Saved to {path}")

    def cmd_show(self, args):
        self.show(self._last())

    def cmd_history(self, args):
        if not self.engine.translations:
            self.write("No translations yet.")
        for i, t in enumerate(self.engine.translations, 1):
            status = "error" if t.error else ("ok" if t.syntax_ok is not False else "syntax error")
            self.write(f"{i:>3}. [{t.language}] {t.request[:60]} — {status}")

    def cmd_stats(self, args):
        self.write(json.dumps(self.engine.brion.get_session_stats(), indent=2))

    def cmd_clear(self, args):
        self.engine.brion.clear_conversation()
        self.write("Conversation cleared.")

    def cmd_help(self, args):
        self.write(HELP)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Natural language to code terminal.")
    parser.add_argument("request", nargs="*", help="translate this one request and exit")
    parser.add_argument("--lang", default="python", help=f"one of: {', '.join(LANGUAGES)}")
    parser.add_argument("--provider", help="anthropic | openai | offline (default: auto)")
    parser.add_argument("--model", help="override the provider's default model")
    args = parser.parse_args(argv)

    engine = NLPEngine(language=args.lang, provider=args.provider, model=args.model)
    if args.request:
        t = engine.translate(" ".join(args.request))
        print(t.code)
        for cmd in t.install_commands():
            print("# missing packages: " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
        return 0 if t.ok else 1

    Terminal(engine).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
