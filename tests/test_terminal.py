"""Tests for the terminal's command handling, with scripted input and output."""

import importlib.util

import pytest

import quantum_bridge
from brion import Brion
from nlp_engine import NLPEngine
from start_terminal import Terminal
from test_nlp_engine import ScriptedProvider


def make_terminal(*replies, answers=(), language="python"):
    brion = Brion(provider="offline")
    brion.provider = ScriptedProvider(*replies)
    out = []
    pending = list(answers)
    term = Terminal(NLPEngine(brion=brion, language=language),
                    read=lambda prompt: pending.pop(0), write=out.append)
    return term, out


def text(out):
    return "\n".join(out)


def test_plain_text_is_translated():
    term, out = make_terminal("```python\nprint(42)\n```")
    term.handle("print the answer")
    assert "print(42)" in text(out) and "/run" in text(out)


def test_run_executes_the_last_translation():
    term, out = make_terminal("```python\nprint(6 * 7)\n```")
    term.handle("multiply")
    term.handle("/run")
    assert "42" in text(out) and "[exit 0]" in text(out)


def test_failed_run_can_be_fixed():
    term, out = make_terminal("```python\nprint(undefined_name)\n```", "print('fixed')")
    term.handle("x")
    term.handle("/run")
    assert "NameError" in text(out)
    term.handle("/fix")
    assert term.engine.last.code == "print('fixed')"


def test_fix_needs_a_failed_run():
    term, out = make_terminal("```python\nprint(1)\n```")
    term.handle("x")
    term.handle("/fix")
    assert "Nothing to fix" in out[-1]


def test_run_before_any_translation_is_explained():
    term, out = make_terminal()
    term.handle("/run")
    assert "No translation yet" in out[-1]


def test_lang_switches_target():
    term, out = make_terminal()
    term.handle("/lang js")
    assert term.engine.language.name == "javascript"


def test_bad_language_does_not_crash():
    term, out = make_terminal()
    term.handle("/lang klingon")
    assert out[-1].startswith("Error: Unknown language")


def test_unknown_command_is_reported():
    term, out = make_terminal()
    term.handle("/frobnicate")
    assert "Unknown command" in out[-1]


def test_install_requires_confirmation():
    term, out = make_terminal("```python\nimport not_a_real_pkg_xyz\n```", answers=["n"])
    term.handle("x")
    term.handle("/install")
    assert "not-a-real-pkg-xyz" in text(out) and out[-1] == "Cancelled."


def test_save_writes_the_code(tmp_path):
    term, out = make_terminal("```python\nx = 1\n```")
    term.handle("x")
    target = tmp_path / "out.py"
    term.handle(f"/save {target}")
    assert target.read_text() == "x = 1\n"


def test_save_will_not_overwrite_without_asking(tmp_path):
    target = tmp_path / "keep.py"
    target.write_text("original")
    term, out = make_terminal("```python\nx = 1\n```", answers=["n"])
    term.handle("x")
    term.handle(f"/save {target}")
    assert target.read_text() == "original"


def test_qrun_refuses_classical_code():
    term, out = make_terminal("```python\nx = 1\n```")
    term.handle("x")
    term.handle("/qrun")
    assert "not a quantum circuit" in out[-1]


def test_hardware_qrun_asks_before_spending_qpu_time(monkeypatch):
    submitted = []
    monkeypatch.setattr(quantum_bridge, "run", lambda *a, **k: submitted.append(k))
    monkeypatch.setattr(quantum_bridge, "to_qasm_text", lambda code, lang: code)
    term, out = make_terminal("```qasm\nOPENQASM 2.0;\n```", answers=["n"], language="openqasm")
    term.handle("bell")
    term.handle("/qrun origin:WK_C180 100")
    assert submitted == [] and out[-1] == "Cancelled."


@pytest.mark.skipif(importlib.util.find_spec("qiskit_aer") is None, reason="qiskit-aer not installed")
def test_qrun_samples_a_generated_circuit():
    code = "```python\nfrom qiskit import QuantumCircuit\ncircuit = QuantumCircuit(1)\ncircuit.x(0)\n```"
    term, out = make_terminal(code, language="qiskit")
    term.handle("flip a qubit")
    term.handle("/qrun simulator 50")
    assert '"1": 50' in out[-1]


def test_history_lists_translations():
    term, out = make_terminal("```python\nx = 1\n```")
    term.handle("make x")
    term.handle("/history")
    assert "make x" in out[-1]
