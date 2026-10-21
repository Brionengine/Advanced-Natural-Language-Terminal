"""
Tests for out-of-process execution and the quantum bridge.

Quantum tests that need Qiskit are skipped when it is not installed; the rest
run on a bare checkout.
"""

import importlib.util

import pytest

import quantum_bridge
from executor import run_code

HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_AER = importlib.util.find_spec("qiskit_aer") is not None

BELL = """
from qiskit import QuantumCircuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
"""

BELL_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""


# -- Executor ---------------------------------------------------------------


def test_run_captures_stdout():
    result = run_code("print('hello')")
    assert result.ok and result.stdout.strip() == "hello"


def test_run_reports_errors_without_raising():
    result = run_code("raise ValueError('boom')")
    assert not result.ok and "ValueError: boom" in result.error_text


def test_sys_exit_does_not_end_the_caller():
    """The old in-process exec() would have taken the terminal down here."""
    assert run_code("import sys; sys.exit(3)").returncode == 3


def test_timeout_is_enforced():
    result = run_code("while True: pass", timeout=1)
    assert result.timed_out and not result.ok


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError):
        run_code("x", language="openqasm")


# -- Quantum bridge ---------------------------------------------------------


def test_non_quantum_language_is_rejected():
    with pytest.raises(ValueError):
        quantum_bridge.to_qasm_text("print(1)", "python")


def test_openqasm_passes_through():
    assert quantum_bridge.to_qasm_text(BELL_QASM, "openqasm") == BELL_QASM


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
def test_qiskit_code_exports_qasm():
    qasm = quantum_bridge.qiskit_to_qasm(BELL)
    assert qasm.startswith("OPENQASM 2.0;") and "cx q[0],q[1];" in qasm


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
def test_missing_circuit_variable_is_explained():
    with pytest.raises(RuntimeError, match="named `circuit`"):
        quantum_bridge.qiskit_to_qasm("from qiskit import QuantumCircuit\nqc = QuantumCircuit(1)")


@pytest.mark.skipif(not HAS_AER, reason="qiskit-aer not installed")
def test_simulator_samples_a_bell_state():
    result = quantum_bridge.run(BELL_QASM, backend="simulator", shots=500, seed=7)
    assert set(result["counts"]) <= {"00", "11"}
    assert sum(result["counts"].values()) == 500


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
def test_hardware_needs_confirmation():
    with pytest.raises(PermissionError):
        quantum_bridge.run(BELL_QASM, backend="origin:WK_C180", shots=10)


@pytest.mark.skipif(not HAS_AER, reason="qiskit-aer not installed")
def test_builtin_backend_is_used_without_the_toolkit(monkeypatch):
    monkeypatch.setattr(quantum_bridge, "toolkit", lambda: None)
    result = quantum_bridge.run(BELL_QASM, shots=100, seed=1)
    assert result["backend"] == "simulator" and sum(result["counts"].values()) == 100


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
def test_builtin_rejects_backends_it_cannot_serve(monkeypatch):
    monkeypatch.setattr(quantum_bridge, "toolkit", lambda: None)
    with pytest.raises(quantum_bridge.QuantumUnavailable):
        quantum_bridge.run(BELL_QASM, backend="ibm:fez", confirm_hardware=True)


def test_origin_without_key_is_explained(monkeypatch):
    monkeypatch.setattr(quantum_bridge, "toolkit", lambda: None)
    monkeypatch.delenv("ORIGIN_QC_API_KEY", raising=False)
    monkeypatch.delenv("ORIGINQ_API_KEY", raising=False)
    origin = quantum_bridge.backends()["origin"]
    assert origin.startswith("unavailable")
