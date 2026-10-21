"""
Brion Quantum - Quantum Bridge
===============================
Runs circuits produced by the NLP engine's quantum targets.

Generated Qiskit code is executed in a child process (see executor.py) and only
the circuit's OpenQASM text comes back into this process, which then samples
it. Two toolkits can do the sampling:

    brions_quantum    Brion's Quantum Tools, when importable. Supports
                      simulator, fake:<ibm-device>, ibm:<backend> and
                      origin:<chip>, and handles its own credentials.
    built-in          Qiskit Aer for "simulator", and pyqpanda3 for
                      origin:<chip> using ORIGIN_QC_API_KEY.

Real hardware spends paid QPU time, so it only runs with
confirm_hardware=True and is asynchronous: you get a job id back, and
job_status() collects the counts.

Developed by Brion Quantum AI Team
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional

from executor import run_code

QASM_MARKER = "----BRION-QASM----"

# Gates Origin's compiler accepts; circuits are transpiled to these first.
ORIGIN_BASIS = ["h", "x", "rz", "ry", "rx", "cx", "cz", "p", "cp", "u", "id"]

_EXPORT_FOOTER = f"""

# -- appended by quantum_bridge: export the circuit, do not run it --
from qiskit import qasm2 as _brion_qasm2
print({QASM_MARKER!r})
print(_brion_qasm2.dumps(circuit))
"""


class QuantumUnavailable(RuntimeError):
    """The packages a quantum operation needs are not installed."""


def toolkit() -> Optional[Any]:
    """brions_quantum.tools if it can be imported, else None."""
    try:
        return importlib.import_module("brions_quantum.tools")
    except ImportError:
        return None


def _require(module: str, install: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise QuantumUnavailable(f"{module} is not installed: pip install {install}") from exc


def qiskit_to_qasm(code: str, timeout: int = 120) -> str:
    """
    Execute generated Qiskit code and return its `circuit` as OpenQASM 2.

    The code runs out of process; a failure raises with the child's stderr so
    the terminal can feed it to the correction loop.
    """
    result = run_code(code + _EXPORT_FOOTER, language="qiskit", timeout=timeout)
    if not result.ok or QASM_MARKER not in result.stdout:
        detail = result.error_text
        if "NameError" in detail and "circuit" in detail:
            detail += "\n(The code must assign its QuantumCircuit to a variable named `circuit`.)"
        raise RuntimeError(detail)
    return result.stdout.split(QASM_MARKER, 1)[1].strip()


def parse_qasm(qasm: str):
    qasm2 = _require("qiskit.qasm2", "qiskit")
    return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def to_qasm_text(code: str, language: str) -> str:
    """OpenQASM 2 for a translation in either quantum target language."""
    if language == "openqasm":
        return code
    if language == "qiskit":
        return qiskit_to_qasm(code)
    raise ValueError(f"{language!r} is not a quantum target")


def run(qasm: str, backend: str = "simulator", shots: int = 1024,
        confirm_hardware: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    """Sample a circuit. Simulators return counts; hardware returns a job id."""
    circuit = parse_qasm(qasm)
    tools = toolkit()
    if tools is not None:
        return tools.run(circuit, backend=backend, shots=shots,
                         confirm_hardware=confirm_hardware, seed=seed)

    kind, _, target = backend.partition(":")
    if kind == "simulator":
        return _aer_run(circuit, shots, seed)
    if kind == "origin":
        if not confirm_hardware:
            raise PermissionError("Real hardware spends paid QPU time: confirm before submitting.")
        return _origin_run(circuit, target, shots)
    raise QuantumUnavailable(
        f"Backend {backend!r} needs Brion's Quantum Tools (brions_quantum); "
        "built-in backends are 'simulator' and 'origin:<chip>'."
    )


def backends() -> Dict[str, Any]:
    tools = toolkit()
    if tools is not None:
        return tools.list_backends()
    out: Dict[str, Any] = {}
    try:
        _require("qiskit_aer", "qiskit-aer")
        out["simulator"] = "Qiskit Aer ideal simulator (local, free)"
    except QuantumUnavailable as exc:
        out["simulator"] = f"unavailable: {exc}"
    try:
        service = _origin_service()
        out["origin"] = {f"origin:{name}": ("online" if up else "offline")
                         for name, up in service.backends().items()}
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        out["origin"] = f"unavailable: {exc}"
    return out


def job_status(job_id: str) -> Dict[str, Any]:
    tools = toolkit()
    if tools is not None:
        return tools.job_status(job_id)
    provider, _, raw = job_id.partition(":")
    if provider != "origin":
        raise QuantumUnavailable("Only origin: job ids are supported without brions_quantum.")
    qcloud = _require("pyqpanda3.qcloud", "pyqpanda3")
    _origin_service()  # authenticates the session the job query uses
    job = qcloud.QCloudJob(raw)
    status = job.status()
    out: Dict[str, Any] = {"job_id": job_id, "status": str(status)}
    if status == qcloud.JobStatus.FINISHED:
        out["probabilities"] = job.result().get_probs()
    elif status == qcloud.JobStatus.FAILED:
        out["error"] = job.result().error_message()
    return out


# -- Built-in backends --------------------------------------------------------


def _with_measurements(circuit):
    if any(inst.operation.name == "measure" for inst in circuit.data):
        return circuit
    measured = circuit.copy()
    measured.measure_all()
    return measured


def _aer_run(circuit, shots: int, seed: Optional[int]) -> Dict[str, Any]:
    aer = _require("qiskit_aer", "qiskit-aer")
    from qiskit import transpile
    sim = aer.AerSimulator()
    compiled = transpile(_with_measurements(circuit), sim, seed_transpiler=seed)
    counts = sim.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
    return {"backend": "simulator", "shots": shots, "kind": "classical simulation",
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


def _origin_service():
    qcloud = _require("pyqpanda3.qcloud", "pyqpanda3")
    key = os.environ.get("ORIGIN_QC_API_KEY") or os.environ.get("ORIGINQ_API_KEY")
    if not key:
        raise QuantumUnavailable("Set ORIGIN_QC_API_KEY to use Origin Quantum backends.")
    return qcloud.QCloudService(api_key=key.strip())


def _origin_run(circuit, chip: str, shots: int) -> Dict[str, Any]:
    from qiskit import qasm2, transpile
    compiler = _require("pyqpanda3.intermediate_compiler", "pyqpanda3")
    compiled = transpile(_with_measurements(circuit), basis_gates=ORIGIN_BASIS,
                         optimization_level=1)
    program = compiler.convert_qasm_string_to_qprog(qasm2.dumps(compiled))
    job = _origin_service().backend(chip).run(program, shots)
    return {"backend": f"origin:{chip}", "job_id": f"origin:{job.job_id()}",
            "status": str(job.status()), "next": "check it with /job <job_id>"}
