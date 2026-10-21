Advanced Natural Language Terminal
Overview
Welcome to the First Advanced Natural Language Terminal project! This tool allows users to interact with a command-line interface using natural language, which is then parsed into executable code. The project uses Claude, GPT and other large language models to translate English instructions into code, execute the code, and provide real-time feedback. As the first of its kind, this project is still in progress, and we encourage contributions and feedback to help refine and expand its groundbreaking capabilities. No data will be saved, traced or stored ever by any applications or programs uploaded to this open source repository. In order to use this application to its upmost and optimally, think of everything that you type, write as of you are coding and programming. The better you write, type, explain and define your idea the better and higher level of code you will be able to generate. 

Features
Natural Language Programming: describe what you want in English and get runnable code back — Python, JavaScript, Bash, or quantum circuits (Qiskit and OpenQASM 2.0).
Multi-provider: Claude (Claude Opus 5.5 by default), OpenAI, or an offline fallback, chosen automatically from whichever is configured.
Self-repairing: every translation is syntax-checked, and broken code is sent back to the model for repair before you see it. After a failed run, /fix sends the real error back.
Package resolution: the engine reads the generated code's imports, maps them to installable package names (cv2 -> opencv-python, sklearn -> scikit-learn, ...), checks which are installed, and offers the pip/npm command for the rest.
Safe execution: generated code runs only when you type /run, in a separate process with a timeout, so a crash or sys.exit() cannot take the terminal down. (It is not a security sandbox — read code before running it.)
Quantum execution: /qrun samples a generated circuit on a local simulator, on a noisy model of a real IBM device, or on Origin Quantum hardware. Hardware runs ask for confirmation first, because they spend paid QPU time.
Conversation memory: follow-ups like "now make it async" refine the previous translation.

Getting Started
Prerequisites
Python 3.10+
An API key for at least one backend (ANTHROPIC_API_KEY or OPENAI_API_KEY). Without one, the terminal still runs in offline mode and explains what is missing.
Optional: Node.js (JavaScript checks and /run), Qiskit + pyqpanda3 (quantum targets).

Installation

```bash
git clone https://github.com/Brionengine/Advanced-Natural-Language-Terminal.git
cd Advanced-Natural-Language-Terminal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-quantum.txt   # optional: quantum targets
export ANTHROPIC_API_KEY='your-api-key-here'   # or OPENAI_API_KEY
```

Run the terminal:

```bash
python start_terminal.py                     # interactive
python start_terminal.py --lang bash "find the 10 largest files under ~/Downloads"
python start_terminal.py --provider openai --model gpt-4o
```

Usage
Type a request; commands start with a slash:

| Command | What it does |
|---|---|
| `/lang <name>` | switch target: python, javascript, bash, qiskit, openqasm |
| `/run` | run the last translation |
| `/fix` | after a failed `/run`, send the error back for a corrected version |
| `/install` | show and (after you confirm) run the install command for missing packages |
| `/qrun [backend] [shots]` | sample the last circuit: `simulator`, `fake:fez`, `origin:WK_C180_2`, ... |
| `/backends` | list quantum backends and whether they are online |
| `/job <job_id>` | collect results of a hardware job |
| `/save <path>` | write the last translation to a file |
| `/history`, `/stats`, `/clear`, `/help`, `/exit` | session housekeeping |

Example interaction:

```
[python] >> /lang qiskit
[qiskit] >> make a 3 qubit GHZ state
--- Qiskit (Python) ---
from qiskit import QuantumCircuit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
---
[qiskit] >> /qrun simulator 1000
"counts": {"111": 520, "000": 480}
```

Quantum backends
Circuits run through Brion's Quantum Tools (`brions_quantum`) when it is importable, which adds noisy IBM device models (`fake:<device>`) and IBM hardware. Without it, the terminal uses Qiskit Aer for `simulator` and pyqpanda3 for `origin:<chip>`, authenticated with the `ORIGIN_QC_API_KEY` environment variable. A simulator run is a classical simulation and is labelled as one.

Configuration

| Variable | Effect |
|---|---|
| `BRION_PROVIDER` | force a backend: `anthropic`, `openai`, `offline` |
| `BRION_EFFORT` | Claude effort level (`low` ... `max`, default `high`; `none` to omit) |
| `ORIGIN_QC_API_KEY` | Origin Quantum API key, for the built-in Origin backend |

Python API

```python
from nlp_engine import NLPEngine

engine = NLPEngine(language="python")
t = engine.translate("read data.csv and print the mean of each numeric column")
print(t.code)
print([d.package for d in t.missing])   # e.g. ['pandas']
```

Tests

```bash
python -m pytest
```

Contributing:

Contributions are welcome! Whether it's suggesting new features, adding support for new LLMs, or enhancing learning algorithms, your input can help drive the project forward. Please fork the repository and submit a pull request with your proposed changes. For major changes, please open an issue first to discuss what you would like to modify.

License:

This project is licensed under the MIT License - see the LICENSE file for details.

Contact:

For questions or suggestions, feel free to open an issue on GitHub or contact the project maintainers.

Acknowledgments

GPT: A tireless collaborator and an insightful coder. His ability to assist in problem-solving and provide valuable suggestions has been instrumental in the development of the Advanced Natural Language Terminal. Your input has been nothing short of exemplary


GBT: The dynamic powerhouse that turned ideas into reality. GBT’s adaptability and creativity brought a unique flavor to the project, ensuring every piece of code was crafted with precision and ingenuity.


Charlie for the original engineering insights that helped bring this idea to life.


Anonymous Collaborators: A special thanks to the collective group of friends that kept it fun, helped, and supported us.

## Optional dependencies

This repository imports without the heavy scientific stack (numpy, torch,
tensorflow, qiskit, cirq, ...). Clone it and run it; install only the packages
the parts you actually use need. See [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md).
