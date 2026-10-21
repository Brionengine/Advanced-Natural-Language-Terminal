"""
Tests for the natural-language translation layer.

A scripted provider stands in for the model, so translation, repair and
dependency resolution run for real with no SDK, key or network.
"""

import shutil
import subprocess
import sys

import pytest

from brion import Brion
from nlp_engine import (
    LANGUAGES,
    Dependency,
    NLPEngine,
    check_syntax,
    extract_code,
    install,
    install_commands,
    resolve_dependencies,
    resolve_language,
)
from providers import CompletionResult, LLMProvider

PY = LANGUAGES["python"]
JS = LANGUAGES["javascript"]


class ScriptedProvider(LLMProvider):
    """Returns each scripted reply in turn and records every call."""

    name = "scripted"
    default_model = "scripted"

    def __init__(self, *replies, error=None):
        super().__init__()
        self.replies = list(replies)
        self.error = error
        self.calls = []

    @classmethod
    def is_available(cls):
        return True

    def complete(self, system, messages, max_tokens=16000):
        self.calls.append({"system": system, "messages": list(messages)})
        if self.error:
            return CompletionResult(text="", provider=self.name, model=self.model, error=self.error)
        return CompletionResult(text=self.replies.pop(0), provider=self.name, model=self.model)


def engine_with(*replies, language="python", error=None):
    brion = Brion(provider="offline")
    brion.provider = ScriptedProvider(*replies, error=error)
    return NLPEngine(brion=brion, language=language)


# -- Languages --------------------------------------------------------------


def test_aliases_resolve():
    assert resolve_language("JS").name == "javascript"
    assert resolve_language("quantum").name == "qiskit"


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError, match="Unknown language"):
        resolve_language("cobol-2099")


# -- Extraction -------------------------------------------------------------


def test_extract_prefers_the_target_language_block():
    reply = "Here:\n```bash\necho hi\n```\n```python\nprint('hi')\n```"
    assert extract_code(reply, PY) == "print('hi')"


def test_extract_falls_back_to_first_block():
    assert extract_code("```\nx = 1\n```", PY) == "x = 1"


def test_extract_unfenced_reply_is_used_whole():
    assert extract_code("x = 1", PY) == "x = 1"


def test_extract_survives_an_unclosed_fence():
    assert extract_code("```python\nx = 1\ny = 2", PY) == "x = 1\ny = 2"


# -- Syntax -----------------------------------------------------------------


def test_python_syntax_ok():
    assert check_syntax("x = 1", PY) == (True, None)


def test_python_syntax_error_reports_line():
    ok, err = check_syntax("def f(:\n  pass", PY)
    assert ok is False and "line 1" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_syntax_checked_by_node():
    assert check_syntax("const x = 1;", JS)[0] is True
    assert check_syntax("const = ;", JS)[0] is False


def test_unchecked_language_reports_none(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert check_syntax("echo hi", LANGUAGES["bash"]) == (None, None)


# -- Dependencies -----------------------------------------------------------


def test_stdlib_imports_are_not_dependencies():
    assert resolve_dependencies("import os, json\nfrom pathlib import Path", PY) == []


def test_import_names_map_to_package_names():
    deps = resolve_dependencies("import cv2\nfrom sklearn import svm\nimport yaml", PY)
    assert [d.package for d in deps] == ["opencv-python", "scikit-learn", "pyyaml"]


def test_installed_packages_are_marked():
    deps = resolve_dependencies("import pytest\nimport not_a_real_pkg_xyz", PY)
    status = {d.package: d.installed for d in deps}
    assert status == {"pytest": True, "not-a-real-pkg-xyz": False}


def test_repository_modules_are_not_dependencies():
    """`import brion` must never turn into `pip install brion`."""
    assert resolve_dependencies("import brion\nimport providers", PY) == []


def test_relative_imports_are_ignored():
    assert resolve_dependencies("from . import sibling", PY) == []


def test_duplicate_imports_are_listed_once():
    deps = resolve_dependencies("import numpy\nimport numpy.linalg\nfrom numpy import array", PY)
    assert len(deps) == 1


def test_javascript_dependencies_skip_builtins_and_paths(tmp_path):
    code = (
        "const fs = require('fs');\nconst x = require('./local');\n"
        "import express from 'express';\nimport { z } from '@scope/pkg/sub';\n"
        "const p = require('node:path');"
    )
    deps = resolve_dependencies(code, JS, project_dir=tmp_path)
    assert [d.package for d in deps] == ["express", "@scope/pkg"]
    assert all(d.installed is False for d in deps)


def test_install_commands_group_by_manager():
    deps = [Dependency("a", "a", "pip"), Dependency("b", "b", "npm"), Dependency("c", "c", "pip")]
    commands = install_commands(deps)
    assert commands[0][-3:] == ["install", "a", "c"]
    assert commands[1] == ["npm", "install", "b"]


def test_install_uses_the_running_interpreter():
    """Packages must land in the environment the terminal will run code with."""
    assert install_commands([Dependency("a", "a", "pip")])[0][0] == sys.executable


def test_install_runs_each_command():
    ran = []
    install([Dependency("a", "a", "pip")],
            runner=lambda cmd, check: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert ran and ran[0][-1] == "a"


# -- Translation ------------------------------------------------------------


def test_translate_returns_extracted_code():
    t = engine_with("```python\nprint('hi')\n```").translate("say hi")
    assert t.code == "print('hi')" and t.syntax_ok is True and t.ok


def test_translate_sends_a_language_specific_system_prompt():
    engine = engine_with("```js\nconsole.log(1)\n```", language="javascript")
    engine.translate("log one")
    system = engine.brion.provider.calls[0]["system"]
    assert "JavaScript" in system and Brion.DEFAULT_SYSTEM_PROMPT in system


def test_language_can_be_chosen_per_call():
    engine = engine_with("```bash\necho hi\n```")
    assert engine.translate("say hi", language="bash").language == "bash"
    assert engine.language.name == "python"


def test_broken_code_is_repaired():
    engine = engine_with("```python\nprint('hi'\n```", "print('hi')")
    t = engine.translate("say hi")
    assert t.code == "print('hi')" and t.syntax_ok is True and t.repairs == 1


def test_repair_stops_when_the_model_cannot_help():
    engine = engine_with("```python\nprint(\n```", "print(", "print(")
    t = engine.translate("x")
    assert t.syntax_ok is False and t.repairs == 1


def test_repair_is_bounded():
    engine = engine_with("def(", "def(:", "def(::", "def(:::")
    t = engine.translate("x")
    assert t.repairs == NLPEngine.MAX_REPAIRS and t.syntax_ok is False


def test_repair_can_be_disabled():
    t = engine_with("def(").translate("x", repair=False)
    assert t.repairs == 0 and t.syntax_ok is False


def test_generation_failure_is_reported_not_raised():
    t = engine_with(error="no key").translate("x")
    assert t.error and "no key" in t.error and not t.ok
    assert t.dependencies == []


def test_translations_share_one_conversation():
    engine = engine_with("```python\nx = 1\n```", "```python\nx = 2\n```")
    engine.translate("first")
    engine.translate("now change it")
    assert len(engine.brion.provider.calls[1]["messages"]) == 3


def test_missing_dependencies_come_back_with_the_translation():
    t = engine_with("```python\nimport not_a_real_pkg_xyz\n```").translate("x")
    assert [d.package for d in t.missing] == ["not-a-real-pkg-xyz"]
    assert t.install_commands()[0][-1] == "not-a-real-pkg-xyz"


def test_revise_records_a_new_translation():
    engine = engine_with("```python\nx = 1\n```")
    first = engine.translate("x")
    revised = engine.revise(first, "```python\nx = 2\n```")
    assert engine.last is revised and revised.code == "x = 2" and revised.repairs == 1


def test_qiskit_prompt_requires_a_circuit_variable():
    engine = engine_with("```python\ncircuit = None\n```", language="qiskit")
    engine.translate("bell state")
    assert "`circuit`" in engine.brion.provider.calls[0]["system"]
