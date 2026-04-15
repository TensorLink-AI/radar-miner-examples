"""Tests for core.validation module."""

import pytest
import sys
import os

# Use any agent dir — all contain identical core/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents", "autonomous"))

# Clear cached core modules to ensure correct resolution
for _k in list(sys.modules.keys()):
    if _k == "core" or _k.startswith("core."):
        del sys.modules[_k]

from core.validation import validate_code as validate


# All task-invariant checks (syntax, forbidden imports, required functions)
# work with an empty task spec — there's no longer an is_harness_task split.
HARNESS_CHALLENGE = {
    "task": {
        "task_params": {
            "context_len": 512,
            "prediction_len": 96,
            "num_variates": 1,
            "quantiles": [0.1, 0.5, 0.9],
        },
    },
}

STANDALONE_CHALLENGE = {"task": {}}


class TestValidateSyntax:
    def test_valid_syntax(self):
        code = '''
def build_model(context_len, prediction_len, num_variates, quantiles):
    return None
def build_optimizer(model):
    return None
x = 1 + 2
'''
        ok, errors = validate(code, STANDALONE_CHALLENGE)
        assert ok
        assert errors == []

    def test_invalid_syntax(self):
        ok, errors = validate("def foo(:", {})
        assert not ok
        assert any("SyntaxError" in e for e in errors)

    def test_empty_code_rejected(self):
        ok, errors = validate("", {})
        assert not ok
        assert any("Empty" in e for e in errors)

    def test_whitespace_only_rejected(self):
        ok, errors = validate("   \n\n  ", {})
        assert not ok
        assert any("Empty" in e for e in errors)


class TestForbiddenImports:
    def test_subprocess_forbidden(self):
        ok, errors = validate("import subprocess", {})
        assert not ok
        assert any("subprocess" in e for e in errors)

    def test_socket_forbidden(self):
        ok, errors = validate("import socket", {})
        assert not ok
        assert any("socket" in e for e in errors)

    def test_ftplib_forbidden(self):
        ok, errors = validate("from ftplib import FTP", {})
        assert not ok
        assert any("ftplib" in e for e in errors)

    def test_allowed_imports(self):
        code = '''
import torch
import numpy as np
def build_model(context_len, prediction_len, num_variates, quantiles):
    return None
def build_optimizer(model):
    return None
'''
        ok, errors = validate(code, STANDALONE_CHALLENGE)
        assert ok

    def test_subprocess_from_import(self):
        ok, errors = validate("from subprocess import run", {})
        assert not ok


class TestHarnessValidation:
    def test_valid_harness_code(self):
        code = '''
import torch
import torch.nn as nn

def build_model(context_len, prediction_len, num_variates, quantiles):
    return nn.Linear(context_len, prediction_len * len(quantiles))

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert ok, f"Unexpected errors: {errors}"

    def test_missing_build_model(self):
        code = '''
import torch
def build_optimizer(model):
    return torch.optim.Adam(model.parameters())
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert not ok
        assert any("build_model" in e for e in errors)

    def test_missing_build_optimizer(self):
        code = '''
import torch.nn as nn
def build_model(context_len, prediction_len, num_variates, quantiles):
    return nn.Linear(10, 10)
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert not ok
        assert any("build_optimizer" in e for e in errors)

    def test_missing_parameter(self):
        code = '''
def build_model(context_len, prediction_len):
    pass
def build_optimizer(model):
    pass
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert not ok
        assert any("num_variates" in e for e in errors)

    def test_harness_check_always_applied(self):
        code = "x = 42"
        ok, errors = validate(code, STANDALONE_CHALLENGE)
        assert not ok
        assert any("build_model" in e for e in errors)

    def test_optional_hooks_not_required(self):
        code = '''
def build_model(context_len, prediction_len, num_variates, quantiles):
    pass
def build_optimizer(model):
    pass
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert ok

    def test_with_optional_hooks(self):
        code = '''
def build_model(context_len, prediction_len, num_variates, quantiles):
    pass
def build_optimizer(model):
    pass
def training_config():
    return {"batch_size": 32}
def init_weights(model):
    pass
def configure_amp():
    return {"enabled": True}
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert ok


class TestMultipleErrors:
    def test_forbidden_import_and_missing_function(self):
        code = '''
import subprocess
def build_optimizer(model):
    pass
'''
        ok, errors = validate(code, HARNESS_CHALLENGE)
        assert not ok
        assert len(errors) >= 2
