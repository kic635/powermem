"""Regression entry point for /system/status dependency probe endpoint tests (Issue #1111).

The required PR test workflow ignores tests/regression/, so the canonical test
implementations live under tests/unit/server/. This module re-exports them for
the regression workflow and `pytest -m regression` runs.
"""

import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.regression,
    pytest.mark.usefixtures("isolated_dependency_probe_state"),
]

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

from unit.server.test_system_status_dependency_probes import *  # noqa: E402, F403
