"""Regression conftest: shared fixtures for status dependency probe endpoint tests."""

import sys
from pathlib import Path

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

from helpers.status_dependency_probes import (  # noqa: E402
    async_client,
    isolated_dependency_probe_state,
    status_endpoint_settings,
    system_app,
)
