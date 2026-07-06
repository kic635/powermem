"""Endpoint-level tests for /system/status dependency probes (Issue #1111).

These tests run in the required PR test workflow (tests/unit/**). They exercise
PR #1085 probe guarantees through the public HTTP endpoints rather than calling
health_check helpers directly.
"""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from server.config import config
from server.models.response import DependencyStatus
from server.utils import health_check

_tests_root = Path(__file__).resolve().parents[2]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

from helpers.status_dependency_probes import (  # noqa: E402
    async_timeout,
    async_client,
    await_threading_event,
    dependency_status,
    isolated_dependency_probe_state,
    status_endpoint_settings,
    system_app,
)

pytestmark = pytest.mark.usefixtures("isolated_dependency_probe_state")


@pytest.mark.asyncio
@async_timeout(10)
async def test_blocked_status_probe_does_not_block_health_endpoint(
    async_client,
    monkeypatch,
):
    release = threading.Event()
    probe_blocked = threading.Event()

    def blocked_database_probe():
        probe_blocked.set()
        release.wait(timeout=2.0)
        return DependencyStatus(name="database", status="healthy")

    def fast_llm_probe():
        return DependencyStatus(name="llm", status="disabled")

    monkeypatch.setattr(health_check, "_check_database_sync", blocked_database_probe)
    monkeypatch.setattr(health_check, "_check_llm_sync", fast_llm_probe)

    status_task = asyncio.create_task(async_client.get("/api/v1/system/status"))
    try:
        assert await await_threading_event(probe_blocked, timeout=1.0)

        start = time.monotonic()
        health_response = await async_client.get("/api/v1/system/health")
        health_elapsed = time.monotonic() - start

        assert health_response.status_code == 200
        assert health_elapsed < 0.2
        assert health_response.json()["data"]["status"] == "healthy"

        status_response = await status_task

        assert status_response.status_code == 200
        body = status_response.json()
        database = dependency_status(body, "database")
        assert database["status"] == "degraded"
        assert "timed out" in database["error_message"]
    finally:
        release.set()
        if not status_task.done():
            await status_task


@pytest.mark.asyncio
@async_timeout(10)
async def test_status_dependency_probes_use_dedicated_executor_and_timeout_in_parallel(
    async_client,
    monkeypatch,
):
    calls = {"database": 0, "llm": 0}
    executor_submits: list[object] = []
    default_executor_calls: list[object] = []
    original_submit = health_check._DEPENDENCY_PROBE_EXECUTOR.submit
    original_run_in_executor = asyncio.BaseEventLoop.run_in_executor

    def tracked_submit(fn, *args, **kwargs):
        future = original_submit(fn, *args, **kwargs)
        executor_submits.append(fn)
        return future

    async def tracked_run_in_executor(self, executor, func, *args):
        if executor is None:
            default_executor_calls.append(func)
        return await original_run_in_executor(self, executor, func, *args)

    def slow_database_probe():
        calls["database"] += 1
        time.sleep(0.2)
        return DependencyStatus(name="database", status="healthy")

    def slow_llm_probe():
        calls["llm"] += 1
        time.sleep(0.2)
        return DependencyStatus(name="llm", status="healthy")

    monkeypatch.setattr(health_check._DEPENDENCY_PROBE_EXECUTOR, "submit", tracked_submit)
    monkeypatch.setattr(asyncio.BaseEventLoop, "run_in_executor", tracked_run_in_executor)
    monkeypatch.setattr(health_check, "_check_database_sync", slow_database_probe)
    monkeypatch.setattr(health_check, "_check_llm_sync", slow_llm_probe)
    monkeypatch.setattr(config, "dependency_status_cache_ttl_seconds", 0.0)

    start = time.monotonic()
    response = await async_client.get("/api/v1/system/status")
    elapsed = time.monotonic() - start

    assert response.status_code == 200
    assert elapsed < 0.2
    assert calls == {"database": 1, "llm": 1}
    assert len(executor_submits) == 2
    assert default_executor_calls == []

    body = response.json()
    assert dependency_status(body, "database")["status"] == "degraded"
    assert dependency_status(body, "llm")["status"] == "degraded"


@pytest.mark.asyncio
@async_timeout(10)
async def test_repeated_status_polls_reuse_blocked_database_probe(
    async_client,
    monkeypatch,
):
    calls = {"database": 0}
    started = threading.Event()
    release = threading.Event()

    def blocked_database_probe():
        calls["database"] += 1
        started.set()
        release.wait(timeout=2.0)
        return DependencyStatus(name="database", status="healthy")

    def fast_llm_probe():
        return DependencyStatus(name="llm", status="disabled")

    monkeypatch.setattr(health_check, "_check_database_sync", blocked_database_probe)
    monkeypatch.setattr(health_check, "_check_llm_sync", fast_llm_probe)
    monkeypatch.setattr(config, "dependency_status_cache_ttl_seconds", 0.0)

    try:
        first = await async_client.get("/api/v1/system/status")
        assert await await_threading_event(started, timeout=1.0)
        assert first.status_code == 200
        assert dependency_status(first.json(), "database")["status"] == "degraded"

        second = await async_client.get("/api/v1/system/status")
        assert second.status_code == 200
        assert dependency_status(second.json(), "database")["status"] == "degraded"
        assert calls == {"database": 1}
    finally:
        release.set()


@pytest.mark.asyncio
@async_timeout(10)
async def test_status_dependency_cache_reuses_within_ttl_then_refreshes(
    async_client,
    monkeypatch,
):
    calls = {"database": 0, "llm": 0}

    def database_probe():
        calls["database"] += 1
        return DependencyStatus(name="database", status="healthy")

    def llm_probe():
        calls["llm"] += 1
        return DependencyStatus(name="llm", status="disabled")

    monkeypatch.setattr(health_check, "_check_database_sync", database_probe)
    monkeypatch.setattr(health_check, "_check_llm_sync", llm_probe)
    monkeypatch.setattr(config, "dependency_status_cache_ttl_seconds", 0.1)

    first = await async_client.get("/api/v1/system/status")
    assert first.status_code == 200
    assert dependency_status(first.json(), "database")["status"] == "healthy"

    second = await async_client.get("/api/v1/system/status")
    assert second.status_code == 200
    assert calls == {"database": 1, "llm": 1}

    # Expire the TTL with real wall-clock time, matching production cache behavior.
    await asyncio.sleep(0.11)

    third = await async_client.get("/api/v1/system/status")
    assert third.status_code == 200
    assert calls == {"database": 2, "llm": 2}


@pytest.mark.asyncio
@async_timeout(10)
async def test_status_endpoint_surfaces_cached_degraded_dependency(
    async_client,
    monkeypatch,
):
    """Cached degraded dependency status and error_message are reused within TTL."""
    calls = {"database": 0, "llm": 0}

    def slow_database_probe():
        calls["database"] += 1
        time.sleep(0.2)
        return DependencyStatus(name="database", status="healthy")

    def fast_llm_probe():
        calls["llm"] += 1
        return DependencyStatus(name="llm", status="disabled")

    monkeypatch.setattr(health_check, "_check_database_sync", slow_database_probe)
    monkeypatch.setattr(health_check, "_check_llm_sync", fast_llm_probe)
    monkeypatch.setattr(config, "dependency_status_cache_ttl_seconds", 60.0)

    first = await async_client.get("/api/v1/system/status")
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["data"]["status"] == "degraded"
    assert dependency_status(first_body, "database")["status"] == "degraded"

    second = await async_client.get("/api/v1/system/status")
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["data"]["status"] == "degraded"
    assert dependency_status(second_body, "database")["status"] == "degraded"
    assert calls == {"database": 1, "llm": 1}
    assert (
        dependency_status(second_body, "database")["error_message"]
        == dependency_status(first_body, "database")["error_message"]
    )


@pytest.mark.asyncio
@async_timeout(10)
async def test_status_endpoint_falls_back_when_dependency_probe_raises(
    async_client,
    monkeypatch,
):
    """Probe failures should hit the /status fallback path without returning 500."""
    internal_error = "dependency probe failed"

    def exploding_database_probe():
        raise RuntimeError(internal_error)

    def fast_llm_probe():
        return DependencyStatus(name="llm", status="disabled")

    monkeypatch.setattr(health_check, "_check_database_sync", exploding_database_probe)
    monkeypatch.setattr(health_check, "_check_llm_sync", fast_llm_probe)

    response = await async_client.get("/api/v1/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["dependencies"] == {}
    assert body["data"]["status"] == "degraded"
    assert body["message"] != "System status retrieved successfully"
    assert internal_error not in json.dumps(body)


__all__ = [
    "test_blocked_status_probe_does_not_block_health_endpoint",
    "test_status_dependency_probes_use_dedicated_executor_and_timeout_in_parallel",
    "test_repeated_status_polls_reuse_blocked_database_probe",
    "test_status_dependency_cache_reuses_within_ttl_then_refreshes",
    "test_status_endpoint_surfaces_cached_degraded_dependency",
    "test_status_endpoint_falls_back_when_dependency_probe_raises",
]
