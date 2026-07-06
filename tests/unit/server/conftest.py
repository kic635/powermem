"""Fixtures for server endpoint tests."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio

from server.config import config
from server.utils import health_check


async def _async_wait_for_in_flight(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        futures = list(health_check._DEPENDENCY_STATUS_IN_FLIGHT.values())
        if not futures or all(future.done() for future in futures):
            return
        await asyncio.sleep(0.05)
    pending = [
        name
        for name, future in health_check._DEPENDENCY_STATUS_IN_FLIGHT.items()
        if not future.done()
    ]
    if pending:
        raise RuntimeError(
            f"dependency probes still in flight after {timeout:g}s: {pending}"
        )


def _reset_probe_executor() -> None:
    executor = health_check._DEPENDENCY_PROBE_EXECUTOR
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except Exception:
        pass
    finally:
        health_check._DEPENDENCY_PROBE_EXECUTOR = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="powermem-dependency-probe",
        )


def _clear_probe_state() -> None:
    health_check._DEPENDENCY_STATUS_CACHE.clear()
    health_check._DEPENDENCY_STATUS_LOCKS.clear()
    health_check._DEPENDENCY_STATUS_IN_FLIGHT.clear()


@pytest_asyncio.fixture
async def isolated_dependency_probe_state():
    await _async_wait_for_in_flight()
    _clear_probe_state()
    _reset_probe_executor()
    yield
    await _async_wait_for_in_flight()
    _clear_probe_state()
    _reset_probe_executor()


@pytest.fixture
def status_endpoint_settings(monkeypatch):
    monkeypatch.setattr(config, "auth_enabled", False)
    monkeypatch.setattr(config, "dependency_check_timeout_seconds", 0.05)
    monkeypatch.setattr(config, "dependency_status_cache_ttl_seconds", 10.0)


@pytest.fixture
def system_app(status_endpoint_settings, monkeypatch):
    pytest.importorskip("fastapi", exc_type=ImportError)
    from fastapi import FastAPI

    from server.api.v1.system import router
    from server.middleware.auth import verify_api_key

    monkeypatch.setattr(
        "server.api.v1.system.auto_config",
        lambda: {
            "vector_store": {"provider": "sqlite"},
            "llm": {"provider": "noop"},
        },
    )

    app = FastAPI()
    app.state.service_ready = True
    app.state.storage_type = "sqlite"
    app.state.service_startup_error = None
    app.dependency_overrides[verify_api_key] = lambda: "anonymous"
    app.include_router(router, prefix="/api/v1")
    return app


@pytest_asyncio.fixture
async def async_client(system_app):
    httpx = pytest.importorskip("httpx", exc_type=ImportError)
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=system_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
