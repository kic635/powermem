"""Shared fixtures and helpers for /system/status dependency probe endpoint tests."""

import asyncio
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio

from server.config import config
from server.models.response import DependencyStatus
from server.utils import health_check


def wait_for_in_flight(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        futures = list(health_check._DEPENDENCY_STATUS_IN_FLIGHT.values())
        if not futures or all(future.done() for future in futures):
            return
        time.sleep(0.05)
    pending = [
        name
        for name, future in health_check._DEPENDENCY_STATUS_IN_FLIGHT.items()
        if not future.done()
    ]
    if pending:
        raise RuntimeError(
            f"dependency probes still in flight after {timeout:g}s: {pending}"
        )


async def async_wait_for_in_flight(timeout: float = 5.0) -> None:
    """Wait without blocking the event loop (sync fixtures must not use time.sleep)."""
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


def reset_probe_executor() -> None:
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


def clear_probe_state() -> None:
    health_check._DEPENDENCY_STATUS_CACHE.clear()
    health_check._DEPENDENCY_STATUS_LOCKS.clear()
    health_check._DEPENDENCY_STATUS_IN_FLIGHT.clear()


def async_timeout(seconds: float = 10):
    def decorator(test_fn):
        @functools.wraps(test_fn)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(test_fn(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


@pytest_asyncio.fixture
async def isolated_dependency_probe_state():
    await async_wait_for_in_flight()
    clear_probe_state()
    reset_probe_executor()
    yield
    await async_wait_for_in_flight()
    clear_probe_state()
    reset_probe_executor()


@pytest.fixture
def status_endpoint_settings(monkeypatch):
    """Status endpoint test defaults: auth off plus bounded probe timeout/TTL."""
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
    # Auth is disabled in these tests; bypass the sync Security dependency so
    # /status does not compete for Starlette's default thread pool in CI.
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


def dependency_status(body: dict, name: str) -> dict:
    data = body.get("data")
    assert data is not None, f"response missing data: {body!r}"
    dependencies = data.get("dependencies")
    assert dependencies is not None, f"response missing dependencies: {body!r}"
    dep = dependencies.get(name)
    assert dep is not None, f"response missing dependency {name!r}: {body!r}"
    return dep


async def await_threading_event(event: threading.Event, timeout: float = 1.0) -> bool:
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, event.wait, timeout),
            timeout=timeout + 0.1,
        )
    except asyncio.TimeoutError:
        pass
    return event.is_set()


