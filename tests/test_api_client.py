"""Unit tests for RobloxPublicAPIClient and SQLiteStorage caching/TTL."""

from __future__ import annotations

from pathlib import Path
import httpx
import pytest

from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.config import EngineConfig
from alt_detector.exceptions import (
    APIClientError,
    AuthenticationRequiredError,
    RateLimitExceededError,
    RequestBudgetExceededError,
)
from alt_detector.storage import SQLiteStorage


def _make_config(tmp_path: Path, **overrides: object) -> EngineConfig:
    params = {
        "cache_dir": tmp_path,
        "db_path": tmp_path / "test_cache.db",
        "max_retries": 2,
        "base_backoff_seconds": 0.01,
        "max_backoff_seconds": 0.05,
        "min_request_interval_seconds": 0.0,
        "request_budget": 20,
    }
    params.update(overrides)
    return EngineConfig(**params)  # type: ignore[arg-type]


def test_http_200_and_sqlite_cache_ttl(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"id": 93910323, "name": "witrecx64"})

    cfg = _make_config(tmp_path, cache_ttl_seconds=60)
    transport = httpx.MockTransport(handler)
    with SQLiteStorage(cfg.db_path, default_ttl_seconds=60) as storage:
        with httpx.Client(transport=transport) as http_client:
            client = RobloxPublicAPIClient(cfg, storage=storage, http_client=http_client)
            res1 = client.request_json(
                "GET",
                "https://users.roblox.com/v1/users/93910323",
                source_name="users.profile",
            )
            assert res1.status_code == 200
            assert res1.attribution.from_cache is False
            assert calls == 1

            # Second call hits SQLite cache
            res2 = client.request_json(
                "GET",
                "https://users.roblox.com/v1/users/93910323",
                source_name="users.profile",
            )
            assert res2.status_code == 200
            assert res2.attribution.from_cache is True
            assert calls == 1
            assert client.telemetry.cache_hits == 1

            # Simulate expiration
            key = SQLiteStorage.make_cache_key("GET", "https://users.roblox.com/v1/users/93910323")
            expired = storage.get_cached_response(key, now_epoch=9_999_999_999.0)
            assert expired is None


def test_http_401_and_403_raise_authentication_required(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "badges" in str(request.url):
            return httpx.Response(
                401,
                json={"errors": [{"code": 9002, "message": "Authentication token is missing"}]},
            )
        return httpx.Response(403, json={"errors": [{"code": 0, "message": "Forbidden"}]})

    cfg = _make_config(tmp_path, use_cache=False)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = RobloxPublicAPIClient(cfg, http_client=http_client)
        with pytest.raises(AuthenticationRequiredError) as exc401:
            client.request_json(
                "GET",
                "https://badges.roblox.com/v1/users/1/badges",
                source_name="badges.user_badges",
            )
        assert exc401.value.status_code == 401

        with pytest.raises(AuthenticationRequiredError) as exc403:
            client.request_json(
                "GET",
                "https://inventory.roblox.com/v2/users/1/inventory/8",
                source_name="inventory.hats",
            )
        assert exc403.value.status_code == 403
        assert "badges.user_badges" in client.telemetry.inaccessible_endpoints


def test_http_404_handling(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": [{"code": 3, "message": "Not found"}]})

    cfg = _make_config(tmp_path, use_cache=False)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = RobloxPublicAPIClient(cfg, http_client=http_client)
        res = client.request_json(
            "GET",
            "https://users.roblox.com/v1/users/99999999999",
            source_name="users.profile",
            allow_404=True,
        )
        assert res.status_code == 404
        assert res.data is None

        with pytest.raises(APIClientError) as exc:
            client.request_json(
                "GET",
                "https://users.roblox.com/v1/users/99999999999",
                source_name="users.profile",
                allow_404=False,
            )
        assert exc.value.status_code == 404


def test_http_429_retry_and_exhaustion(tmp_path: Path) -> None:
    attempts = 0

    def recovering_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, json={"errors": []})
        return httpx.Response(200, json={"ok": True})

    cfg = _make_config(tmp_path, use_cache=False, max_retries=2)
    with httpx.Client(transport=httpx.MockTransport(recovering_handler)) as http_client:
        client = RobloxPublicAPIClient(cfg, http_client=http_client)
        res = client.request_json("GET", "https://users.roblox.com/v1/users/1", source_name="test.429")
        assert res.status_code == 200
        assert client.telemetry.rate_limits_encountered == 1
        assert client.telemetry.retries_performed == 1

    # Persistent 429 raises RateLimitExceededError
    def persistent_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"errors": [{"code": 0, "message": "TooManyRequests"}]})

    with httpx.Client(transport=httpx.MockTransport(persistent_429)) as http_client:
        client2 = RobloxPublicAPIClient(cfg, http_client=http_client)
        with pytest.raises(RateLimitExceededError):
            client2.request_json("GET", "https://users.roblox.com/v1/users/1", source_name="test.429_fail")


def test_http_5xx_and_timeout_retries(tmp_path: Path) -> None:
    step = 0

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        if step == 1:
            raise httpx.ReadTimeout("Simulated timeout", request=request)
        if step == 2:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, json={"recovered": True})

    cfg = _make_config(tmp_path, use_cache=False, max_retries=3)
    with httpx.Client(transport=httpx.MockTransport(flaky_handler)) as http_client:
        client = RobloxPublicAPIClient(cfg, http_client=http_client)
        res = client.request_json("GET", "https://users.roblox.com/v1/users/1", source_name="test.flaky")
        assert res.data == {"recovered": True}
        assert client.telemetry.retries_performed == 2


def test_malformed_json_and_request_budget(tmp_path: Path) -> None:
    def bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<<<not-valid-json>>>")

    cfg = _make_config(tmp_path, use_cache=False, request_budget=5)
    with httpx.Client(transport=httpx.MockTransport(bad_json_handler)) as http_client:
        client = RobloxPublicAPIClient(cfg, http_client=http_client)
        with pytest.raises(APIClientError, match="Malformed JSON"):
            client.request_json("GET", "https://users.roblox.com/v1/users/1", source_name="test.bad_json")

    # Exhaust budget
    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": 1})

    with httpx.Client(transport=httpx.MockTransport(ok_handler)) as http_client:
        client_b = RobloxPublicAPIClient(cfg, http_client=http_client)
        for i in range(5):
            client_b.request_json("GET", f"https://users.roblox.com/v1/users/{i+1}", source_name="test.budget")
        with pytest.raises(RequestBudgetExceededError):
            client_b.request_json("GET", "https://users.roblox.com/v1/users/99", source_name="test.budget")
