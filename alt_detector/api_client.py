"""Resilient HTTP client for verified Roblox public web APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import random
import threading
import time
from typing import Any

import httpx

from alt_detector.config import EngineConfig
from alt_detector.exceptions import (
    APIClientError,
    AuthenticationRequiredError,
    RateLimitExceededError,
    RequestBudgetExceededError,
)
from alt_detector.schemas import DataSourceAttribution
from alt_detector.storage import SQLiteStorage

logger = logging.getLogger("alt_detector")


@dataclass
class APIResponseResult:
    """Normalized wrapper for a verified API response and its provenance."""

    status_code: int
    data: Any
    attribution: DataSourceAttribution


@dataclass
class ClientTelemetry:
    """Runtime statistics for HTTP requests, caching, and rate-limit handling."""

    requests_sent: int = 0
    cache_hits: int = 0
    retries_performed: int = 0
    rate_limits_encountered: int = 0
    status_code_counts: dict[int, int] = field(default_factory=dict)
    unique_endpoints: set[str] = field(default_factory=set)
    inaccessible_endpoints: dict[str, str] = field(default_factory=dict)


class RobloxPublicAPIClient:
    """HTTP client enforcing timeouts, bounded exponential backoff, rate limits, and budgets.

    Strictly queries public interfaces without bypassing authentication, CAPTCHAs,
    or rate limits. Reuses TCP/TLS connections via a persistent `httpx.Client`.
    """

    def __init__(
        self,
        config: EngineConfig,
        storage: SQLiteStorage | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.telemetry = ClientTelemetry()
        self._rng = random.Random(config.random_seed)
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(config.request_timeout_seconds),
            headers={
                "Accept": "application/json",
                "User-Agent": "RobloxAltDetector-LocalOSINT/1.0 (Academic/Research CLI)",
            },
            limits=httpx.Limits(
                max_connections=max(4, config.max_concurrency * 2),
                max_keepalive_connections=max(2, config.max_concurrency),
            ),
        )

    def close(self) -> None:
        """Close the underlying HTTP transport if owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RobloxPublicAPIClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _pace_request(self) -> None:
        """Ensure conservative spacing between outbound live HTTP requests."""
        with self._lock:
            if self.telemetry.requests_sent >= self.config.request_budget:
                raise RequestBudgetExceededError(
                    f"Configured HTTP request budget ({self.config.request_budget}) reached"
                )
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait = self.config.min_request_interval_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_time = time.monotonic()
            self.telemetry.requests_sent += 1

    def _compute_backoff(self, attempt: int, retry_after_header: str | None = None) -> float:
        if retry_after_header:
            try:
                parsed = float(retry_after_header.strip())
                if parsed > 0:
                    return min(self.config.max_backoff_seconds, parsed)
            except ValueError:
                pass
        exp = self.config.base_backoff_seconds * (2**attempt)
        jitter = self._rng.uniform(0.05, 0.35 * self.config.base_backoff_seconds)
        return min(self.config.max_backoff_seconds, exp + jitter)

    def request_json(
        self,
        method: str,
        url: str,
        *,
        source_name: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_404: bool = False,
        cache_ttl_seconds: int | None = None,
        max_retries_override: int | None = None,
    ) -> APIResponseResult:
        """Execute a GET or POST request with caching, schema validation, and retries."""
        method_upper = method.upper()
        full_url = url
        if params:
            req_preview = self._client.build_request(method_upper, url, params=params)
            full_url = str(req_preview.url)

        cache_key = SQLiteStorage.make_cache_key(method_upper, full_url, json_body)
        if self.config.use_cache and self.storage is not None:
            cached = self.storage.get_cached_response(cache_key)
            if cached is not None:
                with self._lock:
                    self.telemetry.cache_hits += 1
                    self.telemetry.unique_endpoints.add(source_name)
                return APIResponseResult(
                    status_code=cached["http_status"],
                    data=cached["payload"],
                    attribution=DataSourceAttribution(
                        source_name=source_name,
                        endpoint_url=full_url,
                        retrieved_at=cached["retrieved_at"],
                        from_cache=True,
                        http_status=cached["http_status"],
                    ),
                )

        effective_retries = (
            self.config.max_retries
            if max_retries_override is None
            else max(0, int(max_retries_override))
        )
        max_attempts = effective_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            self._pace_request()
            try:
                response = self._client.request(
                    method_upper,
                    url,
                    params=params,
                    json=json_body,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    delay = self._compute_backoff(attempt)
                    with self._lock:
                        self.telemetry.retries_performed += 1
                    logger.warning(
                        "Network error on %s (%s); retry %d/%d in %.2fs",
                        source_name,
                        exc.__class__.__name__,
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise APIClientError(
                    f"Request to {full_url} failed after {max_attempts} attempts: {exc}",
                    url=full_url,
                ) from exc

            status = response.status_code
            with self._lock:
                self.telemetry.status_code_counts[status] = (
                    self.telemetry.status_code_counts.get(status, 0) + 1
                )
                self.telemetry.unique_endpoints.add(source_name)

            if status == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise APIClientError(
                        f"Malformed JSON from {full_url} (HTTP 200)",
                        url=full_url,
                        status_code=status,
                    ) from exc

                retrieved_at = datetime.now(tz=timezone.utc).isoformat()
                if self.config.use_cache and self.storage is not None:
                    retrieved_at = self.storage.set_cached_response(
                        cache_key,
                        source_name=source_name,
                        endpoint_url=full_url,
                        http_status=status,
                        payload=payload,
                        ttl_seconds=cache_ttl_seconds,
                    )
                return APIResponseResult(
                    status_code=status,
                    data=payload,
                    attribution=DataSourceAttribution(
                        source_name=source_name,
                        endpoint_url=full_url,
                        retrieved_at=retrieved_at,
                        from_cache=False,
                        http_status=status,
                    ),
                )

            if status == 404 and allow_404:
                retrieved_at = datetime.now(tz=timezone.utc).isoformat()
                return APIResponseResult(
                    status_code=404,
                    data=None,
                    attribution=DataSourceAttribution(
                        source_name=source_name,
                        endpoint_url=full_url,
                        retrieved_at=retrieved_at,
                        from_cache=False,
                        http_status=404,
                    ),
                )

            if status in (401, 403):
                reason = f"HTTP {status} (authentication or privacy restriction)"
                with self._lock:
                    self.telemetry.inaccessible_endpoints[source_name] = reason
                raise AuthenticationRequiredError(
                    f"Endpoint {source_name} returned HTTP {status} (restricted/private)",
                    url=full_url,
                    status_code=status,
                )

            if status == 429:
                with self._lock:
                    self.telemetry.rate_limits_encountered += 1
                if attempt < max_attempts - 1:
                    delay = self._compute_backoff(
                        attempt,
                        response.headers.get("Retry-After"),
                    )
                    with self._lock:
                        self.telemetry.retries_performed += 1
                    logger.warning(
                        "HTTP 429 rate limit on %s; retry %d/%d after %.2fs backoff",
                        source_name,
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                with self._lock:
                    self.telemetry.inaccessible_endpoints[source_name] = (
                        "HTTP 429 (rate limit exceeded)"
                    )
                raise RateLimitExceededError(
                    f"Rate limit (HTTP 429) persisted on {source_name} after {max_attempts} attempts",
                    url=full_url,
                    status_code=429,
                )

            if status >= 500:
                if attempt < max_attempts - 1:
                    delay = self._compute_backoff(attempt)
                    with self._lock:
                        self.telemetry.retries_performed += 1
                    logger.warning(
                        "HTTP %d server error on %s; retry %d/%d in %.2fs",
                        status,
                        source_name,
                        attempt + 1,
                        self.config.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise APIClientError(
                    f"Server error HTTP {status} from {full_url}",
                    url=full_url,
                    status_code=status,
                )

            # Client error such as HTTP 400 or unexpected 404
            raise APIClientError(
                f"HTTP {status} returned by {source_name} ({full_url})",
                url=full_url,
                status_code=status,
            )

        raise APIClientError(
            f"Exhausted retries for {full_url}: {last_error}",
            url=full_url,
        )
