"""Structured exceptions for the Roblox Alt Account Detection Engine."""

from __future__ import annotations


class AltDetectorError(Exception):
    """Base exception for all alt_detector errors."""


class InvalidInputError(AltDetectorError):
    """Raised when the user supplies a malformed username or numeric user ID."""


class UserNotFoundError(AltDetectorError):
    """Raised when a username or user ID cannot be resolved on Roblox."""


class ProfileInaccessibleError(AltDetectorError):
    """Raised when the target profile is terminated, banned, or inaccessible."""


class APIClientError(AltDetectorError):
    """Raised when an HTTP request fails after bounded retries or returns invalid data."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class RateLimitExceededError(APIClientError):
    """Raised when HTTP 429 rate limiting persists beyond bounded retry attempts."""


class AuthenticationRequiredError(APIClientError):
    """Raised when an endpoint requires authentication (HTTP 401/403) that is not used."""


class RequestBudgetExceededError(AltDetectorError):
    """Raised when the configured HTTP request budget for a single run is exhausted."""


class StorageError(AltDetectorError):
    """Raised when local SQLite storage or cache operations fail."""


class ModelError(AltDetectorError):
    """Raised when loading, saving, or evaluating a machine learning model fails."""
