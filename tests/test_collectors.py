"""Unit tests for PublicDataCollector input validation, resolution, pagination, and partial results."""

from __future__ import annotations

from pathlib import Path
import httpx
import pytest

from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.collectors import PublicDataCollector, normalize_identifier
from alt_detector.config import EngineConfig
from alt_detector.exceptions import InvalidInputError, UserNotFoundError


def test_normalize_identifier_valid_and_invalid() -> None:
    assert normalize_identifier("witrecx64") == ("username", "witrecx64")
    assert normalize_identifier("  @Silly_User_123 ") == ("username", "Silly_User_123")
    assert normalize_identifier("93910323") == ("id", 93910323)

    for invalid in ["", "   ", "@", "a", "too_long_username_123456789", "bad!char", "user space", "0", "-5"]:
        with pytest.raises(InvalidInputError):
            normalize_identifier(invalid)


def test_resolve_account_username_and_id(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/v1/usernames/users"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "requestedUsername": "witrecx64",
                            "hasVerifiedBadge": False,
                            "id": 93910323,
                            "name": "witrecx64",
                            "displayName": "SillyWitrecx",
                        }
                    ]
                },
            )
        if url.endswith("/v1/users/93910323"):
            return httpx.Response(
                200,
                json={
                    "id": 93910323,
                    "name": "witrecx64",
                    "displayName": "SillyWitrecx",
                    "description": "Bio",
                    "created": "2015-09-13T02:00:41.39Z",
                    "isBanned": False,
                },
            )
        return httpx.Response(404, json={"errors": []})

    cfg = EngineConfig(cache_dir=tmp_path, use_cache=False, min_request_interval_seconds=0.0)
    with httpx.Client(transport=httpx.MockTransport(handler)) as hc:
        client = RobloxPublicAPIClient(cfg, http_client=hc)
        collector = PublicDataCollector(client, cfg)

        uid, uname, dname = collector.resolve_account("witrecx64")
        assert (uid, uname, dname) == (93910323, "witrecx64", "SillyWitrecx")

        uid2, uname2, dname2 = collector.resolve_account("93910323")
        assert (uid2, uname2, dname2) == (93910323, "witrecx64", "SillyWitrecx")

        with pytest.raises(UserNotFoundError):
            collector.resolve_account("404040404")


def test_group_members_pagination_and_partial_collection(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/groups/235594128/users" in url:
            if "cursor=page2" in url:
                return httpx.Response(
                    200,
                    json={
                        "previousPageCursor": "page1",
                        "nextPageCursor": None,
                        "data": [
                            {
                                "user": {"userId": 202, "username": "MemberTwo", "displayName": "MemberTwo"},
                                "role": {"id": 2, "name": "VIP", "rank": 2},
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "previousPageCursor": None,
                    "nextPageCursor": "page2",
                    "data": [
                        {
                            "user": {"userId": 101, "username": "MemberOne", "displayName": "MemberOne"},
                            "role": {"id": 1, "name": "Owner", "rank": 255},
                        }
                    ],
                },
            )
        if url.endswith("/v1/users/93910323"):
            return httpx.Response(
                200,
                json={
                    "id": 93910323,
                    "name": "witrecx64",
                    "displayName": "SillyWitrecx",
                    "description": "Test",
                    "created": "2015-09-13T02:00:41.39Z",
                    "isBanned": False,
                },
            )
        if "/groups/roles" in url:
            return httpx.Response(200, json={"data": []})
        if url.endswith("/friends"):
            return httpx.Response(200, json={"data": [{"id": 2349691539, "name": "", "displayName": ""}]})
        if url.endswith("/v1/users"):
            return httpx.Response(
                200,
                json={"data": [{"id": 2349691539, "name": "witrecx", "displayName": "witrecx"}]},
            )
        if "/username-history" in url:
            # Simulate rate-limit 429 on optional endpoint to verify graceful partial-data handling
            return httpx.Response(429, json={"errors": [{"code": 0, "message": "Throttled"}]})
        return httpx.Response(200, json={"data": [], "count": 5, "canView": False})

    cfg = EngineConfig(
        cache_dir=tmp_path,
        use_cache=False,
        max_retries=0,
        min_request_interval_seconds=0.0,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as hc:
        client = RobloxPublicAPIClient(cfg, http_client=hc)
        collector = PublicDataCollector(client, cfg)

        members, attrs = collector.fetch_group_members(235594128, limit=10, max_pages=2)
        assert len(members) == 2
        assert len(attrs) == 2
        assert [m["user"]["userId"] for m in members] == [101, 202]

        obs = collector.collect_account(93910323, include_extended=True, enrich_friend_names=True)
        assert obs.user_id == 93910323
        assert obs.friend_names[2349691539] == ("witrecx", "witrecx")
        # Verify partial-result resilience: username_history 429 recorded in inaccessible_sources
        assert obs.previous_usernames is None
        assert "users.username_history" in obs.inaccessible_sources
