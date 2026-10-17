"""Unit and integration tests for CLI execution, exit codes, JSON export, and privacy controls."""

from __future__ import annotations

import json
from pathlib import Path
import httpx

from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.cli import AltDetectionEngine, main
from alt_detector.config import EngineConfig
from alt_detector.reporting import export_json_report, format_terminal_report
from alt_detector.storage import SQLiteStorage


def test_cli_preflight_and_privacy_deletion(tmp_path: Path) -> None:
    db_file = tmp_path / "cli_test.db"

    # 1. Preflight mode returns exit code 0 without network calls
    rc_pre = main(["--preflight", "--db-path", str(db_file)])
    assert rc_pre == 0
    assert db_file.exists()

    # 2. Invalid input returns exit code 2
    rc_bad = main(["bad!username!@#", "--db-path", str(db_file)])
    assert rc_bad == 2

    # 3. Missing target without --preflight returns exit code 2
    rc_missing = main(["--db-path", str(db_file)])
    assert rc_missing == 2

    # 4. Seed a cached record and test --delete-user and --clear-cache
    with SQLiteStorage(db_file) as storage:
        storage.set_cached_response(
            "k1",
            source_name="users.profile",
            endpoint_url="https://users.roblox.com/v1/users/93910323",
            http_status=200,
            payload={"id": 93910323, "name": "witrecx64"},
        )
        assert storage.stats()["http_cache_entries"] == 1

    rc_del = main(["--delete-user", "93910323", "--db-path", str(db_file)])
    assert rc_del == 0
    with SQLiteStorage(db_file) as storage:
        assert storage.stats()["http_cache_entries"] == 0

    with SQLiteStorage(db_file) as storage:
        storage.set_cached_response(
            "k2",
            source_name="users.profile",
            endpoint_url="https://users.roblox.com/v1/users/1",
            http_status=200,
            payload={"id": 1, "name": "Roblox"},
        )
    rc_clr = main(["--clear-cache", "--db-path", str(db_file)])
    assert rc_clr == 0
    with SQLiteStorage(db_file) as storage:
        assert storage.stats()["http_cache_entries"] == 0


def test_end_to_end_engine_with_mock_http_and_json_export(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/v1/usernames/users"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "requestedUsername": "witrecx64",
                            "id": 93910323,
                            "name": "witrecx64",
                            "displayName": "SillyWitrecx",
                            "hasVerifiedBadge": False,
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
                    "description": "Main operator account",
                    "created": "2015-09-13T02:00:41.39Z",
                    "isBanned": False,
                },
            )
        if url.endswith("/v1/users/2349691539"):
            return httpx.Response(
                200,
                json={
                    "id": 2349691539,
                    "name": "witrecx",
                    "displayName": "witrecx",
                    "description": "",
                    "created": "2021-02-05T16:48:07.207Z",
                    "isBanned": False,
                },
            )
        if "/users/93910323/groups/roles" in url or "/users/2349691539/groups/roles" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "group": {
                                "id": 11668202,
                                "name": "Task Force 416",
                                "memberCount": 79,
                                "hasVerifiedBadge": False,
                            },
                            "role": {"id": 82, "name": "E8 | Master Sergeant", "rank": 165},
                        },
                        {
                            "group": {
                                "id": 592800379,
                                "name": "Site-416: Staff Team",
                                "memberCount": 445,
                                "hasVerifiedBadge": False,
                            },
                            "role": {"id": 43, "name": "Senior Moderator", "rank": 253},
                        },
                    ]
                },
            )
        if "/users/93910323/friends" in url and "count" not in url:
            return httpx.Response(
                200,
                json={"data": [{"id": 2349691539, "name": "", "displayName": ""}, {"id": 395422980, "name": "", "displayName": ""}]},
            )
        if "/users/2349691539/friends" in url and "count" not in url:
            return httpx.Response(
                200,
                json={"data": [{"id": 93910323, "name": "", "displayName": ""}, {"id": 395422980, "name": "", "displayName": ""}]},
            )
        if url.endswith("/v1/users"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": 2349691539, "name": "witrecx", "displayName": "witrecx"},
                        {"id": 395422980, "name": "MutualFriend1", "displayName": "MutualFriend1"},
                    ]
                },
            )
        if "/v1/users/search" in url:
            return httpx.Response(
                200,
                json={"data": [{"id": 2349691539, "name": "witrecx", "displayName": "witrecx"}]},
            )
        return httpx.Response(200, json={"data": [], "count": 2, "canView": False})

    cfg = EngineConfig(
        cache_dir=tmp_path,
        db_path=tmp_path / "e2e.db",
        min_request_interval_seconds=0.0,
        max_candidates=5,
        verbose=True,
    )
    with SQLiteStorage(cfg.db_path) as storage:
        with httpx.Client(transport=httpx.MockTransport(handler)) as hc:
            api_client = RobloxPublicAPIClient(cfg, storage=storage, http_client=hc)
            with AltDetectionEngine(cfg, storage=storage, api_client=api_client) as engine:
                report = engine.analyze("witrecx64")

    assert report.resolved_user_id == 93910323
    assert report.candidates_analyzed_count >= 1
    assert report.assessments[0].candidate_user_id == 2349691539
    assert report.assessments[0].confidence_band in ("Strong evidence", "Moderate evidence")

    txt = format_terminal_report(report, verbose=True)
    assert "ROBLOX ALTERNATIVE ACCOUNT DETECTION ENGINE" in txt
    assert "witrecx (ID: 2349691539" in txt

    json_file = tmp_path / "report.json"
    export_json_report(report, json_file)
    parsed = json.loads(json_file.read_text(encoding="utf-8"))
    assert parsed["resolved_user_id"] == 93910323
    assert parsed["schema_version"] == "1.0.0"
