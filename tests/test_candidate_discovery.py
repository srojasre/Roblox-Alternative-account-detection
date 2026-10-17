"""Unit tests for candidate account discovery, deduplication, and priority ranking."""

from __future__ import annotations

from pathlib import Path
import httpx

from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.candidate_discovery import CandidateDiscoveryEngine, extract_lexical_stems
from alt_detector.collectors import PublicDataCollector
from alt_detector.config import EngineConfig
from tests.fixtures.sample_accounts import make_target_account


def test_extract_lexical_stems() -> None:
    stems = extract_lexical_stems("witrecx64", "SillyWitrecx", ["OldWitrecx_01"])
    assert "witrecx" in [s.lower() for s in stems]
    # Generic prefix 'silly' should not be emitted as a standalone search stem
    assert "silly" not in [s.lower() for s in stems]


def test_candidate_discovery_deduplication_and_traceability(tmp_path: Path) -> None:
    target = make_target_account()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/users/search" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": 93910323, "name": "witrecx64", "displayName": "SillyWitrecx"},
                        {"id": 2349691539, "name": "witrecx", "displayName": "witrecx"},
                        {"id": 8814991503, "name": "witrecx_416", "displayName": "witrecx_416"},
                    ]
                },
            )
        if "/groups/235594128/users" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "user": {"userId": 93910323, "username": "witrecx64", "displayName": "SillyWitrecx"},
                            "role": {"id": 1, "name": "Owner", "rank": 255},
                        },
                        {
                            "user": {"userId": 1719913571, "username": "mineturtle_416", "displayName": "SillyLittleTurtle"},
                            "role": {"id": 2, "name": "VIP", "rank": 2},
                        },
                    ]
                },
            )
        if "/groups/11668202/users" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "user": {"userId": 2349691539, "username": "witrecx", "displayName": "witrecx"},
                            "role": {"id": 82, "name": "E8 | Master Sergeant", "rank": 165},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"data": []})

    cfg = EngineConfig(
        cache_dir=tmp_path,
        use_cache=False,
        min_request_interval_seconds=0.0,
        max_candidates=15,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as hc:
        client = RobloxPublicAPIClient(cfg, http_client=hc)
        collector = PublicDataCollector(client, cfg)
        engine = CandidateDiscoveryEngine(collector, cfg)

        candidates = engine.discover_candidates(target)
        cand_ids = [c.user_id for c in candidates]

        # Target itself is excluded, and no duplicate user IDs exist
        assert 93910323 not in cand_ids
        assert len(cand_ids) == len(set(cand_ids))

        # Every candidate must have at least one traceable reason
        for cand in candidates:
            assert len(cand.reasons) >= 1
            for r in cand.reasons:
                assert r.source_endpoint
                assert r.detail

        # 2349691539 (witrecx) was discovered via lexical_search, direct_friend, small_group_co_member, AND favorite_game_creator
        witrecx_rec = next(c for c in candidates if c.user_id == 2349691539)
        strategies = {r.strategy for r in witrecx_rec.reasons}
        assert "lexical_search" in strategies
        assert "direct_friend" in strategies
        assert "small_group_co_member" in strategies
        assert candidates[0].user_id == 2349691539
