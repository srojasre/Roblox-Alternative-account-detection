"""Modular public data collection layer for verified Roblox endpoints."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import ValidationError

from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.config import EngineConfig
from alt_detector.exceptions import (
    APIClientError,
    AuthenticationRequiredError,
    InvalidInputError,
    ProfileInaccessibleError,
    RateLimitExceededError,
    RequestBudgetExceededError,
    UserNotFoundError,
)
from alt_detector.schemas import (
    AccountObservation,
    AvatarSummary,
    DataSourceAttribution,
    GameRecord,
    GroupMembershipRecord,
    InventorySampleRecord,
    RobloxMultiUserResponse,
    RobloxUserGroupsResponse,
    RobloxUserProfileResponse,
)
from alt_detector.storage import SQLiteStorage

logger = logging.getLogger("alt_detector")

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{2,20}$")

# Endpoints verified to require cookie/OAuth authentication as of 2026
KNOWN_AUTH_RESTRICTED_SOURCES: dict[str, str] = {
    "badges.user_badges": (
        "https://badges.roblox.com/v1/users/{userId}/badges requires authentication (HTTP 401 code 9002)"
    ),
    "friends.followers_list": (
        "https://friends.roblox.com/v1/users/{userId}/followers requires authentication (HTTP 401 code 9002)"
    ),
    "friends.followings_list": (
        "https://friends.roblox.com/v1/users/{userId}/followings requires authentication (HTTP 401 code 9002)"
    ),
}


def normalize_identifier(identifier: str) -> tuple[Literal["id", "username"], int | str]:
    """Validate and normalize a CLI input identifier (username or numeric user ID)."""
    if not isinstance(identifier, str):
        raise InvalidInputError("Identifier must be a string")
    cleaned = identifier.strip()
    if cleaned.startswith("@"):
        cleaned = cleaned[1:].strip()
    if not cleaned:
        raise InvalidInputError("Identifier cannot be empty")

    if cleaned.isdigit():
        uid = int(cleaned)
        if uid <= 0:
            raise InvalidInputError(f"Numeric user ID must be positive: {identifier}")
        return "id", uid

    if not _USERNAME_RE.match(cleaned):
        raise InvalidInputError(
            f"Invalid Roblox username '{identifier}': must be 2-20 alphanumeric or underscore characters"
        )
    return "username", cleaned


class PublicDataCollector:
    """Collects verified public observations for Roblox accounts with partial-result resilience."""

    def __init__(
        self,
        client: RobloxPublicAPIClient,
        config: EngineConfig,
        storage: SQLiteStorage | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.storage = storage
        self._throttled_optional_sources: set[str] = set()

    def resolve_account(self, identifier: str) -> tuple[int, str, str]:
        """Resolve a username or numeric ID to `(user_id, canonical_username, display_name)`."""
        kind, value = normalize_identifier(identifier)
        if kind == "id":
            uid = int(value)
            res = self.client.request_json(
                "GET",
                f"https://users.roblox.com/v1/users/{uid}",
                source_name="users.get_user_by_id",
                allow_404=True,
            )
            if res.status_code == 404 or not res.data:
                raise UserNotFoundError(f"Roblox user ID {uid} does not exist")
            try:
                parsed = RobloxUserProfileResponse.model_validate(res.data)
            except ValidationError as exc:
                raise APIClientError(f"Invalid profile schema for user ID {uid}: {exc}") from exc
            return parsed.id, parsed.name, parsed.displayName or parsed.name

        username = str(value)
        res = self.client.request_json(
            "POST",
            "https://users.roblox.com/v1/usernames/users",
            source_name="users.resolve_usernames",
            json_body={"usernames": [username], "excludeBannedUsers": False},
        )
        try:
            parsed_multi = RobloxMultiUserResponse.model_validate(res.data)
        except ValidationError as exc:
            raise APIClientError(f"Invalid response schema from username resolution: {exc}") from exc

        if not parsed_multi.data:
            raise UserNotFoundError(f"Roblox username '{username}' could not be resolved")

        item = parsed_multi.data[0]
        return item.id, item.name, item.displayName or item.name

    def batch_resolve_user_ids(
        self,
        user_ids: list[int],
    ) -> dict[int, tuple[str, str, bool]]:
        """Batch-resolve numeric user IDs via `POST https://users.roblox.com/v1/users`."""
        unique_ids = sorted({int(uid) for uid in user_ids if int(uid) > 0})
        resolved: dict[int, tuple[str, str, bool]] = {}
        for idx in range(0, len(unique_ids), 100):
            chunk = unique_ids[idx : idx + 100]
            try:
                res = self.client.request_json(
                    "POST",
                    "https://users.roblox.com/v1/users",
                    source_name="users.batch_get_by_ids",
                    json_body={"userIds": chunk, "excludeBannedUsers": False},
                )
                parsed = RobloxMultiUserResponse.model_validate(res.data)
                for item in parsed.data:
                    resolved[item.id] = (
                        item.name,
                        item.displayName or item.name,
                        item.hasVerifiedBadge,
                    )
            except (APIClientError, ValidationError, RequestBudgetExceededError) as exc:
                logger.warning("Batch user ID resolution partial failure: %s", exc)
                break
        return resolved

    def collect_account(
        self,
        user_id: int,
        *,
        include_extended: bool = True,
        enrich_friend_names: bool = False,
        require_accessible: bool = False,
    ) -> AccountObservation:
        """Collect all permitted, publicly accessible profile signals for `user_id`."""
        sources: list[DataSourceAttribution] = []
        inaccessible: dict[str, str] = dict(KNOWN_AUTH_RESTRICTED_SOURCES)

        # 1. Core profile metadata
        res_profile = self.client.request_json(
            "GET",
            f"https://users.roblox.com/v1/users/{user_id}",
            source_name="users.profile",
            allow_404=True,
        )
        if res_profile.status_code == 404 or not res_profile.data:
            raise UserNotFoundError(f"User ID {user_id} was not found")

        try:
            profile = RobloxUserProfileResponse.model_validate(res_profile.data)
        except ValidationError as exc:
            raise APIClientError(f"Schema validation failed on user {user_id}: {exc}") from exc

        sources.append(res_profile.attribution)
        if profile.isBanned and require_accessible:
            logger.warning("Target account %s (%d) is marked as banned/terminated", profile.name, user_id)

        obs = AccountObservation(
            user_id=profile.id,
            username=profile.name,
            display_name=profile.displayName or profile.name,
            description=profile.description if profile.description is not None else "",
            created_at=profile.created,
            is_banned=profile.isBanned,
            has_verified_badge=profile.hasVerifiedBadge,
            sources=sources,
            inaccessible_sources=inaccessible,
        )

        # 2. Public group memberships & roles
        try:
            res_groups = self.client.request_json(
                "GET",
                f"https://groups.roblox.com/v1/users/{user_id}/groups/roles",
                source_name="groups.user_roles",
            )
            sources.append(res_groups.attribution)
            parsed_groups = RobloxUserGroupsResponse.model_validate(res_groups.data)
            obs.groups = [
                GroupMembershipRecord(
                    group_id=entry.group.id,
                    group_name=entry.group.name,
                    member_count=entry.group.memberCount,
                    role_id=entry.role.id,
                    role_name=entry.role.name,
                    role_rank=entry.role.rank,
                    owner_id=entry.group.owner.userId if entry.group.owner else None,
                    owner_username=entry.group.owner.username if entry.group.owner else None,
                    has_verified_badge=entry.group.hasVerifiedBadge,
                )
                for entry in parsed_groups.data
            ]
        except (APIClientError, ValidationError, RequestBudgetExceededError) as exc:
            obs.inaccessible_sources["groups.user_roles"] = str(exc)

        # 3. Public friends list
        try:
            res_friends = self.client.request_json(
                "GET",
                f"https://friends.roblox.com/v1/users/{user_id}/friends",
                source_name="friends.list",
            )
            sources.append(res_friends.attribution)
            raw_friends = res_friends.data.get("data", []) if isinstance(res_friends.data, dict) else []
            friend_ids: list[int] = []
            friend_names: dict[int, tuple[str, str]] = {}
            for item in raw_friends:
                if isinstance(item, dict) and isinstance(item.get("id"), int):
                    fid = int(item["id"])
                    friend_ids.append(fid)
                    fname = str(item.get("name") or "").strip()
                    fdname = str(item.get("displayName") or "").strip()
                    if fname:
                        friend_names[fid] = (fname, fdname or fname)
            obs.friends = friend_ids
            obs.friends_count = len(friend_ids)

            if enrich_friend_names and friend_ids:
                missing_name_ids = [fid for fid in friend_ids if fid not in friend_names]
                if missing_name_ids:
                    resolved_map = self.batch_resolve_user_ids(missing_name_ids)
                    for fid, (fname, fdname, _) in resolved_map.items():
                        friend_names[fid] = (fname, fdname)
            obs.friend_names = friend_names
        except (APIClientError, RequestBudgetExceededError) as exc:
            obs.inaccessible_sources["friends.list"] = str(exc)

        # 4. Public avatar details
        try:
            res_avatar = self.client.request_json(
                "GET",
                f"https://avatar.roblox.com/v1/users/{user_id}/avatar",
                source_name="avatar.details",
            )
            sources.append(res_avatar.attribution)
            if isinstance(res_avatar.data, dict):
                av_data = res_avatar.data
                raw_assets = av_data.get("assets", []) or []
                asset_ids: list[int] = []
                asset_names: dict[int, str] = {}
                asset_types: dict[int, str] = {}
                for asset in raw_assets:
                    if isinstance(asset, dict) and isinstance(asset.get("id"), int):
                        aid = int(asset["id"])
                        asset_ids.append(aid)
                        asset_names[aid] = str(asset.get("name") or "")
                        atype = asset.get("assetType")
                        if isinstance(atype, dict):
                            asset_types[aid] = str(atype.get("name") or "")
                scales = {
                    str(k): float(v)
                    for k, v in (av_data.get("scales") or {}).items()
                    if isinstance(v, (int, float))
                }
                body_colors = {
                    str(k): int(v)
                    for k, v in (av_data.get("bodyColors") or {}).items()
                    if isinstance(v, int)
                }
                obs.avatar = AvatarSummary(
                    avatar_type=str(av_data.get("playerAvatarType") or "Unknown"),
                    scales=scales,
                    body_colors=body_colors,
                    asset_ids=sorted(set(asset_ids)),
                    asset_names=asset_names,
                    asset_types=asset_types,
                )
        except (APIClientError, RequestBudgetExceededError) as exc:
            obs.inaccessible_sources["avatar.details"] = str(exc)

        if include_extended:
            self._collect_extended_signals(obs)

        if self.config.store_account_summaries and self.storage is not None:
            try:
                self.storage.save_account_observation(obs)
            except Exception as exc:
                logger.debug("Could not persist account observation for %d: %s", user_id, exc)

        if require_accessible and obs.is_banned and not obs.groups and not obs.friends:
            raise ProfileInaccessibleError(
                f"Account {obs.username} ({obs.user_id}) is banned/terminated and has no accessible public associations"
            )
        return obs

    def _collect_extended_signals(self, obs: AccountObservation) -> None:
        """Collect supplementary public activity signals (games, favorites, counts, history)."""
        uid = obs.user_id

        # Social counts (followers / followings)
        for key, attr_name in (
            ("followers", "followers_count"),
            ("followings", "followings_count"),
        ):
            try:
                res_cnt = self.client.request_json(
                    "GET",
                    f"https://friends.roblox.com/v1/users/{uid}/{key}/count",
                    source_name=f"friends.{key}_count",
                )
                obs.sources.append(res_cnt.attribution)
                if isinstance(res_cnt.data, dict) and isinstance(res_cnt.data.get("count"), int):
                    setattr(obs, attr_name, int(res_cnt.data["count"]))
            except (APIClientError, RequestBudgetExceededError) as exc:
                obs.inaccessible_sources[f"friends.{key}_count"] = str(exc)

        # Publicly created games
        try:
            res_games = self.client.request_json(
                "GET",
                f"https://games.roblox.com/v2/users/{uid}/games",
                params={"limit": 50, "sortOrder": "Asc"},
                source_name="games.user_created",
            )
            obs.sources.append(res_games.attribution)
            raw_games = res_games.data.get("data", []) if isinstance(res_games.data, dict) else []
            obs.created_games = [
                GameRecord(
                    game_id=int(g["id"]),
                    name=str(g.get("name") or ""),
                    description=g.get("description"),
                    creator_id=int(g["creator"]["id"])
                    if isinstance(g.get("creator"), dict) and isinstance(g["creator"].get("id"), int)
                    else uid,
                    creator_type=str(g["creator"].get("type") or "User")
                    if isinstance(g.get("creator"), dict)
                    else "User",
                    created_at=g.get("created"),
                    updated_at=g.get("updated"),
                    place_visits=g.get("placeVisits"),
                )
                for g in raw_games
                if isinstance(g, dict) and isinstance(g.get("id"), int)
            ]
        except (APIClientError, RequestBudgetExceededError) as exc:
            obs.inaccessible_sources["games.user_created"] = str(exc)

        # Publicly favorited games
        try:
            res_favs = self.client.request_json(
                "GET",
                f"https://games.roblox.com/v2/users/{uid}/favorite/games",
                params={"limit": 50},
                source_name="games.user_favorites",
            )
            obs.sources.append(res_favs.attribution)
            raw_favs = res_favs.data.get("data", []) if isinstance(res_favs.data, dict) else []
            obs.favorite_games = [
                GameRecord(
                    game_id=int(g["id"]),
                    name=str(g.get("name") or ""),
                    description=g.get("description"),
                    creator_id=int(g["creator"]["id"])
                    if isinstance(g.get("creator"), dict) and isinstance(g["creator"].get("id"), int)
                    else None,
                    creator_type=str(g["creator"].get("type") or "")
                    if isinstance(g.get("creator"), dict)
                    else None,
                    created_at=g.get("created"),
                    updated_at=g.get("updated"),
                )
                for g in raw_favs
                if isinstance(g, dict) and isinstance(g.get("id"), int)
            ]
        except (APIClientError, RequestBudgetExceededError) as exc:
            obs.inaccessible_sources["games.user_favorites"] = str(exc)

        # Username history (strictly rate-limited by Roblox; degrade gracefully on 429)
        if "users.username_history" in self._throttled_optional_sources:
            obs.inaccessible_sources["users.username_history"] = (
                "Skipped due to active HTTP 429 flood-limit circuit breaker on users.username_history"
            )
        else:
            try:
                res_hist = self.client.request_json(
                    "GET",
                    f"https://users.roblox.com/v1/users/{uid}/username-history",
                    params={"limit": 50, "sortOrder": "Desc"},
                    source_name="users.username_history",
                    max_retries_override=0,
                )
                obs.sources.append(res_hist.attribution)
                raw_hist = res_hist.data.get("data", []) if isinstance(res_hist.data, dict) else []
                obs.previous_usernames = [
                    str(item["name"])
                    for item in raw_hist
                    if isinstance(item, dict) and item.get("name")
                ]
            except RateLimitExceededError as exc:
                self._throttled_optional_sources.add("users.username_history")
                obs.inaccessible_sources["users.username_history"] = str(exc)
            except (APIClientError, RequestBudgetExceededError) as exc:
                obs.inaccessible_sources["users.username_history"] = str(exc)

        # Public inventory visibility & sample hats (assetTypeId=8)
        if "inventory.can_view" in self._throttled_optional_sources:
            obs.inaccessible_sources["inventory.can_view"] = (
                "Skipped due to active HTTP 429 flood-limit circuit breaker on inventory.can_view"
            )
        else:
            try:
                res_vis = self.client.request_json(
                    "GET",
                    f"https://inventory.roblox.com/v1/users/{uid}/can-view-inventory",
                    source_name="inventory.can_view",
                    max_retries_override=0,
                )
                obs.sources.append(res_vis.attribution)
                if isinstance(res_vis.data, dict) and isinstance(res_vis.data.get("canView"), bool):
                    obs.can_view_inventory = bool(res_vis.data["canView"])
                    if obs.can_view_inventory:
                        res_inv = self.client.request_json(
                            "GET",
                            f"https://inventory.roblox.com/v2/users/{uid}/inventory/8",
                            params={"limit": 25, "sortOrder": "Desc"},
                            source_name="inventory.hats_sample",
                            max_retries_override=0,
                        )
                        obs.sources.append(res_inv.attribution)
                        raw_inv = res_inv.data.get("data", []) if isinstance(res_inv.data, dict) else []
                        obs.inventory_sample = [
                            InventorySampleRecord(
                                asset_id=int(item["assetId"]),
                                asset_name=str(item.get("assetName") or ""),
                                acquired_at=item.get("created"),
                            )
                            for item in raw_inv
                            if isinstance(item, dict) and isinstance(item.get("assetId"), int)
                        ]
                    else:
                        obs.inaccessible_sources["inventory.items"] = (
                            "User inventory privacy setting is private (canView=false)"
                        )
            except RateLimitExceededError as exc:
                self._throttled_optional_sources.add("inventory.can_view")
                obs.inaccessible_sources["inventory.can_view"] = str(exc)
            except (AuthenticationRequiredError, APIClientError, RequestBudgetExceededError) as exc:
                obs.inaccessible_sources["inventory.can_view"] = str(exc)

    def search_users_by_keyword(
        self,
        keyword: str,
        *,
        limit: int = 10,
    ) -> tuple[list[dict[str, Any]], DataSourceAttribution | None]:
        """Search public users by keyword (`GET https://users.roblox.com/v1/users/search`)."""
        valid_limit = 10 if limit <= 10 else (25 if limit <= 25 else 50)
        try:
            res = self.client.request_json(
                "GET",
                "https://users.roblox.com/v1/users/search",
                params={"keyword": keyword, "limit": valid_limit},
                source_name="users.search",
            )
            rows = res.data.get("data", []) if isinstance(res.data, dict) else []
            return [r for r in rows if isinstance(r, dict)], res.attribution
        except (APIClientError, RequestBudgetExceededError) as exc:
            logger.warning("Keyword user search for '%s' unavailable: %s", keyword, exc)
            return [], None

    def fetch_group_members(
        self,
        group_id: int,
        *,
        limit: int = 50,
        max_pages: int = 1,
    ) -> tuple[list[dict[str, Any]], list[DataSourceAttribution]]:
        """Enumerate members of a public group with cursor pagination."""
        members: list[dict[str, Any]] = []
        attributions: list[DataSourceAttribution] = []
        cursor: str | None = None
        valid_limit = 10 if limit <= 10 else (25 if limit <= 25 else (50 if limit <= 50 else 100))

        for _ in range(max_pages):
            params: dict[str, Any] = {"limit": valid_limit, "sortOrder": "Asc"}
            if cursor:
                params["cursor"] = cursor
            try:
                res = self.client.request_json(
                    "GET",
                    f"https://groups.roblox.com/v1/groups/{group_id}/users",
                    params=params,
                    source_name="groups.members",
                )
                attributions.append(res.attribution)
                if not isinstance(res.data, dict):
                    break
                page_data = res.data.get("data", [])
                if isinstance(page_data, list):
                    members.extend(m for m in page_data if isinstance(m, dict))
                cursor = res.data.get("nextPageCursor")
                if not cursor:
                    break
            except (APIClientError, RequestBudgetExceededError) as exc:
                logger.debug("Stopping group %d member enumeration: %s", group_id, exc)
                break

        return members, attributions

    def fetch_group_role_members(
        self,
        group_id: int,
        role_id: int,
        *,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], DataSourceAttribution | None]:
        """Enumerate members holding a specific role in a group."""
        valid_limit = 10 if limit <= 10 else (25 if limit <= 25 else (50 if limit <= 50 else 100))
        try:
            res = self.client.request_json(
                "GET",
                f"https://groups.roblox.com/v1/groups/{group_id}/roles/{role_id}/users",
                params={"limit": valid_limit, "sortOrder": "Asc"},
                source_name="groups.role_members",
            )
            rows = res.data.get("data", []) if isinstance(res.data, dict) else []
            return [r for r in rows if isinstance(r, dict)], res.attribution
        except (APIClientError, RequestBudgetExceededError) as exc:
            logger.debug("Could not fetch role %d members in group %d: %s", role_id, group_id, exc)
            return [], None
