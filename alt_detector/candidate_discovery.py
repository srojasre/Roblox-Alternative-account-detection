"""Defensible, budget-bounded candidate account discovery from public Roblox signals."""

from __future__ import annotations

import logging
import math
import re

from alt_detector.collectors import PublicDataCollector
from alt_detector.config import EngineConfig
from alt_detector.schemas import (
    AccountObservation,
    CandidateDiscoveryReason,
    CandidateRecord,
)

logger = logging.getLogger("alt_detector")

_COMMON_AFFIXES = {
    "silly",
    "real",
    "the",
    "itz",
    "its",
    "not",
    "official",
    "mr",
    "ms",
    "dr",
    "lord",
    "king",
    "queen",
    "sir",
    "pro",
    "alt",
    "main",
    "yt",
    "ttv",
    "roblox",
    "player",
    "user",
    "gaming",
    "gamer",
}


def extract_lexical_stems(
    username: str,
    display_name: str = "",
    previous_usernames: list[str] | None = None,
) -> list[str]:
    """Extract distinctive lexical stems from a target's username and display name.

    Avoids short or generic stems so keyword searches remain specific and low-noise.
    """
    candidates: list[str] = []
    raw_names = [username]
    if display_name and display_name.lower() != username.lower():
        raw_names.append(display_name)
    if previous_usernames:
        raw_names.extend(previous_usernames[:5])

    for name in raw_names:
        cleaned = name.strip()
        if not cleaned:
            continue
        # Strip leading/trailing digits and underscores
        core = re.sub(r"^[0-9_]+|[0-9_]+$", "", cleaned)
        if len(core) >= 4 and core.lower() not in _COMMON_AFFIXES:
            candidates.append(core)

        # Split by underscore or camelCase boundaries
        parts = re.split(r"_+|(?<=[a-z])(?=[A-Z])", core)
        for part in parts:
            stripped = re.sub(r"[0-9]+", "", part).strip()
            if len(stripped) >= 4 and stripped.lower() not in _COMMON_AFFIXES:
                candidates.append(stripped)

    # Deduplicate case-insensitively while preserving order
    seen: set[str] = set()
    unique_stems: list[str] = []
    for stem in candidates:
        key = stem.lower()
        if key not in seen:
            seen.add(key)
            unique_stems.append(stem)
    return unique_stems


def _quick_stem_overlap(
    name_a: str,
    display_a: str,
    stems_b: list[str],
) -> str | None:
    """Return the matched stem if `name_a` or `display_a` contains any target stem."""
    low_u = name_a.lower()
    low_d = display_a.lower()
    for stem in stems_b:
        s_low = stem.lower()
        if len(s_low) >= 4 and (s_low in low_u or s_low in low_d):
            return stem
    return None


class CandidateDiscoveryEngine:
    """Discovers candidate accounts with traceable provenance and prevalence weighting."""

    def __init__(
        self,
        collector: PublicDataCollector,
        config: EngineConfig,
    ) -> None:
        self.collector = collector
        self.config = config

    def discover_candidates(
        self,
        target: AccountObservation,
    ) -> list[CandidateRecord]:
        """Discover and prioritize candidate accounts for `target` within configured bounds."""
        registry: dict[int, CandidateRecord] = {}
        stems = extract_lexical_stems(
            target.username,
            target.display_name,
            target.previous_usernames,
        )

        def register_candidate(
            uid: int,
            uname: str,
            dname: str,
            reason: CandidateDiscoveryReason,
        ) -> None:
            if uid <= 0 or uid == target.user_id:
                return
            if uid not in registry:
                registry[uid] = CandidateRecord(
                    user_id=uid,
                    username=uname or f"User_{uid}",
                    display_name=dname or uname or f"User_{uid}",
                    reasons=[reason],
                )
            else:
                rec = registry[uid]
                if uname and not rec.username.startswith("User_"):
                    rec.username = uname
                if dname and not rec.display_name.startswith("User_"):
                    rec.display_name = dname
                existing_keys = {(r.strategy, r.detail) for r in rec.reasons}
                if (reason.strategy, reason.detail) not in existing_keys:
                    rec.reasons.append(reason)

        # Strategy 1: Lexical stem search via users.roblox.com/v1/users/search
        for stem in stems[: self.config.max_lexical_queries]:
            search_rows, attr = self.collector.search_users_by_keyword(stem, limit=10)
            endpoint_url = (
                attr.endpoint_url
                if attr is not None
                else f"https://users.roblox.com/v1/users/search?keyword={stem}&limit=10"
            )
            for row in search_rows:
                uid = int(row.get("id") or 0)
                uname = str(row.get("name") or "")
                dname = str(row.get("displayName") or uname)
                if uid <= 0 or uid == target.user_id:
                    continue
                matched = _quick_stem_overlap(uname, dname, [stem])
                weight = 0.72 if matched else 0.45
                register_candidate(
                    uid,
                    uname,
                    dname,
                    CandidateDiscoveryReason(
                        strategy="lexical_search",
                        detail=f"Public user search matched lexical stem '{stem}' (username='{uname}')",
                        source_endpoint=endpoint_url,
                        prior_weight=weight,
                    ),
                )

        # Strategy 2: Direct friends list (with lexical & rarity prioritization)
        if target.friends:
            for fid in target.friends:
                fname, fdname = target.friend_names.get(fid, ("", ""))
                matched_stem = _quick_stem_overlap(fname, fdname, stems) if fname else None
                prior = 0.75 if matched_stem else 0.38
                detail = (
                    f"Direct friend of target and shares lexical stem '{matched_stem}'"
                    if matched_stem
                    else f"Direct public friend of target ({target.username})"
                )
                register_candidate(
                    fid,
                    fname,
                    fdname,
                    CandidateDiscoveryReason(
                        strategy="direct_friend",
                        detail=detail,
                        source_endpoint=f"https://friends.roblox.com/v1/users/{target.user_id}/friends",
                        prior_weight=prior,
                    ),
                )

        # Strategy 3: Small-group co-members, group owners, and shared elevated roles
        if target.groups:
            sorted_groups = sorted(
                target.groups,
                key=lambda g: g.member_count if (g.member_count and g.member_count > 0) else 999_999_999,
            )

            enumerated_groups = 0
            for grp in sorted_groups:
                mc = grp.member_count or 999_999_999
                # Group owner check for non-massive groups or groups where target is staff
                if (
                    grp.owner_id
                    and grp.owner_id != target.user_id
                    and (mc <= self.config.max_group_size_for_expansion or grp.role_rank >= 100)
                ):
                    owner_prior = min(
                        0.75,
                        max(0.30, 1.0 / (1.0 + math.log10(max(mc, 2)))),
                    )
                    register_candidate(
                        grp.owner_id,
                        grp.owner_username or "",
                        grp.owner_username or "",
                        CandidateDiscoveryReason(
                            strategy="group_owner",
                            detail=(
                                f"Owner of group '{grp.group_name}' (ID {grp.group_id}, "
                                f"members={grp.member_count}, target role='{grp.role_name}' rank={grp.role_rank})"
                            ),
                            source_endpoint=f"https://groups.roblox.com/v1/users/{target.user_id}/groups/roles",
                            prior_weight=round(owner_prior, 3),
                        ),
                    )

                # Small group member enumeration
                if (
                    mc <= self.config.max_group_size_for_expansion
                    and enumerated_groups < self.config.max_groups_to_enumerate
                ):
                    enumerated_groups += 1
                    members, _ = self.collector.fetch_group_members(
                        grp.group_id,
                        limit=self.config.max_role_members_to_fetch,
                        max_pages=1,
                    )
                    rarity_prior = min(
                        0.72,
                        max(0.28, 0.90 / (1.0 + math.log10(max(mc, 2)))),
                    )
                    for m in members:
                        u_obj = m.get("user") if isinstance(m.get("user"), dict) else {}
                        r_obj = m.get("role") if isinstance(m.get("role"), dict) else {}
                        muid = int(u_obj.get("userId") or 0)
                        muname = str(u_obj.get("username") or "")
                        mdname = str(u_obj.get("displayName") or muname)
                        mrank = int(r_obj.get("rank") or 0)
                        mrole_name = str(r_obj.get("name") or "Member")
                        if muid <= 0 or muid == target.user_id:
                            continue
                        role_bonus = 0.12 if (mrank >= 100 and grp.role_rank >= 100) else 0.0
                        stem_match = _quick_stem_overlap(muname, mdname, stems)
                        stem_bonus = 0.15 if stem_match else 0.0
                        register_candidate(
                            muid,
                            muname,
                            mdname,
                            CandidateDiscoveryReason(
                                strategy="small_group_co_member",
                                detail=(
                                    f"Co-member in small group '{grp.group_name}' "
                                    f"(ID {grp.group_id}, members={mc}, candidate role='{mrole_name}' rank={mrank})"
                                ),
                                source_endpoint=f"https://groups.roblox.com/v1/groups/{grp.group_id}/users",
                                prior_weight=round(min(0.92, rarity_prior + role_bonus + stem_bonus), 3),
                            ),
                        )

                # Elevated role co-member enumeration in medium groups
                elif grp.role_rank >= 100 and mc <= 5000 and enumerated_groups < self.config.max_groups_to_enumerate + 2:
                    role_members, _ = self.collector.fetch_group_role_members(
                        grp.group_id,
                        grp.role_id,
                        limit=25,
                    )
                    for m in role_members:
                        u_obj = m.get("user") if isinstance(m.get("user"), dict) else {}
                        muid = int(u_obj.get("userId") or 0)
                        muname = str(u_obj.get("username") or "")
                        mdname = str(u_obj.get("displayName") or muname)
                        if muid <= 0 or muid == target.user_id:
                            continue
                        register_candidate(
                            muid,
                            muname,
                            mdname,
                            CandidateDiscoveryReason(
                                strategy="shared_role_co_member",
                                detail=(
                                    f"Shares elevated role '{grp.role_name}' (rank {grp.role_rank}) "
                                    f"in group '{grp.group_name}' (ID {grp.group_id}, members={mc})"
                                ),
                                source_endpoint=(
                                    f"https://groups.roblox.com/v1/groups/{grp.group_id}/roles/{grp.role_id}/users"
                                ),
                                prior_weight=0.62,
                            ),
                        )

        # Strategy 4: Individual creators of games favorited by the target
        if target.favorite_games:
            for fav in target.favorite_games:
                if (
                    fav.creator_type == "User"
                    and fav.creator_id
                    and fav.creator_id != target.user_id
                ):
                    register_candidate(
                        fav.creator_id,
                        "",
                        "",
                        CandidateDiscoveryReason(
                            strategy="favorite_game_creator",
                            detail=f"Creator of experience '{fav.name}' (ID {fav.game_id}) favorited by target",
                            source_endpoint=f"https://games.roblox.com/v2/users/{target.user_id}/favorite/games",
                            prior_weight=0.35,
                        ),
                    )

        # Resolve any candidate missing a username before ranking
        missing_uids = [
            uid for uid, rec in registry.items() if rec.username.startswith("User_") or not rec.username
        ]
        if missing_uids:
            resolved = self.collector.batch_resolve_user_ids(missing_uids)
            for uid, (uname, dname, _) in resolved.items():
                if uid in registry:
                    registry[uid].username = uname
                    registry[uid].display_name = dname

        # Sort candidates deterministically by priority_score DESC, reasons count DESC, user_id ASC
        ordered = sorted(
            registry.values(),
            key=lambda c: (-c.priority_score, -len(c.reasons), c.user_id),
        )
        return ordered[: self.config.max_candidates]
