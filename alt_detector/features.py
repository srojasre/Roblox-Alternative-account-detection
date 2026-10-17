"""Feature engineering module for pairwise Roblox account association analysis."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

from alt_detector import FEATURE_SCHEMA_VERSION
from alt_detector.graph_analysis import COMMON_DEFAULT_AVATAR_ASSETS
from alt_detector.schemas import AccountObservation, PairwiseFeatures

FEATURE_NAMES: tuple[str, ...] = (
    # Family 1: lexical_naming
    "username_char_ngram_sim",
    "username_lcs_ratio",
    "display_name_similarity",
    "shared_distinctive_affix_score",
    # Family 2: group_structure
    "group_jaccard",
    "rarity_weighted_group_sim",
    "small_group_co_membership_score",
    "shared_elevated_role_score",
    # Family 3: social_graph
    "is_direct_friend",
    "mutual_friends_jaccard",
    "mutual_friends_containment",
    # Family 4: profile_avatar
    "description_similarity",
    "avatar_asset_overlap",
    "avatar_style_similarity",
    "creation_burst_proximity",
    # Family 5: activity_games
    "favorite_games_overlap",
    "created_games_similarity",
    "inventory_asset_overlap",
    "activity_timestamp_overlap",
    # Family 6: graph_topology
    "graph_adamic_adar",
    "graph_resource_allocation",
    "graph_personalized_pagerank",
    "graph_same_community",
)

FEATURE_TO_FAMILY: dict[str, str] = {
    "username_char_ngram_sim": "lexical_naming",
    "username_lcs_ratio": "lexical_naming",
    "display_name_similarity": "lexical_naming",
    "shared_distinctive_affix_score": "lexical_naming",
    "group_jaccard": "group_structure",
    "rarity_weighted_group_sim": "group_structure",
    "small_group_co_membership_score": "group_structure",
    "shared_elevated_role_score": "group_structure",
    "is_direct_friend": "social_graph",
    "mutual_friends_jaccard": "social_graph",
    "mutual_friends_containment": "social_graph",
    "description_similarity": "profile_avatar",
    "avatar_asset_overlap": "profile_avatar",
    "avatar_style_similarity": "profile_avatar",
    "creation_burst_proximity": "profile_avatar",
    "favorite_games_overlap": "activity_games",
    "created_games_similarity": "activity_games",
    "inventory_asset_overlap": "activity_games",
    "activity_timestamp_overlap": "activity_games",
    "graph_adamic_adar": "graph_topology",
    "graph_resource_allocation": "graph_topology",
    "graph_personalized_pagerank": "graph_topology",
    "graph_same_community": "graph_topology",
}

_COMMON_NUMBERS = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "24",
    "32",
    "64",
    "69",
    "77",
    "88",
    "99",
    "100",
    "101",
    "123",
    "1234",
    "360",
    "404",
    "420",
    "666",
    "777",
    "888",
    "999",
    "1000",
    *(str(y) for y in range(1990, 2030)),
}


def char_ngram_cosine(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram cosine similarity in [0.0, 1.0]."""
    a = re.sub(r"\s+", " ", text_a.lower().strip())
    b = re.sub(r"\s+", " ", text_b.lower().strip())
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    padded_a = f"^{a}$"
    padded_b = f"^{b}$"
    grams_a = Counter(
        padded_a[i : i + n] for i in range(max(1, len(padded_a) - n + 1))
    )
    grams_b = Counter(
        padded_b[i : i + n] for i in range(max(1, len(padded_b) - n + 1))
    )
    intersection = set(grams_a.keys()) & set(grams_b.keys())
    dot = sum(grams_a[k] * grams_b[k] for k in intersection)
    norm_a = math.sqrt(sum(v * v for v in grams_a.values()))
    norm_b = math.sqrt(sum(v * v for v in grams_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return round(min(1.0, max(0.0, dot / (norm_a * norm_b))), 4)


def strip_common_affixes(username: str) -> str:
    """Strip underscores and trailing/leading digits to isolate the core lexical handle."""
    core = re.sub(r"^[0-9_]+|[0-9_]+$", "", username.lower().strip())
    core = core.replace("_", "")
    return core if len(core) >= 3 else username.lower().strip()


def longest_common_substring_length(a: str, b: str) -> int:
    """Return the length of the longest contiguous common substring between `a` and `b`."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ca = a[i - 1]
        for j in range(1, n + 1):
            if ca == b[j - 1]:
                val = prev[j - 1] + 1
                curr[j] = val
                if val > best:
                    best = val
        prev = curr
    return best


def compute_core_lcs_ratio(name_a: str, name_b: str) -> float:
    """Compute core-handle Longest Common Substring ratio, suppressing short coincidences."""
    core_a = strip_common_affixes(name_a)
    core_b = strip_common_affixes(name_b)
    if not core_a or not core_b:
        return 0.0
    if core_a == core_b:
        return 1.0
    lcs_len = longest_common_substring_length(core_a, core_b)
    if lcs_len < 4:
        return 0.0
    shorter = min(len(core_a), len(core_b))
    longer = max(len(core_a), len(core_b))
    containment = lcs_len / max(shorter, 1)
    length_balance = shorter / max(longer, 1)
    # High score when one handle's core stem is fully or mostly contained in the other
    score = containment * (0.75 + 0.25 * length_balance)
    return round(min(1.0, max(0.0, score)), 4)


def _extract_distinctive_tokens(obs: AccountObservation) -> tuple[set[str], set[str]]:
    """Return `(numeric_tokens, word_tokens)` from an account's names and small-group names."""
    num_tokens: set[str] = set()
    word_tokens: set[str] = set()

    name_pool = [obs.username, obs.display_name]
    if obs.previous_usernames:
        name_pool.extend(obs.previous_usernames)

    for text in name_pool:
        for num in re.findall(r"\d{2,6}", text):
            if num not in _COMMON_NUMBERS:
                num_tokens.add(num)
        for token in re.findall(r"[A-Za-z]{4,}", text):
            word_tokens.add(token.lower())

    # Also extract camelCase prefix/suffix tokens from display_name/username
    for text in name_pool:
        camel_parts = re.findall(r"[A-Z][a-z]{3,}", text)
        for part in camel_parts:
            word_tokens.add(part.lower())

    # Small/medium group numeric tags (e.g., Site-416, Task Force 416)
    group_num_tokens: set[str] = set()
    if obs.groups:
        for grp in obs.groups:
            mc = grp.member_count or 999_999
            if mc <= 15_000 or grp.role_rank >= 100:
                for num in re.findall(r"\d{3,6}", grp.group_name):
                    if num not in _COMMON_NUMBERS:
                        group_num_tokens.add(num)

    return num_tokens | group_num_tokens, word_tokens


def extract_pairwise_features(
    target: AccountObservation,
    candidate: AccountObservation,
    graph_metrics: dict[str, float] | None = None,
) -> PairwiseFeatures:
    """Extract normalized pairwise features with explicit missing-value separation."""
    values: dict[str, float | None] = {}
    provenance: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Family 1: Lexical & Identity Naming Signals
    # ------------------------------------------------------------------
    target_usernames = [target.username] + (target.previous_usernames or [])
    cand_usernames = [candidate.username] + (candidate.previous_usernames or [])

    best_ngram = 0.0
    best_lcs = 0.0
    best_pair = (target.username, candidate.username)
    for tu in target_usernames:
        for cu in cand_usernames:
            ng = char_ngram_cosine(tu, cu, n=3)
            lcs = compute_core_lcs_ratio(tu, cu)
            if ng > best_ngram:
                best_ngram = ng
            if lcs > best_lcs:
                best_lcs = lcs
                best_pair = (tu, cu)

    values["username_char_ngram_sim"] = round(best_ngram, 4)
    values["username_lcs_ratio"] = round(best_lcs, 4)
    provenance["username_char_ngram_sim"] = "users.profile,users.username_history"
    provenance["username_lcs_ratio"] = "users.profile,users.username_history"
    diagnostics["matched_username_pair"] = best_pair

    # Cross-compare display names and usernames
    target_all_names = list(dict.fromkeys(target_usernames + [target.display_name]))
    cand_all_names = list(dict.fromkeys(cand_usernames + [candidate.display_name]))
    best_disp_sim = 0.0
    for tn in target_all_names:
        for cn in cand_all_names:
            sim = max(
                char_ngram_cosine(tn, cn, n=3),
                compute_core_lcs_ratio(tn, cn),
            )
            if sim > best_disp_sim:
                best_disp_sim = sim
    values["display_name_similarity"] = round(best_disp_sim, 4)
    provenance["display_name_similarity"] = "users.profile"

    # Shared distinctive numeric or word affixes (e.g., "416", "Silly", clan tags)
    t_nums, t_words = _extract_distinctive_tokens(target)
    c_nums, c_words = _extract_distinctive_tokens(candidate)
    # Candidate's explicit handle numeric tokens
    cand_handle_nums = {
        n
        for n in re.findall(r"\d{2,6}", f"{candidate.username} {candidate.display_name}")
        if n not in _COMMON_NUMBERS
    }
    target_handle_nums = {
        n
        for n in re.findall(r"\d{2,6}", f"{target.username} {target.display_name}")
        if n not in _COMMON_NUMBERS
    }
    shared_nums = (cand_handle_nums & t_nums) | (target_handle_nums & c_nums)
    shared_words = t_words & c_words

    affix_score = 0.0
    if shared_nums:
        affix_score += 0.65
    if shared_words:
        # Check if it's a core stem vs secondary stylistic prefix like "silly"
        longest_w = max((len(w) for w in shared_words), default=0)
        affix_score += 0.55 if longest_w >= 6 else 0.30
    values["shared_distinctive_affix_score"] = round(min(1.0, affix_score), 4)
    provenance["shared_distinctive_affix_score"] = "users.profile,groups.user_roles"
    diagnostics["shared_numeric_affixes"] = sorted(shared_nums)
    diagnostics["shared_word_tokens"] = sorted(shared_words)

    # ------------------------------------------------------------------
    # Family 2: Group & Role Structure (Rarity-Weighted)
    # ------------------------------------------------------------------
    if target.groups is None or candidate.groups is None:
        values["group_jaccard"] = None
        values["rarity_weighted_group_sim"] = None
        values["small_group_co_membership_score"] = None
        values["shared_elevated_role_score"] = None
    else:
        t_groups = {g.group_id: g for g in target.groups}
        c_groups = {g.group_id: g for g in candidate.groups}
        union_gids = set(t_groups.keys()) | set(c_groups.keys())
        shared_gids = set(t_groups.keys()) & set(c_groups.keys())

        if not union_gids:
            # Neither account belongs to any public group: absence of group data, not dissimilarity
            values["group_jaccard"] = None
            values["rarity_weighted_group_sim"] = None
            values["small_group_co_membership_score"] = None
            values["shared_elevated_role_score"] = None
        else:
            values["group_jaccard"] = round(len(shared_gids) / len(union_gids), 4)

            def _grp_weight(gid: int) -> float:
                g_obj = t_groups.get(gid) or c_groups.get(gid)
                mc = (g_obj.member_count if g_obj and g_obj.member_count else 50_000)
                w = 1.0 / math.log(max(mc, 2) + 1.0)
                if mc > 50_000:
                    w *= 0.12
                elif mc > 10_000:
                    w *= 0.35
                return w

            shared_w = sum(_grp_weight(gid) for gid in shared_gids)
            union_w = sum(_grp_weight(gid) for gid in union_gids)
            t_total_w = sum(_grp_weight(gid) for gid in t_groups)
            c_total_w = sum(_grp_weight(gid) for gid in c_groups)
            # Regularize denominator so an account whose only group is a mega-group
            # (tiny rarity weight) cannot achieve 1.0 containment on that mega-group alone
            min_side_w = max(min(t_total_w, c_total_w), 0.35)

            w_jaccard = (shared_w / union_w) if union_w > 0 else 0.0
            w_containment = (shared_w / min_side_w) if min_side_w > 0 else 0.0
            values["rarity_weighted_group_sim"] = round(
                min(1.0, 0.55 * w_jaccard + 0.45 * w_containment), 4
            )

            # Small-group Adamic-Adar score & elevated role score
            small_aa = 0.0
            elevated_role_raw = 0.0
            shared_group_details: list[dict[str, Any]] = []

            for gid in sorted(shared_gids):
                tg = t_groups[gid]
                cg = c_groups[gid]
                mc = tg.member_count or cg.member_count or 50_000
                shared_group_details.append(
                    {
                        "group_id": gid,
                        "group_name": tg.group_name,
                        "member_count": mc,
                        "target_role": tg.role_name,
                        "target_rank": tg.role_rank,
                        "candidate_role": cg.role_name,
                        "candidate_rank": cg.role_rank,
                    }
                )
                if mc <= 2_500:
                    small_aa += 1.5 / math.log(max(mc, 2) + 1.0)

                # Elevated role evaluation
                if mc <= 10_000 and (tg.role_rank >= 100 or cg.role_rank >= 100):
                    if tg.role_rank >= 100 and cg.role_rank >= 100:
                        rank_diff = abs(tg.role_rank - cg.role_rank)
                        rank_closeness = 1.0 - (rank_diff / 255.0)
                        elevated_role_raw += (
                            (1.6 * rank_closeness) / math.log10(max(mc, 3) + 1.0)
                        )
                    elif (tg.role_rank == 255 or cg.role_rank == 255) and mc <= 25:
                        # Tiny personal group owned by one account where the other is a member
                        elevated_role_raw += 0.95 / math.log10(max(mc, 2) + 1.0)

            values["small_group_co_membership_score"] = round(
                min(1.0, 1.0 - math.exp(-small_aa / 0.65)) if small_aa > 0 else 0.0,
                4,
            )
            values["shared_elevated_role_score"] = round(
                min(1.0, 1.0 - math.exp(-elevated_role_raw / 0.70))
                if elevated_role_raw > 0
                else 0.0,
                4,
            )
            diagnostics["shared_groups"] = shared_group_details

        for k in (
            "group_jaccard",
            "rarity_weighted_group_sim",
            "small_group_co_membership_score",
            "shared_elevated_role_score",
        ):
            provenance[k] = "groups.user_roles"

    # ------------------------------------------------------------------
    # Family 3: Social Graph & Friendship Overlap
    # ------------------------------------------------------------------
    if target.friends is None and candidate.friends is None:
        values["is_direct_friend"] = None
        values["mutual_friends_jaccard"] = None
        values["mutual_friends_containment"] = None
    else:
        t_friends = set(target.friends or [])
        c_friends = set(candidate.friends or [])
        is_friend = (
            candidate.user_id in t_friends or target.user_id in c_friends
        )
        values["is_direct_friend"] = 1.0 if is_friend else 0.0
        provenance["is_direct_friend"] = "friends.list"

        if target.friends is not None and candidate.friends is not None:
            # Exclude target and candidate themselves when computing mutual third-party friends
            t_third = t_friends - {target.user_id, candidate.user_id}
            c_third = c_friends - {target.user_id, candidate.user_id}
            mutuals = t_third & c_third
            union_f = t_third | c_third
            diagnostics["mutual_friends_count"] = len(mutuals)
            diagnostics["mutual_friend_ids"] = sorted(mutuals)[:20]

            if not union_f:
                values["mutual_friends_jaccard"] = None
                values["mutual_friends_containment"] = None
            else:
                values["mutual_friends_jaccard"] = round(len(mutuals) / len(union_f), 4)
                denom = max(min(len(t_third), len(c_third)), 4)
                values["mutual_friends_containment"] = round(
                    min(1.0, len(mutuals) / denom), 4
                )
                provenance["mutual_friends_jaccard"] = "friends.list"
                provenance["mutual_friends_containment"] = "friends.list"
        else:
            values["mutual_friends_jaccard"] = None
            values["mutual_friends_containment"] = None

    # ------------------------------------------------------------------
    # Family 4: Profile Description, Avatar & Creation Lifecycle
    # ------------------------------------------------------------------
    t_desc = (target.description or "").strip()
    c_desc = (candidate.description or "").strip()
    if t_desc and c_desc:
        values["description_similarity"] = char_ngram_cosine(t_desc, c_desc, n=3)
        provenance["description_similarity"] = "users.profile"
    else:
        # Empty description on either account is missing signal, not dissimilarity
        values["description_similarity"] = None

    if target.avatar is not None and candidate.avatar is not None:
        t_assets = set(target.avatar.asset_ids)
        c_assets = set(candidate.avatar.asset_ids)
        t_custom = t_assets - COMMON_DEFAULT_AVATAR_ASSETS
        c_custom = c_assets - COMMON_DEFAULT_AVATAR_ASSETS
        union_custom = t_custom | c_custom
        shared_custom = t_custom & c_custom
        diagnostics["shared_custom_avatar_assets"] = sorted(shared_custom)

        if union_custom:
            values["avatar_asset_overlap"] = round(
                len(shared_custom) / len(union_custom), 4
            )
        elif t_assets | c_assets:
            # Both only wear default free animations/assets -> very low discriminative weight
            values["avatar_asset_overlap"] = round(
                0.10 * (len(t_assets & c_assets) / len(t_assets | c_assets)), 4
            )
        else:
            values["avatar_asset_overlap"] = None
        provenance["avatar_asset_overlap"] = "avatar.details"

        # Avatar body color + rig type similarity
        same_rig = 1.0 if target.avatar.avatar_type == candidate.avatar.avatar_type else 0.0
        t_colors = target.avatar.body_colors
        c_colors = candidate.avatar.body_colors
        color_keys = set(t_colors.keys()) & set(c_colors.keys())
        if color_keys:
            color_matches = sum(1 for k in color_keys if t_colors[k] == c_colors[k])
            color_sim = color_matches / len(color_keys)
            # Discount if monochromatic single-color default
            if len(set(t_colors.values())) == 1 and len(set(c_colors.values())) == 1:
                color_sim *= 0.5
        else:
            color_sim = 0.0
        values["avatar_style_similarity"] = round(0.35 * same_rig + 0.65 * color_sim, 4)
        provenance["avatar_style_similarity"] = "avatar.details"
    else:
        values["avatar_asset_overlap"] = None
        values["avatar_style_similarity"] = None

    t_dt = target.created_datetime
    c_dt = candidate.created_datetime
    if t_dt is not None and c_dt is not None:
        diff_days = abs((t_dt - c_dt).total_seconds()) / 86400.0
        diagnostics["creation_time_diff_days"] = round(diff_days, 2)
        values["creation_burst_proximity"] = round(math.exp(-diff_days / 45.0), 4)
        provenance["creation_burst_proximity"] = "users.profile"
    else:
        values["creation_burst_proximity"] = None

    # ------------------------------------------------------------------
    # Family 5: Experiences, Favorites & Inventory Activity
    # ------------------------------------------------------------------
    if target.favorite_games is not None and candidate.favorite_games is not None:
        t_favs = {g.game_id for g in target.favorite_games}
        c_favs = {g.game_id for g in candidate.favorite_games}
        # Check if one account favorited a game created by the other account
        t_created_ids = {g.game_id for g in (target.created_games or [])}
        c_created_ids = {g.game_id for g in (candidate.created_games or [])}
        cross_fav = bool((t_favs & c_created_ids) or (c_favs & t_created_ids))
        diagnostics["cross_favorited_created_game"] = cross_fav

        if t_favs or c_favs or cross_fav:
            union_fav = t_favs | c_favs
            jacc_fav = (len(t_favs & c_favs) / len(union_fav)) if union_fav else 0.0
            cross_bonus = 0.65 if cross_fav else 0.0
            values["favorite_games_overlap"] = round(min(1.0, jacc_fav + cross_bonus), 4)
            provenance["favorite_games_overlap"] = "games.user_favorites"
        else:
            values["favorite_games_overlap"] = None
    else:
        values["favorite_games_overlap"] = None

    if target.created_games and candidate.created_games:
        t_titles = " ".join(
            g.name for g in target.created_games if " place" not in g.name.lower()
        ).strip()
        c_titles = " ".join(
            g.name for g in candidate.created_games if " place" not in g.name.lower()
        ).strip()
        if t_titles and c_titles:
            values["created_games_similarity"] = char_ngram_cosine(t_titles, c_titles, n=3)
            provenance["created_games_similarity"] = "games.user_created"
        else:
            values["created_games_similarity"] = None
    else:
        values["created_games_similarity"] = None

    if target.inventory_sample and candidate.inventory_sample:
        t_inv = {i.asset_id for i in target.inventory_sample}
        c_inv = {i.asset_id for i in candidate.inventory_sample}
        union_inv = t_inv | c_inv
        values["inventory_asset_overlap"] = (
            round(len(t_inv & c_inv) / len(union_inv), 4) if union_inv else None
        )
        provenance["inventory_asset_overlap"] = "inventory.hats_sample"
    else:
        values["inventory_asset_overlap"] = None

    # Timestamped activity window overlap (created_games + inventory_sample)
    t_years: set[int] = set()
    c_years: set[int] = set()
    for g in target.created_games or []:
        if g.updated_at and len(g.updated_at) >= 4 and g.updated_at[:4].isdigit():
            t_years.add(int(g.updated_at[:4]))
    for i in target.inventory_sample or []:
        if i.acquired_at and len(i.acquired_at) >= 4 and i.acquired_at[:4].isdigit():
            t_years.add(int(i.acquired_at[:4]))
    for g in candidate.created_games or []:
        if g.updated_at and len(g.updated_at) >= 4 and g.updated_at[:4].isdigit():
            c_years.add(int(g.updated_at[:4]))
    for i in candidate.inventory_sample or []:
        if i.acquired_at and len(i.acquired_at) >= 4 and i.acquired_at[:4].isdigit():
            c_years.add(int(i.acquired_at[:4]))

    if t_years and c_years:
        values["activity_timestamp_overlap"] = round(
            len(t_years & c_years) / len(t_years | c_years), 4
        )
        provenance["activity_timestamp_overlap"] = "games.user_created,inventory.hats_sample"
    else:
        values["activity_timestamp_overlap"] = None

    # ------------------------------------------------------------------
    # Family 6: Candidate Relationship Graph Structural Signals
    # ------------------------------------------------------------------
    gm = graph_metrics or {}
    for g_feat in (
        "graph_adamic_adar",
        "graph_resource_allocation",
        "graph_personalized_pagerank",
        "graph_same_community",
    ):
        if g_feat in gm:
            values[g_feat] = round(float(gm[g_feat]), 4)
            provenance[g_feat] = "candidate_relationship_graph"
        else:
            values[g_feat] = None

    available = [k for k in FEATURE_NAMES if values.get(k) is not None]
    missing = [k for k in FEATURE_NAMES if values.get(k) is None]

    return PairwiseFeatures(
        target_user_id=target.user_id,
        candidate_user_id=candidate.user_id,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        values=values,
        available_features=available,
        missing_features=missing,
        provenance=provenance,
        diagnostics=diagnostics,
    )
