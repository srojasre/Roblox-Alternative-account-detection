"""Synthetic and realistic fixtures for deterministic unit and regression testing."""

from __future__ import annotations

from alt_detector.schemas import (
    AccountObservation,
    AvatarSummary,
    DataSourceAttribution,
    GameRecord,
    GroupMembershipRecord,
    InventorySampleRecord,
    LabeledAccountPair,
)


def make_target_account() -> AccountObservation:
    """Create a realistic target account fixture (`witrecx64`, ID 93910323)."""
    return AccountObservation(
        user_id=93910323,
        username="witrecx64",
        display_name="SillyWitrecx",
        description="I still play roblox and I have more than the double of the mean robloxian player age.",
        created_at="2015-09-13T02:00:41.39Z",
        is_banned=False,
        has_verified_badge=False,
        previous_usernames=[],
        friends=[2349691539, 1719913571, 395422980, 419857724, 5306681491, 159708565, 3813551317],
        friend_names={
            2349691539: ("witrecx", "witrecx"),
            1719913571: ("mineturtle_416", "SillyLittleTurtle"),
            395422980: ("AlphaOperator", "Alpha"),
            419857724: ("BravoOperator", "Bravo"),
            5306681491: ("CharliePeer", "Charlie"),
            159708565: ("StarFireIce29", "StarFireIce29"),
            3813551317: ("ImLordOxide", "IWillWatchAsYouBurn"),
        },
        friends_count=7,
        followers_count=167,
        followings_count=137,
        groups=[
            GroupMembershipRecord(
                group_id=235594128,
                group_name="Paragon Armaments",
                member_count=9,
                role_id=573848063,
                role_name="Owner",
                role_rank=255,
                owner_id=93910323,
                owner_username="witrecx64",
            ),
            GroupMembershipRecord(
                group_id=11668202,
                group_name="Task Force 416",
                member_count=79,
                role_id=82998303,
                role_name="E8 | Master Sergeant",
                role_rank=165,
                owner_id=5001,
                owner_username="TFLeader",
            ),
            GroupMembershipRecord(
                group_id=11575035,
                group_name="Grand Cicada Front",
                member_count=128,
                role_id=67899939,
                role_name="Grand Cicada Front | Vice President",
                role_rank=254,
                owner_id=5002,
                owner_username="CicadaPres",
            ),
            GroupMembershipRecord(
                group_id=592800379,
                group_name="Site-416: Staff Team",
                member_count=445,
                role_id=432814003,
                role_name="Senior Moderator",
                role_rank=253,
                owner_id=5003,
                owner_username="SiteDirector",
            ),
            GroupMembershipRecord(
                group_id=34783288,
                group_name="Arc I: Warriors",
                member_count=920100,
                role_id=135408120,
                role_name="Member",
                role_rank=1,
            ),
        ],
        avatar=AvatarSummary(
            avatar_type="R15",
            scales={"height": 1.0, "width": 1.0, "head": 1.0},
            body_colors={
                "headColorId": 1001,
                "torsoColorId": 1001,
                "rightArmColorId": 1001,
                "leftArmColorId": 1001,
                "rightLegColorId": 1001,
                "leftLegColorId": 1001,
            },
            asset_ids=[2510242378, 10647852134, 11600317961, 14488197116],
        ),
        created_games=[
            GameRecord(
                game_id=10348213966,
                name="Clicker & Flex Simulator",
                creator_id=93910323,
                creator_type="User",
                created_at="2026-06-18T03:49:33Z",
                updated_at="2026-07-05T03:23:31Z",
            )
        ],
        favorite_games=[
            GameRecord(
                game_id=6409513651,
                name="Anime Warriors III",
                creator_id=99999,
                creator_type="Group",
            ),
            GameRecord(
                game_id=777001,
                name="Site-416 Testing Grounds",
                creator_id=2349691539,
                creator_type="User",
            ),
        ],
        can_view_inventory=True,
        inventory_sample=[
            InventorySampleRecord(
                asset_id=102611803,
                asset_name="Verified, Bonafide, Plaidafied",
                acquired_at="2015-09-26T13:54:32Z",
            ),
            InventorySampleRecord(
                asset_id=11600317961,
                asset_name="Tactical Visor 416",
                acquired_at="2021-05-10T10:00:00Z",
            ),
        ],
        sources=[
            DataSourceAttribution(
                source_name="users.profile",
                endpoint_url="https://users.roblox.com/v1/users/93910323",
                retrieved_at="2026-10-16T16:50:00Z",
                http_status=200,
            )
        ],
    )


def make_strong_alt_candidate() -> AccountObservation:
    """Create a candidate with strong multi-family corroboration (`witrecx`, ID 2349691539)."""
    return AccountObservation(
        user_id=2349691539,
        username="witrecx",
        display_name="witrecx",
        description="",
        created_at="2021-02-05T16:48:07.207Z",
        is_banned=False,
        has_verified_badge=False,
        previous_usernames=[],
        friends=[93910323, 395422980, 419857724, 5306681491, 159708565, 3813551317],
        friends_count=6,
        groups=[
            GroupMembershipRecord(
                group_id=11668202,
                group_name="Task Force 416",
                member_count=79,
                role_id=82998303,
                role_name="E8 | Master Sergeant",
                role_rank=165,
            ),
            GroupMembershipRecord(
                group_id=11575035,
                group_name="Grand Cicada Front",
                member_count=128,
                role_id=67899938,
                role_name="Grand Cicada Front | High Marshal",
                role_rank=253,
            ),
            GroupMembershipRecord(
                group_id=592800379,
                group_name="Site-416: Staff Team",
                member_count=445,
                role_id=432814003,
                role_name="Senior Moderator",
                role_rank=253,
            ),
        ],
        avatar=AvatarSummary(
            avatar_type="R15",
            scales={"height": 1.0, "width": 1.0, "head": 1.0},
            body_colors={"headColorId": 1001, "torsoColorId": 1001},
            asset_ids=[11600317961, 14488197116],
        ),
        created_games=[
            GameRecord(
                game_id=777001,
                name="Site-416 Testing Grounds",
                creator_id=2349691539,
                creator_type="User",
                created_at="2021-03-01T00:00:00Z",
                updated_at="2026-06-20T00:00:00Z",
            )
        ],
        favorite_games=[
            GameRecord(
                game_id=10348213966,
                name="Clicker & Flex Simulator",
                creator_id=93910323,
                creator_type="User",
            )
        ],
        can_view_inventory=True,
        inventory_sample=[
            InventorySampleRecord(
                asset_id=11600317961,
                asset_name="Tactical Visor 416",
                acquired_at="2021-05-12T10:00:00Z",
            )
        ],
        sources=[
            DataSourceAttribution(
                source_name="users.profile",
                endpoint_url="https://users.roblox.com/v1/users/2349691539",
                retrieved_at="2026-10-16T16:50:00Z",
                http_status=200,
            )
        ],
    )


def make_lexical_only_candidate() -> AccountObservation:
    """Create a sparse candidate that ONLY shares a lexical handle (`witrecx_416`, 0 friends, 0 groups)."""
    return AccountObservation(
        user_id=8814991503,
        username="witrecx_416",
        display_name="witrecx_416",
        description="",
        created_at="2025-07-02T07:58:57.566Z",
        is_banned=False,
        has_verified_badge=False,
        previous_usernames=[],
        friends=[],
        friends_count=0,
        groups=[],
        avatar=AvatarSummary(
            avatar_type="R15",
            scales={"height": 1.0, "width": 1.0},
            body_colors={"headColorId": 1, "torsoColorId": 2},
            asset_ids=[2510242378],
        ),
        created_games=[],
        favorite_games=[],
        can_view_inventory=False,
    )


def make_popular_group_only_candidate() -> AccountObservation:
    """Create an unrelated control account that only shares a 920,100-member mega-group."""
    return AccountObservation(
        user_id=555000111,
        username="RandomPlayer90210",
        display_name="RandomPlayer",
        description="Just playing random games.",
        created_at="2019-04-12T10:00:00Z",
        is_banned=False,
        has_verified_badge=False,
        friends=[1001, 1002, 1003],
        friends_count=3,
        groups=[
            GroupMembershipRecord(
                group_id=34783288,
                group_name="Arc I: Warriors",
                member_count=920100,
                role_id=135408120,
                role_name="Member",
                role_rank=1,
            )
        ],
        avatar=AvatarSummary(
            avatar_type="R6",
            scales={"height": 1.0},
            body_colors={"headColorId": 24, "torsoColorId": 23},
            asset_ids=[1028859],
        ),
        created_games=[],
        favorite_games=[],
        can_view_inventory=False,
    )


def make_labeled_dataset_fixture() -> list[LabeledAccountPair]:
    """Generate a deterministic 24-pair multi-cluster dataset for grouped CV tests."""
    pairs: list[LabeledAccountPair] = []
    # 8 positive pairs across 4 distinct operator clusters (2 pairs per cluster)
    for cluster_idx in range(4):
        cluster_id = f"op_pos_{cluster_idx}"
        for sub_idx in range(2):
            uid_a = 10_000 + cluster_idx * 100 + sub_idx
            uid_b = 20_000 + cluster_idx * 100 + sub_idx
            pairs.append(
                LabeledAccountPair(
                    user_id_a=uid_a,
                    user_id_b=uid_b,
                    username_a=f"operator{cluster_idx}_main_{sub_idx}",
                    username_b=f"operator{cluster_idx}_alt_{sub_idx}",
                    label=1,
                    operator_cluster_id=cluster_id,
                    label_confidence=0.95,
                    adjudication_status="verified",
                    provenance="consented_multi_account_benchmark",
                    features={
                        "username_char_ngram_sim": 0.72 - 0.04 * sub_idx,
                        "username_lcs_ratio": 0.85,
                        "display_name_similarity": 0.78,
                        "shared_distinctive_affix_score": 0.65,
                        "group_jaccard": 0.45,
                        "rarity_weighted_group_sim": 0.76,
                        "small_group_co_membership_score": 0.82,
                        "shared_elevated_role_score": 0.74,
                        "is_direct_friend": 1.0,
                        "mutual_friends_jaccard": 0.38,
                        "mutual_friends_containment": 0.65,
                        "description_similarity": None,
                        "avatar_asset_overlap": 0.50,
                        "avatar_style_similarity": 0.65,
                        "creation_burst_proximity": 0.15,
                        "favorite_games_overlap": 0.45,
                        "created_games_similarity": None,
                        "inventory_asset_overlap": 0.40,
                        "activity_timestamp_overlap": 0.60,
                        "graph_adamic_adar": 0.78,
                        "graph_resource_allocation": 0.70,
                        "graph_personalized_pagerank": 0.85,
                        "graph_same_community": 1.0,
                    },
                )
            )

    # 14 negative pairs across 4 distinct negative operator clusters
    for cluster_idx in range(4):
        cluster_id = f"op_neg_{cluster_idx}"
        for sub_idx in range(3 if cluster_idx < 2 else 4):
            uid_a = 30_000 + cluster_idx * 100 + sub_idx
            uid_b = 40_000 + cluster_idx * 100 + sub_idx
            # Include hard negatives (e.g., clanmate with shared group OR stranger with mildly similar name)
            is_clanmate = (sub_idx == 0)
            pairs.append(
                LabeledAccountPair(
                    user_id_a=uid_a,
                    user_id_b=uid_b,
                    username_a=f"control_a_{cluster_idx}_{sub_idx}",
                    username_b=f"control_b_{cluster_idx}_{sub_idx}",
                    label=0,
                    operator_cluster_id=cluster_id,
                    label_confidence=1.0,
                    adjudication_status="verified",
                    provenance="unrelated_control_pair",
                    features={
                        "username_char_ngram_sim": 0.08 if is_clanmate else 0.25,
                        "username_lcs_ratio": 0.0 if is_clanmate else 0.30,
                        "display_name_similarity": 0.10,
                        "shared_distinctive_affix_score": 0.30 if is_clanmate else 0.0,
                        "group_jaccard": 0.15 if is_clanmate else 0.02,
                        "rarity_weighted_group_sim": 0.32 if is_clanmate else 0.01,
                        "small_group_co_membership_score": 0.35 if is_clanmate else 0.0,
                        "shared_elevated_role_score": 0.15 if is_clanmate else 0.0,
                        "is_direct_friend": 0.0,
                        "mutual_friends_jaccard": 0.05 if is_clanmate else 0.0,
                        "mutual_friends_containment": 0.10 if is_clanmate else 0.0,
                        "description_similarity": 0.02,
                        "avatar_asset_overlap": 0.0,
                        "avatar_style_similarity": 0.20,
                        "creation_burst_proximity": 0.01,
                        "favorite_games_overlap": 0.0,
                        "created_games_similarity": None,
                        "inventory_asset_overlap": None,
                        "activity_timestamp_overlap": None,
                        "graph_adamic_adar": 0.22 if is_clanmate else 0.02,
                        "graph_resource_allocation": 0.18 if is_clanmate else 0.01,
                        "graph_personalized_pagerank": 0.20 if is_clanmate else 0.05,
                        "graph_same_community": 1.0 if is_clanmate else 0.0,
                    },
                )
            )

    # 2 uncertain pairs (label == -1) to verify they are tracked and excluded from supervised loss
    pairs.append(
        LabeledAccountPair(
            user_id_a=90001,
            user_id_b=90002,
            username_a="ambiguous_1",
            username_b="ambiguous_2",
            label=-1,
            operator_cluster_id="op_uncertain_1",
            label_confidence=0.3,
            adjudication_status="uncertain",
            provenance="unverified_observation",
            features={"username_char_ngram_sim": 0.50, "username_lcs_ratio": 0.60},
        )
    )
    return pairs
