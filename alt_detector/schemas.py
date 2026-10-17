"""Data schemas and boundary validation models for the Roblox Alt Detection Engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alt_detector import FEATURE_SCHEMA_VERSION, MODEL_VERSION, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# External Roblox API Boundary Schemas (Validated on ingestion)
# ---------------------------------------------------------------------------


class RobloxResolvedUserItem(BaseModel):
    """Item returned by POST https://users.roblox.com/v1/usernames/users or /v1/users."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    displayName: str = ""
    hasVerifiedBadge: bool = False
    requestedUsername: str | None = None


class RobloxMultiUserResponse(BaseModel):
    """Envelope returned by POST /v1/usernames/users and POST /v1/users."""

    model_config = ConfigDict(extra="ignore")

    data: list[RobloxResolvedUserItem] = Field(default_factory=list)


class RobloxUserProfileResponse(BaseModel):
    """Schema for GET https://users.roblox.com/v1/users/{userId}."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    displayName: str = ""
    description: str | None = ""
    created: str | None = None
    isBanned: bool = False
    hasVerifiedBadge: bool = False
    externalAppDisplayName: str | None = None


class RobloxGroupOwner(BaseModel):
    """Owner object embedded in group responses."""

    model_config = ConfigDict(extra="ignore")

    userId: int | None = None
    username: str | None = None
    displayName: str | None = None
    hasVerifiedBadge: bool = False


class RobloxGroupSummary(BaseModel):
    """Group object in GET https://groups.roblox.com/v1/users/{userId}/groups/roles."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    description: str | None = ""
    memberCount: int | None = None
    hasVerifiedBadge: bool = False
    publicEntryAllowed: bool | None = None
    owner: RobloxGroupOwner | None = None


class RobloxGroupRoleSummary(BaseModel):
    """Role object in GET https://groups.roblox.com/v1/users/{userId}/groups/roles."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    rank: int = 0


class RobloxUserGroupRoleEntry(BaseModel):
    """Single membership entry in GET /v1/users/{userId}/groups/roles."""

    model_config = ConfigDict(extra="ignore")

    group: RobloxGroupSummary
    role: RobloxGroupRoleSummary


class RobloxUserGroupsResponse(BaseModel):
    """Envelope for GET /v1/users/{userId}/groups/roles."""

    model_config = ConfigDict(extra="ignore")

    data: list[RobloxUserGroupRoleEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical Internal & Output Domain Models
# ---------------------------------------------------------------------------


class DataSourceAttribution(BaseModel):
    """Provenance record for a single public API observation."""

    source_name: str
    endpoint_url: str
    retrieved_at: str
    from_cache: bool = False
    http_status: int = 200


class GroupMembershipRecord(BaseModel):
    """Normalized group membership and role record for an account."""

    group_id: int
    group_name: str
    member_count: int | None = None
    role_id: int
    role_name: str
    role_rank: int
    owner_id: int | None = None
    owner_username: str | None = None
    has_verified_badge: bool = False


class AvatarSummary(BaseModel):
    """Normalized public avatar configuration."""

    avatar_type: str = "Unknown"
    scales: dict[str, float] = Field(default_factory=dict)
    body_colors: dict[str, int] = Field(default_factory=dict)
    asset_ids: list[int] = Field(default_factory=list)
    asset_names: dict[int, str] = Field(default_factory=dict)
    asset_types: dict[int, str] = Field(default_factory=dict)


class GameRecord(BaseModel):
    """Normalized public experience (created or favorited) record."""

    game_id: int
    name: str
    description: str | None = None
    creator_id: int | None = None
    creator_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    place_visits: int | None = None


class InventorySampleRecord(BaseModel):
    """Normalized public inventory item record with acquisition timestamp."""

    asset_id: int
    asset_name: str
    acquired_at: str | None = None


class AccountObservation(BaseModel):
    """Canonical collection of publicly observable signals for a single Roblox account.

    Explicitly separates `None` (unavailable / not queried / restricted) from
    empty collections `[]` (confirmed empty).
    """

    user_id: int
    username: str
    display_name: str
    description: str | None = None
    created_at: str | None = None
    is_banned: bool = False
    has_verified_badge: bool = False

    previous_usernames: list[str] | None = None
    friends: list[int] | None = None
    friend_names: dict[int, tuple[str, str]] = Field(default_factory=dict)
    friends_count: int | None = None
    followers_count: int | None = None
    followings_count: int | None = None

    groups: list[GroupMembershipRecord] | None = None
    avatar: AvatarSummary | None = None
    created_games: list[GameRecord] | None = None
    favorite_games: list[GameRecord] | None = None
    can_view_inventory: bool | None = None
    inventory_sample: list[InventorySampleRecord] | None = None

    sources: list[DataSourceAttribution] = Field(default_factory=list)
    inaccessible_sources: dict[str, str] = Field(default_factory=dict)

    @property
    def created_datetime(self) -> datetime | None:
        """Parse ISO-8601 `created_at` timestamp into a UTC datetime if available."""
        if not self.created_at:
            return None
        try:
            normalized = self.created_at.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None


class CandidateDiscoveryReason(BaseModel):
    """Traceable justification for why a candidate account was included."""

    strategy: Literal[
        "lexical_search",
        "direct_friend",
        "small_group_co_member",
        "shared_role_co_member",
        "group_owner",
        "favorite_game_creator",
        "explicit_comparison",
    ]
    detail: str
    source_endpoint: str
    prior_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class CandidateRecord(BaseModel):
    """Discovered candidate account prior to full pairwise evaluation."""

    user_id: int
    username: str
    display_name: str = ""
    reasons: list[CandidateDiscoveryReason] = Field(default_factory=list)

    @property
    def priority_score(self) -> float:
        """Rank candidates for budget-bounded deep collection."""
        if not self.reasons:
            return 0.0
        weights = sorted((r.prior_weight for r in self.reasons), reverse=True)
        # Combine strongest reason with diminishing returns from independent reasons
        base = weights[0]
        bonus = sum(w * 0.25 for w in weights[1:])
        return min(1.0, round(base + bonus, 4))


class PairwiseFeatures(BaseModel):
    """Extracted pairwise features between target account and candidate account."""

    target_user_id: int
    candidate_user_id: int
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    values: dict[str, float | None] = Field(default_factory=dict)
    available_features: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """Single human-explainable evidence factor for or against association."""

    signal_name: str
    family: str
    direction: Literal["supporting", "against", "missing"]
    strength: Literal["high", "medium", "low", "info"]
    value_summary: str
    explanation: str
    contribution: float = 0.0


ConfidenceBand = Literal[
    "Strong evidence",
    "Moderate evidence",
    "Weak evidence",
    "Insufficient evidence",
]


class CandidateAssessment(BaseModel):
    """Structured explainable assessment for a single candidate account."""

    candidate_user_id: int
    candidate_username: str
    candidate_display_name: str
    discovery_reasons: list[CandidateDiscoveryReason] = Field(default_factory=list)

    baseline_score: float = Field(ge=0.0, le=1.0)
    ml_score: float | None = Field(default=None)
    combined_score: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand
    confidence_interpretation: str
    threshold_status: Literal["validated", "provisional"] = "provisional"
    scoring_mode: str = "heuristic_baseline_primary"

    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    conflicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_evidence: list[EvidenceItem] = Field(default_factory=list)

    family_scores: dict[str, float] = Field(default_factory=dict)
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    raw_features: dict[str, float | None] = Field(default_factory=dict)
    graph_metrics: dict[str, float] = Field(default_factory=dict)
    unsupervised_signals: dict[str, float | str] = Field(default_factory=dict)

    uncertainty_notes: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)

    model_version: str = MODEL_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION


class LabeledAccountPair(BaseModel):
    """Training and evaluation pair with provenance and group-leakage cluster ID."""

    user_id_a: int
    user_id_b: int
    username_a: str = ""
    username_b: str = ""
    label: Literal[0, 1, -1]  # 1=positive, 0=negative, -1=uncertain/unlabeled
    operator_cluster_id: str  # Used by GroupKFold to prevent leakage across splits
    label_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    adjudication_status: Literal["verified", "weakly_labeled", "uncertain"] = "verified"
    provenance: str = ""
    features: dict[str, float | None] = Field(default_factory=dict)

    @field_validator("user_id_b")
    @classmethod
    def check_distinct_users(cls, v: int, info: Any) -> int:
        if "user_id_a" in info.data and v == info.data["user_id_a"]:
            raise ValueError("user_id_a and user_id_b must be distinct accounts")
        return v


class AnalysisReport(BaseModel):
    """Complete analysis output schema for CLI rendering and JSON export."""

    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    generated_at: str

    input_identifier: str
    resolved_user_id: int
    resolved_username: str
    resolved_display_name: str
    status: Literal["completed", "partial", "no_candidates", "failed"]

    sources_queried_count: int
    unique_endpoints_queried: list[str] = Field(default_factory=list)
    inaccessible_endpoints: dict[str, str] = Field(default_factory=dict)
    candidates_discovered_count: int
    candidates_analyzed_count: int

    scoring_methodology: str
    threshold_status: Literal["validated", "provisional"]
    assessments: list[CandidateAssessment] = Field(default_factory=list)
    graph_summary: dict[str, Any] = Field(default_factory=dict)

    data_freshness: dict[str, Any] = Field(default_factory=dict)
    known_limitations: list[str] = Field(default_factory=list)
    runtime_seconds: float
    resource_usage: dict[str, Any] = Field(default_factory=dict)
