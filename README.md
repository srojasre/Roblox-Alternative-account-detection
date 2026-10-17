# Roblox Alternative Account Detection Engine (`alt_detector`)

A local, explainable, evidence-based OSINT and entity-resolution engine that evaluates publicly observable signals to estimate whether two or more Roblox accounts may be operated by the same person or closely coordinated operators.

The engine operates strictly as a local command-line application (`python -m alt_detector <username_or_id>`), uses local SQLite caching and local CPU-efficient machine learning (`scikit-learn` + `networkx`), and never bypasses authentication, rate limits, privacy settings, or security controls.

---

## 1. Mission & Analytical Scope

**Objective:** Estimate whether publicly observable evidence supports a possible relationship between Roblox accounts, with full provenance tracking, feature-level attribution, and explicit alternative explanations.

**Non-Objectives & Ethical Boundaries:**
- Does **not** claim definitive real-world identity attribution. Every assessment is an analytical lead bounded by public telemetry.
- Does **not** use `.ROBLOSECURITY` cookies, stolen credentials, leaked databases, private telemetry, IP addresses, or device fingerprints.
- Does **not** rotate proxies/identities to evade HTTP 429 rate limits.
- Does **not** include a web frontend, public API server, Discord bot, or automated moderation/reporting triggers.

---

## 2. Verified Roblox Public API Landscape (As of 2026-10-16)

All endpoints were live-tested and cross-referenced against official Roblox Creator Hub and Open Cloud documentation prior to implementation:
- Official Open Cloud Reference: <https://create.roblox.com/docs/cloud>
- Official Cloud API Index: <https://create.roblox.com/docs/cloud/llms.txt>
- Official Users Feature Reference: <https://create.roblox.com/docs/en-us/cloud/reference/features/users.md>
- Official Groups Feature Reference: <https://create.roblox.com/docs/en-us/cloud/reference/features/groups.md>
- Official Connections/Friends Reference: <https://create.roblox.com/docs/cloud/reference/features/friends.md>
- Official Avatars Reference: <https://create.roblox.com/docs/cloud/reference/features/avatars.md>
- Official Inventories Reference: <https://create.roblox.com/docs/cloud/reference/features/inventories.md>
- Official Badges Reference: <https://create.roblox.com/docs/cloud/reference/features/badges.md>

### Verified Endpoint Matrix

| Source ID | Verifiable URL | Data Exposed | Auth Required | Documentation Status | Rate Limits & Restrictions | Freshness & Limitations | Permitted Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `users.resolve_usernames` | `POST https://users.roblox.com/v1/usernames/users` | Resolves usernames (including historical handles) to canonical `id`, `name`, `displayName`, `hasVerifiedBadge` | None (Public) | Documented on Creator Hub (`MultiGetByUsernameRequest`) | ~200 req/min per IP; max 100 usernames/call | Real-time; resolves banned accounts when `excludeBannedUsers=false` | Permitted |
| `users.batch_get_by_ids` | `POST https://users.roblox.com/v1/users` | Batch-resolves numeric `userIds` to `id`, `name`, `displayName`, `hasVerifiedBadge` | None (Public) | Documented on Creator Hub (`MultiGetByUserIdRequest`) | ~200 req/min per IP; max 100 IDs/call | Real-time; required because `friends.list` omits usernames in 2026 | Permitted |
| `users.profile` | `GET https://users.roblox.com/v1/users/{userId}` | `id`, `name`, `displayName`, `description`, `created` (ISO-8601), `isBanned`, `hasVerifiedBadge` | None (Public) | Documented domain API (`users.roblox.com`) | ~120 req/min per IP | Real-time profile state | Permitted |
| `users.username_history` | `GET https://users.roblox.com/v1/users/{userId}/username-history` | Paginated list of previous usernames (`name`) | None (Public) | Documented domain API | **Strict flood limit** (~5–10 rapid calls/IP before HTTP 429) | Guarded by a per-run HTTP 429 circuit breaker; treated as optional | Permitted |
| `users.search` | `GET https://users.roblox.com/v1/users/search?keyword={kw}&limit=10` | Keyword search returning `id`, `name`, `displayName`, `previousUsernames` | None (Public) | Documented on Creator Hub (`SearchGetUserResponse`) | Strict flood limit (HTTP 429 code 4); keyword length/filter validation (HTTP 400 codes 5, 6) | Capped to `max_lexical_queries=2` per run | Permitted |
| `groups.user_roles` | `GET https://groups.roblox.com/v1/users/{userId}/groups/roles` | All public group memberships (`group.id`, `name`, `memberCount`, `owner`) and user's `role` (`id`, `name`, `rank` 1..255) | None (Public) | Documented on Creator Hub | ~120 req/min per IP | Real-time; exposes exact rank hierarchy | Permitted |
| `groups.members` / `groups.role_members` | `GET https://groups.roblox.com/v1/groups/{groupId}/users` & `/roles/{roleId}/users` | Paginated group or role member roster (`userId`, `username`, `displayName`, `role`) | None (Public) | Documented domain API | ~100 req/min; cursor-paginated | Restricted to small groups (`<= 500` members) or elevated roles (`rank >= 100`) to avoid platform crawling | Permitted |
| `friends.list` & counts | `GET https://friends.roblox.com/v1/users/{userId}/friends` (`.../count`, `followers/count`, `followings/count`) | Public friend user IDs and aggregate social counts | None (Public) | Documented on Creator Hub | ~120 req/min per IP | As of 2026, `/friends` returns IDs with empty `name: ""` strings; enriched via `POST /v1/users` | Permitted |
| `avatar.details` | `GET https://avatar.roblox.com/v1/users/{userId}/avatar` | `playerAvatarType` (`R6`/`R15`), `scales`, `bodyColors`, equipped `assets` (`id`, `name`, `assetType`) | None (Public) | Documented domain API | ~120 req/min per IP | Real-time currently worn avatar configuration | Permitted |
| `games.user_created` & `games.user_favorites` | `GET https://games.roblox.com/v2/users/{userId}/games` & `/favorite/games` | Public experiences created (`id`, `name`, `created`, `updated`, `placeVisits`) and favorited by user | None (Public) | Documented domain API | ~100 req/min per IP | Real-time public places and favorite games | Permitted |
| `inventory.can_view` & `inventory.hats_sample` | `GET https://inventory.roblox.com/v1/users/{userId}/can-view-inventory` & `/v2/users/{userId}/inventory/8` | Boolean `canView` and timestamped asset acquisition records (`assetId`, `created`) when public | None when `canView=true` (HTTP 403 if private) | Documented domain API | **Strict flood limit** (HTTP 429); guarded by circuit breaker | Only available for users who leave inventory visibility public | Permitted |
| `badges.user_badges` *(Restricted)* | `GET https://badges.roblox.com/v1/users/{userId}/badges` | User badge list | **Requires Auth Cookie/Token (HTTP 401 code 9002)** | Documented | Returns HTTP 401 without authentication | Explicitly recorded as restricted/unavailable; never bypassed | Not queried without auth |
| `friends.followers_list` / `followings_list` *(Restricted)* | `GET https://friends.roblox.com/v1/users/{userId}/followers` | Follower/following user lists | **Requires Auth Cookie/Token (HTTP 401 code 9002)** | Documented | Returns HTTP 401 without authentication (only `/count` is public) | Explicitly recorded as restricted/unavailable; never bypassed | Not queried without auth |

---

## 3. Architecture & Layered Detection Methodology

### Package Structure

```text
alt_detector/
    __init__.py              # Package metadata and schema/model version constants
    __main__.py              # Module execution entrypoint (python -m alt_detector)
    cli.py                   # CLI parser, real-time logger, and AltDetectionEngine orchestrator
    config.py                # EngineConfig and local CPU/RAM/GPU hardware detection
    exceptions.py            # Structured exception hierarchy
    schemas.py               # Boundary validation and internal/output Pydantic v2 models
    storage.py               # Thread-safe SQLite WAL cache, TTL expiration, and privacy deletion
    api_client.py            # Connection-reusing httpx client with backoff, jitter, and budgets
    collectors.py            # Modular public profile, group, friend, avatar, and game collectors
    candidate_discovery.py   # Traceable multi-strategy candidate generation and prioritization
    graph_analysis.py        # Layer D heterogeneous rarity-weighted graph analysis
    features.py              # 23-feature pairwise extraction with missing-value separation
    scoring.py               # Layer A TransparentBaselineScorer and evidence synthesis
    model.py                 # Layer B supervised models (GroupKFold) and Layer C IsolationForest/LOF
    reporting.py             # Concise terminal formatter and versioned JSON report exporter
    validation.py            # Isolated 24-account real-world validation and benchmark runner
tests/
    fixtures/sample_accounts.py
    test_api_client.py
    test_collectors.py
    test_candidate_discovery.py
    test_features.py
    test_scoring.py
    test_model.py
    test_cli.py
```

### Candidate Discovery Strategy (`candidate_discovery.py`)
A username alone does not list alternative accounts. The engine discovers candidates within a strict request budget (`default=90` requests) using four traceable public strategies:
1. **Lexical Stem Search (`lexical_search`)**: Extracts non-generic handle stems (`len >= 4`) from `username`, `display_name`, and `previous_usernames` and queries `GET /v1/users/search`.
2. **Direct Friendship Graph (`direct_friend`)**: Retrieves `GET /v1/users/{id}/friends`, batch-resolves friend usernames via `POST /v1/users`, and boosts friends exhibiting lexical stem or numeric affix overlap.
3. **Rarity-Filtered Group & Role Enumeration (`small_group_co_member`, `shared_role_co_member`, `group_owner`)**: Enumerates members of the target's smallest groups (`memberCount <= 500`) and co-holders of elevated roles (`rank >= 100` in groups `<= 5,000` members), weighting each candidate by $1 / \log_{10}(\max(N_g, 2) + 1)$.
4. **Experience Creator Cross-References (`favorite_game_creator`)**: Identifies individual user creators of experiences favorited by the target.

### Feature Engineering (`features.py`)
23 normalized pairwise features are organized into **6 orthogonal evidence families** to prevent collinear signals from inflating confidence:
1. **`lexical_naming`** (weight `0.30`): `username_char_ngram_sim` (character 3-gram cosine), `username_lcs_ratio` (core handle Longest Common Substring containment), `display_name_similarity`, `shared_distinctive_affix_score` (shared non-common numeric tags like `416` or lexical tokens).
2. **`group_structure`** (weight `0.30`): `group_jaccard`, `rarity_weighted_group_sim` (inverse-log-popularity weighted Jaccard + regularized containment), `small_group_co_membership_score` (capped Adamic-Adar over groups $\le 2,500$ members), `shared_elevated_role_score` (co-holding ranks $\ge 100$ or owner/officer roles in small/medium groups).
3. **`social_graph`** (weight `0.20`): `is_direct_friend`, `mutual_friends_jaccard`, `mutual_friends_containment` ($|F_u \cap F_v| / \max(\min(|F_u|, |F_v|), 4)$).
4. **`profile_avatar`** (weight `0.08`): `description_similarity`, `avatar_asset_overlap` (excluding common default free Rthro animations), `avatar_style_similarity`, `creation_burst_proximity` ($\exp(-\Delta t_{\text{days}} / 45)$).
5. **`activity_games`** (weight `0.05`): `favorite_games_overlap` (including cross-favorited user-created places), `created_games_similarity`, `inventory_asset_overlap`, `activity_timestamp_overlap`.
6. **`graph_topology`** (weight `0.07`): `graph_adamic_adar`, `graph_resource_allocation`, `graph_personalized_pagerank`, `graph_same_community`.

**Missing-Value Handling:** Unavailable signals (e.g., empty profile descriptions, private inventories, rate-limited endpoints) are represented as `None` (`NaN`), never as `0.0` dissimilarity.

### Layered Detection Pipeline (`scoring.py`, `model.py`, `graph_analysis.py`)
- **Layer A (`TransparentBaselineScorer`)**: Computes bounded within-family scores and applies a **multi-family corroboration guardrail**: a candidate with only a single active signal family (such as a similar username with zero shared groups/friends, or a single shared group with zero lexical/social overlap) is capped below the `Moderate evidence` threshold (`< 0.45`).
- **Layer B (`SupervisedPairwiseModel`)**: Evaluates `LogisticRegression`, `RandomForestClassifier`, and `GradientBoostingClassifier` (each with explicit missingness indicator columns via `SimpleImputer(add_indicator=True)`) using group-isolated `StratifiedGroupKFold` / `GroupKFold` keyed by `operator_cluster_id` to prevent multi-account leakage across train/test splits.
- **Layer C (`UnsupervisedAccountAnalyzer`)**: Computes `IsolationForest` anomaly indices and `LocalOutlierFactor` flags on intrinsic account activity vectors to identify `sparse_minimal_profile` vs `standard_active_profile` archetypes, kept strictly separate from linkage scores.
- **Layer D (`CandidateGraphAnalyzer`)**: Builds a typed heterogeneous `networkx` graph over users, groups, elevated roles, friends, custom avatar assets, and games to compute Adamic-Adar, Resource Allocation, Personalized PageRank, and greedy modularity communities.

---

## 4. Installation, Setup & CLI Usage

### Requirements & Installation
Requires Python `>= 3.11` (verified on Python `3.14.0` on Windows 11).

```powershell
pip install -e .[dev]
```

### CLI Commands

```powershell
# 1. Verify local hardware (CPU/RAM/GPU), SQLite storage, and configuration (Dry-run / Preflight)
python -m alt_detector --preflight

# 2. Analyze a target account by Roblox username (or numeric user ID)
python -m alt_detector witrecx64

# 3. Analyze with verbose diagnostics and export machine-readable JSON report
python -m alt_detector witrecx64 -v --json-out report_witrecx64.json

# 4. Explicitly compare specific accounts alongside discovered candidates
python -m alt_detector witrecx64 --compare witrecx witrecx_416 mineturtle_416

# 5. Privacy controls: delete cached records for a specific user or clear the entire local SQLite cache
python -m alt_detector --delete-user witrecx64
python -m alt_detector --clear-cache

# 6. Run the isolated 24-account real-world validation & ML benchmark suite
python -m alt_detector --validate
```

---

## 5. Automated Testing & Real-World Validation Results

### Automated Unit & Integration Test Suite (`pytest`)
Run the full deterministic test suite (uses `httpx.MockTransport` and synthetic/realistic fixtures; does not hit live Roblox servers):

```powershell
python -m pytest -v
```

- **Actual Result:** `19 passed in 7.12s` across 7 test modules (`test_api_client.py`, `test_collectors.py`, `test_candidate_discovery.py`, `test_features.py`, `test_scoring.py`, `test_model.py`, `test_cli.py`).

### Real-Account Validation & Grouped ML Benchmark (`alt_detector/validation.py`)
Executed on `2026-10-16` against **24 distinct real Roblox accounts** in an isolated SQLite database (`.alt_detector_cache/validation/validation_isolated.db`):
- **Accounts Tested:** `24 targeted / 24 collected successfully` (`0` account-level failures).
- **Categories Covered:**
  - Primary target & strongly linked candidate (`witrecx64` [93910323], `witrecx` [2349691539])
  - Sparse public profile (`witrecx_416` [8814991503] — 0 friends, 0 groups, empty bio)
  - Small-group peers (`mineturtle_416`, `yonacool6`, `StarFireIce29`, `HAAGAs`, `ImLordOxide`, `Skyattack397`, `Marius55555555`, `Nominalmetal` in 9-member group *Paragon Armaments*)
  - High-activity developer & co-founder accounts (`asimo3089` [68603365], `badcc` [13961135], `builderman` [156], `Shedletsky` [261])
  - Popular mega-group owners/members (`Enclamatic` [1220158530] in 920k-member *Arc I: Warriors*, `UristMcSparks` [6783205] in 3.7M-member *Roblox Wiki*)
  - Unrelated social controls (`1725548849`, `395422980`, `5306681491`, `419857724`)
  - Inaccessible/locked/private-inventory accounts (`Roblox` [1], `John Doe` [2], `Jane Doe` [3])

#### Empirical Pairwise Validation Highlights
| Account Pair | Relationship Category | Baseline Score | Confidence Band | Key Evidence & Guardrail Behavior |
| :--- | :--- | :---: | :--- | :--- |
| `witrecx64 (93910323) <-> witrecx (2349691539)` | Strongly linked candidate | `0.887` | **Strong evidence** | Exact root handle (`LCS=1.0`), direct friend + 15 mutuals, identical elevated ranks (`Senior Moderator` rank 253 in *Site-416: Staff Team*, `E8 Master Sergeant` rank 165 in *Task Force 416*, `Vice President`/`High Marshal` in *Grand Cicada Front*) |
| `witrecx64 (93910323) <-> mineturtle_416 (1719913571)` | Small-group clan peer | `0.403` | **Weak evidence** | Shares 9-member group *Paragon Armaments*, mutual friends, `_416` suffix, and `Silly` display prefix, but distinct root handle (`mineturtle` vs `witrecx`) |
| `witrecx64 (93910323) <-> StarFireIce29 (159708565)` | Small-group peer & friend | `0.383` | **Weak evidence** | Co-member in *Paragon Armaments* and mutual friend, zero lexical similarity |
| `witrecx64 (93910323) <-> witrecx_416 (8814991503)` | Uncertain sparse handle match | `0.228` | **Insufficient evidence** | High lexical score (`0.881`), but `0` groups and `0` friends; single-family guardrail prevents false high-confidence accusation |
| `asimo3089 (68603365) <-> badcc (13961135)` | Known distinct studio co-founders (Hard negative) | `0.078` | **Insufficient evidence** | Correctly avoids conflating distinct human co-developers |
| `witrecx64 (93910323) <-> Enclamatic (1220158530)` | Shared 920k-member mega-group | `0.017` | **Insufficient evidence** | Popular group membership downweighted (`group_structure=0.0032`) |

#### Group-Isolated Model vs. Baseline Benchmark (4-Fold `StratifiedGroupKFold`, 29 Operator Clusters, 46 Evaluable Pairs + 2 Uncertain Excluded)

| Model Name | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier Score | ECE | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `transparent_baseline` (Layer A) | `1.000` | `0.818` | `0.900` | `0.894` | `0.907` | `0.062` | `0.077` | `[[35, 0], [2, 9]]` |
| `logistic_regression` (Layer B) | `0.909` | `0.909` | `0.909` | `0.944` | `0.966` | `0.040` | `0.031` | `[[34, 1], [1, 10]]` |
| `random_forest` (Layer B - Selected) | `1.000` | `0.818` | `0.900` | `0.951` | `0.966` | `0.039` | `0.032` | `[[35, 0], [2, 9]]` |
| `hist_gradient_boosting` (Layer B) | `1.000` | `0.818` | `0.900` | `0.873` | `0.894` | `0.044` | `0.041` | `[[35, 0], [2, 9]]` |

*Note on Weak Supervision:* Because public Roblox telemetry does not provide large-scale, independently consented ground-truth identity labels, the system defaults to `heuristic_baseline_primary` / `research_weak_supervision` mode and labels decision thresholds (`Strong >= 0.68`, `Moderate >= 0.45`, `Weak >= 0.25`) as **provisional**. Scores are never described as calibrated probabilities of human identity.

---

## 6. Statement of Capabilities and Limitations

### What the Engine Can Reliably Detect
- **Corroborated Multi-Signal Alt Accounts:** Accounts that reuse a lexical handle stem or naming convention **and** intersect the target's small organizational groups, elevated group roles, friend circle, or created/favorited experiences (e.g., `witrecx64` and `witrecx`).
- **Small-Group & Clan Network Peers:** Accounts operating within the same small group (`N <= 500`) or staff hierarchy, explicitly separated from strong alt predictions when lexical identity signals are absent and accompanied by clan/colleague alternative explanations.
- **Popular vs. Rare Shared Attributes:** Rarity-weighted group and graph metrics suppress noise from mega-groups (e.g., >100k-member game fan groups) and default free avatar animations.

### What the Engine Cannot Reliably Detect
- **Compartmentalized ("Clean") Alt Accounts:** An alternative account with a completely unrelated username, default/distinct avatar, zero shared friends, and zero shared small groups cannot be discovered or linked using public APIs.
- **Shared Household / Sibling / Co-Developer Accounts vs. Single Operator:** Two distinct people who share a naming theme, co-administer the same small groups, and share the same friend circle produce structural signals indistinguishable from a single operator's alt account using public data alone.
- **Private / Auth-Restricted Signals:** User badge lists (`badges.roblox.com`), follower/following lists (`friends.roblox.com`), private inventories (`canView=false`), IP addresses, and hardware identifiers are inaccessible without authentication or privileged platform telemetry.
