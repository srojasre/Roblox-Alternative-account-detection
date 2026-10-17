"""Configuration and local hardware resource management for alt_detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import platform
import shutil
import subprocess
from typing import Any

import psutil


DEFAULT_FAMILY_WEIGHTS: dict[str, float] = {
    "lexical_naming": 0.30,
    "group_structure": 0.30,
    "social_graph": 0.20,
    "profile_avatar": 0.08,
    "activity_games": 0.05,
    "graph_topology": 0.07,
}

DEFAULT_SCORE_THRESHOLDS: dict[str, float] = {
    "strong": 0.68,
    "moderate": 0.45,
    "weak": 0.25,
}


@dataclass(frozen=True)
class HardwareResources:
    """Snapshot of locally available CPU, RAM, and optional GPU resources."""

    os_name: str
    python_version: str
    logical_cpus: int
    physical_cpus: int
    total_ram_mb: float
    available_ram_mb: float
    gpu_available: bool
    gpu_name: str | None
    recommended_workers: int


def detect_local_resources(max_workers_cap: int = 4) -> HardwareResources:
    """Inspect local CPU, RAM, and GPU availability without accessing private files."""
    logical_cpus = os.cpu_count() or 1
    physical_cpus = psutil.cpu_count(logical=False) or logical_cpus
    vm = psutil.virtual_memory()
    total_ram_mb = round(vm.total / (1024 * 1024), 1)
    available_ram_mb = round(vm.available / (1024 * 1024), 1)

    gpu_available = False
    gpu_name: str | None = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpu_available = True
                gpu_name = proc.stdout.strip().splitlines()[0].strip()
        except Exception:
            gpu_available = False

    recommended_workers = max(1, min(physical_cpus, max_workers_cap))
    return HardwareResources(
        os_name=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        logical_cpus=logical_cpus,
        physical_cpus=physical_cpus,
        total_ram_mb=total_ram_mb,
        available_ram_mb=available_ram_mb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        recommended_workers=recommended_workers,
    )


@dataclass
class EngineConfig:
    """Configuration for the Roblox Alt Account Detection Engine.

    Kept separate from business logic; supports deterministic overrides via CLI
    flags or environment variables.
    """

    cache_dir: Path = field(default_factory=lambda: Path(".alt_detector_cache"))
    db_path: Path | None = None
    model_dir: Path | None = None

    # Caching and retention policies
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_entries: int = 5000
    store_account_summaries: bool = True

    # HTTP & rate-limit resilience
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    base_backoff_seconds: float = 0.75
    max_backoff_seconds: float = 8.0
    min_request_interval_seconds: float = 0.08
    max_concurrency: int = 4
    request_budget: int = 110

    # Candidate discovery and expansion bounds
    max_candidates: int = 18
    max_group_size_for_expansion: int = 500
    max_groups_to_enumerate: int = 4
    max_role_members_to_fetch: int = 50
    max_lexical_queries: int = 2
    extended_profile_top_k: int = 5

    # Scoring & ML configuration
    enable_ml: bool = True
    random_seed: int = 42
    family_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FAMILY_WEIGHTS)
    )
    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SCORE_THRESHOLDS)
    )
    threshold_status: str = "provisional"
    weights_version: str = "1.0.0"

    # Execution & reporting
    verbose: bool = False

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if self.db_path is None:
            self.db_path = self.cache_dir / "alt_detector.db"
        else:
            self.db_path = Path(self.db_path)

        if self.model_dir is None:
            self.model_dir = self.cache_dir / "models"
        else:
            self.model_dir = Path(self.model_dir)

        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.request_budget < 5:
            raise ValueError("request_budget must be at least 5")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the configuration."""
        return {
            "cache_dir": str(self.cache_dir),
            "db_path": str(self.db_path),
            "model_dir": str(self.model_dir),
            "use_cache": self.use_cache,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_entries": self.max_cache_entries,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "base_backoff_seconds": self.base_backoff_seconds,
            "max_concurrency": self.max_concurrency,
            "request_budget": self.request_budget,
            "max_candidates": self.max_candidates,
            "max_group_size_for_expansion": self.max_group_size_for_expansion,
            "max_groups_to_enumerate": self.max_groups_to_enumerate,
            "enable_ml": self.enable_ml,
            "random_seed": self.random_seed,
            "family_weights": dict(self.family_weights),
            "thresholds": dict(self.thresholds),
            "threshold_status": self.threshold_status,
            "weights_version": self.weights_version,
            "verbose": self.verbose,
        }
