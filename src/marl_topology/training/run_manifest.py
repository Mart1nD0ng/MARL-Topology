"""Active run manifest + comparison-arms registry (v2 Engineering-Plan §R0).

Every training run emits a manifest that makes it reproducible and its mechanism
state auditable: the git revision, the environment-math version, the action-
distribution version, the dataset manifest, the seed/split, the activated
mechanisms (D6), and the evaluator-call budget (the primary fairness budget).

This is a NEW *active* manifest, distinct from the frozen ``run_manifest_contract``
/ ``run_manifest_validator`` (the Stage-5.9/5.10 dry-run *design* contract, which is
``planned_not_active`` and forbids writing). It lives under ``training/`` (gate-
exempt) and is imported only by training entrypoints, never by a deployed path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from .config_tiers import TIER_NAMES

# Bumped whenever the environment math changes in a way that makes prior numbers
# incomparable (Spec §0.1; v2 plan principle 10). The v2-corrected env math
# (safe quorum + fixed-B + one-hop relay + tri-state + phase accounting).
ENVIRONMENT_MATH_VERSION = "v2-corrected-2026-06-23"

# The action distribution in the production rollout path. R6/R7 landed the unordered
# Budget-Conditioned Subset Policy (BCSP) in the graph-mappo production arm (the prior
# REVISE is resolved); the legacy ema/rloo baselines still use ordered Plackett-Luce.
ACTION_DISTRIBUTION_VERSION = "bcsp_v1_unordered_subset"

# The four registered comparison arms (v2 plan R0 work-item 3 / R7 fairness table).
# evaluator_calls_per_scene is the primary fairness budget (Spec §9.8): it must be
# equal across arms in a fair A/B; RLOO needs M>=2 evaluator calls per scene.
ARMS: dict[str, dict[str, object]] = {
    "ema": {
        "description": "REINFORCE with an EMA baseline (M=1) -- required baseline (Spec §15)",
        "evaluator_calls_per_scene": 1,
        "is_production_candidate": False,
    },
    "rloo": {
        "description": "REINFORCE leave-one-out (M>=2 samples/scene) -- required baseline (Spec §15)",
        "evaluator_calls_per_scene": 2,
        "is_production_candidate": False,
    },
    "graph-mappo": {
        "description": "CTDE Graph-MAPPO + BCSP: per-agent ratio, central graph critic (Phase 7)",
        "evaluator_calls_per_scene": 1,
        "is_production_candidate": True,
    },
    "production": {
        "description": (
            "the single production trunk (transitional: the REINFORCE-EMA/RLOO entry; "
            "becomes the CTDE-SCQ entry at Phase 13)"
        ),
        "evaluator_calls_per_scene": 1,
        "is_production_candidate": True,
    },
}

REQUIRED_MANIFEST_FIELDS = (
    "git_revision",
    "environment_math_version",
    "action_distribution_version",
    "dataset_manifest",
    "seed",
    "split",
    "mechanism_activation",
    "evaluator_call_budget",
    "tier",
    "arm",
    "created_at_utc",
    "headline_eligible",
)


@dataclass(frozen=True)
class RunManifest:
    """An active, reproducible run manifest (one per training run)."""

    git_revision: str
    environment_math_version: str
    action_distribution_version: str
    dataset_manifest: dict
    seed: int
    split: str
    mechanism_activation: dict
    evaluator_call_budget: int
    tier: str
    arm: str
    created_at_utc: str
    headline_eligible: bool = False

    def to_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RunManifest":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_run_manifest(
    *,
    git_revision: str,
    seed: int,
    split: str,
    dataset_manifest: dict,
    mechanism_activation: dict,
    evaluator_call_budget: int,
    tier: str,
    arm: str,
    created_at_utc: str,
    headline_eligible: bool = False,
    environment_math_version: str = ENVIRONMENT_MATH_VERSION,
    action_distribution_version: str = ACTION_DISTRIBUTION_VERSION,
) -> RunManifest:
    """Build + validate a run manifest, stamping the env-math / action-dist versions."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; must be one of {sorted(ARMS)}")
    if tier not in TIER_NAMES:
        raise ValueError(f"unknown tier {tier!r}; must be one of {TIER_NAMES}")
    if tier in ("smoke", "pilot") and headline_eligible:
        raise ValueError(f"tier {tier!r} cannot be headline_eligible (smoke/pilot are never headlines)")
    if not isinstance(evaluator_call_budget, int) or isinstance(evaluator_call_budget, bool):
        raise ValueError("evaluator_call_budget must be an int")
    if evaluator_call_budget < 0:
        raise ValueError("evaluator_call_budget must be >= 0")
    if not isinstance(dataset_manifest, dict) or not dataset_manifest:
        raise ValueError("dataset_manifest must be a non-empty dict")
    if not isinstance(mechanism_activation, dict) or not mechanism_activation:
        raise ValueError("mechanism_activation must be a non-empty dict (D6)")
    return RunManifest(
        git_revision=git_revision,
        environment_math_version=environment_math_version,
        action_distribution_version=action_distribution_version,
        dataset_manifest=dataset_manifest,
        seed=seed,
        split=split,
        mechanism_activation=mechanism_activation,
        evaluator_call_budget=evaluator_call_budget,
        tier=tier,
        arm=arm,
        created_at_utc=created_at_utc,
        headline_eligible=headline_eligible,
    )
