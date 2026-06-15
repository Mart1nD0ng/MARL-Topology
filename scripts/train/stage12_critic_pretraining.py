#!/usr/bin/env python3
"""Stage 12 centralized critic and edge-delta pretraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.critic_pretrainer import (  # noqa: E402
    STAGE12_CRITIC_PRETRAINING_STAGE_ID,
    CriticPretrainingConfig,
    pretrain_centralized_critic,
)
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)
from marl_topology.training.supervised_batching import (  # noqa: E402
    build_supervised_batch_from_evidence,
    load_learning_evidence_json,
)


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_stage12_manifest(run_id: str) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save/stage12_critic_pretraining",
        overrides={
            "run_id": run_id,
            "stage_id": STAGE12_CRITIC_PRETRAINING_STAGE_ID,
            "owner_approval_id": "active_goal_stage_9_to_15_owner_approved",
            "config_id": "stage12_critic_pretraining_config_v1",
            "scenario_set_id": "stage7_completion_learning_evidence_dataset_v1",
            "split_id": "stage12_full_evidence_fit_for_fidelity_smoke",
            "seed": 12,
            "seed_group_id": "stage12_single_seed_smoke",
            "model_id": "centralized_mlp_critic_stage12_pretrained",
            "data_ids": ["stage7_completion_learning_evidence_dataset_v1"],
            "objective_id": "objective_contract_stage5_0_tau_requirement_min_0_9",
            "reward_id": "not_used_critic_pretraining",
            "architecture_id": "centralized_mlp_critic_baseline_v1",
            "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
        },
    )


def build_report() -> dict[str, object]:
    evidence = load_learning_evidence_json(EVIDENCE_PATH)
    batch = build_supervised_batch_from_evidence(evidence)
    _model, result = pretrain_centralized_critic(batch, config=CriticPretrainingConfig())
    manifest = build_stage12_manifest("stage12_critic_pretraining_seed12")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)
    if not validation.is_valid:
        raise RuntimeError(f"manifest validation failed: {validation.error_codes()}")
    return {
        "stage": STAGE12_CRITIC_PRETRAINING_STAGE_ID,
        "manifest_validation": validation.to_dict(),
        "training_result": result.to_dict(),
        "dataset_id": evidence["dataset_id"],
        "critic_training_performed": True,
        "actor_training_performed": False,
        "rl_training_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
    }


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
