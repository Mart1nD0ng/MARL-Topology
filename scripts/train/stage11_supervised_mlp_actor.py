#!/usr/bin/env python3
"""Stage 11 supervised Local MLP actor warm start."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.run_manifest_validator import (  # noqa: E402
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)
from marl_topology.training.supervised_batching import load_learning_evidence_json  # noqa: E402
from marl_topology.training.supervised_actor_trainer import (  # noqa: E402
    STAGE11_SUPERVISED_ACTOR_STAGE_ID,
    SupervisedActorTrainingConfig,
    run_supervised_actor_training,
)


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_stage11_manifest(run_id: str) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save/stage11_supervised_mlp_actor",
        overrides={
            "run_id": run_id,
            "stage_id": STAGE11_SUPERVISED_ACTOR_STAGE_ID,
            "owner_approval_id": "active_goal_stage_9_to_15_owner_approved",
            "config_id": "stage11_supervised_mlp_actor_config_v1",
            "scenario_set_id": "stage7_completion_learning_evidence_dataset_v1",
            "split_id": "deterministic_row_order_75_25",
            "seed": 11,
            "seed_group_id": "stage11_single_seed_smoke",
            "model_id": "local_mlp_edge_scorer_stage11_supervised",
            "data_ids": ["stage7_completion_learning_evidence_dataset_v1"],
            "objective_id": "objective_contract_stage5_0_tau_requirement_min_0_9",
            "reward_id": "not_used_supervised_actor_warm_start",
            "architecture_id": "local_mlp_edge_scorer_v2_model_package",
            "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
            "artifact_paths": [
                "result_save/stage11_supervised_mlp_actor/"
                f"{run_id}/manifest.json",
                "result_save/stage11_supervised_mlp_actor/"
                f"{run_id}/training_report.json",
            ],
        },
    )


def build_report(*, write_artifacts: bool = True) -> dict[str, object]:
    evidence = load_learning_evidence_json(EVIDENCE_PATH)
    rows = tuple(dict(row) for row in evidence["rows"])  # type: ignore[index]
    config = SupervisedActorTrainingConfig()
    _model, result = run_supervised_actor_training(rows, config=config)
    manifest = build_stage11_manifest("stage11_supervised_mlp_actor_seed11")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)
    if not validation.is_valid:
        raise RuntimeError(f"manifest validation failed: {validation.error_codes()}")
    report = {
        "stage": STAGE11_SUPERVISED_ACTOR_STAGE_ID,
        "manifest_validation": validation.to_dict(),
        "training_result": result.to_dict(),
        "dataset_id": evidence["dataset_id"],
        "actor_only_training": True,
        "critic_training_performed": False,
        "rl_training_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
    }
    if write_artifacts:
        artifact_dir = ROOT / "result_save" / "stage11_supervised_mlp_actor" / str(manifest["run_id"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        )
        report = {**report, "artifact_written": True}
        (artifact_dir / "training_report.json").write_bytes(
            json.dumps(report, indent=2, sort_keys=True).encode("utf-8")
        )
    return report


def main() -> int:
    print(json.dumps(build_report(write_artifacts=False), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
