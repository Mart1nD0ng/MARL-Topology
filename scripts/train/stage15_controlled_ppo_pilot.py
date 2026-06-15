#!/usr/bin/env python3
"""Stage 15 controlled clipped policy-gradient pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.ppo_pilot import (  # noqa: E402
    STAGE15_POLICY_PILOT_STAGE_ID,
    PolicyPilotConfig,
    run_controlled_policy_pilot,
)
from marl_topology.models import LOCAL_GNN_EDGE_SCORER_MODEL_ID  # noqa: E402
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)


EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def build_stage15_manifest(run_id: str) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save/stage15_controlled_policy_pilot",
        overrides={
            "run_id": run_id,
            "stage_id": STAGE15_POLICY_PILOT_STAGE_ID,
            "owner_approval_id": "active_goal_stage_9_to_15_owner_approved",
            "config_id": "stage15_controlled_policy_pilot_config_v1",
            "scenario_set_id": "demo_stage2_single_fixture_pilot",
            "split_id": "stage15_single_rollout_smoke",
            "seed": 15,
            "seed_group_id": "stage15_single_seed_smoke",
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "data_ids": ["stage7_completion_learning_evidence_dataset_v1"],
            "objective_id": "objective_contract_stage5_0_tau_requirement_min_0_9",
            "reward_id": "stage15_pilot_equal_component_weights_not_calibrated",
            "architecture_id": "local_message_passing_gnn_edge_scorer_v2",
            "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
        },
    )


def build_report() -> dict[str, object]:
    result = run_controlled_policy_pilot(EVIDENCE_PATH, config=PolicyPilotConfig())
    manifest = build_stage15_manifest("stage15_policy_pilot_seed15")
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)
    if not validation.is_valid:
        raise RuntimeError(f"manifest validation failed: {validation.error_codes()}")
    return {
        "stage": STAGE15_POLICY_PILOT_STAGE_ID,
        "manifest_validation": validation.to_dict(),
        "pilot_result": result.to_dict(),
        "logprob_semantics": {
            "policy_action": "sampled directed-edge proposal indicators from actor logits",
            "environment_transition": "projected topology from ConflictAwareGreedyAssembler",
            "projected_topology_logprob_exact": False,
            "projection_diagnostics_logged": True,
        },
        "coma_introduced": False,
        "transformer_introduced": False,
        "large_sweep_performed": False,
        "final_tau_selected": False,
        "checkpoint_written": False,
        "artifact_written": False,
    }


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
