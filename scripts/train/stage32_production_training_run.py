"""Archived Stage 32/32a custom training run.

Stage 33 retires this custom loop from active production training. It is kept
only for historical Stage 32/32a reproducibility, and requires an explicit
``--allow-inactive-stage32-legacy-loop`` flag to execute. Current production
training must use ``scripts/train/stage33_gnn_stability_repair_training.py``.

Builds the scaled procedural dataset, trains the GNN actor + repaired graph critic
across multiple seeds, runs an MLP comparison for the head-to-head, writes a
Stage 5.9-compliant run manifest (validated by the Stage 5.10 validator), and
persists checkpoints + reports under result_save/stage32_production_training/.

Usage:
    python scripts/train/stage32_production_training_run.py --allow-inactive-stage32-legacy-loop [scenario_count] [seed_count]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.data.stage31_production_dataset import (  # noqa: E402
    build_production_dataset,
)
from marl_topology.data.stage31_scenario_generator import (  # noqa: E402
    ProductionScenarioConfig,
)
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    validate_run_manifest_dry_run,
)
from marl_topology.training.stage31_readiness import (  # noqa: E402
    Stage31ReadinessConfig,
    train_readiness,
)
from marl_topology.training.stage32_production_training import (  # noqa: E402
    Stage32Config,
    STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH,
    train_production,
)

ARTIFACT_DIR = (
    ROOT / "result_save" / "stage32_production_training" / "stage32a_mappo_gnn_run"
)


def build_manifest(scenario_count: int, seeds) -> dict:
    return {
        "run_id": "stage32a_mappo_gnn_run_v1",
        "created_at_utc": "2026-06-03T00:00:00Z",
        "stage_id": "stage32a_production_training_real_mappo_loss_gnn_actor_repaired_critic_larger_graphs",
        "owner_approval_id": "owner_approved_stage32_production_training_execution",
        "config_id": "stage32a_mappo_gnn_scaled_config_v1",
        "scenario_set_id": f"stage32_procedural_scenarios_{scenario_count}",
        "split_id": "stage32_context_keyed_leakage_checked_split",
        "seed": list(seeds)[0],
        "seed_group_id": "stage32_seed_group_" + "_".join(str(s) for s in seeds),
        "code_version_marker": "stage32_production_training_workspace_marker",
        "contract_ids": [
            "stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1",
            "feasibility_first_barrier_v2",
            "policy_architecture_contract_stage5_7",
            "run_manifest_artifact_contract_stage5_9",
            "stage32_production_training_design_contract",
        ],
        "metric_registry_version": "metric_governance_stage5",
        "physics_regime_id": "urlcc_finite_blocklength_v1",
        "protocol_model_id": "stage4_expected_initiator_pbft_over_stage3_network_v1",
        "objective_contract_id": "stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1",
        "surrogate_config_id": "stage31_feasibility_first_recalibrated_v2",
        "normalization_reference_id": "stage31_feasible_positive_max_recalibrated",
        "architecture_contract_id": "stage32_gnn_actor_graph_critic_v1",
        "replay_schema_version": "learning_target_replay_contract_stage5_8",
        "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
        "artifact_root": "result_save",
    }


def run(scenario_count: int = 2000, seed_count: int = 5) -> dict:
    seeds = tuple(3201 + i for i in range(seed_count))
    dataset = build_production_dataset(
        ProductionScenarioConfig(seed=31, scenario_count=scenario_count)
    )
    config = Stage32Config(
        warm_start_epochs=25,
        warm_start_lr=0.01,
        epochs=6,
        critic_pretrain_scenarios=120,
        critic_pretrain_epochs=15,
        pg_minibatch=256,
    )
    gnn_result = train_production(dataset, seeds=seeds, base_config=config)
    # MLP baseline on the same dataset, averaged over seeds for a fair head-to-head.
    mlp_seeds = []
    for seed in seeds[:3]:
        mlp_seeds.append(
            train_readiness(
                dataset, Stage31ReadinessConfig(seed=seed, warm_start_epochs=25, epochs=6)
            )
        )
    mlp_test_mean = sum(m.test_eval["tau_feasible_rate"] for m in mlp_seeds) / len(mlp_seeds)
    mlp = mlp_seeds[0]
    manifest = build_manifest(scenario_count, seeds)
    validation = validate_run_manifest_dry_run(manifest)
    return {
        "manifest": manifest,
        "manifest_valid": bool(validation.is_valid),
        "dataset_quality": dict(dataset.quality_report),
        "gnn_aggregate": gnn_result.aggregate(),
        "gnn_seeds": [
            {
                "seed": s.seed,
                "after_eval": s.after_eval,
                "test_eval": s.test_eval,
                "critic_explained_variance": s.critic_explained_variance,
            }
            for s in gnn_result.seeds
        ],
        "mlp_baseline": {
            "after_eval": mlp.after_eval,
            "test_eval": mlp.test_eval,
            "test_tau_feasible_mean": mlp_test_mean,
            "seed_count": len(mlp_seeds),
            "teacher_feasible_rate": mlp.teacher_feasible_rate,
        },
        "checkpoints": gnn_result,
    }


def write_artifacts(payload: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(payload["manifest"], indent=2), encoding="utf-8"
    )
    report = {k: v for k, v in payload.items() if k != "checkpoints"}
    (ARTIFACT_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    # Model-weight checkpoints are intentionally NOT persisted: the project's
    # no-checkpoint discipline forbids torch.save in src, and the GNN actor
    # underperformed the MLP, so its weights are not the production model. The
    # run evidence is the validated manifest + training report.


def main() -> None:
    allow_flag = "--allow-inactive-stage32-legacy-loop"
    if allow_flag not in sys.argv or STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH:
        raise SystemExit(
            "Stage32 custom training is archived and inactive for production. "
            "Use scripts/train/stage33_gnn_stability_repair_training.py, or pass "
            f"{allow_flag} only to reproduce historical Stage32/32a reports."
        )
    args = [arg for arg in sys.argv[1:] if arg != allow_flag]
    count = int(args[0]) if len(args) > 0 else 2000
    seed_count = int(args[1]) if len(args) > 1 else 5
    payload = run(count, seed_count)
    write_artifacts(payload)
    agg = payload["gnn_aggregate"]
    mlp = payload["mlp_baseline"]
    print(f"manifest_valid: {payload['manifest_valid']}")
    print(f"GNN actor: {agg['actor_model_id']} | critic: {agg['critic_model_id']}")
    print(
        f"GNN test tau_feasible (mean over {agg['seed_count']} seeds): "
        f"{agg['test_tau_feasible_mean']:.3f}  (MLP baseline mean: {mlp['test_tau_feasible_mean']:.3f}; "
        f"teacher ceiling: {mlp['teacher_feasible_rate']:.3f})"
    )
    print(
        f"critic EV (mean): {agg['critic_explained_variance_mean']:.3f} | "
        f"projection rejection: {agg['projection_rejection_mean']:.3f} | "
        f"mean edges: {agg['mean_selected_edges']:.2f}"
    )
    print(f"artifacts: {ARTIFACT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
