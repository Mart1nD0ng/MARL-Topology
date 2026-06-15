from pathlib import Path

from marl_topology.data.stage33_graph_structure_dataset import (
    Stage33GraphStructureConfig,
    build_stage33_graph_structure_dataset,
)
from marl_topology.models import ACTIVE_STAGE33_GNN_MODEL_ID
from marl_topology.training.production_mappo_adapter import (
    STAGE32_CUSTOM_LOOP_MODULE,
    STAGE33_CONFIG_ID,
    Stage33ProductionMappoAdapter,
    build_stage33_manifest,
)
from marl_topology.training.run_manifest_validator import validate_run_manifest_dry_run


ROOT = Path(__file__).resolve().parents[2]


def test_stage33_adapter_reports_official_mappo_components() -> None:
    report = Stage33ProductionMappoAdapter().official_component_report()

    assert report["official_mappo_trainer_path"] == "marl_topology.training.mappo"
    assert report["official_rollout_collector"] == "_collect_repaired_rollout_batch"
    assert report["official_advantage_function"] == "compute_gae_returns"
    assert report["official_loss_function"] == "clipped_policy_value_loss"
    assert report["official_repaired_critic_loss_adapter"] == "_loss_for_repaired_indices"
    assert report["stage32_custom_training_loop_called"] is False
    assert report["stage32_custom_training_loop_module"] == STAGE32_CUSTOM_LOOP_MODULE
    assert report["reimplements_ppo_loss_locally"] is False
    assert report["hand_rolled_reinforce_update"] is False


def test_stage33_adapter_builds_actor_safe_row_contexts_with_family_metadata() -> None:
    dataset = build_stage33_graph_structure_dataset(
        Stage33GraphStructureConfig(scenario_count=7, node_count_choices=(6,))
    )
    row, _context = Stage33ProductionMappoAdapter().build_row_contexts(dataset, "train")[0]

    assert row.structural_family
    assert row.actor_safe_view
    assert all(
        actor_row["stage33_structural_family"] == row.structural_family
        for actor_row in row.actor_safe_view
    )


def test_stage33_manifest_validates_and_selects_active_gnn() -> None:
    manifest = build_stage33_manifest()
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)

    assert validation.is_valid
    assert manifest["config_id"] == STAGE33_CONFIG_ID
    assert manifest["model_id"] == ACTIVE_STAGE33_GNN_MODEL_ID
    assert manifest["checkpoint_creation_allowed"] is False
