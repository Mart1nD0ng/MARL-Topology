import json
from pathlib import Path

from marl_topology.models import CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
from marl_topology.training.mappo.stage28_repaired_critic_pilot import (
    STAGE28_CONFIG_ID,
    Stage28PilotConfig,
    build_stage28_manifest,
    run_stage28_preflight,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    ROOT
    / "result_save"
    / "stage28_repaired_critic_mappo_rerun"
    / "stage28_repaired_critic_stage25_base_protocol"
    / "training_report.json"
)


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_stage28_config_reuses_stage25_fixed_base_protocol() -> None:
    config = Stage28PilotConfig()
    payload = config.to_payload()

    assert payload["config_id"] == STAGE28_CONFIG_ID
    assert payload["inherits_stage25_fixed_base_protocol"] is True
    assert payload["train_scenarios"] == 16
    assert payload["eval_scenarios"] == 8
    assert payload["seeds"] == [2501, 2502, 2503]
    assert payload["rollout_steps"] == 16
    assert payload["transitions_per_update"] == 256
    assert payload["critic_model_id"] == CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
    assert payload["hyperparameter_tuning_performed"] is False
    assert payload["scale_up_training"] is False


def test_stage28_preflight_uses_selected_stage27_graph_critic() -> None:
    preflight = run_stage28_preflight(project_root=ROOT)

    assert preflight["preflight_passed"] is True
    gate = preflight["gates"]["stage27_selected_graph_critic_available"]
    assert gate["passed"] is True
    assert gate["selected_critic_id"] == CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
    assert gate["active_future_value_critics"] == [CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID]


def test_stage28_manifest_is_report_validated_no_checkpoint_scope() -> None:
    manifest = build_stage28_manifest()

    assert manifest["stage_id"] == "stage_28_rerun_small_scale_mappo_with_repaired_critic"
    assert manifest["config_id"] == STAGE28_CONFIG_ID
    assert manifest["critic_model_id"] == CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
    assert manifest["artifact_write_allowed"] == "manifest_validated_reports_only"
    assert manifest["checkpoint_creation_allowed"] is False
    assert manifest["training_scale_up_allowed"] is False


def test_stage28_report_passes_with_repaired_critic_health() -> None:
    report = _report()
    gate = report["pass_fail_gate"]
    critic = gate["critic_health"]

    assert report["pass_gate"] is True
    assert report["verdict"] == "stage28_pass_repaired_critic_small_scale_rerun_complete"
    assert gate["issues"] == []
    assert gate["completed_seed_count"] == 3
    assert gate["stage27_selected_critic_reused"] is True
    assert critic["mean_update_explained_variance"] > 0.10
    assert critic["mean_value_return_correlation"] > 0.30
    assert report["aggregate"]["completed_seed_count"] == 3
    assert report["stage25_historical_comparison"]["repaired_critic_improved_old_critic_result"] is True
