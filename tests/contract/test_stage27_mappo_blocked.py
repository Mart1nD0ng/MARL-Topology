import json
from pathlib import Path

from marl_topology.training.critic_repair_trainer import (
    Stage27CriticRepairTrainConfig,
    run_stage27_critic_repair,
)

from stage27_synthetic import make_stage27_synthetic_dataset


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    ROOT
    / "result_save"
    / "stage27_critic_baseline_repair"
    / "stage27_critic_repair_dataset_and_train_config"
    / "stage27_critic_repair_report.json"
)


def test_stage27_does_not_authorize_policy_training_automatically() -> None:
    report = _load_report_or_synthetic()
    readiness = report["mappo_readiness"]

    assert readiness["active_critic_selected"] is True
    assert readiness["next_stage_allowed_with_owner_decision"] is True
    assert readiness["recommended_next_task"] == (
        "stage_28_rerun_small_scale_mappo_with_repaired_critic"
    )
    assert readiness["policy_gradient_update_performed"] is False
    assert readiness["actor_update_performed"] is False


def test_stage27_project_state_keeps_scale_up_and_lstm_blocked_after_closeout() -> None:
    state_text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required_terms = [
        "post_stage_27_complete_critic_repaired_awaiting_owner_decision",
        "stage_27_critic_baseline_repair",
        "stage27_critic_selection_gate",
        "stage27_no_actor_update_gate",
        "stage_28_rerun_small_scale_mappo_with_repaired_critic",
        "Scale-up training remains blocked",
        "LSTM/recurrent PPO remains blocked",
    ]
    missing = [term for term in required_terms if term not in state_text]
    assert missing == []


def test_stage27_actor_policy_and_sampler_remain_active_stack_unchanged() -> None:
    report = _load_report_or_synthetic()

    assert report["dataset_health"]["actor_policy_id"] == (
        "stage27_frozen_local_gnn_policy_no_actor_update"
    )
    assert report["dataset_health"]["sampler_id"] == "physical_plackett_luce_top_k_sampler"
    assert report["mappo_readiness"]["reward_weights_changed"] is False
    assert report["mappo_readiness"]["sampler_switched"] is False


def _load_report_or_synthetic() -> dict[str, object]:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return run_stage27_critic_repair(
        dataset=make_stage27_synthetic_dataset(),
        config=Stage27CriticRepairTrainConfig(
            epochs=2,
            min_eval_explained_variance=-10.0,
            min_eval_value_return_correlation=-10.0,
            min_bias_reduction_fraction=-10.0,
            min_advantage_variance_reduction=-10.0,
        ),
    )
