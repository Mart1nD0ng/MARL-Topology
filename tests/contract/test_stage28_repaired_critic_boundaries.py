import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    ROOT
    / "result_save"
    / "stage28_repaired_critic_mappo_rerun"
    / "stage28_repaired_critic_stage25_base_protocol"
    / "training_report.json"
)
STAGE28_SOURCE = ROOT / "src" / "marl_topology" / "training" / "mappo" / "stage28_repaired_critic_pilot.py"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_stage28_keeps_forbidden_work_blocked() -> None:
    report = _report()

    assert report["reward_weight_tuning_performed"] is False
    assert report["sampler_switched"] is False
    assert report["final_tau_selected"] is False
    assert report["coma_introduced"] is False
    assert report["transformer_introduced"] is False
    assert report["gru_lstm_or_recurrent_ppo_introduced"] is False
    assert report["scale_up_training_performed"] is False
    assert report["checkpoint_written"] is False
    assert report["uncontrolled_artifact_written"] is False
    assert report["v5_modified"] is False


def test_stage28_actor_and_sampler_stack_remain_fixed() -> None:
    report = _report()

    assert report["active_policy_gradient_sampler_id"] == "physical_plackett_luce_top_k_sampler"
    assert report["config"]["critic_model_id"] == "centralized_message_passing_graph_value_critic_v1"
    assert report["stage27_repaired_critic_report"]["selected_critic_id"] == (
        "centralized_message_passing_graph_value_critic_v1"
    )
    assert report["stage27_repaired_critic_report"]["actor_update_performed"] is False
    assert report["stage27_repaired_critic_report"]["policy_gradient_update_performed"] is False


def test_stage28_source_avoids_forbidden_architecture_and_checkpoint_patterns() -> None:
    text = STAGE28_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "class COMA",
        "class Transformer",
        "class LSTM",
        "class GRU",
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "D:\\PhD_works\\v5",
        "reward_weight_tuning_performed\": true",
        "sampler_switched\": true",
    )
    hits = [term for term in forbidden if term in text]

    assert hits == []
