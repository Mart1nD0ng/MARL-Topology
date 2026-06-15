from pathlib import Path

from marl_topology.evaluation.stage29_pre_scale_decision import (
    build_stage29_pre_scale_decision_report,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE29_SOURCE = ROOT / "src" / "marl_topology" / "evaluation" / "stage29_pre_scale_decision.py"
STAGE29_SCRIPT = ROOT / "scripts" / "replay" / "stage29_pre_scale_decision_review.py"


def test_stage29_report_does_not_authorize_forbidden_work() -> None:
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)
    packet = report["decision_packet"]

    assert report["pass_gate"] is True
    assert packet["larger_pilot_approved"] is False
    assert packet["scale_up_approved"] is False
    assert report["forbidden_action_flags"]["new_training_run"] is False
    assert report["forbidden_action_flags"]["scale_up_training_run"] is False
    assert report["forbidden_action_flags"]["reward_weights_changed"] is False
    assert report["forbidden_action_flags"]["hyperparameter_tuning_performed"] is False
    assert report["forbidden_action_flags"]["sampler_switched"] is False
    assert report["forbidden_action_flags"]["checkpoint_created"] is False
    assert report["forbidden_action_flags"]["legacy_v5_modified"] is False
    assert report["forbidden_action_flags"]["recurrent_policy_introduced"] is False


def test_stage29_source_and_script_avoid_training_checkpoint_and_architecture_patterns() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (STAGE29_SOURCE, STAGE29_SCRIPT)
    )
    forbidden = (
        "torch.save",
        "torch.load",
        "checkpoint_path",
        "optimizer",
        "backward(",
        "train_loop",
        "class COMA",
        "class Transformer",
        "class LSTM",
        "class GRU",
        "D:\\PhD_works\\v5",
    )
    hits = [term for term in forbidden if term in combined]

    assert hits == []


def test_stage29_does_not_create_stage29_result_save_directory() -> None:
    assert not (ROOT / "result_save" / "stage29_pre_scale_decision_review").exists()
