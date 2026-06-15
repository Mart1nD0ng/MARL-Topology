from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage21_harness_task_declares_required_gates_and_boundaries() -> None:
    task = yaml.safe_load(
        _read("harness/tasks/stage21_objective_stack_aligned_assembler_aware_training_rerun.yaml")
    )

    assert task["id"] == "stage21_objective_stack_aligned_assembler_aware_training_rerun"
    assert "controlled_object" in task
    assert "desired_state" in task
    assert "policy_gradient_conditional_gate" in task
    negative = " ".join(task["negative_checks"])
    assert "SimpleLinkModel" in negative
    assert "Policy-gradient runs after any hard Stage 21 gate fails" in negative
    assert "COMA" in negative
    assert "Transformer" in negative
    assert "v5 code is modified" in negative


def test_stage21_rationale_document_explains_failure_rebuild_fairness_and_conditional_pilot() -> None:
    text = _read("docs/STAGE21_OBJECTIVE_STACK_ALIGNED_ASSEMBLER_AWARE_RERUN.md")

    required = [
        "Why Stage 20 Failed",
        "Why Evidence Is Rebuilt",
        "Why The Same Assembler Is Required",
        "Why Targets Are Assembler-Aware",
        "Why Policy-Gradient Is Conditional",
        "Pass / Fail Meaning",
        "Stage 3 finite-blocklength",
        "Stage 4 expected-initiator PBFT",
    ]
    missing = [term for term in required if term not in text]
    assert not missing


def test_stage21_script_prints_report_and_writes_no_artifact_or_checkpoint() -> None:
    text = _read("scripts/train/stage21_objective_stack_aligned_assembler_aware_rerun.py")

    assert "run_stage21_objective_stack_aligned_assembler_aware_rerun" in text
    assert "print(json.dumps" in text
    forbidden = [
        "torch.save",
        "checkpoint_path",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits


def test_stage21_blocked_closeout_is_registered_in_project_state() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_21_blocked_awaiting_owner_decision",
        "recommended_next_task: stage_22_actor_feature_or_supervised_training_repair_based_on_stage21_report",
        "stage21_actor_performance_failed_gate",
        "stage21_policy_gradient_conditional_gate_failed",
        "stage_21_policy_gradient_pilot_after_failed_gate",
        "STAGE21_FAILURE_REVIEW_AND_ROOT_CAUSE",
        "owner_decision_required: true",
        "Stage 21 did not run PPO/MAPPO",
    ]
    missing = [term for term in required if term not in text]
    assert not missing
