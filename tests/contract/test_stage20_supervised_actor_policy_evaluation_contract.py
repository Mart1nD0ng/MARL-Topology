from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage20_document_records_assembler_evaluation_and_failed_readiness_gate() -> None:
    text = _read("docs/STAGE20_SUPERVISED_ACTOR_POLICY_EVALUATION.md")

    required = [
        "ConflictAwareGreedyAssembler",
        "environment-side topology assembler",
        "actor outputs remain edge scores only",
        "full graph baseline",
        "oracle diagnostic",
        "GNN remains the best full-row supervised actor",
        "Policy-gradient readiness: failed",
        "stage_21_assembler_aware_supervised_target_refinement",
        "No PPO/MAPPO",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 20 doc missing terms: {missing}"


def test_stage20_harness_task_exists_and_declares_negative_checks() -> None:
    task = yaml.safe_load(_read("harness/tasks/stage20_supervised_actor_policy_evaluation.yaml"))

    assert task["id"] == "stage20_supervised_actor_policy_evaluation"
    expected = " ".join(task["expected_evidence"])
    assert "ConflictAwareGreedyAssembler" in task["prompt"]
    assert "full graph is a baseline and not an oracle" in expected
    assert "policy-gradient readiness gate" in expected
    negative = " ".join(task["negative_checks"])
    assert "PPO MAPPO" in negative
    assert "COMA" in negative
    assert "Transformer" in negative
    assert "deployment assembler consumes objective oracle reward consensus latency or energy" in negative


def test_stage20_project_state_marks_stage20_complete_and_stage21_owner_gate() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_20_complete_awaiting_owner_decision_for_stage_21",
        "stage_20_supervised_actor_policy_evaluation_with_environment_assembler",
        "recommended_next_task: stage_21_assembler_aware_supervised_target_refinement",
        "stage20_environment_assembler_policy_evaluation_gate",
        "stage20_policy_gradient_readiness_failed_gate",
        "PPO_MAPPO_rerun_until_stage21_refinement_and_owner_approval",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing Stage 20 state: {missing}"


def test_stage20_script_prints_report_and_writes_no_artifact() -> None:
    text = _read("scripts/train/stage20_supervised_actor_policy_evaluation.py")

    required = [
        "run_stage20_supervised_actor_policy_evaluation",
        "print(json.dumps",
        "Stage 20 supervised actor policy evaluation",
    ]
    missing = [term for term in required if term not in text]
    assert not missing

    forbidden = [
        "torch.save",
        "checkpoint_path",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits, f"Stage 20 script introduced forbidden writes: {hits}"


def test_stage20_source_does_not_add_forbidden_policy_gradient_or_global_actor_paths() -> None:
    paths = [
        "src/marl_topology/evaluation/stage20_supervised_actor_policy_evaluation.py",
        "scripts/train/stage20_supervised_actor_policy_evaluation.py",
    ]
    text = "\n".join(_read(path) for path in paths)

    forbidden = [
        "PPOTrainer",
        "MAPPOTrainer",
        "class COMA",
        "class Transformer",
        "torch.save",
        "checkpoint_path",
        "reward_weight_tuning = True",
        "final_tau_selected = True",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits, f"Stage 20 source introduced forbidden paths: {hits}"
