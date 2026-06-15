import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_7_document_exists_and_states_no_implementation() -> None:
    text = _read_doc("STAGE5_7_POLICY_ARCHITECTURE_CONTRACT.md")

    required = [
        "Stage 5.7 Policy Architecture Contract Without Implementation",
        "architecture_contract_frozen_implementation_blocked",
        "implementation_allowed = false",
        "training_execution_allowed = false",
        "local EdgeActionDecision values only",
        "Credit assignment is not selected",
        "Checkpoint creation remains blocked",
        "Full candidate-graph state",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.7 doc missing terms: {missing}"


def test_stage5_7_updates_contracts_and_project_state() -> None:
    policy = _read_doc("POLICY_ARCHITECTURE_CONTRACT.md")
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")
    replay = _read_doc("REPLAY_DATASET_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "Stage 5.7 does not implement actor networks" in policy
    assert "Stage 5.7 does not authorize policy or critic implementation" in training
    assert "Stage 5.7 does not add metric names" in metric
    assert "Stage 5.7 Policy Architecture Contract" in dec_pomdp
    assert "Stage 5.7 Policy Architecture Contract adds no replay columns" in replay
    for expected in [
        "post_stage_5_7_awaiting_owner_decision_for_stage_5_8",
        "stage_5_7_policy_architecture_contract_without_implementation",
        "stage5_7_policy_architecture_contract_gate",
        "stage_5_8_learning_target_replay_contract_without_implementation",
        "recommended_next_task: stage_5_8_learning_target_replay_contract_without_implementation",
        "training_runs",
        "actor_critic_coma_gnn_lstm_implementation",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage5_7_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_7_policy_architecture_contract.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_7_policy_architecture_contract"
    for field in [
        "required_artifacts",
        "expected_evidence",
        "negative_checks",
        "relevant_lessons",
        "required_tests",
        "forbidden_v5_inheritance",
        "expected_outputs",
    ]:
        assert field in task
        assert task[field]
    negative_text = " ".join(task["negative_checks"])
    assert "actor or critic implementation is added" in negative_text
    assert "centralized training fields enter deployment actor inputs" in negative_text
    assert "credit assignment is selected without calibration evidence" in negative_text


def test_stage5_7_replay_script_prints_architecture_contract() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_7_policy_architecture_contract.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_5_7_policy_architecture_contract_without_implementation"' in completed.stdout
    assert '"verdict": "architecture_contract_frozen_implementation_blocked"' in completed.stdout
    assert '"implementation_allowed": false' in completed.stdout
    assert '"deployment_actor_boundary_local_only": true' in completed.stdout
    assert '"recommended_next_task": "stage_5_8_learning_target_replay_contract_without_implementation"' in completed.stdout


def test_stage5_7_source_keeps_training_model_v5_and_legacy_metric_out() -> None:
    banned_terms = [
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "optimizer",
        "train_loop",
        "D:\\PhD_works\\v5",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "COMA",
        "MAPPO",
    ]
    hits: list[str] = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 5.7 source introduced forbidden terms: {hits}"
