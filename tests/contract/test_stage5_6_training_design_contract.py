import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_6_document_exists_and_states_design_only_boundary() -> None:
    text = _read_doc("STAGE5_6_TRAINING_DESIGN_CONTRACT.md")

    required = [
        "Stage 5.6 Training Design Contract Without Execution",
        "design_contract_frozen_training_execution_blocked",
        "training_execution_allowed = false",
        "constraint_first_component_policy_v1",
        "no reliability bonus above tau",
        "at least five seeds before learning claims",
        "Full graph remains a baseline, not an oracle",
        "Stage 5.6 writes no checkpoint",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.6 doc missing terms: {missing}"


def test_stage5_6_updates_contracts_and_project_state() -> None:
    training = _read_doc("TRAINING_CONTRACT.md")
    reward = _read_doc("REWARD_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")
    replay = _read_doc("REPLAY_DATASET_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "Stage 5.6 does not authorize training execution" in training
    assert "Stage 5.6 Training Design Contract Without Execution" in reward
    assert "Stage 5.6 Training Design Contract Without Execution" in surrogate
    assert "Stage 5.6 does not add metric names" in metric
    assert "Stage 5.6 Training Design Contract" in dec_pomdp
    assert "Stage 5.6 Training Design Contract" in replay
    for expected in [
        "post_stage_5_6_awaiting_owner_decision_for_stage_5_7",
        "stage_5_6_training_design_contract_without_execution",
        "stage5_6_training_design_contract_gate",
        "stage_5_7_policy_architecture_contract_without_implementation",
        "recommended_next_task: stage_5_7_policy_architecture_contract_without_implementation",
        "training_runs",
        "reward_weight_calibration",
        "actor_critic_coma_gnn_lstm_implementation",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage5_6_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_6_training_design_contract.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_6_training_design_contract"
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
    assert "training execution is allowed" in negative_text
    assert "scalarization weights are selected" in negative_text
    assert "return advantage or value-target columns are admitted" in negative_text


def test_stage5_6_replay_script_prints_design_contract() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_6_training_design_contract.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_5_6_training_design_contract_without_execution"' in completed.stdout
    assert '"verdict": "design_contract_frozen_training_execution_blocked"' in completed.stdout
    assert '"training_execution_allowed": false' in completed.stdout
    assert '"policy_id": "constraint_first_component_policy_v1"' in completed.stdout
    assert '"recommended_next_task": "stage_5_7_policy_architecture_contract_without_implementation"' in completed.stdout


def test_stage5_6_source_keeps_training_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 5.6 source introduced forbidden terms: {hits}"
