import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_8_document_exists_and_states_no_replay_implementation() -> None:
    text = _read_doc("STAGE5_8_LEARNING_TARGET_REPLAY_CONTRACT.md")

    required = [
        "Stage 5.8 Learning Target And Replay Contract Without Implementation",
        "learning_target_replay_contract_frozen_implementation_blocked",
        "dataset_writer_allowed = false",
        "replay_buffer_allowed = false",
        "learner_batch_allowed = false",
        "training_execution_allowed = false",
        "return",
        "advantage",
        "value_target",
        "active replay schema must still reject",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.8 doc missing terms: {missing}"


def test_stage5_8_updates_contracts_and_project_state() -> None:
    learning = _read_doc("LEARNING_TARGET_REPLAY_CONTRACT.md")
    replay = _read_doc("REPLAY_DATASET_CONTRACT.md")
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "The active learning-target replay contract is design-only" in learning
    assert "Stage 5.8 Learning Target And Replay Contract" in replay
    assert "Stage 5.8 does not authorize dataset writer" in training
    assert "Stage 5.8 does not add metric names" in metric
    assert "Stage 5.8 Learning Target And Replay Contract" in dec_pomdp
    for expected in [
        "post_stage_5_8_awaiting_owner_decision_for_stage_5_9",
        "stage_5_8_learning_target_replay_contract_without_implementation",
        "stage5_8_learning_target_replay_contract_gate",
        "stage_5_9_training_run_manifest_artifact_contract_without_execution",
        "recommended_next_task: stage_5_9_training_run_manifest_artifact_contract_without_execution",
        "training_runs",
        "actor_critic_coma_gnn_lstm_implementation",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage5_8_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_8_learning_target_replay_contract.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_8_learning_target_replay_contract"
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
    assert "dataset writer replay buffer or learner batch is introduced" in negative_text
    assert "planned target columns enter deployment actor inputs" in negative_text
    assert "return advantage or value-target columns become active" in negative_text


def test_stage5_8_replay_script_prints_learning_target_contract() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_8_learning_target_replay_contract.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_5_8_learning_target_replay_contract_without_implementation"' in completed.stdout
    assert '"verdict": "learning_target_replay_contract_frozen_implementation_blocked"' in completed.stdout
    assert '"dataset_writer_allowed": false' in completed.stdout
    assert '"planned_learning_targets_active": false' in completed.stdout
    assert '"recommended_next_task": "stage_5_9_training_run_manifest_artifact_contract_without_execution"' in completed.stdout


def test_stage5_8_source_keeps_training_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 5.8 source introduced forbidden terms: {hits}"
