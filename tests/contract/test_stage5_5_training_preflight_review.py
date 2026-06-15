import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_5_document_exists_and_blocks_training_execution() -> None:
    text = _read_doc("STAGE5_5_TRAINING_PREFLIGHT_REVIEW.md")

    required = [
        "Stage 5.5 Training Preflight Review Without Training",
        "verdict = not_ready_for_training_execution",
        "training_execution_allowed = false",
        "training_design_contract_allowed = true",
        "stage_5_6_training_design_contract_without_execution",
        "owner_decision_required = true",
        "multi-seed",
        "must not",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.5 doc missing terms: {missing}"


def test_stage5_5_updates_contracts_and_project_state() -> None:
    reward = _read_doc("REWARD_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")
    replay = _read_doc("REPLAY_DATASET_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "Stage 5.5 Training Preflight Review Without Training" in reward
    assert "Stage 5.5 Training Preflight Review Without Training" in surrogate
    assert "Stage 5.5 does not add metric names" in metric
    assert "Stage 5.5 Training Preflight Review" in dec_pomdp
    assert "Stage 5.5 Training Preflight Review" in replay
    for expected in [
        "post_stage_5_5_awaiting_owner_decision_for_stage_5_6",
        "stage_5_5_training_preflight_review_without_training",
        "stage5_5_training_preflight_review_gate",
        "stage_5_6_training_design_contract_without_execution",
        "recommended_next_task: stage_5_6_training_design_contract_without_execution",
        "training_runs",
        "reward_weight_calibration",
        "actor_critic_coma_gnn_lstm_implementation",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage5_5_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_5_training_preflight_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_5_training_preflight_review"
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
    assert "reward weights are calibrated" in negative_text
    assert "return advantage or value-target columns are admitted" in negative_text


def test_stage5_5_replay_script_prints_preflight_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_5_training_preflight_review.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_5_5_training_preflight_review_without_training"' in completed.stdout
    assert '"verdict": "not_ready_for_training_execution"' in completed.stdout
    assert '"training_execution_allowed": false' in completed.stdout
    assert '"recommended_next_task": "stage_5_6_training_design_contract_without_execution"' in completed.stdout


def test_stage5_5_source_keeps_training_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 5.5 source introduced forbidden terms: {hits}"
