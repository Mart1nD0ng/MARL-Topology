import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_4_document_exists_and_sets_report_boundary() -> None:
    text = _read_doc("STAGE5_4_REWARD_REPORT_INTEGRATION.md")

    required = [
        "Stage 5.4 Reward Report Integration Without Training",
        "component_only_no_scalar_reward_v1",
        "does not report scalar surrogate",
        "training-only diagnostics",
        "not evaluation metrics",
        "does not calibrate reward weights",
        "does not train",
        "does not migrate v5 code",
        "Stage 5.3 normalization references",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.4 doc missing terms: {missing}"


def test_stage5_4_updates_contracts_and_project_state() -> None:
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")
    reward = _read_doc("REWARD_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "Stage 5.4 Reward Report Integration Without Training" in surrogate
    assert "Stage 5.4 Reward Report Integration Without Training" in reward
    assert "Stage 5.4 does not add metric names" in metric
    for expected in [
        "post_stage_5_4_awaiting_owner_decision_for_stage_5_5",
        "stage_5_4_reward_report_integration_without_training",
        "stage5_4_reward_report_integration_gate",
        "stage_5_5_training_preflight_review_without_training",
        "recommended_next_task: stage_5_5_training_preflight_review_without_training",
        "reward_weight_calibration",
        "training_runs",
        "v5_code_migration",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage5_4_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_4_reward_report_integration.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_4_reward_report_integration"
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
    assert "scalar surrogate is reported as an evaluation metric" in negative_text
    assert "reward weights are calibrated" in negative_text
    assert "training actor critic COMA GNN LSTM or model code is introduced" in negative_text


def test_stage5_4_replay_script_prints_component_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_4_reward_report_integration.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_5_4_reward_report_integration_without_training"' in completed.stdout
    assert '"component_policy": "component_only_no_scalar_reward_v1"' in completed.stdout
    assert '"scalar_surrogate_reported": false' in completed.stdout
    assert '"training_run": false' in completed.stdout


def test_stage5_4_source_keeps_training_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 5.4 source introduced forbidden terms: {hits}"
