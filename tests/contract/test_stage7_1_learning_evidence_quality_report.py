import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage7_1_report_document_exists_and_states_quality_boundary() -> None:
    text = _read_doc("STAGE7_1_LEARNING_EVIDENCE_DATA_QUALITY_REPORT.md")

    required = [
        "Stage 7.1 Learning Evidence Data Quality Report",
        "not a training stage",
        "Blocking gates",
        "Warnings",
        "Stage 7 is closed",
        "stage_8_0_actor_policy_interface_contract_with_owner_approval",
        "Training execution remains blocked",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 7.1 doc missing terms: {missing}"


def test_stage7_1_project_state_closes_stage7_and_recommends_stage8() -> None:
    state = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_7_1_stage_7_closed_awaiting_owner_decision_for_stage_8",
        "stage_7_1_learning_evidence_data_quality_report_with_owner_approval",
        "stage_7_completion_exit_review",
        "stage7_1_learning_evidence_quality_gate",
        "stage7_completion_exit_gate",
        "stage_8_0_actor_policy_interface_contract_with_owner_approval",
        "Stage 7 is closed",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in state]
    assert not missing, f"PROJECT_STATE missing Stage 7.1 terms: {missing}"


def test_stage7_1_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage7_1_learning_evidence_data_quality_report.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage7_1_learning_evidence_data_quality_report"
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
    assert "minimal evidence is claimed sufficient for training execution" in negative_text
    assert "another Stage 7 planning loop" in negative_text


def test_stage7_1_replay_script_prints_quality_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage7_1_learning_evidence_quality_report.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert '"stage": "stage_7_1_learning_evidence_data_quality_report_with_owner_approval"' in stdout
    assert '"stage7_exit_ready": true' in stdout
    assert '"stage8_policy_interface_ready": true' in stdout
    assert '"training_execution_ready": false' in stdout
    assert '"blocking_issue_count": 0' in stdout
    assert '"no_feasible_rows_at_current_tau"' in stdout
    assert '"recommended_next_task": "stage_8_0_actor_policy_interface_contract_with_owner_approval"' in stdout


def test_stage7_1_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage7_1_source_keeps_model_training_checkpoint_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 7.1 source introduced forbidden terms: {hits}"
