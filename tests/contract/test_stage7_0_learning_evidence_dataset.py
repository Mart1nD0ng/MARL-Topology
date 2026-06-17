import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage7_0_contract_documents_exist_and_define_views() -> None:
    contract = _read_doc("STAGE7_LEARNING_EVIDENCE_DATASET_CONTRACT.md")
    report = _read_doc("STAGE7_0_LEARNING_EVIDENCE_DATASET.md")

    required_contract_terms = [
        "actor_safe_view",
        "critic_centralized_view",
        "learning_target_view",
        "diagnostics_view",
        "oracle labels in `actor_safe_view`",
        "objective metrics in `actor_safe_view`",
        "future outcomes in `actor_safe_view`",
        "reward surrogate components in `actor_safe_view`",
        "global topology in `actor_safe_view`",
        "evidence_dataset_only",
    ]
    missing = [term for term in required_contract_terms if term not in contract]
    assert not missing, f"Stage 7 contract missing terms: {missing}"

    required_report_terms = [
        "Stage 7.0 Learning Evidence Dataset Generation",
        "Actor policy interface work is deferred to Stage 8",
        "LearningEvidenceRow",
        "EdgeDeltaTarget",
        "write_learning_evidence_artifact",
        "not write artifacts",
    ]
    missing_report = [term for term in required_report_terms if term not in report]
    assert not missing_report, f"Stage 7 report doc missing terms: {missing_report}"


def test_stage7_0_project_state_corrects_next_stage_and_defers_actor_interface() -> None:
    state = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_7_0_awaiting_owner_decision_for_stage_7_1",
        "stage_7_0_learning_evidence_dataset_generation_with_owner_approval",
        "stage7_0_learning_evidence_dataset_gate",
        "stage_7_1_learning_evidence_data_quality_report_with_owner_approval",
        "stage_7_0_local_actor_policy_interface_contract_with_owner_approval",
        "Actor policy interface work is deferred to Stage 8",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in state]
    assert not missing, f"PROJECT_STATE missing Stage 7.0 terms: {missing}"


def test_stage7_0_contract_updates_preserve_training_metric_and_dec_pomdp_boundaries() -> None:
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")

    assert "Stage 7 Learning Evidence Boundary" in training
    assert "Stage 7.0 does not add metric names" in metric
    assert "Stage 7.0 Learning Evidence Dataset" in dec_pomdp
    assert "Those values must not enter `actor_safe_view`" in metric
    assert "Actor policy interface work is deferred to Stage 8" in dec_pomdp


def test_stage7_0_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage7_0_learning_evidence_dataset_generation.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage7_0_learning_evidence_dataset_generation"
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
    assert "oracle labels objective metrics future outcomes" in negative_text
    assert "artifact writing bypasses owner approval" in negative_text


def test_stage7_0_replay_script_prints_evidence_report_without_project_write() -> None:
    script = ROOT / "scripts" / "replay" / "stage7_0_learning_evidence_report.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert '"stage": "stage_7_0_learning_evidence_dataset_generation_with_owner_approval"' in stdout
    assert '"evidence_generated": true' in stdout
    assert '"actor_safe_view_separated": true' in stdout
    assert '"learning_target_view_separated": true' in stdout
    assert '"artifact_written": false' in stdout
    assert '"training_execution_allowed": false' in stdout
    assert '"recommended_next_task": "stage_7_1_learning_evidence_data_quality_report_with_owner_approval"' in stdout


def test_stage7_0_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage7_0_source_keeps_model_training_checkpoint_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 7.0 source introduced forbidden terms: {hits}"
