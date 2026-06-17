import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def _result_save_entries() -> set[str]:
    return {path.name for path in (ROOT / "result_save").iterdir()}


def test_stage7_completion_document_contains_plan_and_exit_criteria() -> None:
    text = _read_doc("STAGE7_COMPLETION_LEARNING_EVIDENCE_EXIT_GATE.md")

    required = [
        "Stage 7 Completion",
        "Detailed Execution Plan",
        "empty",
        "random",
        "sparse_heuristic",
        "full_graph",
        "greedy",
        "oracle_candidate",
        "add edge, remove edge, keep edge, and swap edge",
        "actor-observable predictability risk",
        "Exit Criteria",
        "result_save/evidence_dataset_only",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 7 completion doc missing terms: {missing}"


def test_stage7_completion_project_state_recommends_stage8_after_real_exit_gate() -> None:
    state = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_7_completion_closed_awaiting_owner_decision_for_stage_8",
        "stage_7_completion_learning_evidence_dataset_build_quality_exit_gate",
        "stage7_completion_learning_evidence_exit_gate",
        "stage_8_0_actor_policy_interface_contract_with_owner_approval",
        "multi-scenario topology evidence",
        "manifest-validated evidence-only artifact",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in state]
    assert not missing, f"PROJECT_STATE missing Stage 7 completion terms: {missing}"


def test_stage7_completion_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage7_completion_learning_evidence_exit_gate.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage7_completion_learning_evidence_exit_gate"
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
    assert "Stage 7 closes on smoke-only evidence" in negative_text
    assert "swap edge target is omitted" in negative_text


def test_stage7_completion_script_writes_manifest_validated_evidence_artifact() -> None:
    script = ROOT / "scripts" / "replay" / "stage7_completion_evidence_dataset_report.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--write-artifact"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert '"stage": "stage_7_completion_learning_evidence_dataset_build_quality_exit_gate"' in stdout
    assert '"stage7_exit_ready": true' in stdout
    assert '"learnable_signal_present": true' in stdout
    assert '"artifact_written": true' in stdout
    assert '"manifest_validated": true' in stdout
    assert '"training_execution_ready": false' in stdout

    artifact_dir = (
        ROOT
        / "result_save"
        / "evidence_dataset_only"
        / "stage7_completion_learning_evidence_dataset_v1"
    )
    assert artifact_dir.exists()
    assert {path.name for path in artifact_dir.iterdir()} == {
        "learning_evidence.json",
        "manifest.json",
        "quality_report.json",
    }


def test_stage7_completion_result_save_contains_only_allowed_evidence_scope() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage7_completion_source_keeps_model_training_checkpoint_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 7 completion source introduced forbidden terms: {hits}"
