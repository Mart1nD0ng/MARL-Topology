import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_10_document_exists_and_states_exit_gate_boundaries() -> None:
    text = _read_doc("STAGE5_10_RUN_MANIFEST_VALIDATOR.md")

    required = [
        "Stage 5.10 Run Manifest Validator Exit Gate",
        "stage5_closed_manifest_validator_exit_gate_passable",
        "required manifest fields present",
        "owner_approval_id",
        "artifact root under result_save",
        "no path escape",
        "legacy reference path is rejected",
        "Stage 5 is closed",
        "Stage 6 may begin only with owner approval",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.10 doc missing terms: {missing}"


def test_stage5_10_project_state_closes_stage5_and_recommends_stage6() -> None:
    state = _read_doc("PROJECT_STATE.md")

    expected = [
        "post_stage_5_10_stage_5_closed_awaiting_owner_decision_for_stage_6",
        "stage_5_10_run_manifest_validator_exit_gate_without_execution",
        "stage5_10_run_manifest_validator_exit_gate",
        "stage_6_0_minimal_training_stack_implementation_with_manifest_guard",
        "Stage 5 is closed",
        "owner_decision_required: true",
    ]
    for term in expected:
        assert term in state
    assert "Do not open additional Stage 5.x planning tasks" in state


def test_stage5_10_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_10_run_manifest_validator_exit_gate.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_10_run_manifest_validator_exit_gate"
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
    assert "path escape" in negative_text
    assert "result_save receives new files beyond .gitkeep" in negative_text
    assert "training execution is allowed" in negative_text


def test_stage5_10_replay_script_prints_exit_gate_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_10_run_manifest_validator_exit_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage5_closed": true' in completed.stdout
    assert '"stage6_allowed_with_owner_approval": true' in completed.stdout
    assert '"writes_performed": false' in completed.stdout
    assert '"training_execution_allowed": false' in completed.stdout


def test_stage5_10_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage5_10_source_keeps_execution_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 5.10 source introduced forbidden terms: {hits}"
