import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage6_0_document_exists_and_states_boundaries() -> None:
    text = _read_doc("STAGE6_0_MINIMAL_TRAINING_STACK_WITH_MANIFEST_GUARD.md")

    required = [
        "Stage 6.0 Minimal Training Stack With Manifest Guard",
        "manifest guard -> contract reference check -> dry-run stack readiness report",
        "minimal_training_stack_guard_ready_execution_blocked",
        "training_execution_allowed = false",
        "model_implementation_allowed = false",
        "Stage 5.10 manifest validation",
        "actor-safe batch builder",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 6.0 doc missing terms: {missing}"


def test_stage6_0_project_state_advances_to_stage6_1_recommendation() -> None:
    state = _read_doc("PROJECT_STATE.md")

    expected = [
        "post_stage_6_0_awaiting_owner_decision_for_stage_6_1",
        "stage_6_0_minimal_training_stack_implementation_with_manifest_guard",
        "stage6_0_minimal_training_stack_guard",
        "stage_6_1_actor_safe_batch_builder_without_model_or_training",
        "Stage 5 is closed",
        "additional_stage5_planning_tasks_after_exit_gate",
        "owner_decision_required: true",
    ]
    for term in expected:
        assert term in state


def test_stage6_0_contract_docs_are_updated_without_new_metrics() -> None:
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")

    assert "Stage 6.0 adds only a minimal dry-run training-stack guard" in training
    assert "Stage 6.0 does not add metric names" in metric
    assert "Stage 6.0 adds only a dry-run stack guard" in dec_pomdp


def test_stage6_0_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage6_0_minimal_training_stack_guard.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage6_0_minimal_training_stack_guard"
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
    assert "bypasses the Stage 5.10 manifest validator" in negative_text
    assert "missing owner approval is accepted" in negative_text
    assert "result_save receives new files beyond .gitkeep" in negative_text


def test_stage6_0_replay_script_prints_guard_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage6_0_minimal_training_stack_guard.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_6_0_minimal_training_stack_implementation_with_manifest_guard"' in completed.stdout
    assert '"stack_ready": true' in completed.stdout
    assert '"writes_performed": false' in completed.stdout
    assert '"training_execution_allowed": false' in completed.stdout
    assert '"recommended_next_task": "stage_6_1_actor_safe_batch_builder_without_model_or_training"' in completed.stdout


def test_stage6_0_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage6_0_source_keeps_execution_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 6.0 source introduced forbidden terms: {hits}"
