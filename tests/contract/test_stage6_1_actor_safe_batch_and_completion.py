import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage6_1_document_exists_and_states_actor_batch_boundary() -> None:
    text = _read_doc("STAGE6_1_ACTOR_SAFE_BATCH_BUILDER.md")

    required = [
        "Stage 6.1 Actor-Safe Batch Builder Without Model Or Training",
        "actor_safe_batch_builder_ready_model_training_blocked",
        "agent_id",
        "local_neighbor_observations",
        "global topology",
        "No model",
        "Recommended Next Stage",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 6.1 doc missing terms: {missing}"


def test_stage6_completion_review_closes_stage6() -> None:
    text = _read_doc("STAGE6_COMPLETION_REVIEW.md")

    required = [
        "Stage 6 Completion Review",
        "Stage 6 is closed",
        "Do not add more Stage 6.x planning tasks",
        "stage_7_0_local_actor_policy_interface_contract_with_owner_approval",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 6 completion doc missing terms: {missing}"


def test_stage6_1_project_state_marks_stage6_closed_and_recommends_stage7() -> None:
    state = _read_doc("PROJECT_STATE.md")

    expected = [
        "post_stage_6_closed_awaiting_owner_decision_for_stage_7",
        "stage_6_1_actor_safe_batch_builder_without_model_or_training",
        "stage_6_completion_exit_review",
        "stage6_1_actor_safe_batch_builder_gate",
        "stage6_completion_exit_gate",
        "stage_7_0_local_actor_policy_interface_contract_with_owner_approval",
        "Stage 6 is closed",
        "owner_decision_required: true",
    ]
    for term in expected:
        assert term in state


def test_stage6_1_contract_docs_are_updated_without_new_metrics() -> None:
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    dec_pomdp = _read_doc("DEC_POMDP_CONTRACT.md")

    assert "Stage 6.1 adds only an in-memory actor-safe batch builder" in training
    assert "Stage 6.1 does not add metric names" in metric
    assert "Stage 6.1 implements actor-safe in-memory batch projection" in dec_pomdp


def test_stage6_1_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage6_1_actor_safe_batch_builder_and_exit_gate.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage6_1_actor_safe_batch_builder_and_exit_gate"
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
    assert "actor batch contains centralized training fields" in negative_text
    assert "Stage 6 remains open" in negative_text
    assert "result_save receives new files beyond .gitkeep" in negative_text


def test_stage6_1_replay_script_prints_batch_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage6_1_actor_safe_batch_report.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_6_1_actor_safe_batch_builder_without_model_or_training"' in completed.stdout
    assert '"batch_ready": true' in completed.stdout
    assert '"writes_performed": false' in completed.stdout
    assert '"training_execution_allowed": false' in completed.stdout
    assert '"recommended_next_stage": "stage_7_0_local_actor_policy_interface_contract_with_owner_approval"' in completed.stdout


def test_stage6_1_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage6_1_source_keeps_execution_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 6.1 source introduced forbidden terms: {hits}"
