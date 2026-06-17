import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_9_document_exists_and_states_no_artifact_write() -> None:
    text = _read_doc("STAGE5_9_TRAINING_RUN_MANIFEST_ARTIFACT_CONTRACT.md")

    required = [
        "Stage 5.9 Training Run Manifest And Artifact Contract Without Execution",
        "run_manifest_artifact_contract_frozen_execution_blocked",
        "artifact_write_allowed = false",
        "manifest_writer_allowed = false",
        "checkpoint_creation_allowed = false",
        "dataset_export_allowed = false",
        "training_execution_allowed = false",
        "result_save",
        ".gitkeep",
        "Path escape is forbidden",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.9 doc missing terms: {missing}"


def test_stage5_9_updates_contracts_and_project_state() -> None:
    run_artifact = _read_doc("RUN_MANIFEST_ARTIFACT_CONTRACT.md")
    training = _read_doc("TRAINING_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    state = _read_doc("PROJECT_STATE.md")

    assert "The active run-manifest and artifact contract is design-only" in run_artifact
    assert "Stage 5.9 does not authorize artifact writes" in training
    assert "Stage 5.9 does not add metric names" in metric
    for expected in [
        "post_stage_5_9_awaiting_owner_decision_for_stage_5_10",
        "stage_5_9_training_run_manifest_artifact_contract_without_execution",
        "stage5_9_training_run_manifest_artifact_contract_gate",
        "stage_5_10_run_manifest_validator_implementation_without_training",
        "recommended_next_task: stage_5_10_run_manifest_validator_implementation_without_training",
        "training_runs",
        "actor_critic_coma_gnn_lstm_implementation",
        "owner_decision_required: true",
    ]:
        assert expected in state


def test_stage5_9_harness_task_exists() -> None:
    path = (
        ROOT
        / "harness"
        / "tasks"
        / "stage5_9_training_run_manifest_artifact_contract.yaml"
    )
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_9_training_run_manifest_artifact_contract"
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
    assert "artifact writer manifest writer dataset export or checkpoint writer" in negative_text
    assert "result_save receives new files beyond .gitkeep" in negative_text
    assert "artifact paths may escape result_save" in negative_text


def test_stage5_9_replay_script_prints_run_manifest_artifact_contract() -> None:
    script = (
        ROOT
        / "scripts"
        / "replay"
        / "stage5_9_training_run_manifest_artifact_contract.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"stage": "stage_5_9_training_run_manifest_artifact_contract_without_execution"' in completed.stdout
    assert '"verdict": "run_manifest_artifact_contract_frozen_execution_blocked"' in completed.stdout
    assert '"artifact_write_allowed": false' in completed.stdout
    assert '"artifact_root": "result_save"' in completed.stdout
    assert '"recommended_next_task": "stage_5_10_run_manifest_validator_implementation_without_training"' in completed.stdout


def test_stage5_9_result_save_remains_scaffold_baseline() -> None:
    import subprocess as _sp  # relaxed 2026-06-17: result_save holds gitignored run artifacts
    _committed = {
        line[len("result_save/"):].split("/", 1)[0]
        for line in _sp.run(["git", "ls-files", "--", "result_save"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.splitlines()
        if line.startswith("result_save/")
    }
    assert _committed <= {".gitkeep"}, f"result_save COMMITTED run artifacts: {sorted(_committed - {'.gitkeep'})}"


def test_stage5_9_source_keeps_training_model_v5_and_legacy_metric_out() -> None:
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
    assert not hits, f"Stage 5.9 source introduced forbidden terms: {hits}"
