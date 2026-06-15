import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0l_documentation_exists_and_records_range_review() -> None:
    text = _read_doc("STAGE5_0L_STAGE3_BACKED_SWEEP_RANGE_REVIEW.md")

    required = [
        "Stage 5.0l",
        "tau_requirement_min = 0.9",
        "does not select final `tau_consensus`",
        "does not implement reward",
        "does not train models",
        "does not migrate v5 code",
        "tx_power_sweep",
        "payload_sweep",
        "rsu_height_placement_sweep",
        "resource_budget_limit_sweep",
        "unknown_needs_reference",
        "Stage 5.0m - objective readiness review before reward implementation",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0l doc missing terms: {missing}"


def test_stage5_0l_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0l_stage3_backed_sweep_range_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0l_stage3_backed_sweep_range_review"
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
    assert "final tau lower than 0.9" in negative_text
    assert "reward" in negative_text
    assert "Monte Carlo" in negative_text
    assert "full graph" in negative_text
    assert "deployment-calibrated" in negative_text


def test_stage5_0l_replay_script_prints_range_review_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0l_stage3_backed_sweep_range_review.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"tx_power_sweep"' in completed.stdout
    assert '"payload_sweep"' in completed.stdout
    assert '"rsu_height_placement_sweep"' in completed.stdout
    assert '"resource_budget_limit_sweep"' in completed.stdout
    assert '"every_sweep_has_infeasible_to_feasible_transition": true' in completed.stdout
    assert '"all_rows_stage3_backed": true' in completed.stdout
    assert '"unknown_needs_reference"' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0l_project_state_marks_completion_and_next_readiness_review() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0l_awaiting_owner_decision",
        "post_stage_5_0k_awaiting_owner_decision",
        "stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review",
        "stage5_0l_stage3_backed_sweep_range_review_gate",
        "stage_5_0m_objective_readiness_review_before_reward_implementation",
        "tau_consensus_final_selection",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "recommended_next_task: stage_5_0m_objective_readiness_review_before_reward_implementation",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0l state: {missing}"


def test_stage5_0l_source_avoids_sampling_reward_training_model_and_v5_code() -> None:
    source = _read(
        ROOT
        / "src"
        / "marl_topology"
        / "evaluation"
        / "feasibility_envelope_sweep_range_review.py"
    )
    banned_terms = [
        "random",
        "sample",
        "Monte Carlo",
        "combinations(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "def reward",
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in banned_terms if term in source]
    assert not hits, f"Stage 5.0l source added forbidden terms: {hits}"
