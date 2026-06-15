import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0h_documentation_exists_and_answers_required_questions() -> None:
    text = _read_doc("STAGE5_0H_REQUIREMENT_ANCHORED_FEASIBILITY_DIAGNOSIS.md")

    required = [
        "tau_requirement_min = 0.9",
        "tau_stress_candidates = [0.95, 0.99]",
        "tau_diagnostic_values = [0.5, 0.75]",
        "consensus_success_probability >= tau_requirement_min",
        "Infeasible Row Classification",
        "Parameter Sanity Table",
        "Feasibility Envelope Plan",
        "Yes: `tau_requirement_min = 0.9` can be written into the objective contract",
        "Reward implementation should remain blocked",
        "the next technical task should be a feasibility envelope sweep",
        "does not migrate v5 code",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0h doc missing terms: {missing}"


def test_stage5_0h_objective_contract_records_requirement_baseline() -> None:
    text = _read_doc("OBJECTIVE_CONTRACT.md")

    required = [
        "Stage 5.0h",
        "tau_requirement_min = 0.9",
        "requirement baseline",
        "not a fitted calibration value",
        "Lower tau values may be used only as `tau_diagnostic_values`",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"OBJECTIVE_CONTRACT missing Stage 5.0h terms: {missing}"


def test_stage5_0h_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0h_requirement_feasibility_diagnosis.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0h_requirement_feasibility_diagnosis"
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
    assert "lower diagnostic tau values" in negative_text
    assert "simulation parameters are changed" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_stage5_0h_replay_script_prints_requirement_diagnosis() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0h_requirement_feasibility_diagnosis.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0h_requirement_anchored_feasibility_diagnosis"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"final_tau_below_requirement_selected": false' in completed.stdout
    assert '"all_infeasible_rows_have_failure_reason": true' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0h_project_state_marks_completion_and_blocks_reward_training() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0h_awaiting_owner_decision",
        "post_stage_5_0g_awaiting_owner_decision",
        "stage_5_0h_requirement_anchored_feasibility_diagnosis",
        "stage5_0h_requirement_feasibility_diagnosis_gate",
        "tau_consensus_final_selection",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "recommended_next_task: stage_5_0i_feasibility_envelope_sweep_design",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0h state: {missing}"


def test_stage5_0h_does_not_add_forbidden_reward_training_model_or_v5_code() -> None:
    banned_terms = [
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
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
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = _read(path)
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 5.0h added forbidden source terms: {offenders}"
