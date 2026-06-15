import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0i_documentation_exists_and_freezes_sweep_design() -> None:
    text = _read_doc("STAGE5_0I_FEASIBILITY_ENVELOPE_SWEEP_DESIGN.md")

    required = [
        "Stage 5.0i",
        "tau_requirement_min = 0.9",
        "does not lower",
        "does not implement reward",
        "does not train models",
        "does not migrate v5 code",
        "Sweep Axes",
        "Required Future Sweep Row Schema",
        "Comparison Policy",
        "Acceptance Sensors",
        "Negative Controls",
        "Stage 5.0j - minimal executable feasibility envelope sweep",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0i doc missing terms: {missing}"


def test_stage5_0i_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0i_feasibility_envelope_sweep_design.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0i_feasibility_envelope_sweep_design"
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


def test_stage5_0i_replay_script_prints_design_manifest_without_sweep_execution() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0i_feasibility_envelope_sweep_design.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0i_feasibility_envelope_sweep_design"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"required_sweeps_present": true' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"simulation_parameters_changed_to_force_feasibility": false' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0i_project_state_marks_completion_and_next_executable_sweep() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0i_awaiting_owner_decision",
        "post_stage_5_0h_awaiting_owner_decision",
        "stage_5_0i_feasibility_envelope_sweep_design",
        "stage5_0i_feasibility_envelope_sweep_design_gate",
        "stage_5_0j_minimal_executable_feasibility_envelope_sweep",
        "tau_consensus_final_selection",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "recommended_next_task: stage_5_0j_minimal_executable_feasibility_envelope_sweep",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0i state: {missing}"


def test_stage5_0i_does_not_add_forbidden_reward_training_model_or_v5_code() -> None:
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

    assert not offenders, f"Stage 5.0i added forbidden source terms: {offenders}"
