import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0j_documentation_exists_and_records_executable_sweep() -> None:
    text = _read_doc("STAGE5_0J_MINIMAL_FEASIBILITY_ENVELOPE_SWEEP.md")

    required = [
        "Stage 5.0j",
        "tau_requirement_min = 0.9",
        "does not select final `tau_consensus`",
        "does not implement reward",
        "does not train models",
        "does not migrate v5 code",
        "Executed Sweep Subset",
        "Deferred",
        "Increased bandwidth-like alpha control",
        "Conservative `remove_largest` does not improve reliability",
        "Stage 5.0k - Stage 3-backed feasibility envelope sweep hardening",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0j doc missing terms: {missing}"


def test_stage5_0j_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0j_minimal_feasibility_envelope_sweep.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0j_minimal_feasibility_envelope_sweep"
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
    assert "v5" in negative_text


def test_stage5_0j_replay_script_prints_minimal_sweep_report() -> None:
    script = ROOT / "scripts" / "replay" / "stage5_0j_minimal_feasibility_envelope_sweep.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0j_minimal_executable_feasibility_envelope_sweep"' in completed.stdout
    assert '"tau_requirement_min": 0.9' in completed.stdout
    assert '"bandwidth_sweep"' in completed.stdout
    assert '"deadline_sweep"' in completed.stdout
    assert '"resource_orthogonalization_sweep"' in completed.stdout
    assert '"fault_filter_mode_comparison"' in completed.stdout
    assert '"topology_candidate_expansion"' in completed.stdout
    assert '"at_least_one_infeasible_to_feasible_transition": true' in completed.stdout
    assert '"conservative_fault_filter_does_not_increase_reliability": true' in completed.stdout
    assert '"final_tau_selected": false' in completed.stdout
    assert '"reward_implemented": false' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0j_project_state_marks_completion_and_next_hardening() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0j_awaiting_owner_decision",
        "post_stage_5_0i_awaiting_owner_decision",
        "stage_5_0j_minimal_executable_feasibility_envelope_sweep",
        "stage5_0j_minimal_feasibility_envelope_sweep_gate",
        "stage_5_0k_stage3_backed_feasibility_envelope_sweep_hardening",
        "tau_consensus_final_selection",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "recommended_next_task: stage_5_0k_stage3_backed_feasibility_envelope_sweep_hardening",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0j state: {missing}"


def test_stage5_0j_source_avoids_sampling_reward_training_model_and_v5_code() -> None:
    source = _read(ROOT / "src" / "marl_topology" / "evaluation" / "feasibility_envelope_sweep.py")
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
    assert not hits, f"Stage 5.0j source added forbidden terms: {hits}"
