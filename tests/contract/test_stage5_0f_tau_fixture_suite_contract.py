import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0f_documentation_exists_and_keeps_boundaries() -> None:
    text = _read_doc("STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md")

    required = [
        "Stage 5.0f",
        "Minimal Tau Calibration Fixture Suite Implementation",
        "does not select `tau_consensus`",
        "does not implement reward",
        "does not train models",
        "does not migrate v5 code",
        "Reflection On Over-Conservatism",
        "smallest executable implementation",
        "Stage 5.0d report builder consumes the new fixture rows",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0f doc missing terms: {missing}"


def test_stage5_0f_docs_update_prior_stage_contracts() -> None:
    docs = {
        "TAU_CONSENSUS_CALIBRATION_PLAN": _read_doc("TAU_CONSENSUS_CALIBRATION_PLAN.md"),
        "STAGE5_0D": _read_doc("STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md"),
        "STAGE5_0E": _read_doc("STAGE5_0E_TAU_CONSENSUS_FIXTURE_FAMILY_DESIGN.md"),
        "OBJECTIVE_CONTRACT": _read_doc("OBJECTIVE_CONTRACT.md"),
        "REWARD_CONTRACT": _read_doc("REWARD_CONTRACT.md"),
        "REWARD_SURROGATE_CONTRACT": _read_doc("REWARD_SURROGATE_CONTRACT.md"),
        "METRIC_CONTRACT": _read_doc("METRIC_CONTRACT.md"),
        "STAGE5_REWARD_OBJECTIVE_FREEZE": _read_doc("STAGE5_REWARD_OBJECTIVE_FREEZE.md"),
    }

    for name, text in docs.items():
        assert "Stage 5.0f" in text, f"{name} missing Stage 5.0f reference"
    assert "The metric-valued list above is intentionally de-duplicated" in docs["STAGE5_0E"]


def test_stage5_0f_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0f_tau_calibration_fixture_suite.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0f_tau_calibration_fixture_suite"
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
    assert "final tau_consensus" in negative_text
    assert "reward" in negative_text
    assert "full graph" in negative_text
    assert "v5" in negative_text


def test_stage5_0f_replay_script_prints_fixture_suite_report() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_fixture_suite_report.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"stage": "stage_5_0f_minimal_tau_calibration_fixture_suite"' in completed.stdout
    assert '"required_families_present": true' in completed.stdout
    assert '"training_run": false' in completed.stdout
    assert '"v5_code_migrated": false' in completed.stdout


def test_stage5_0d_replay_script_can_use_stage5_0f_fixture_source() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--source", "stage5_0f", "--tau", "0.05", "--tau", "0.5"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"source_kind": "stage5_0f_alpha_fixture_suite"' in completed.stdout
    assert '"source_is_alpha_fixture_suite": true' in completed.stdout
    assert '"tau_selected": false' in completed.stdout
    assert '"final_tau_consensus": null' in completed.stdout


def test_stage5_0f_project_state_marks_completion_and_recommends_report_run() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0f_awaiting_owner_decision",
        "post_stage_5_0e_awaiting_owner_decision",
        "stage_5_0f_minimal_tau_calibration_fixture_suite",
        "stage5_0f_tau_calibration_fixture_suite_gate",
        "stage_5_0g_tau_consensus_calibration_report_run_with_owner_supplied_candidates_without_selection",
        "recommended_next_task: stage_5_0g_tau_consensus_calibration_report_run_with_owner_supplied_candidates_without_selection",
        "tau_consensus_final_selection",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0f state: {missing}"


def test_stage5_0f_does_not_add_forbidden_reward_training_model_or_v5_code() -> None:
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

    assert not offenders, f"Stage 5.0f added forbidden source terms: {offenders}"
