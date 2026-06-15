import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_0d_report_documentation_exists_and_keeps_tau_unselected() -> None:
    text = _read_doc("STAGE5_0D_TAU_CONSENSUS_CALIBRATION_REPORT.md")

    required = [
        "Stage 5.0d",
        "without tau selection",
        "owner-declared candidate tau values",
        "Stage 4.8 smoke-test rows",
        "not a final calibration set",
        "does not select `tau_consensus`",
        "does not implement reward",
        "does not train models",
        "scripts/replay/tau_consensus_calibration_report.py --tau",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0d doc missing terms: {missing}"


def test_stage5_0d_source_and_replay_script_exist_with_explicit_tau_only() -> None:
    source = ROOT / "src" / "marl_topology" / "evaluation" / "tau_consensus_calibration_report.py"
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    assert source.exists()
    assert script.exists()

    source_text = _read(source)
    script_text = _read(script)
    assert "build_tau_consensus_calibration_report" in source_text
    assert "required=True" in script_text
    assert "--tau" in script_text
    assert "reliability_threshold = 0.2" not in source_text
    assert "tau_candidates: Iterable[float | TauCandidate]" in source_text


def test_stage5_0d_replay_script_requires_tau_and_prints_json() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    missing_tau = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_tau.returncode != 0
    assert "--tau" in missing_tau.stderr

    with_tau = subprocess.run(
        [sys.executable, str(script), "--tau", "0.15", "--tau", "0.55"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"final_tau_consensus": null' in with_tau.stdout
    assert '"tau_selected": false' in with_tau.stdout
    assert '"source_scope": "stage4_8_smoke_test_only_not_calibration_set"' in with_tau.stdout


def test_stage5_0d_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0d_tau_consensus_calibration_report_implementation.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0d_tau_consensus_calibration_report_implementation"
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
    assert "default numeric tau candidates" in negative_text
    assert "final tau_consensus" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_stage5_0d_project_state_marks_completion_and_keeps_next_owner_decision() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0d_awaiting_owner_decision",
        "post_stage_5_0c_awaiting_owner_decision",
        "stage_5_0d_tau_consensus_calibration_report_implementation_without_tau_selection",
        "stage5_0d_tau_consensus_calibration_report_gate",
        "reward_implementation",
        "reward_weight_calibration",
        "training_runs",
        "owner_decision_required: true",
        "recommended_next_task: stage_5_0e_tau_consensus_fixture_family_design",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0d state: {missing}"


def test_stage5_0d_does_not_add_forbidden_reward_training_model_or_v5_code() -> None:
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

    assert not offenders, f"Stage 5.0d added forbidden source terms: {offenders}"
