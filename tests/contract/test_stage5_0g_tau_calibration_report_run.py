import subprocess
import sys
from pathlib import Path

import yaml

from marl_topology.evaluation import (
    build_stage5_0f_tau_calibration_fixture_suite_report,
    build_tau_consensus_calibration_report,
)


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def _stage5_0g_report() -> dict[str, object]:
    return build_tau_consensus_calibration_report(
        [0.9],
        tau_source="owner_supplied_stage5_0g",
        tau_owner_note=(
            "owner supplied tau_consensus <= 0.9; interpreted for Stage 5.0g "
            "as candidate tau=0.9, not final selection"
        ),
        source_report=build_stage5_0f_tau_calibration_fixture_suite_report(),
    )


def test_stage5_0g_documentation_records_report_run_without_selection() -> None:
    text = _read_doc("STAGE5_0G_TAU_CONSENSUS_CALIBRATION_REPORT_RUN.md")

    required = [
        "Stage 5.0g",
        "tau_candidate = 0.9",
        "consensus_success_probability >= tau_consensus",
        "source_kind: stage5_0f_alpha_fixture_suite",
        "tau_selected: False",
        "final_tau_consensus: None",
        "does not select final `tau_consensus`",
        "does not implement reward",
        "does not train models",
        "does not migrate v5 code",
        "Owner Decision Required",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.0g doc missing terms: {missing}"


def test_stage5_0g_report_uses_owner_candidate_and_stage5_source_flags() -> None:
    report = _stage5_0g_report()

    assert report["source_kind"] == "stage5_0f_alpha_fixture_suite"
    assert report["calibration_ready"] is True
    assert report["tau_selected"] is False
    assert report["final_tau_consensus"] is None
    assert report["owner_decision_packet"]["recommended_tau_candidate"] is None
    assert report["owner_decision_packet"]["owner_selected_tau"] is None

    candidate_rows = report["candidate_tau_rows"]
    assert candidate_rows == [
        {
            "tau_candidate": 0.9,
            "tau_source": "owner_supplied_stage5_0g",
            "tau_owner_note": (
                "owner supplied tau_consensus <= 0.9; interpreted for Stage 5.0g "
                "as candidate tau=0.9, not final selection"
            ),
            "is_default": False,
            "is_selected": False,
        }
    ]

    all_flags = {
        flag
        for row in report["tau_feasibility_summary"]
        for flag in row["diagnostic_flags"]
    }
    assert "stage5_0f_alpha_fixture_suite" in all_flags
    assert "stage4_8_smoke_only" not in all_flags


def test_stage5_0g_report_has_expected_alpha_fixture_feasibility_signal() -> None:
    report = _stage5_0g_report()
    summary = {
        row["scenario_family"]: row
        for row in report["tau_feasibility_summary"]
    }

    assert len(summary) == 8
    assert summary["clear_free_space_reference"]["feasible_non_full_topology_count"] == 1
    assert summary["same_resource_interference"]["full_graph_feasible"] is False
    assert summary["sparse_vs_dense_tradeoff"]["feasible_topology_count"] == 2
    assert summary["weak_primary_distribution"]["weak_primary_case_count"] == 2
    assert summary["near_threshold_link_budget"]["feasible_topology_count"] == 0

    checks = report["checks"]
    assert checks["source_is_alpha_fixture_suite"] is True
    assert checks["no_default_tau_candidates"] is True
    assert checks["no_final_tau_selected"] is True
    assert checks["sparse_better_than_full_graph_for_some_family"] is True
    assert checks["full_graph_not_oracle"] is True
    assert checks["oracle_labels_not_actor_inputs"] is True
    assert checks["reward_implemented"] is False
    assert checks["training_run"] is False
    assert checks["v5_code_migrated"] is False


def test_stage5_0g_replay_script_accepts_owner_tau_and_prints_report() -> None:
    script = ROOT / "scripts" / "replay" / "tau_consensus_calibration_report.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            "stage5_0f",
            "--tau",
            "0.9",
            "--tau-source",
            "owner_supplied_stage5_0g",
            "--tau-owner-note",
            "owner supplied tau_consensus <= 0.9",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"source_kind": "stage5_0f_alpha_fixture_suite"' in completed.stdout
    assert '"tau_candidate": 0.9' in completed.stdout
    assert '"tau_source": "owner_supplied_stage5_0g"' in completed.stdout
    assert '"source_is_alpha_fixture_suite": true' in completed.stdout
    assert '"tau_selected": false' in completed.stdout
    assert '"final_tau_consensus": null' in completed.stdout
    assert '"stage4_8_smoke_only"' not in completed.stdout


def test_stage5_0g_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_0g_tau_consensus_calibration_report_run.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_0g_tau_consensus_calibration_report_run"
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
    assert "wrong inequality direction" in negative_text
    assert "Stage 5.0f fixture rows" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_stage5_0g_project_state_marks_completion_and_keeps_final_tau_blocked() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_0g_awaiting_owner_decision",
        "post_stage_5_0f_awaiting_owner_decision",
        "stage_5_0g_tau_consensus_calibration_report_run_with_owner_supplied_candidates_without_selection",
        "stage5_0g_tau_consensus_calibration_report_run_gate",
        "tau_consensus_final_selection",
        "reward_implementation",
        "training_runs",
        "v5_code_migration",
        "recommended_next_task: stage_5_0h_owner_tau_decision_or_calibration_hardening",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.0g state: {missing}"
