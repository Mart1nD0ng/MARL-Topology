from pathlib import Path

import yaml

from marl_topology.evaluation import (
    STAGE4_7_REPORT_STAGE_ID,
    Stage47EvaluationReportConfig,
    Stage47TopologySummary,
    build_stage4_7_pbft_application_evaluation_report,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_7_doc_records_report_boundary() -> None:
    text = _read_doc("STAGE4_7_PBFT_APPLICATION_EVALUATION_REPORT.md")

    required = [
        "Stage 4.7 PBFT Application Evaluation Report",
        "stage_4_7_pbft_application_evaluation_report",
        "build_stage4_7_pbft_application_evaluation_report",
        "consensus_success_probability",
        "lowest feasible latency",
        "lowest feasible energy",
        "does not add metric names",
        "does not",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 4.7 doc missing terms: {missing}"


def test_protocol_metric_and_stage4_plan_record_stage4_7() -> None:
    protocol = _read_doc("PROTOCOL_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    plan = _read_doc("STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md")

    protocol_required = [
        "Stage 4.7 PBFT Application Evaluation Report",
        "stage_4_7_pbft_application_evaluation_report",
        "does not compute a new PBFT reliability formula",
    ]
    metric_required = [
        "Stage 4.7 PBFT Application Evaluation Report",
        "does not add metric names",
        "lowest-latency",
        "lowest-energy",
    ]
    plan_required = [
        "Stage 4.7: PBFT Application Evaluation Report",
        "Implementation status",
        "single deterministic evaluation sensor",
    ]
    missing_protocol = [item for item in protocol_required if item not in protocol]
    missing_metric = [item for item in metric_required if item not in metric]
    missing_plan = [item for item in plan_required if item not in plan]

    assert not missing_protocol, f"PROTOCOL_CONTRACT missing Stage 4.7 terms: {missing_protocol}"
    assert not missing_metric, f"METRIC_CONTRACT missing Stage 4.7 terms: {missing_metric}"
    assert not missing_plan, f"Stage 4 plan missing Stage 4.7 terms: {missing_plan}"


def test_stage4_7_public_interfaces_exist() -> None:
    report = build_stage4_7_pbft_application_evaluation_report()

    assert STAGE4_7_REPORT_STAGE_ID == "stage_4_7_pbft_application_evaluation_report"
    assert Stage47EvaluationReportConfig
    assert Stage47TopologySummary
    assert report["stage"] == STAGE4_7_REPORT_STAGE_ID


def test_stage4_7_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage4_7_pbft_application_evaluation_report.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage4_7_pbft_application_evaluation_report"
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
    assert "unregistered metric" in negative_text
    assert "reward" in negative_text
    assert "v5" in negative_text


def test_project_state_marks_stage4_7_complete_and_waits_for_owner() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_4_7_awaiting_owner_decision",
        "stage_4_7_pbft_application_evaluation_report",
        "stage4_7_application_evaluation_report_gate",
        "recommended_next_task: stage_5_0_reward_objective_contract_freeze",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 4.7 state: {missing}"


def test_stage4_7_replay_script_is_print_only() -> None:
    text = (
        ROOT / "scripts" / "replay" / "stage4_7_pbft_application_evaluation_report.py"
    ).read_text(encoding="utf-8")

    assert "build_stage4_7_pbft_application_evaluation_report" in text
    assert "print(" in text
    banned_terms = ["to_csv", "result_save", "pickle", "torch.save", "open("]
    hits = [term for term in banned_terms if term in text]
    assert not hits, f"Stage 4.7 replay script writes outputs: {hits}"


def test_stage4_7_source_does_not_add_forbidden_routes() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_application_report.py",
        ROOT / "scripts" / "replay" / "stage4_7_pbft_application_evaluation_report.py",
    ]
    banned_terms = [
        "D:\\PhD_works\\v5",
        "import v5",
        "from v5",
        "P_eff",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "def reward",
        "class Reward",
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 4.7 added forbidden routes: {offenders}"
