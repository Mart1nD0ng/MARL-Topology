from pathlib import Path

from marl_topology.evaluation import (
    STAGE4_7_REPORT_STAGE_ID,
    build_stage4_7_pbft_application_evaluation_report,
)
from marl_topology.protocol import PBFT_PROTOCOL_ACCOUNTING_MODEL_ID


ROOT = Path(__file__).resolve().parents[2]


def _summaries_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["name"]): row for row in report["topology_summaries"]}


def test_stage4_7_report_combines_reliability_latency_energy_and_diagnostics() -> None:
    report = build_stage4_7_pbft_application_evaluation_report()

    assert report["stage"] == STAGE4_7_REPORT_STAGE_ID
    assert report["source_stage"] == "stage_4_5_baseline_and_oracle_review"
    assert report["metric_governance"]["metric_rows_are_registered"] is True
    assert report["metric_governance"]["new_metric_names_introduced"] == []
    assert report["application_summary"]["evaluated_topology_count"] >= 4
    assert report["application_summary"]["feasible_topology_count"] >= 1


def test_stage4_7_keeps_reliability_constraint_separate_from_objectives() -> None:
    report = build_stage4_7_pbft_application_evaluation_report()
    threshold = report["application_summary"]["reliability_threshold"]

    for row in report["topology_summaries"]:
        reliability = row["consensus_success_probability"]
        assert row["reliability_feasible"] == (reliability >= threshold)
        assert "latency_s" in row
        assert "energy_j" in row
    assert report["checks"]["reliability_constraint_separate_from_latency_energy"] is True


def test_stage4_7_reports_best_feasible_latency_and_energy_without_new_metrics() -> None:
    report = build_stage4_7_pbft_application_evaluation_report()
    feasible = [row for row in report["topology_summaries"] if row["reliability_feasible"]]
    lowest_latency = report["application_summary"]["lowest_latency_feasible_topology"]
    lowest_energy = report["application_summary"]["lowest_energy_feasible_topology"]

    assert lowest_latency["latency_s"] == min(row["latency_s"] for row in feasible)
    assert lowest_energy["energy_j"] == min(row["energy_j"] for row in feasible)
    assert {row["metric_name"] for row in report["metric_table"]} == {
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    }


def test_stage4_7_preserves_full_graph_and_oracle_boundaries() -> None:
    report = build_stage4_7_pbft_application_evaluation_report()
    rows = _summaries_by_name(report)

    assert rows["full"]["is_full_graph_baseline"] is True
    assert rows["full"]["is_oracle_candidate"] is False
    assert report["application_summary"]["oracle_candidate"]["is_deployment_actor_input"] is False
    assert report["checks"]["full_graph_is_baseline_not_oracle"] is True
    assert report["checks"]["oracle_candidate_is_not_actor_input"] is True


def test_stage4_7_includes_stage4_6_protocol_accounting_sensor() -> None:
    report = build_stage4_7_pbft_application_evaluation_report()

    assert report["checks"]["protocol_accounting_present"] is True
    for row in report["metric_table"]:
        if row["metric_name"] != "topology_diagnostics":
            continue
        diagnostics = row["metric_value"]
        accounting = diagnostics["protocol_accounting"]
        assert accounting["protocol_accounting_model_id"] == PBFT_PROTOCOL_ACCOUNTING_MODEL_ID
        assert accounting["exports_consensus_probability"] is False
        assert accounting["implements_reward"] is False


def test_stage4_7_source_has_no_forbidden_routes() -> None:
    source = (
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_application_report.py"
    ).read_text(encoding="utf-8")
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
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.7 report source uses forbidden routes: {hits}"
