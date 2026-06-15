from pathlib import Path

from marl_topology.evaluation import (
    STAGE4_5_REVIEW_STAGE_ID,
    build_stage4_5_baseline_oracle_review,
)


ROOT = Path(__file__).resolve().parents[2]


def _rows_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["name"]): row for row in report["topology_rows"]}


def test_stage4_5_report_evaluates_expected_baselines_and_candidate() -> None:
    report = build_stage4_5_baseline_oracle_review()
    rows = _rows_by_name(report)

    assert report["stage"] == STAGE4_5_REVIEW_STAGE_ID
    assert {"empty", "sparse_star", "sparse_chain", "full"} <= set(rows)
    assert report["oracle_candidate"]["status"] == "feasible"
    assert report["oracle_candidate"]["row"] is not None
    assert report["oracle_candidate"]["is_deployment_actor_input"] is False


def test_empty_failure_does_not_prove_infeasible_when_candidate_exists() -> None:
    report = build_stage4_5_baseline_oracle_review()
    rows = _rows_by_name(report)

    assert rows["empty"]["reliability_feasible"] is False
    assert rows["empty"]["metrics"]["consensus_success"] == 0
    assert report["oracle_candidate"]["status"] == "feasible"
    assert report["checks"]["empty_baseline_fails_but_does_not_prove_infeasible"] is True


def test_full_graph_is_baseline_not_oracle_candidate() -> None:
    report = build_stage4_5_baseline_oracle_review()
    full = _rows_by_name(report)["full"]
    candidate = report["oracle_candidate"]["row"]

    assert full["is_full_graph_baseline"] is True
    assert full["is_oracle_candidate"] is False
    assert candidate["is_full_graph_baseline"] is False
    assert candidate["is_oracle_candidate"] is True
    assert report["checks"]["full_graph_is_baseline_not_oracle"] is True


def test_reliability_constraint_is_separate_from_latency_and_energy() -> None:
    report = build_stage4_5_baseline_oracle_review()
    threshold = report["protocol"]["reliability_threshold"]

    for row in report["topology_rows"]:
        reliability = row["metrics"]["consensus_success_probability"]
        assert row["reliability_feasible"] == (reliability >= threshold)
        assert "latency" in row["metrics"]
        assert "energy" in row["metrics"]
    assert report["checks"]["reliability_constraint_separate_from_latency_energy"] is True


def test_stage4_5_metric_rows_are_registered_and_no_new_metric_names_added() -> None:
    report = build_stage4_5_baseline_oracle_review()

    assert report["metric_governance"]["metric_rows_are_registered"] is True
    assert report["metric_governance"]["new_metric_names_introduced"] == []
    for row in report["topology_rows"]:
        metric_names = {metric_row["metric_name"] for metric_row in row["metric_rows"]}
        assert metric_names == {
            "consensus_success",
            "consensus_success_probability",
            "latency",
            "energy",
            "topology_diagnostics",
        }


def test_stage4_5_source_has_no_training_or_v5_dependency() -> None:
    source = (
        ROOT / "src" / "marl_topology" / "evaluation" / "stage4_baseline_oracle_review.py"
    ).read_text(encoding="utf-8")
    banned_terms = [
        "D:\\PhD_works\\v5",
        "import v5",
        "from v5",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "P_eff",
        "def reward",
        "class Reward",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.5 review source uses forbidden routes: {hits}"
