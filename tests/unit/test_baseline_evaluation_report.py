from marl_topology.evaluation import build_stage2_baseline_evaluation_report
from marl_topology.metrics import REGISTERED_METRICS


def _rows_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["name"]: row for row in report["baseline_rows"]}


def test_baseline_report_contains_global_and_decentralized_rows() -> None:
    report = build_stage2_baseline_evaluation_report()
    rows = _rows_by_name(report)

    expected = {
        "empty",
        "full",
        "greedy_reliability",
        "random_seed_7",
        "decentralized_no_edges",
        "decentralized_all_local_edges",
        "decentralized_top1_reliability",
        "decentralized_threshold_p65",
        "decentralized_random_seed_13_p40",
    }
    assert expected <= set(rows)
    assert report["stage"] == "stage_2_4_baseline_evaluation_report"


def test_baseline_report_uses_registered_metrics_only() -> None:
    report = build_stage2_baseline_evaluation_report()

    assert report["metric_governance"]["new_metric_names_introduced"] == []
    assert report["metric_governance"]["metric_rows_are_registered"] is True
    for row in report["baseline_rows"]:
        assert set(row["metrics"]) == set(REGISTERED_METRICS)
        assert {metric_row["metric_name"] for metric_row in row["metric_rows"]} == set(
            REGISTERED_METRICS
        )
        assert all("metric_level" in metric_row for metric_row in row["metric_rows"])
        assert all("used_for" in metric_row for metric_row in row["metric_rows"])


def test_baseline_report_keeps_full_graph_out_of_oracle_label() -> None:
    report = build_stage2_baseline_evaluation_report()
    rows = _rows_by_name(report)

    assert report["checks"]["full_graph_is_baseline_not_oracle"] is True
    assert rows["full"]["is_oracle"] is False
    assert rows["full"]["topology_id"] == "baseline:full"
    assert rows["full"]["decision_diagnostics"]["is_full_graph_baseline"] is True
    assert report["oracle_reference"]["oracle_name"] != "full"
    assert report["oracle_reference"]["is_deployment_actor_input"] is False


def test_decentralized_report_rows_use_env_step_and_actor_observation_boundary() -> None:
    report = build_stage2_baseline_evaluation_report()
    rows = _rows_by_name(report)

    decentralized_names = [
        name for name, row in rows.items() if row["family"] == "decentralized_non_learning"
    ]
    assert decentralized_names
    for name in decentralized_names:
        row = rows[name]
        assert row["decision_source"] == "actor_observation_to_edge_action_decision"
        assert row["decision_diagnostics"]["uses_actor_observation"] is True
        assert row["decision_diagnostics"]["actor_observation_count"] == 4
        assert row["decision_diagnostics"]["step_time"] == 1
        assert row["topology_id"] == f"baseline:{name}"


def test_baseline_report_is_deterministic_and_uses_unique_topology_ids() -> None:
    first = build_stage2_baseline_evaluation_report()
    second = build_stage2_baseline_evaluation_report()

    assert first == second
    topology_ids = [row["topology_id"] for row in first["baseline_rows"]]
    assert len(topology_ids) == len(set(topology_ids))


def test_baseline_report_validates_parameters() -> None:
    invalid_calls = [
        lambda: build_stage2_baseline_evaluation_report(
            threshold_min_success_probability=-0.01
        ),
        lambda: build_stage2_baseline_evaluation_report(
            local_random_edge_probability=1.01
        ),
        lambda: build_stage2_baseline_evaluation_report(top_k=-1),
    ]

    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid baseline report parameter accepted")
