from marl_topology.evaluation import (
    build_fixture_stack,
    build_stage2_scenario_fixture_report,
)
from marl_topology.link import (
    STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID,
    validate_link_record_against_regime,
)
from marl_topology.scenario import iter_scenario_fixtures


def test_scenario_fixture_ids_are_unique_and_match_scene_ids() -> None:
    fixtures = iter_scenario_fixtures()
    fixture_ids = [fixture.fixture_id for fixture in fixtures]

    assert len(fixtures) >= 3
    assert len(fixture_ids) == len(set(fixture_ids))
    for fixture in fixtures:
        assert fixture.fixture_id == fixture.scene.scenario_id
        assert fixture.scene.physics_regime == "stage2_deterministic_distance"
        assert fixture.max_candidate_distance_m >= 0


def test_scenario_fixture_graphs_are_deterministic() -> None:
    for fixture in iter_scenario_fixtures():
        first = build_fixture_stack(fixture)
        second = build_fixture_stack(fixture)

        assert first.graph.node_ids == second.graph.node_ids
        assert first.graph.edge_ids == second.graph.edge_ids
        assert first.graph.physics_regime == "stage2_deterministic_distance"


def test_scenario_fixture_expected_oracle_statuses_hold() -> None:
    statuses = {}
    for fixture in iter_scenario_fixtures():
        stack = build_fixture_stack(fixture)
        result = stack.oracle.solve(random_seed=7)
        statuses[fixture.fixture_id] = result.status
        assert result.status == fixture.expected_oracle_status

    assert statuses["demo_stage2"] == "feasible"
    assert statuses["sparse_chain_stage2"] == "feasible"
    assert statuses["quorum_blocked_stage2"] == "infeasible"


def test_scenario_fixture_expected_full_graph_success_values_hold() -> None:
    for fixture in iter_scenario_fixtures():
        stack = build_fixture_stack(fixture)
        full = stack.oracle.evaluate_baselines(random_seed=7)["full"]

        assert bool(full.metrics["consensus_success"]) is fixture.expected_full_graph_success
        assert full.topology_id == "baseline:full"
        assert full.diagnostics["is_full_graph_baseline"] is True


def test_scenario_fixture_link_records_match_active_regime() -> None:
    for fixture in iter_scenario_fixtures():
        stack = build_fixture_stack(fixture)

        assert stack.evaluator.link_records
        for record in stack.evaluator.link_records.values():
            validate_link_record_against_regime(record)
            assert record.physics_regime == STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID


def test_stage2_scenario_fixture_report_checks_pass() -> None:
    report = build_stage2_scenario_fixture_report()

    assert report["stage"] == "stage_2_5_scenario_fixture_contract"
    assert set(report["fixture_ids"]) == {
        "demo_stage2",
        "sparse_chain_stage2",
        "quorum_blocked_stage2",
    }
    assert report["checks"]["fixture_ids_are_unique"] is True
    assert report["checks"]["all_expected_oracle_statuses_match"] is True
    assert report["checks"]["all_expected_full_graph_success_values_match"] is True
    assert report["checks"]["all_metric_rows_are_registered"] is True
    assert report["checks"]["full_graph_is_baseline_not_oracle_all"] is True
    assert report["checks"]["training_run"] is False
    assert report["checks"]["v5_code_migrated"] is False
