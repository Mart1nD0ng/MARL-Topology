from marl_topology.evaluation.reward_surface_analysis import (
    STAGE25_OBJECTIVE_ORDERING,
    STAGE25_REWARD_SURFACE_ANALYSIS_ID,
    build_reward_surface_analysis,
)
from marl_topology.training.policy_gradient import pilot_runner as stage23_pg


def _reward_config():
    return stage23_pg.build_stage23_reward_surrogate_config()


def test_stage25_reward_surface_checks_objective_alignment_without_weight_tuning() -> None:
    records = [
        {
            "row_id": "s0_sparse_feasible",
            "policy_label": "sparse_feasible",
            "scenario_id": "scenario_0",
            "selected_edge_count": 3,
            "candidate_edge_count": 6,
            "consensus_success_probability": 0.92,
            "latency": 0.004,
            "energy": 0.006,
        },
        {
            "row_id": "s0_full_feasible",
            "policy_label": "projected_full",
            "scenario_id": "scenario_0",
            "selected_edge_count": 6,
            "candidate_edge_count": 6,
            "consensus_success_probability": 0.92,
            "latency": 0.006,
            "energy": 0.009,
        },
        {
            "row_id": "s0_dominated_infeasible",
            "policy_label": "dominated_infeasible",
            "scenario_id": "scenario_0",
            "selected_edge_count": 2,
            "candidate_edge_count": 6,
            "consensus_success_probability": 0.80,
            "latency": 0.007,
            "energy": 0.010,
        },
        {
            "row_id": "s0_empty",
            "policy_label": "empty_diagnostic",
            "scenario_id": "scenario_0",
            "selected_edge_count": 0,
            "candidate_edge_count": 6,
            "consensus_success_probability": 0.0,
            "latency": 0.0,
            "energy": 0.0,
        },
    ]

    analysis = build_reward_surface_analysis(
        records,
        reward_config=_reward_config(),
        tau_requirement_min=0.9,
    )

    assert analysis["analysis_id"] == STAGE25_REWARD_SURFACE_ANALYSIS_ID
    assert analysis["objective_ordering"] == STAGE25_OBJECTIVE_ORDERING
    assert analysis["weight_tuning_performed"] is False
    assert analysis["row_count"] == 4
    assert analysis["alignment_passed"] is True
    assert analysis["checks"]["above_tau_reliability_component_zero"]["passed"] is True
    assert analysis["checks"]["empty_topology_not_high_reward"]["passed"] is True
    assert analysis["checks"]["full_topology_not_automatic_sparse_dominator"]["passed"] is True


def test_stage25_reward_surface_dominance_is_scenario_local() -> None:
    records = [
        {
            "row_id": "scenario_a_feasible",
            "policy_label": "sparse_feasible",
            "scenario_id": "scenario_a",
            "selected_edge_count": 3,
            "candidate_edge_count": 5,
            "consensus_success_probability": 0.99,
            "latency": 0.001,
            "energy": 0.001,
        },
        {
            "row_id": "scenario_b_infeasible",
            "policy_label": "infeasible_other_scenario",
            "scenario_id": "scenario_b",
            "selected_edge_count": 1,
            "candidate_edge_count": 5,
            "consensus_success_probability": 0.10,
            "latency": 0.900,
            "energy": 0.900,
        },
    ]

    analysis = build_reward_surface_analysis(
        records,
        reward_config=_reward_config(),
        tau_requirement_min=0.9,
    )
    rows = {row["row_id"]: row for row in analysis["rows"]}

    assert rows["scenario_b_infeasible"]["dominated"] is False
    assert analysis["checks"]["feasible_not_ranked_below_dominated_infeasible"]["passed"] is True
