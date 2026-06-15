"""Stage 31 Phase B: procedural scenario generator with a measured tau gradient."""

from __future__ import annotations

import pytest

from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    ProductionScenarioConfig,
    ProceduralGeneratorViolation,
    generate_production_scenarios,
    measure_reliable_range_m,
    scenario_fixture_from_spec,
    summarize_feasibility_distribution,
)

TAU = 0.9


def _specs(count: int = 48, seed: int = 31):
    return generate_production_scenarios(
        ProductionScenarioConfig(seed=seed, scenario_count=count)
    )


def test_reliable_range_is_finite_and_realistic() -> None:
    r = measure_reliable_range_m(PhysicsRegime())
    assert 30.0 < r < 1000.0  # an urban V2X short-range, not infinite/faked


def test_generator_is_deterministic() -> None:
    a = _specs(seed=7)
    b = _specs(seed=7)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert [round(s.best_feasible_psucc, 6) for s in a] == [
        round(s.best_feasible_psucc, 6) for s in b
    ]


def test_produces_real_feasibility_gradient_not_saturated() -> None:
    summary = summarize_feasibility_distribution(_specs(64))
    # Both feasible and infeasible must be materially present (no saturation).
    assert 0.25 <= summary["feasible_fraction"] <= 0.85
    assert 0.15 <= summary["infeasible_fraction"] <= 0.75


def test_tau_0_9_is_reachable_on_feasible_scenarios() -> None:
    specs = _specs(64)
    feasible = [s for s in specs if s.feasible_exists]
    assert feasible, "expected some feasible scenarios"
    # Every feasible scenario actually reaches the 0.9 constraint by measurement.
    assert all(s.best_feasible_psucc >= TAU for s in feasible)


def test_infeasible_scenarios_stay_below_tau() -> None:
    specs = _specs(64)
    infeasible = [s for s in specs if not s.feasible_exists]
    assert infeasible, "expected some genuinely infeasible scenarios"
    assert all(s.best_feasible_psucc < TAU for s in infeasible)


def test_problem_is_non_trivial_full_graph_not_always_optimal() -> None:
    # Under shared-spectrum interference the dense full graph usually fails, so
    # the controller cannot trivially select all edges.
    summary = summarize_feasibility_distribution(_specs(64))
    assert summary["full_graph_feasible_fraction"] < 0.6
    # In a large fraction of scenarios a sparser topology is strictly better.
    assert summary["sparse_beats_full_fraction"] >= 0.3


def test_no_degenerate_quorum_and_unique_ids() -> None:
    specs = _specs(48)
    assert all(s.quorum_size >= 2 for s in specs)
    assert all(len(s.scene.nodes) >= 4 for s in specs)
    assert len({s.scenario_id for s in specs}) == len(specs)


def test_spec_to_fixture_bridge_is_valid() -> None:
    spec = _specs(8)[0]
    fixture = scenario_fixture_from_spec(spec)
    assert fixture.fixture_id == spec.scenario_id
    assert fixture.scene is spec.scene
    assert fixture.quorum_size == spec.quorum_size
    assert fixture.expected_oracle_status in {"feasible", "infeasible"}


def test_config_rejects_tau_change_and_bad_fractions() -> None:
    with pytest.raises(ProceduralGeneratorViolation):
        ProductionScenarioConfig(tau_requirement_min=0.8)
    with pytest.raises(ProceduralGeneratorViolation):
        ProductionScenarioConfig(
            target_feasible_fraction=0.5,
            target_near_threshold_fraction=0.3,
            target_infeasible_fraction=0.3,
        )
