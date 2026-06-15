"""Stage 31 Phase C: feasibility-first reward surrogate (B1 repair)."""

from __future__ import annotations

import pytest

from marl_topology.data.stage31_surrogate_recalibration import (
    build_recalibrated_stage31_surrogate,
)
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioConfig,
    generate_production_scenarios,
)
from marl_topology.evaluation.reward_surface_analysis import build_reward_surface_analysis
from marl_topology.objectives.surrogate_signal import (
    SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
    SURROGATE_STRUCTURE_FLAT_WEIGHTED_SUM_V1,
    SurrogateSignalConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
)

TAU = 0.9


def _ff_config(**overrides):
    base = dict(
        tau=TAU,
        reliability_weight=1.0,
        latency_weight=1.0,
        energy_weight=1.0,
        latency_reference_s=0.004,
        energy_reference_j=0.024,
        clip_min=0.0,
        clip_max=4.0,
        reliability_penalty_power=1.0,
        structure=SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
        feasibility_margin=1.0,
    )
    base.update(overrides)
    return SurrogateSignalConfig(**base)


def _reward(cfg, psucc, latency, energy):
    return evaluate_reward_surrogate(
        SurrogateSignalInput(
            consensus_success_probability=psucc, latency=latency, energy=energy
        ),
        cfg,
    ).training_signal_value


def test_requires_finite_clip_max() -> None:
    with pytest.raises(ValueError):
        _ff_config(clip_max=None)
    with pytest.raises(ValueError):
        _ff_config(feasibility_margin=0.0)


def test_feasibility_barrier_every_feasible_beats_every_infeasible() -> None:
    cfg = _ff_config()
    # The original pathology: a cheap-but-infeasible topology outscoring an
    # expensive-but-feasible one. The barrier forbids it.
    feasible_expensive = _reward(cfg, psucc=0.90, latency=0.004, energy=0.024)
    infeasible_cheap = _reward(cfg, psucc=0.89, latency=1e-6, energy=1e-6)
    assert feasible_expensive > infeasible_cheap
    # Even the worst feasible beats the best infeasible (psucc just below tau).
    worst_feasible = _reward(cfg, psucc=0.90, latency=0.004, energy=0.024)
    best_infeasible = _reward(cfg, psucc=0.8999, latency=1e-9, energy=1e-9)
    assert worst_feasible > best_infeasible


def test_reliability_plateaus_above_tau() -> None:
    cfg = _ff_config()
    # Same latency/energy, psucc 0.90 vs 1.0 -> identical reward (plateau).
    at_tau = _reward(cfg, psucc=0.90, latency=0.003, energy=0.02)
    above_tau = _reward(cfg, psucc=1.0, latency=0.003, energy=0.02)
    assert at_tau == pytest.approx(above_tau)


def test_within_feasible_prefers_lower_latency_then_energy() -> None:
    cfg = _ff_config()
    cheap = _reward(cfg, psucc=0.95, latency=0.001, energy=0.005)
    pricey = _reward(cfg, psucc=0.95, latency=0.004, energy=0.024)
    assert cheap > pricey


def test_within_infeasible_prefers_higher_psucc() -> None:
    cfg = _ff_config()
    # Closer to tau is less bad; latency/energy do not matter while infeasible.
    near = _reward(cfg, psucc=0.85, latency=0.004, energy=0.024)
    far = _reward(cfg, psucc=0.50, latency=0.0001, energy=0.0001)
    assert near > far


def test_infeasible_reward_ignores_latency_energy() -> None:
    cfg = _ff_config()
    a = _reward(cfg, psucc=0.7, latency=0.0001, energy=0.0001)
    b = _reward(cfg, psucc=0.7, latency=0.004, energy=0.024)
    # No latency/energy gradient while infeasible -> avoids the edge-shedding
    # pathology where the policy trades reliability for cheaper resources.
    assert a == pytest.approx(b)


def test_decomposition_invariant_holds() -> None:
    cfg = _ff_config()
    for psucc, lat, en in [(0.95, 0.002, 0.01), (0.8, 0.003, 0.02)]:
        rec = evaluate_reward_surrogate(
            SurrogateSignalInput(
                consensus_success_probability=psucc, latency=lat, energy=en
            ),
            cfg,
        )
        assert rec.training_signal_value == pytest.approx(
            -(rec.reliability_penalty + rec.latency_penalty + rec.energy_penalty)
        )
        assert rec.structure == SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2


def test_recalibrated_surrogate_reduces_inversions_on_real_data() -> None:
    specs = generate_production_scenarios(
        ProductionScenarioConfig(seed=31, scenario_count=36)
    )
    config_ff, refs, records = build_recalibrated_stage31_surrogate(specs)
    flat = SurrogateSignalConfig(
        tau=TAU,
        reliability_weight=1.0,
        latency_weight=1.0,
        energy_weight=1.0,
        latency_reference_s=refs.latency_reference_s,
        energy_reference_j=refs.energy_reference_j,
        clip_max=2.0,
        reliability_penalty_power=2.0,
        structure=SURROGATE_STRUCTURE_FLAT_WEIGHTED_SUM_V1,
    )

    def inversion_rate(cfg):
        analysis = build_reward_surface_analysis(
            records, reward_config=cfg, tau_requirement_min=TAU
        )
        return analysis["checks"]["reward_rank_broadly_matches_objective_order"][
            "inversion_rate"
        ]

    ff_rate = inversion_rate(config_ff)
    flat_rate = inversion_rate(flat)
    assert ff_rate <= flat_rate
    assert ff_rate < 0.06  # tight objective/reward alignment


def test_no_feasible_ranked_below_infeasible_on_real_data() -> None:
    specs = generate_production_scenarios(
        ProductionScenarioConfig(seed=5, scenario_count=24)
    )
    config_ff, _refs, records = build_recalibrated_stage31_surrogate(specs)
    rewards = []
    for r in records:
        val = evaluate_reward_surrogate(
            SurrogateSignalInput(
                consensus_success_probability=r["consensus_success_probability"],
                latency=r["latency"],
                energy=r["energy"],
            ),
            config_ff,
        ).training_signal_value
        rewards.append((r["consensus_success_probability"] >= TAU, val))
    feasible_rewards = [v for ok, v in rewards if ok]
    infeasible_rewards = [v for ok, v in rewards if not ok]
    if feasible_rewards and infeasible_rewards:
        assert min(feasible_rewards) > max(infeasible_rewards)
