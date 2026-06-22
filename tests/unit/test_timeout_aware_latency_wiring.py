"""Recalibration step 2b: the evaluator's timeout-aware latency wiring (Spec S4.10).

With ``timeout_aware_latency=True`` a FAILED topology pays the full phase budget per phase
(a timeout), instead of the degenerate ``min(max_all_pairs, budget)`` which charged a failed
topology ~0. Default off keeps the legacy latency byte-identical.
"""

from __future__ import annotations

import dataclasses

import pytest

from marl_topology.data.stage21_objective_stack_evidence import Stage21ObjectiveStackEvaluator
from marl_topology.data.stage31_production_dataset import build_scenario_evaluator
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioConfig,
    generate_production_scenarios,
)


def _scene():
    spec = generate_production_scenarios(
        ProductionScenarioConfig(seed=5, scenario_count=1, node_count_choices=(8,))
    )[0]
    graph, canonical = build_scenario_evaluator(spec)
    return spec, graph, canonical


def test_failed_topology_pays_full_timeout_when_enabled() -> None:
    spec, graph, canonical = _scene()
    budget = canonical.config.phase_budget_s
    legacy_lat = canonical.evaluate(set()).metrics["latency"]  # empty topology, degenerate latency
    ta = Stage21ObjectiveStackEvaluator(
        scene=spec.scene, graph=graph,
        config=dataclasses.replace(canonical.config, timeout_aware_latency=True),
    )
    ta_lat = ta.evaluate(set()).metrics["latency"]
    # empty topology -> consensus never completes -> 3 phases each pay the full budget.
    assert ta_lat == pytest.approx(3.0 * budget)
    assert ta_lat != pytest.approx(legacy_lat)  # the degenerate one charged ~0


def test_default_off_is_byte_identical() -> None:
    spec, graph, canonical = _scene()
    # default config has timeout_aware_latency=False -> latency unchanged from the legacy path.
    again = Stage21ObjectiveStackEvaluator(scene=spec.scene, graph=graph, config=canonical.config)
    full = tuple(graph.edge_ids)
    assert again.evaluate(set(full)).metrics["latency"] == pytest.approx(
        canonical.evaluate(set(full)).metrics["latency"]
    )


def test_timeout_latency_is_bounded_by_three_budgets() -> None:
    spec, graph, canonical = _scene()
    budget = canonical.config.phase_budget_s
    ta = Stage21ObjectiveStackEvaluator(
        scene=spec.scene, graph=graph,
        config=dataclasses.replace(canonical.config, timeout_aware_latency=True),
    )
    for topology in ((), tuple(graph.edge_ids)):
        lat = ta.evaluate(set(topology)).metrics["latency"]
        assert 0.0 <= lat <= 3.0 * budget + 1e-12
