"""Equivalence test: the vectorized Stage-21 evaluator must match the canonical one on the
consensus probability (and latency/energy) to within numerical noise, across random topologies
and node counts. This is the contract that lets the training flow use the fast path safely."""

import random

import pytest

from marl_topology.data.stage31_production_dataset import build_scenario_evaluator
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioConfig,
    generate_production_scenarios,
)
from marl_topology.data.vectorized_objective_stack_evaluator import VectorizedStage21Evaluator


def _random_topologies(edge_ids, rng, count):
    edge_ids = list(edge_ids)
    topos = [(), tuple(edge_ids)]  # empty + full
    for _ in range(count):
        k = rng.randint(0, len(edge_ids))
        topos.append(tuple(sorted(rng.sample(edge_ids, k))))
    return topos


@pytest.mark.slow
def test_vectorized_evaluator_matches_canonical_on_consensus() -> None:
    specs = generate_production_scenarios(
        ProductionScenarioConfig(seed=17, scenario_count=4, node_count_choices=(4, 6, 8))
    )
    rng = random.Random(0)
    compared = 0
    for spec in specs:
        graph, canonical = build_scenario_evaluator(spec)
        fast = VectorizedStage21Evaluator(
            scene=spec.scene, graph=graph, config=canonical.config, ref=canonical
        )
        for topology in _random_topologies(graph.edge_ids, rng, count=4):
            ref_eval = canonical.evaluate(set(topology), topology_id=f"ref:{compared}")
            fast_eval = fast.evaluate(set(topology), topology_id=f"fast:{compared}")
            ref_p = ref_eval.metrics["consensus_success_probability"]
            fast_p = fast_eval.metrics["consensus_success_probability"]
            assert abs(ref_p - fast_p) < 1e-9, (
                f"consensus mismatch on {spec.scenario_id} topo={topology}: "
                f"ref={ref_p} fast={fast_p}"
            )
            assert abs(ref_eval.metrics["latency"] - fast_eval.metrics["latency"]) < 1e-9
            assert abs(ref_eval.metrics["energy"] - fast_eval.metrics["energy"]) < 1e-9
            compared += 1
    assert compared > 0


@pytest.mark.slow
def test_vectorized_evaluator_is_drop_in_type() -> None:
    spec = generate_production_scenarios(
        ProductionScenarioConfig(seed=3, scenario_count=1, node_count_choices=(5,))
    )[0]
    graph, canonical = build_scenario_evaluator(spec)
    fast = VectorizedStage21Evaluator(scene=spec.scene, graph=graph, config=canonical.config, ref=canonical)
    # Same public surface the flow relies on.
    assert fast.link_records is canonical.link_records
    assert fast.validator_ids == canonical.validator_ids
    evaluation = fast.evaluate(set(graph.edge_ids))
    assert "consensus_success_probability" in evaluation.metrics
    assert evaluation.evaluator_id == canonical.config.evaluator_id
