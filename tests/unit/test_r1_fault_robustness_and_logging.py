"""R1 (v2 Engineering-Plan §R1, Spec §4.7): honest fixed-B robustness + auditable f/q logging.

Pins four R1 contracts:
  - the worst case is searched over ALL |B| <= f, not only |B| = f (Spec §4.7.1);
  - honest-primary averaging is pointwise non-monotone (the counterexample that makes
    "the worst case must be at |B| = f" unjustified);
  - greedy is an OPTIMISTIC approximation, never marked exact or certified (Spec §4.7.2);
  - the production evaluator logs validator_count / configured_f / effective_f / q /
    q_external / fault_strategy / fault_set_count / enumeration_exact / is_certified.
"""

from __future__ import annotations

import dataclasses
from math import comb

import pytest

from marl_topology.protocol.fault_set_robustness import (
    consensus_given_fault_set,
    enumerate_fault_sets_up_to,
    robust_consensus_reliability,
)
from marl_topology.protocol.quorum_spec import PBFTQuorumSpec
from marl_topology.data.stage21_objective_stack_evidence import Stage21ObjectiveStackEvaluator
from marl_topology.data.stage31_production_dataset import build_scenario_evaluator
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioConfig,
    generate_production_scenarios,
)

NODES8 = tuple(f"n{i}" for i in range(8))
NODES7 = tuple(f"n{i}" for i in range(7))


def _complete(p: float, nodes: tuple[str, ...]) -> dict[tuple[str, str], float]:
    return {(a, b): p for a in nodes for b in nodes if a != b}


# --- (1) the search covers all sizes 0..f, not just |B| = f -------------------------------

def test_fault_set_min_checks_all_sizes_when_required() -> None:
    m = _complete(0.86, NODES8)
    res = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m, fault_tolerance=2,
    )
    # enumerated EVERY |B| <= 2 (1 + 8 + 28 = 37), not just the 28 size-2 sets
    assert res.fault_set_count == sum(comb(8, r) for r in range(3)) == 37
    spec = PBFTQuorumSpec(node_count=8, fault_tolerance=2)
    brute = min(
        consensus_given_fault_set(
            NODES8, b, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
            quorum=spec.quorum, external_quorum=spec.external_quorum,
        )
        for b in enumerate_fault_sets_up_to(NODES8, 2)  # ALL sizes
    )
    assert res.consensus_success_probability == pytest.approx(brute)


# --- (2) honest-primary averaging is pointwise non-monotone (the counterexample) ----------

def test_asymmetric_primary_breaks_naive_monotonicity() -> None:
    # n0 is a weak primary AND weak voter; the rest are strong. Removing n0 (calling it
    # Byzantine) drops a below-average primary from the honest-average denominator, so
    # C_honest({n0}) RISES above C_honest({}) -- monotonicity in B fails, hence one cannot
    # assume the worst case lies at |B| = f. (n=7, f=1; deterministic, no RNG.)
    spec = PBFTQuorumSpec(node_count=7, fault_tolerance=1)
    m = {(a, b): (0.2 if a == "n0" else 0.97) for a in NODES7 for b in NODES7 if a != b}

    def C(*faulty):
        return consensus_given_fault_set(
            NODES7, frozenset(faulty), pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
            quorum=spec.quorum, external_quorum=spec.external_quorum,
        )

    assert C("n0") > C() + 1e-6           # removing the weak primary RAISES C: non-monotone
    assert C("n1") < C() - 1e-6           # removing a strong/critical node lowers C
    # the robust search therefore enumerates ALL sizes 0..1 (the empty set included), so it
    # does not silently rely on the broken monotonicity assumption.
    res = robust_consensus_reliability(
        NODES7, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m, fault_tolerance=1,
    )
    assert res.fault_set_count == sum(comb(7, r) for r in range(2)) == 8


# --- (3) greedy is an optimistic approximation: never exact, never certified --------------

def test_greedy_is_not_marked_exact_or_certified() -> None:
    m = _complete(0.9, NODES8)
    greedy = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=2, strategy="greedy",
    )
    assert greedy.strategy == "greedy"
    assert greedy.enumeration_exact is False
    assert greedy.is_certified is False
    # greedy is an OPTIMISTIC bound: C(B_greedy) >= the exact min
    exact = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=2, strategy="exact",
    )
    assert exact.enumeration_exact is True
    assert exact.is_certified is True
    assert greedy.consensus_success_probability >= exact.consensus_success_probability - 1e-9
    # softmin is exact-enumeration but a smoothed surrogate -> NOT a certificate. Use a
    # well-conditioned instance (n=4,f=1,m=0.99) so the surrogate stays in [0,1].
    nodes4 = ("n0", "n1", "n2", "n3")
    m4 = _complete(0.99, nodes4)
    soft = robust_consensus_reliability(
        nodes4, pre_prepare_matrix=m4, prepare_matrix=m4, commit_matrix=m4,
        fault_tolerance=1, strategy="exact", reduction="softmin",
    )
    assert soft.enumeration_exact is True
    assert soft.is_certified is False


# --- (4) the production evaluator logs the effective f / q accounting ----------------------

def _scene(node_count: int = 8):
    spec = generate_production_scenarios(
        ProductionScenarioConfig(seed=5, scenario_count=1, node_count_choices=(node_count,))
    )[0]
    graph, canonical = build_scenario_evaluator(spec)
    return spec, graph, canonical


def test_production_effective_f_q_are_logged() -> None:
    spec, graph, canonical = _scene(8)
    metrics = canonical.evaluate(set(graph.edge_ids)).metrics
    fa = metrics["fault_accounting"]
    for field in ("validator_count", "configured_fault_tolerance", "effective_fault_tolerance",
                  "quorum", "external_quorum", "fault_strategy", "fault_set_count",
                  "enumeration_exact", "is_certified", "worst_case_fault_set"):
        assert field in fa
    n = fa["validator_count"]
    assert fa["effective_fault_tolerance"] <= fa["configured_fault_tolerance"]
    assert fa["effective_fault_tolerance"] <= max(0, (n - 1) // 3)  # n-clamped, Spec §4.7
    assert fa["quorum"] == (n + fa["effective_fault_tolerance"]) // 2 + 1  # safe_generalized q
    assert fa["external_quorum"] == fa["quorum"] - 1
    # default fault strategy is the remove_largest heuristic -> NOT a certified fixed-B worst case
    assert fa["fault_strategy"] == "remove_largest"
    assert fa["is_certified"] is False


def test_production_effective_f_clamp_engages_and_is_logged() -> None:
    # configured f=3 at N=8 must be CLAMPED to effective f = (8-1)//3 = 2 (PBFT n >= 3f+1),
    # and the discrepancy must be VISIBLE in the log -- not the trivial configured==effective case.
    spec, graph, canonical = _scene(8)
    over = Stage21ObjectiveStackEvaluator(
        scene=spec.scene, graph=graph,
        config=dataclasses.replace(canonical.config, fault_tolerance=3),
    )
    fa = over.evaluate(set(graph.edge_ids)).metrics["fault_accounting"]
    n = fa["validator_count"]
    assert fa["configured_fault_tolerance"] == 3
    assert fa["effective_fault_tolerance"] == (n - 1) // 3      # the clamp actually bit
    assert fa["effective_fault_tolerance"] < fa["configured_fault_tolerance"]
    assert fa["quorum"] == (n + fa["effective_fault_tolerance"]) // 2 + 1


def test_production_fixed_set_is_certified_and_logs_worst_case() -> None:
    spec, graph, canonical = _scene(8)
    fixed = Stage21ObjectiveStackEvaluator(
        scene=spec.scene, graph=graph,
        config=dataclasses.replace(canonical.config, fault_model="fixed_set"),
    )
    fa = fixed.evaluate(set(graph.edge_ids)).metrics["fault_accounting"]
    assert fa["fault_strategy"] == "fixed_set"
    assert fa["enumeration_exact"] is True          # small N -> exact enumeration
    assert fa["is_certified"] is True               # exact hard-min over all |B| <= f
    assert fa["fault_set_count"] == sum(comb(fa["validator_count"], r)
                                        for r in range(fa["effective_fault_tolerance"] + 1))
    assert isinstance(fa["worst_case_fault_set"], tuple)
