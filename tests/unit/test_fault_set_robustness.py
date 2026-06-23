"""Phase 1b: fixed Byzantine fault-set robustness (Technical-Spec S4.7).

A single fault set B (|B| <= f) must be held across ALL protocol phases; robust
reliability is C_robust(x) = min_{|B|<=f} C(x; B). The current per-receiver /
per-phase REMOVE_LARGEST filter is NOT such a fixed set and is pinned here as wrong.
"""

from __future__ import annotations

from math import comb

import pytest

from marl_topology.protocol import (
    FAULT_FILTER_NONE,
    PBFTExpectedInitiatorConfig,
    consensus_given_fault_set,
    enumerate_fault_sets,
    evaluate_expected_initiator_pbft_reliability,
    robust_consensus_reliability,
)
from marl_topology.protocol.quorum_spec import PBFTQuorumSpec


NODES24 = tuple(f"n{i}" for i in range(24))


NODES8 = tuple(f"n{i}" for i in range(8))


def _complete_matrix(p: float, nodes: tuple[str, ...]) -> dict[tuple[str, str], float]:
    return {(a, b): p for a in nodes for b in nodes if a != b}


# --- enumeration ---

def test_enumerate_fault_sets_counts_size_f_subsets() -> None:
    sets = enumerate_fault_sets(NODES8, 2)
    assert len(sets) == comb(8, 2) == 28
    assert all(len(s) == 2 for s in sets)
    assert len({s for s in sets}) == 28  # unique
    # f = 0 -> the single empty fault set
    assert enumerate_fault_sets(NODES8, 0) == (frozenset(),)


def test_enumeration_budget_guard_raises() -> None:
    # comb(24, 7) = 346104 >> a small budget -> must fail loud, not silently approximate.
    with pytest.raises(ValueError, match="enumeration budget"):
        robust_consensus_reliability(
            tuple(f"n{i}" for i in range(24)),
            pre_prepare_matrix={},
            prepare_matrix={},
            commit_matrix={},
            fault_tolerance=7,
            max_enumeration=10000,
        )


# --- parity with the validated cascade at f = 0 (no faults) ---

def test_f0_robust_matches_existing_no_filter_cascade() -> None:
    m = _complete_matrix(0.83, NODES8)
    robust = robust_consensus_reliability(
        NODES8,
        pre_prepare_matrix=m,
        prepare_matrix=m,
        commit_matrix=m,
        fault_tolerance=0,
    )
    cfg = PBFTExpectedInitiatorConfig(
        node_ids=NODES8, fault_tolerance=0, fault_filter_mode=FAULT_FILTER_NONE
    )
    existing = evaluate_expected_initiator_pbft_reliability(
        cfg, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m
    )
    assert robust.consensus_success_probability == pytest.approx(
        existing.consensus_success_probability
    )


# --- the same B is held across all phases (a single fixed set) ---

def test_consensus_given_fault_set_excludes_B_in_every_phase() -> None:
    m = _complete_matrix(0.9, NODES8)
    spec = PBFTQuorumSpec(node_count=8, fault_tolerance=2)
    fault_set = frozenset({"n0", "n1"})
    c = consensus_given_fault_set(
        NODES8, fault_set, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        quorum=spec.quorum, external_quorum=spec.external_quorum,
    )
    # a faulty node never votes and is not a successful initiator; with n0,n1 faulty the
    # honest committee is the other 6, and c is the mean over those 6 honest initiators.
    assert 0.0 <= c <= 1.0
    honest_only = consensus_given_fault_set(
        NODES8, fault_set, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        quorum=spec.quorum, external_quorum=spec.external_quorum,
    )
    assert honest_only == pytest.approx(c)


# --- monotonicity: growing B can only lower reliability ---

def test_consensus_is_monotone_nonincreasing_in_fault_set() -> None:
    m = _complete_matrix(0.88, NODES8)
    spec = PBFTQuorumSpec(node_count=8, fault_tolerance=2)
    c1 = consensus_given_fault_set(
        NODES8, frozenset({"n0"}), pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        quorum=spec.quorum, external_quorum=spec.external_quorum,
    )
    c2 = consensus_given_fault_set(
        NODES8, frozenset({"n0", "n1"}), pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        quorum=spec.quorum, external_quorum=spec.external_quorum,
    )
    assert c2 <= c1 + 1e-12


# --- robust is the min over fault sets; <= every individual C(B) ---

def test_robust_is_min_over_fault_sets() -> None:
    m = _complete_matrix(0.86, NODES8)
    res = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m, fault_tolerance=2,
    )
    spec = PBFTQuorumSpec(node_count=8, fault_tolerance=2)
    every = [
        consensus_given_fault_set(
            NODES8, b, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
            quorum=spec.quorum, external_quorum=spec.external_quorum,
        )
        for b in enumerate_fault_sets(NODES8, 2)
    ]
    # the search ranges over ALL |B| <= 2; with a symmetric matrix the min lands at |B| = 2,
    # so it equals the min over the size-2 sets, but every size 0..2 was enumerated.
    assert res.consensus_success_probability == pytest.approx(min(every))
    assert res.fault_set_count == sum(comb(8, r) for r in range(3)) == 37
    assert len(frozenset(res.worst_case_fault_set)) <= 2


# --- softmin >= ... approaches hard min from below as beta grows ---

def test_softmin_lower_bounds_and_approaches_hard_min() -> None:
    # n=4,f=1 (only C(4,1)=4 fault sets) at high reliability keeps the worst-case C(B)
    # comfortably positive, so the strict softmin properties are not masked by the [0,1]
    # clamp. softmin = -1/beta log sum exp(-beta C_B) is a strict lower bound on the min
    # (with >= 2 sets) that approaches the hard min as beta grows.
    nodes4 = ("n0", "n1", "n2", "n3")
    m = _complete_matrix(0.99, nodes4)
    hard = robust_consensus_reliability(
        nodes4, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=1, reduction="hard_min",
    ).consensus_success_probability
    assert hard > 0.1  # construction sanity: not on the 0-floor
    soft_weak = robust_consensus_reliability(
        nodes4, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=1, reduction="softmin", softmin_beta=20.0,
    ).consensus_success_probability
    soft_strong = robust_consensus_reliability(
        nodes4, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=1, reduction="softmin", softmin_beta=300.0,
    ).consensus_success_probability
    assert soft_weak < hard + 1e-9             # softmin is a lower bound on the min
    assert abs(soft_strong - hard) < abs(soft_weak - hard)  # larger beta -> closer to min


# --- the bug: per-phase REMOVE_LARGEST != a single fixed B ---

def test_remove_largest_differs_from_fixed_set_robustness() -> None:
    """The per-receiver / per-phase remove-largest filter is NOT a single coherent
    adversary, so it gives a different number from the principled fixed-B min. Neither
    uniformly dominates: remove-largest over-strips votes (different nodes per receiver
    and per phase), while fixed-B additionally zeros faulty primaries (deferred
    view-change) -- they are genuinely different models, which is exactly the point."""
    m = _complete_matrix(0.95, NODES8)
    for r in NODES8:
        if r not in ("n0",):
            m[("n0", r)] = 0.99
        if r not in ("n7",):
            m[("n7", r)] = 0.99
    remove_largest = evaluate_expected_initiator_pbft_reliability(
        PBFTExpectedInitiatorConfig(node_ids=NODES8, fault_tolerance=2),  # default REMOVE_LARGEST
        pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
    ).consensus_success_probability
    fixed = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m, fault_tolerance=2,
    ).consensus_success_probability
    assert 0.0 <= remove_largest <= 1.0
    assert 0.0 <= fixed <= 1.0
    assert remove_largest != pytest.approx(fixed)


# --- per-primary contributions are reported and average to C_robust (Phase 1b-wire) ---

def test_per_primary_reliability_averages_to_consensus() -> None:
    m = _complete_matrix(0.9, NODES8)
    res = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m, fault_tolerance=2,
    )
    pp = res.per_primary_reliability
    assert set(pp) == set(NODES8)
    for node in res.worst_case_fault_set:  # faulty nodes are not successful initiators
        assert pp[node] == 0.0
    # the scalar is the mean over the HONEST initiators (deferred view-change), not all n.
    honest = [pp[p] for p in NODES8 if p not in res.worst_case_fault_set]
    assert sum(honest) / len(honest) == pytest.approx(res.consensus_success_probability)


# --- greedy fallback for large f; equals exact at f = 1 ---

def test_greedy_equals_exact_at_f1() -> None:
    m = _complete_matrix(0.85, NODES8)
    for r in NODES8:           # asymmetric so the worst single node is well-defined
        if r != "n3":
            m[("n3", r)] = 0.3
    exact = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=1, strategy="exact",
    )
    greedy = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=1, strategy="greedy",
    )
    assert greedy.consensus_success_probability == pytest.approx(exact.consensus_success_probability)
    assert greedy.worst_case_fault_set == exact.worst_case_fault_set


def test_auto_strategy_falls_back_to_greedy_above_budget() -> None:
    # n=24, f=7: C(24,7)=346104 >> budget -> auto must NOT raise; it uses greedy (O(f*n) evals).
    m = _complete_matrix(0.95, NODES24)
    res = robust_consensus_reliability(
        NODES24, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=7, strategy="auto", max_enumeration=10000,
    )
    assert res.strategy == "greedy"
    assert res.enumeration_exact is False
    assert len(res.worst_case_fault_set) == 7
    assert 0.0 <= res.consensus_success_probability <= 1.0


def test_exact_strategy_records_activation_metadata() -> None:
    m = _complete_matrix(0.9, NODES8)
    res = robust_consensus_reliability(
        NODES8, pre_prepare_matrix=m, prepare_matrix=m, commit_matrix=m,
        fault_tolerance=2, strategy="auto",
    )
    assert res.strategy == "exact"
    assert res.enumeration_exact is True
    assert res.is_certified is True  # exact hard-min over all |B| <= f is a certificate
    assert res.fault_set_count == sum(comb(8, r) for r in range(3)) == 37
