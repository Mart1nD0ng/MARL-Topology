"""Phase 3: tri-state solvability + train-only witness memory (Technical-Spec S5).

A finite-search MISS must be ``unknown``, never ``certified_infeasible`` (hard-constraint
#11). ``certified_infeasible`` requires a PROVEN optimistic upper bound below tau (S5.2).
Witness memories are split-isolated; held/test discoveries must never feed training (S5.3).
"""

from __future__ import annotations

import pytest

from marl_topology.solvability import (
    CERTIFIED_INFEASIBLE,
    UNKNOWN,
    WITNESS_FEASIBLE,
    Witness,
    WitnessMemory,
    classify_solvability,
    solvability_from_finite_search,
)


TAU = 0.9


# --- classifier: the three states ---

def test_witness_at_or_above_tau_is_feasible() -> None:
    v = classify_solvability(lower_bound=0.95, tau=TAU)
    assert v.status == WITNESS_FEASIBLE
    assert classify_solvability(lower_bound=0.90, tau=TAU).status == WITNESS_FEASIBLE  # boundary


def test_search_miss_without_proof_is_unknown_not_infeasible() -> None:
    # THE Phase-3 fix: no discovered witness reaches tau, and there is NO proven upper bound.
    v = classify_solvability(lower_bound=0.3, tau=TAU, upper_bound=None)
    assert v.status == UNKNOWN
    assert v.status != CERTIFIED_INFEASIBLE


def test_proven_upper_bound_below_tau_is_certified_infeasible() -> None:
    v = classify_solvability(lower_bound=0.2, tau=TAU, upper_bound=0.5)
    assert v.status == CERTIFIED_INFEASIBLE


def test_low_witness_high_upper_bound_is_unknown() -> None:
    # best found 0.4, but the optimistic relaxation could reach 0.99 -> still in the dark.
    assert classify_solvability(lower_bound=0.4, tau=TAU, upper_bound=0.99).status == UNKNOWN


# --- a finite search can only ever produce W or U, never I (hard-constraint #11) ---

def test_finite_search_never_certifies_infeasible() -> None:
    assert solvability_from_finite_search(best_witness_consensus=0.95, tau=TAU).status == WITNESS_FEASIBLE
    miss = solvability_from_finite_search(best_witness_consensus=0.0, tau=TAU)
    assert miss.status == UNKNOWN
    assert miss.upper_bound is None  # a finite search proves no upper bound


def test_binary_feasible_exists_false_maps_to_unknown() -> None:
    # The legacy scenario generator sets feasible_exists=False on a search miss; under the
    # corrected semantics that is UNKNOWN (a candidate may still exist), never infeasible.
    legacy_feasible_exists = False
    verdict = solvability_from_finite_search(best_witness_consensus=0.55, tau=TAU)
    assert (verdict.status == WITNESS_FEASIBLE) == legacy_feasible_exists  # both False here
    assert verdict.status == UNKNOWN


# --- validation guards ---

def test_lower_bound_above_upper_bound_raises() -> None:
    with pytest.raises(ValueError, match="lower_bound.*upper_bound"):
        classify_solvability(lower_bound=0.8, tau=TAU, upper_bound=0.5)


def test_out_of_range_inputs_raise() -> None:
    with pytest.raises(ValueError):
        classify_solvability(lower_bound=1.2, tau=TAU)
    with pytest.raises(ValueError):
        classify_solvability(lower_bound=0.5, tau=0.0)


# --- witness memory: monotone, split-isolated ---

def test_witness_memory_keeps_best_and_is_monotone() -> None:
    mem = WitnessMemory(split="train")
    assert mem.lower_bound("s1") == 0.0
    assert mem.update("s1", Witness(topology=("a--b",), reliability=0.7, energy=1.0, latency=0.01)) is True
    assert mem.lower_bound("s1") == pytest.approx(0.7)
    # a worse witness does not replace the best
    assert mem.update("s1", Witness(topology=("a--c",), reliability=0.5, energy=0.5, latency=0.01)) is False
    assert mem.lower_bound("s1") == pytest.approx(0.7)
    # a better witness does
    assert mem.update("s1", Witness(topology=("a--d",), reliability=0.93, energy=2.0, latency=0.02)) is True
    assert mem.lower_bound("s1") == pytest.approx(0.93)


def test_witness_memory_drives_solvability() -> None:
    mem = WitnessMemory(split="train")
    mem.update("s1", Witness(topology=("a--b",), reliability=0.93, energy=1.0, latency=0.01))
    assert classify_solvability(mem.lower_bound("s1"), tau=TAU).status == WITNESS_FEASIBLE
    assert classify_solvability(mem.lower_bound("unseen"), tau=TAU).status == UNKNOWN


def test_split_isolation_blocks_test_to_train_feedback() -> None:
    train = WitnessMemory(split="train")
    test = WitnessMemory(split="test")
    test.update("s9", Witness(topology=("x--y",), reliability=0.99, energy=1.0, latency=0.01))
    # held/test discoveries must NOT feed training (S5.3) -- merging test into train fails fast.
    with pytest.raises(ValueError, match="held/test|isolation|train"):
        train.merge_from(test)
    # merging another train memory is allowed
    other_train = WitnessMemory(split="train")
    other_train.update("s9", Witness(topology=("x--y",), reliability=0.95, energy=1.0, latency=0.01))
    train.merge_from(other_train)
    assert train.lower_bound("s9") == pytest.approx(0.95)
