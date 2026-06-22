"""Tri-state solvability (Phase 3, Technical-Spec S5).

witness_feasible / certified_infeasible / unknown, with a train-only split-isolated witness
memory. A finite-search miss is ``unknown`` (never ``certified_infeasible``); infeasibility
requires a proven optimistic upper bound below tau.
"""

from .status import (
    CERTIFIED_INFEASIBLE,
    SOLVABILITY_STATES,
    UNKNOWN,
    WITNESS_FEASIBLE,
    SolvabilityVerdict,
    classify_solvability,
    solvability_from_finite_search,
)
from .witness_memory import WITNESS_SPLITS, Witness, WitnessMemory

__all__ = [
    "CERTIFIED_INFEASIBLE",
    "SOLVABILITY_STATES",
    "UNKNOWN",
    "WITNESS_FEASIBLE",
    "SolvabilityVerdict",
    "classify_solvability",
    "solvability_from_finite_search",
    "WITNESS_SPLITS",
    "Witness",
    "WitnessMemory",
]
