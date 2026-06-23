"""Tri-state solvability control for the training trunk (v2 Engineering-Plan §R3, Spec §5).

The trunk must stop skipping ``unknown`` scenes via the binary ``feasible_exists`` label
(D8 / hard-constraint #8: an unknown scene is NOT deleted). Under tri-state solvability a
finite candidate search yields only a witness or ``unknown`` -- never ``certified_infeasible``
(#11 / Spec §5.1) -- so:

  - ``witness_feasible`` scenes train (a feasible topology is known);
  - ``unknown`` scenes train BY DEFAULT (they enter exploration; the dense reward gives a
    gradient toward higher consensus and the policy may DISCOVER a witness, U -> W);
  - ``certified_infeasible`` scenes never train (a PROVEN-infeasible scene has no learnable
    feasibility signal) -- currently none exist (no proven optimistic upper bound is computed,
    so certified-infeasible is explicitly UNAVAILABLE, Spec §5.2 / §R3 work-item 6).

This is training-side governance (gate-exempt ``training/``); it touches no deployed path.
"""

from __future__ import annotations

from collections.abc import Mapping

from marl_topology.solvability.status import (
    CERTIFIED_INFEASIBLE,
    SOLVABILITY_STATES,
    UNKNOWN,
    WITNESS_FEASIBLE,
)


def solvability_status_for_label(label: Mapping[str, object]) -> str:
    """Tri-state status of a trunk teacher-label.

    Honours an explicit ``label['solvability_status']`` when present; otherwise derives it from
    the finite-search outcome ``feasible_exists`` (a budget-feasible witness reached tau). A
    finite-search MISS is ``unknown`` -- never ``certified_infeasible`` (#11): the search cannot
    prove infeasibility.
    """
    existing = label.get("solvability_status")
    if existing in SOLVABILITY_STATES:
        return str(existing)
    return WITNESS_FEASIBLE if bool(label.get("feasible_exists")) else UNKNOWN


def trainable_under_tristate(status: str, *, exclude_unknown: bool = False) -> bool:
    """Whether a scene of the given solvability ``status`` enters training.

    ``witness_feasible`` always trains; ``unknown`` trains by DEFAULT (D8 -- it is exploration,
    not a certified violation), unless an explicit ablation excludes it; ``certified_infeasible``
    never trains (no learnable feasibility signal).
    """
    if status not in SOLVABILITY_STATES:
        raise ValueError(f"unknown solvability status {status!r}; must be one of {SOLVABILITY_STATES}")
    if status == CERTIFIED_INFEASIBLE:
        return False
    if status == UNKNOWN and exclude_unknown:
        return False
    return True


def assert_split_isolation(*splits: set) -> None:
    """Runtime activation assertion (Spec §5.3 / §R3 work-item 4): the train/val/held scenario-id
    sets must be pairwise DISJOINT, so a held/val witness discovery can never feed training.
    """
    seen: set = set()
    for split in splits:
        overlap = seen & set(split)
        if overlap:
            raise AssertionError(f"witness-memory isolation violated: scenes in >1 split: {sorted(overlap)[:5]}")
        seen |= set(split)
