"""Tri-state solvability status (Phase 3, Technical-Spec S5.1-5.2).

A scene's solvability is one of:

* ``witness_feasible``     -- a concrete topology with consensus C >= tau has been FOUND.
* ``certified_infeasible`` -- a PROVEN optimistic upper bound is < tau (no topology can reach
                              tau). The bound must come from a proven relaxation, not a model.
* ``unknown``              -- neither: no witness yet AND no proof of infeasibility.

The decisive rule (hard-constraint #11): a finite search that fails to find a feasible
topology yields ``unknown``, NEVER ``certified_infeasible`` -- absence of a witness is not a
proof of infeasibility. ``unknown`` scenes are kept, never silently deleted (#12).
"""

from __future__ import annotations

from dataclasses import dataclass


WITNESS_FEASIBLE = "witness_feasible"
CERTIFIED_INFEASIBLE = "certified_infeasible"
UNKNOWN = "unknown"
SOLVABILITY_STATES = (WITNESS_FEASIBLE, CERTIFIED_INFEASIBLE, UNKNOWN)


@dataclass(frozen=True, slots=True)
class SolvabilityVerdict:
    """A scene's tri-state solvability with the bounds that justified it."""

    status: str
    lower_bound: float
    tau: float
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if self.status not in SOLVABILITY_STATES:
            raise ValueError(f"status must be one of {SOLVABILITY_STATES}")
        if not 0.0 <= self.lower_bound <= 1.0:
            raise ValueError("lower_bound must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.upper_bound is not None and not 0.0 <= self.upper_bound <= 1.0:
            raise ValueError("upper_bound must be in [0, 1]")


def classify_solvability(
    lower_bound: float,
    tau: float,
    upper_bound: float | None = None,
) -> SolvabilityVerdict:
    """Classify a scene from its feasibility bounds.

    ``lower_bound``: the best WITNESSED consensus reliability -- ``max`` over discovered
    feasible topologies of ``C(s, x)`` (0.0 if no witness). A finite search produces this.
    ``upper_bound``: a PROVEN optimistic upper bound on achievable ``C`` (from a proven
    relaxation), or ``None`` if no such proof exists.

    Returns ``witness_feasible`` iff ``lower_bound >= tau``; ``certified_infeasible`` iff a
    proven ``upper_bound < tau``; otherwise ``unknown``.
    """

    if not 0.0 <= lower_bound <= 1.0:
        raise ValueError("lower_bound must be in [0, 1]")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    if upper_bound is not None:
        if not 0.0 <= upper_bound <= 1.0:
            raise ValueError("upper_bound must be in [0, 1]")
        if lower_bound > upper_bound + 1e-12:
            raise ValueError(
                "lower_bound cannot exceed upper_bound (a witness cannot beat the optimistic "
                f"bound): lower_bound={lower_bound} upper_bound={upper_bound}"
            )

    if lower_bound >= tau:
        status = WITNESS_FEASIBLE
    elif upper_bound is not None and upper_bound < tau:
        status = CERTIFIED_INFEASIBLE
    else:
        status = UNKNOWN
    return SolvabilityVerdict(status=status, lower_bound=lower_bound, tau=tau, upper_bound=upper_bound)


def solvability_from_finite_search(
    best_witness_consensus: float,
    tau: float,
) -> SolvabilityVerdict:
    """Classify from a FINITE search alone (no infeasibility proof).

    A finite search can only ever discover a witness; it can never certify infeasibility, so
    the result is ``witness_feasible`` (found) or ``unknown`` (miss) -- never
    ``certified_infeasible`` (hard-constraint #11). ``upper_bound`` is always ``None`` here.
    """

    return classify_solvability(lower_bound=best_witness_consensus, tau=tau, upper_bound=None)
