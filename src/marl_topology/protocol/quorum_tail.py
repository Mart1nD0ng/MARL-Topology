"""Heterogeneous quorum-tail utilities for Stage 4.1."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Iterable


QUORUM_TAIL_EVALUATOR_ID = "stage4_heterogeneous_quorum_tail_v1"


@dataclass(frozen=True, slots=True)
class QuorumTailResult:
    """Diagnostic record for an analytic heterogeneous quorum-tail evaluation."""

    quorum_size: int
    input_count: int
    probability: float
    evaluator_id: str = QUORUM_TAIL_EVALUATOR_ID

    def __post_init__(self) -> None:
        if self.quorum_size < 0:
            raise ValueError("quorum_size must be nonnegative")
        if self.input_count < 0:
            raise ValueError("input_count must be nonnegative")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not self.evaluator_id:
            raise ValueError("evaluator_id must be non-empty")


def heterogeneous_quorum_tail(
    probabilities: Iterable[float],
    quorum_size: int,
) -> float:
    """Return the probability that at least ``quorum_size`` inputs succeed.

    The evaluator computes coefficients of the generating polynomial
    ``prod_j ((1 - p_j) + p_j z)`` with a capped tail bucket. It supports
    heterogeneous Bernoulli probabilities without assuming a common success
    probability and without enumerating success subsets.
    """

    values = _checked_probabilities(probabilities)
    quorum = _checked_quorum_size(quorum_size)
    if quorum == 0:
        return 1.0
    if quorum > len(values):
        return 0.0

    buckets = [0.0] * (quorum + 1)
    buckets[0] = 1.0
    for probability in values:
        failure_probability = 1.0 - probability
        next_buckets = [0.0] * (quorum + 1)
        next_buckets[0] = buckets[0] * failure_probability
        for success_count in range(1, quorum):
            next_buckets[success_count] = fsum(
                (
                    buckets[success_count] * failure_probability,
                    buckets[success_count - 1] * probability,
                )
            )
        next_buckets[quorum] = fsum(
            (
                buckets[quorum],
                buckets[quorum - 1] * probability,
            )
        )
        buckets = next_buckets

    return _clamp_probability(buckets[quorum])


def evaluate_quorum_tail(
    probabilities: Iterable[float],
    quorum_size: int,
) -> QuorumTailResult:
    """Return a diagnostic result for the heterogeneous quorum-tail utility."""

    values = _checked_probabilities(probabilities)
    quorum = _checked_quorum_size(quorum_size)
    return QuorumTailResult(
        quorum_size=quorum,
        input_count=len(values),
        probability=heterogeneous_quorum_tail(values, quorum),
    )


def remove_largest_probabilities(
    probabilities: Iterable[float],
    remove_count: int,
) -> tuple[float, ...]:
    """Remove the largest probabilities as a conservative byzantine filter."""

    values = _checked_probabilities(probabilities)
    count = _checked_remove_count(remove_count)
    if count == 0:
        return values
    if count >= len(values):
        return ()

    ranked = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    removed_indices = {index for index, _ in ranked[:count]}
    return tuple(value for index, value in enumerate(values) if index not in removed_indices)


def conservative_quorum_tail(
    probabilities: Iterable[float],
    quorum_size: int,
    fault_count: int,
) -> float:
    """Evaluate quorum tail after conservative largest-probability filtering."""

    filtered = remove_largest_probabilities(probabilities, fault_count)
    return heterogeneous_quorum_tail(filtered, quorum_size)


def _checked_probabilities(probabilities: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in probabilities)
    for value in values:
        if not isfinite(value):
            raise ValueError("probabilities must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
    return values


def _checked_quorum_size(quorum_size: int) -> int:
    if isinstance(quorum_size, bool) or not isinstance(quorum_size, int):
        raise TypeError("quorum_size must be an integer")
    if quorum_size < 0:
        raise ValueError("quorum_size must be nonnegative")
    return quorum_size


def _checked_remove_count(remove_count: int) -> int:
    if isinstance(remove_count, bool) or not isinstance(remove_count, int):
        raise TypeError("remove_count must be an integer")
    if remove_count < 0:
        raise ValueError("remove_count must be nonnegative")
    return remove_count


def _clamp_probability(value: float) -> float:
    if value < 0.0 and value > -1e-15:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-15:
        return 1.0
    return value
