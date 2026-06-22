"""Fixed Byzantine fault-set robustness for PBFT consensus (Phase 1b, Spec S4.7).

The current ``remove_largest_probabilities`` filter strips the ``f`` highest-delivery
senders *independently at every receiver and every protocol phase*. That does not
correspond to any single coherent adversary: a real faulty node is faulty everywhere.

The principled quantity fixes a single fault set ``B`` (``|B| <= f``) for the whole
protocol and takes the worst case::

    C_robust(x) = min_{B subset of V, |B| <= f}  C(x; B)

where ``C(x; B)`` is the uniform-initiator expected reliability when exactly the nodes
in ``B`` are faulty: a faulty node never delivers a valid vote and (under the deferred
view-change model) a faulty primary's view makes no progress, so it contributes 0 to
the uniform average over the ``n`` possible primaries.

``C(x; B)`` reuses the validated heterogeneous quorum-tail DP; only the orchestration
is local (restricted to the honest sub-committee ``H = V \\ B``). A parity test pins
``f = 0`` against the existing cascade.

Exact enumeration is ``C(n, f)`` fault sets; for small ``f`` this is the reference. A
hard ``min`` is used for evaluation; a ``softmin`` is available for training smoothness.
A budget guard fails loud when the enumeration is too large (no silent approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, exp, fsum, log
from typing import Mapping

from .quorum_spec import QUORUM_MODE_SAFE, PBFTQuorumSpec
from .quorum_tail import heterogeneous_quorum_tail


FAULT_SET_ROBUSTNESS_MODEL_ID = "fixed_fault_set_robust_consensus_v1"
REDUCTION_HARD_MIN = "hard_min"
REDUCTION_SOFTMIN = "softmin"
REDUCTIONS = (REDUCTION_HARD_MIN, REDUCTION_SOFTMIN)
DEFAULT_MAX_ENUMERATION = 50000
DEFAULT_SOFTMIN_BETA = 50.0

MessageMatrix = Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class FaultSetRobustnessResult:
    """Worst-case (or soft-worst-case) consensus reliability over fixed fault sets."""

    model_id: str
    consensus_success_probability: float
    worst_case_fault_set: tuple[str, ...]
    fault_set_count: int
    reduction: str
    fault_tolerance: int
    quorum: int
    external_quorum: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if self.fault_set_count < 1:
            raise ValueError("fault_set_count must be positive")
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}")


def enumerate_fault_sets(node_ids: tuple[str, ...], fault_tolerance: int) -> tuple[frozenset, ...]:
    """All Byzantine fault sets of size exactly ``f`` (the worst case lies at ``|B| = f``).

    ``C(x; B)`` is monotone non-increasing in ``B`` (adding a faulty node only removes
    honest votes), so ``min_{|B| <= f}`` is attained at ``|B| = f``. ``f = 0`` yields the
    single empty set.
    """

    if fault_tolerance < 0:
        raise ValueError("fault_tolerance must be nonnegative")
    if fault_tolerance == 0:
        return (frozenset(),)
    if fault_tolerance > len(node_ids):
        raise ValueError("fault_tolerance cannot exceed the committee size")
    return tuple(frozenset(combo) for combo in combinations(node_ids, fault_tolerance))


def consensus_given_fault_set(
    node_ids: tuple[str, ...],
    fault_set: frozenset,
    *,
    pre_prepare_matrix: MessageMatrix,
    prepare_matrix: MessageMatrix,
    commit_matrix: MessageMatrix,
    quorum: int,
    external_quorum: int,
) -> float:
    """Uniform-initiator expected reliability with the *single* fault set ``B`` fixed.

    The same ``B`` is excluded as both sender and successful participant in every phase.
    A faulty primary contributes 0 (deferred view-change). Quorum thresholds are the
    committee-level ``quorum`` / ``external_quorum`` (over the original ``n``), applied to
    the honest sub-committee.
    """

    honest = tuple(node_id for node_id in node_ids if node_id not in fault_set)
    contributions: list[float] = []
    for primary in node_ids:
        if primary in fault_set:
            contributions.append(0.0)
            continue
        pre_ready = {
            node_id: (1.0 if node_id == primary else _matrix_probability(pre_prepare_matrix, primary, node_id))
            for node_id in honest
        }
        prepared = _honest_cascade(honest, pre_ready, prepare_matrix, external_quorum)
        committed = _honest_cascade(honest, prepared, commit_matrix, external_quorum)
        contributions.append(
            heterogeneous_quorum_tail([committed[node_id] for node_id in honest], quorum)
        )
    return fsum(contributions) / len(node_ids)


def robust_consensus_reliability(
    node_ids: tuple[str, ...],
    *,
    pre_prepare_matrix: MessageMatrix,
    prepare_matrix: MessageMatrix,
    commit_matrix: MessageMatrix,
    fault_tolerance: int,
    quorum_mode: str = QUORUM_MODE_SAFE,
    reduction: str = REDUCTION_HARD_MIN,
    softmin_beta: float = DEFAULT_SOFTMIN_BETA,
    max_enumeration: int = DEFAULT_MAX_ENUMERATION,
) -> FaultSetRobustnessResult:
    """``C_robust = min_{|B| <= f} C(x; B)`` (or its softmin) by exact enumeration."""

    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    spec = PBFTQuorumSpec(node_count=len(node_ids), fault_tolerance=fault_tolerance, mode=quorum_mode)
    count = comb(len(node_ids), fault_tolerance)
    if count > max_enumeration:
        raise ValueError(
            "fault-set enumeration budget exceeded: "
            f"C({len(node_ids)},{fault_tolerance})={count} > max_enumeration={max_enumeration}; "
            "use a smaller f or a documented approximation"
        )

    fault_sets = enumerate_fault_sets(node_ids, fault_tolerance)
    values: list[float] = []
    worst_value = 2.0
    worst_set: frozenset = fault_sets[0]
    for fault_set in fault_sets:
        value = consensus_given_fault_set(
            node_ids,
            fault_set,
            pre_prepare_matrix=pre_prepare_matrix,
            prepare_matrix=prepare_matrix,
            commit_matrix=commit_matrix,
            quorum=spec.quorum,
            external_quorum=spec.external_quorum,
        )
        values.append(value)
        if value < worst_value:
            worst_value = value
            worst_set = fault_set

    if reduction == REDUCTION_HARD_MIN:
        result_value = worst_value
    else:
        result_value = _softmin(values, softmin_beta)

    return FaultSetRobustnessResult(
        model_id=FAULT_SET_ROBUSTNESS_MODEL_ID,
        consensus_success_probability=_clamp(result_value),
        worst_case_fault_set=tuple(sorted(worst_set)),
        fault_set_count=len(fault_sets),
        reduction=reduction,
        fault_tolerance=fault_tolerance,
        quorum=spec.quorum,
        external_quorum=spec.external_quorum,
    )


def _honest_cascade(
    honest: tuple[str, ...],
    sender_readiness: Mapping[str, float],
    message_matrix: MessageMatrix,
    external_quorum: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for receiver_id in honest:
        incoming = [
            sender_readiness[sender_id] * _matrix_probability(message_matrix, sender_id, receiver_id)
            for sender_id in honest
            if sender_id != receiver_id
        ]
        result[receiver_id] = sender_readiness[receiver_id] * heterogeneous_quorum_tail(
            incoming, external_quorum
        )
    return result


def _softmin(values: list[float], beta: float) -> float:
    if beta <= 0.0:
        raise ValueError("softmin_beta must be positive")
    minimum = min(values)
    # softmin_beta{C} = -1/beta * log sum_B exp(-beta C_B), stabilized around the min.
    shifted = fsum(exp(-beta * (value - minimum)) for value in values)
    return minimum - log(shifted) / beta


def _matrix_probability(matrix: MessageMatrix, sender_id: str, receiver_id: str) -> float:
    if sender_id == receiver_id:
        return 0.0
    return matrix.get((sender_id, receiver_id), 0.0)


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0 if value > -1e-12 else value
    if value > 1.0:
        return 1.0 if value < 1.0 + 1e-12 else value
    return value
