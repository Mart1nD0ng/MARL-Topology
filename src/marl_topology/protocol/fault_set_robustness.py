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
STRATEGY_EXACT = "exact"
STRATEGY_GREEDY = "greedy"
STRATEGY_AUTO = "auto"
STRATEGIES = (STRATEGY_EXACT, STRATEGY_GREEDY, STRATEGY_AUTO)
DEFAULT_MAX_ENUMERATION = 50000
DEFAULT_SOFTMIN_BETA = 50.0

MessageMatrix = Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class FaultSetRobustnessResult:
    """Worst-case (or soft-worst-case) consensus reliability over fixed fault sets."""

    model_id: str
    consensus_success_probability: float
    worst_case_fault_set: tuple[str, ...]
    per_primary_reliability: Mapping[str, float]
    fault_set_count: int
    reduction: str
    strategy: str
    enumeration_exact: bool
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
        if self.strategy not in (STRATEGY_EXACT, STRATEGY_GREEDY):
            raise ValueError("strategy must resolve to exact or greedy")


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
    """Reliability over *honest* initiators with the single fault set ``B`` fixed.

    The same ``B`` is excluded as both sender and successful participant in every phase.
    Under the deferred-view-change model a faulty primary is replaced by an honest one, so
    the metric is the mean of ``C_p(B)`` over the ``n - |B|`` honest primaries (NOT zeroed
    over all ``n`` -- zeroing would cap reliability at ``(n-f)/n`` and make ``tau >= 0.9``
    unreachable at small ``n``). With perfect links and ``B = {}`` this returns 1.0, matching
    the existing uniform-initiator cascade. Quorum thresholds are the committee-level
    ``quorum`` / ``external_quorum`` (over the original ``n``), applied to the honest set.
    """

    return _consensus_and_per_primary(
        node_ids,
        fault_set,
        pre_prepare_matrix=pre_prepare_matrix,
        prepare_matrix=prepare_matrix,
        commit_matrix=commit_matrix,
        quorum=quorum,
        external_quorum=external_quorum,
    )[0]


def _consensus_and_per_primary(
    node_ids: tuple[str, ...],
    fault_set: frozenset,
    *,
    pre_prepare_matrix: MessageMatrix,
    prepare_matrix: MessageMatrix,
    commit_matrix: MessageMatrix,
    quorum: int,
    external_quorum: int,
) -> tuple[float, dict[str, float]]:
    honest = tuple(node_id for node_id in node_ids if node_id not in fault_set)
    if not honest:
        return 0.0, {node_id: 0.0 for node_id in node_ids}
    # Faulty primaries report 0 in the per-primary diagnostic (they cannot be a successful
    # initiator); the scalar averages over the honest initiators only.
    per_primary: dict[str, float] = {node_id: 0.0 for node_id in fault_set if node_id in node_ids}
    honest_values: list[float] = []
    for primary in honest:
        pre_ready = {
            node_id: (1.0 if node_id == primary else _matrix_probability(pre_prepare_matrix, primary, node_id))
            for node_id in honest
        }
        prepared = _honest_cascade(honest, pre_ready, prepare_matrix, external_quorum)
        committed = _honest_cascade(honest, prepared, commit_matrix, external_quorum)
        value = heterogeneous_quorum_tail([committed[node_id] for node_id in honest], quorum)
        per_primary[primary] = value
        honest_values.append(value)
    scalar = fsum(honest_values) / len(honest)
    return scalar, per_primary


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
    strategy: str = STRATEGY_EXACT,
    max_enumeration: int = DEFAULT_MAX_ENUMERATION,
) -> FaultSetRobustnessResult:
    """``C_robust = min_{|B| <= f} C(x; B)`` (or its softmin).

    ``strategy``: ``exact`` enumerates all ``C(n, f)`` fault sets (raises if that exceeds
    ``max_enumeration``); ``greedy`` builds ``B`` one node at a time (each step adds the
    node whose inclusion most lowers ``C`` -- ``O(f * n)`` evaluations, an upper bound on
    the true min); ``auto`` is ``exact`` when ``C(n, f) <= max_enumeration`` else ``greedy``.
    """

    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    spec = PBFTQuorumSpec(node_count=len(node_ids), fault_tolerance=fault_tolerance, mode=quorum_mode)
    count = comb(len(node_ids), fault_tolerance)

    resolved = strategy
    if strategy == STRATEGY_AUTO:
        resolved = STRATEGY_EXACT if count <= max_enumeration else STRATEGY_GREEDY
    if resolved == STRATEGY_EXACT and count > max_enumeration:
        raise ValueError(
            "fault-set enumeration budget exceeded: "
            f"C({len(node_ids)},{fault_tolerance})={count} > max_enumeration={max_enumeration}; "
            "use strategy='auto'/'greedy' or a larger budget"
        )

    if resolved == STRATEGY_EXACT:
        worst_set, worst_pp, worst_value, soft_value, enumerated = _exact_worst_case(
            node_ids, fault_tolerance, pre_prepare_matrix, prepare_matrix, commit_matrix,
            spec, reduction, softmin_beta,
        )
    else:
        worst_set, worst_pp, worst_value, enumerated = _greedy_worst_case(
            node_ids, fault_tolerance, pre_prepare_matrix, prepare_matrix, commit_matrix, spec,
        )
        soft_value = worst_value  # softmin over a greedy trace is undefined; use the hard min

    result_value = worst_value if reduction == REDUCTION_HARD_MIN else soft_value

    return FaultSetRobustnessResult(
        model_id=FAULT_SET_ROBUSTNESS_MODEL_ID,
        consensus_success_probability=_clamp(result_value),
        worst_case_fault_set=tuple(sorted(worst_set)),
        per_primary_reliability=dict(worst_pp),
        fault_set_count=enumerated,
        reduction=reduction,
        strategy=resolved,
        enumeration_exact=(resolved == STRATEGY_EXACT),
        fault_tolerance=fault_tolerance,
        quorum=spec.quorum,
        external_quorum=spec.external_quorum,
    )


def _exact_worst_case(
    node_ids, fault_tolerance, pre_prepare_matrix, prepare_matrix, commit_matrix,
    spec, reduction, softmin_beta,
):
    fault_sets = enumerate_fault_sets(node_ids, fault_tolerance)
    values: list[float] = []
    worst_value = 2.0
    worst_set = fault_sets[0]
    worst_pp: dict[str, float] = {}
    for fault_set in fault_sets:
        value, per_primary = _consensus_and_per_primary(
            node_ids, fault_set,
            pre_prepare_matrix=pre_prepare_matrix, prepare_matrix=prepare_matrix,
            commit_matrix=commit_matrix, quorum=spec.quorum, external_quorum=spec.external_quorum,
        )
        values.append(value)
        if value < worst_value:
            worst_value, worst_set, worst_pp = value, fault_set, per_primary
    soft_value = _softmin(values, softmin_beta) if reduction == REDUCTION_SOFTMIN else worst_value
    return worst_set, worst_pp, worst_value, soft_value, len(fault_sets)


def _greedy_worst_case(
    node_ids, fault_tolerance, pre_prepare_matrix, prepare_matrix, commit_matrix, spec,
):
    """Build B one node at a time; each step adds the node that most lowers C (O(f*n))."""

    chosen: set = set()
    worst_pp: dict[str, float] = {}
    worst_value = 1.0
    evaluations = 0
    for _ in range(fault_tolerance):
        step_best = 2.0
        step_node = None
        step_pp: dict[str, float] = {}
        for candidate in node_ids:
            if candidate in chosen:
                continue
            value, per_primary = _consensus_and_per_primary(
                node_ids, frozenset(chosen | {candidate}),
                pre_prepare_matrix=pre_prepare_matrix, prepare_matrix=prepare_matrix,
                commit_matrix=commit_matrix, quorum=spec.quorum, external_quorum=spec.external_quorum,
            )
            evaluations += 1
            if value < step_best:
                step_best, step_node, step_pp = value, candidate, per_primary
        chosen.add(step_node)
        worst_pp, worst_value = step_pp, step_best
    if not chosen:  # f == 0
        worst_value, worst_pp = _consensus_and_per_primary(
            node_ids, frozenset(),
            pre_prepare_matrix=pre_prepare_matrix, prepare_matrix=prepare_matrix,
            commit_matrix=commit_matrix, quorum=spec.quorum, external_quorum=spec.external_quorum,
        )
        evaluations = 1
    return frozenset(chosen), worst_pp, worst_value, evaluations


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
