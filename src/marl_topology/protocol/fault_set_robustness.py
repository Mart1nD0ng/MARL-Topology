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

The worst case is searched over **all** sizes ``0 <= |B| <= f``, NOT only ``|B| = f``.
Two distinct facts (do not conflate them): (i) as a function of the SET ``B``, ``C(x; B)``
is NOT monotone (Spec S4.7.1) -- adding a *weak* primary to ``B`` drops it from the honest-
average denominator and can RAISE ``C_honest(B)`` (e.g. ``C({}) = 0.847 < C({weak}) = 0.884``),
so the argmin set is not guaranteed to have size ``f``; that is why all sizes are enumerated.
(ii) Separately, the size-wise minimum ``min_{|B|=r} C`` happens to be non-increasing in ``r``
for this cascade (extend the size-``r-1`` argmin by the strongest remaining primary: it drops an
above-average term AND shrinks every quorum tail), so the *certified value* coincides with the
old ``|B|=f``-only value -- but the code does NOT rely on (ii); enumerating all sizes is correct
regardless. Exact enumeration is therefore ``sum_{r=0}^{f} C(n, r)`` fault sets; for
small ``f`` this is the certified reference (``is_certified`` is true only for an exact
hard-min). A ``softmin`` is available for training smoothness (not a certificate). A
budget guard fails loud when the enumeration is too large; ``greedy`` is then an
explicitly-OPTIMISTIC approximation (``C(B_greedy) >= min_B C(B)``), never certified.
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
    # certified == an EXACT hard-min over all |B| <= f (a true worst-case certificate).
    # greedy (optimistic approximation) and softmin (training surrogate) are NOT certified.
    is_certified: bool = False

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
    """The Byzantine fault sets of size *exactly* ``f`` (a utility).

    NOTE: the robust worst-case search does NOT use this alone -- it searches all sizes
    ``0..f`` via :func:`enumerate_fault_sets_up_to`, because under honest-primary averaging
    the minimum is not guaranteed to lie at ``|B| = f`` (Spec S4.7.1). ``f = 0`` yields the
    single empty set.
    """

    if fault_tolerance < 0:
        raise ValueError("fault_tolerance must be nonnegative")
    if fault_tolerance == 0:
        return (frozenset(),)
    if fault_tolerance > len(node_ids):
        raise ValueError("fault_tolerance cannot exceed the committee size")
    return tuple(frozenset(combo) for combo in combinations(node_ids, fault_tolerance))


def enumerate_fault_sets_up_to(node_ids: tuple[str, ...], fault_tolerance: int) -> tuple[frozenset, ...]:
    """ALL Byzantine fault sets with ``0 <= |B| <= f`` (the set the worst case ranges over).

    The robust reliability is ``min_{|B| <= f} C(x; B)``; under honest-primary averaging
    this minimum can occur at *any* size (Spec S4.7.1), so every size 0..f is enumerated.
    The empty set (``|B| = 0``, the no-fault case) is included.
    """

    if fault_tolerance < 0:
        raise ValueError("fault_tolerance must be nonnegative")
    if fault_tolerance > len(node_ids):
        raise ValueError("fault_tolerance cannot exceed the committee size")
    return tuple(
        frozenset(combo)
        for r in range(fault_tolerance + 1)
        for combo in combinations(node_ids, r)
    )


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

    The minimum ranges over ALL sizes ``0 <= |B| <= f`` (Spec S4.7.1: honest-primary
    averaging is not monotone in ``B``), so exact enumeration is ``sum_{r=0}^{f} C(n, r)``
    fault sets.

    ``strategy``: ``exact`` enumerates every ``|B| <= f`` fault set (raises if that exceeds
    ``max_enumeration``) and yields a CERTIFIED hard-min; ``greedy`` builds ``B`` one node
    at a time and tracks the running min over the path -- ``O(f * n)`` evaluations, an
    OPTIMISTIC approximation ``C(B_greedy) >= min_B C(B)`` that is NOT certified; ``auto`` is
    ``exact`` when the enumeration fits ``max_enumeration`` else ``greedy``.
    """

    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    spec = PBFTQuorumSpec(node_count=len(node_ids), fault_tolerance=fault_tolerance, mode=quorum_mode)
    count = sum(comb(len(node_ids), r) for r in range(fault_tolerance + 1))

    resolved = strategy
    if strategy == STRATEGY_AUTO:
        resolved = STRATEGY_EXACT if count <= max_enumeration else STRATEGY_GREEDY
    if resolved == STRATEGY_EXACT and count > max_enumeration:
        raise ValueError(
            "fault-set enumeration budget exceeded: "
            f"sum_r<={fault_tolerance} C({len(node_ids)},r)={count} > max_enumeration={max_enumeration}; "
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
        # certified iff an EXACT hard-min over all |B| <= f; greedy / softmin are not certificates.
        is_certified=(resolved == STRATEGY_EXACT and reduction == REDUCTION_HARD_MIN),
    )


def _exact_worst_case(
    node_ids, fault_tolerance, pre_prepare_matrix, prepare_matrix, commit_matrix,
    spec, reduction, softmin_beta,
):
    fault_sets = enumerate_fault_sets_up_to(node_ids, fault_tolerance)
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
    """Greedy path search over ``|B| <= f``: start from the empty set and at each step add the
    node that most lowers ``C``; track the running MIN over the whole path (every prefix size
    ``0..f``, since the honest-primary average is not monotone -- a prefix can be lower than the
    final set). This is an OPTIMISTIC approximation: ``C(B_greedy) >= min_{|B|<=f} C`` -- it is
    NOT a certificate (``is_certified`` is false). ``O(f*n)`` evaluations.
    """

    def consensus(fault_set):
        return _consensus_and_per_primary(
            node_ids, fault_set,
            pre_prepare_matrix=pre_prepare_matrix, prepare_matrix=prepare_matrix,
            commit_matrix=commit_matrix, quorum=spec.quorum, external_quorum=spec.external_quorum,
        )

    chosen: set = set()
    best_value, best_pp = consensus(frozenset())  # the |B| = 0 case is part of |B| <= f
    best_set: frozenset = frozenset()
    evaluations = 1
    for _ in range(fault_tolerance):
        step_best, step_node, step_pp = 2.0, None, {}
        for candidate in node_ids:
            if candidate in chosen:
                continue
            value, per_primary = consensus(frozenset(chosen | {candidate}))
            evaluations += 1
            if value < step_best:
                step_best, step_node, step_pp = value, candidate, per_primary
        if step_node is None:
            break
        chosen.add(step_node)
        if step_best < best_value:
            best_value, best_pp, best_set = step_best, step_pp, frozenset(chosen)
    return best_set, best_pp, best_value, evaluations


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
