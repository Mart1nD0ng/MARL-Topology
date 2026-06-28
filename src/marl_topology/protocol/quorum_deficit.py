"""Q3 (POMDP-QP-FAR): quorum-shortfall feasibility distance ``D_quorum`` (Spec S8).

The final reliability ``C`` is a whole-network multi-phase quorum-tail conjunction: deep in the
infeasible region ``C ~ 0`` AND a single local edit barely moves it (a flat sub-feasible plateau). The
quorum DEFICIT measures, per phase ``h`` and receiver ``j``, how many votes the receiver is still
SHORT of its quorum, in expectation:

    S_{j,h} = sum_{i != j} Y_{ij}^h,   Y_{ij}^h ~ Bernoulli(p_{ij}^h)
    delta_{j,h} = E[(q_h - S_{j,h})_+] = sum_{k=0}^{q_h-1} (q_h - k) P(S_{j,h} = k)

This is a Poisson-binomial expectation, computed EXACTLY by the same DP the reliability uses -- the
tail bucket of :func:`poisson_binomial_low_pmf` is identically ``heterogeneous_quorum_tail`` (pinned by
a test), so ``D_quorum`` is the SAME math as ``C``, not a toy proxy. Unlike ``C``, the deficit has a
gradient in the plateau: connecting one more node toward its quorum reduces its ``delta`` even while the
whole-network product stays ~0.

Phases mirrored EXACTLY from :mod:`marl_topology.protocol.pbft_reliability`:
  * prepare: receiver j needs ``external_quorum`` of ``{alpha_1[i] * P_prepare(i,j)}_{i!=j}``
  * commit:  receiver j needs ``external_quorum`` of ``{alpha_2[i] * P_commit(i,j)}_{i!=j}``
  * global:  the network needs ``total_quorum`` of the filtered ``{alpha_3[j]}`` (== consensus tail)
with the SAME ``remove_largest`` fault filter and the SAME alpha cascade.

HARD INVARIANT (Spec S6 / Contract): ``D_quorum`` is a TRAINING-ONLY proxy. It is NOT the final metric,
and (Spec S8.3 / S7) it MUST pass the Q4 alignment test against the true ``C`` BEFORE it may enter any
reward. This module only COMPUTES the diagnostic; it wires into nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .pbft_reliability import (
    FAULT_FILTER_REMOVE_LARGEST,
    PBFTThreePhaseConfig,
    _fault_filtered_values,
    _matrix_probability,
    evaluate_pbft_three_phase_reliability,
)
from .quorum_tail import heterogeneous_quorum_tail


# -- pure Poisson-binomial primitives ---------------------------------------------------------------

def poisson_binomial_low_pmf(probabilities: Iterable[float], quorum_size: int) -> list[float]:
    """Return ``[P(S=0), ..., P(S=q-1), P(S>=q)]`` (length ``q+1``) for ``S = sum Bernoulli(p_i)``.

    Exact DP, O(n*q). The final bucket ``[q]`` is the quorum tail ``P(S>=q)`` and is IDENTICAL to
    :func:`heterogeneous_quorum_tail` (pinned by a test) -- the deficit shares the reliability's math.
    """
    q = _checked_quorum(quorum_size)
    values = _checked_probs(probabilities)
    if q == 0:
        return [1.0]                                   # everything is "at or above" a 0 quorum
    buckets = [0.0] * (q + 1)
    buckets[0] = 1.0
    for p in values:
        nf = 1.0 - p
        nxt = [0.0] * (q + 1)
        for k in range(q + 1):
            bk = buckets[k]
            if bk == 0.0:
                continue
            nxt[k] += bk * nf                          # this input fails
            if k < q:
                nxt[k + 1] += bk * p                   # this input succeeds
            else:
                nxt[q] += bk * p                       # already at/above quorum -> stays in the tail
        buckets = nxt
    return buckets


def expected_quorum_shortfall(probabilities: Iterable[float], quorum_size: int) -> float:
    """``delta = E[(q - S)_+] = sum_{k<q} (q-k) P(S=k)`` -- the expected number of votes short of quorum."""
    q = _checked_quorum(quorum_size)
    if q == 0:
        return 0.0
    pmf = poisson_binomial_low_pmf(probabilities, q)
    return math.fsum((q - k) * pmf[k] for k in range(q))


# -- aggregates -------------------------------------------------------------------------------------

def deficit_cvar(values: Iterable[float], alpha: float) -> float:
    """Upper-tail CVaR of a deficit sample = mean of the worst ``ceil((1-alpha)*n)`` deficits (the
    largest shortfalls). Matches the project's ``tail_mean_shortfall`` convention (Spec S6.3). Empty
    -> 0.0; ``alpha`` in [0, 1)."""
    xs = sorted((float(v) for v in values), reverse=True)
    if not xs:
        return 0.0
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    k = max(1, math.ceil(round((1.0 - alpha) * len(xs), 9)))   # round guards float ceil (3.0000..04 -> 3)
    return math.fsum(xs[:k]) / k


@dataclass(frozen=True, slots=True)
class QuorumDeficitSummary:
    """Aggregate quorum-deficit diagnostic for one topology (Spec S8.2)."""
    d_mean: float
    d_max: float
    d_cvar: float
    worst_phase: str
    worst_receiver: str
    per_phase_receiver: Mapping[tuple[str, str], float]
    global_deficit: float
    cvar_alpha: float

    def as_dict(self) -> dict:
        return {
            "d_quorum_mean": round(self.d_mean, 6), "d_quorum_max": round(self.d_max, 6),
            "d_quorum_cvar": round(self.d_cvar, 6), "worst_phase": self.worst_phase,
            "worst_receiver": self.worst_receiver, "global_deficit": round(self.global_deficit, 6),
            "cvar_alpha": self.cvar_alpha, "n_terms": len(self.per_phase_receiver),
        }


def aggregate_deficits(per_phase_receiver: Mapping[tuple[str, str], float], *,
                       global_deficit: float, cvar_alpha: float = 0.9) -> QuorumDeficitSummary:
    """Reduce ``{(phase, receiver): delta}`` (+ the global deficit) to mean / max / CVaR + the worst term."""
    items = dict(per_phase_receiver)
    items[("global", "*")] = float(global_deficit)
    vals = list(items.values())
    worst_key = max(items, key=lambda k: items[k]) if items else ("none", "")
    return QuorumDeficitSummary(
        d_mean=(math.fsum(vals) / len(vals)) if vals else 0.0,
        d_max=(max(vals) if vals else 0.0),
        d_cvar=deficit_cvar(vals, cvar_alpha),
        worst_phase=worst_key[0], worst_receiver=worst_key[1],
        per_phase_receiver=items, global_deficit=float(global_deficit), cvar_alpha=cvar_alpha,
    )


# -- per-primary / expected-initiator deficits (mirror the reliability cascade) ---------------------

def per_primary_quorum_deficit(
    node_ids: tuple[str, ...], primary_id: str, fault_tolerance: int, *,
    pre_prepare_matrix: Mapping[tuple[str, str], float],
    prepare_matrix: Mapping[tuple[str, str], float],
    commit_matrix: Mapping[tuple[str, str], float],
    fault_filter_mode: str = FAULT_FILTER_REMOVE_LARGEST,
    cvar_alpha: float = 0.9,
) -> QuorumDeficitSummary:
    """Per-(phase, receiver) quorum deficit for a GIVEN primary, mirroring
    :func:`evaluate_pbft_three_phase_reliability` EXACTLY (same alpha cascade, same fault filter, same
    quorum sizes). prepare/commit use ``external_quorum``; the global uses ``total_quorum``."""
    config = PBFTThreePhaseConfig(node_ids=node_ids, primary_id=primary_id,
                                  fault_tolerance=fault_tolerance, fault_filter_mode=fault_filter_mode)
    record = evaluate_pbft_three_phase_reliability(
        config, pre_prepare_matrix=pre_prepare_matrix,
        prepare_matrix=prepare_matrix, commit_matrix=commit_matrix)
    alpha_1 = record.pre_prepare_readiness          # sender readiness into PREPARE
    alpha_2 = record.prepared_probability           # sender readiness into COMMIT
    alpha_3 = record.committed_probability          # inputs to the global quorum
    eq = config.external_quorum
    deficits: dict[tuple[str, str], float] = {}
    for receiver in node_ids:
        prep_in = _fault_filtered_values(
            tuple(alpha_1[s] * _matrix_probability(prepare_matrix, s, receiver)
                  for s in node_ids if s != receiver), config)
        deficits[("prepare", receiver)] = expected_quorum_shortfall(prep_in, eq)
        commit_in = _fault_filtered_values(
            tuple(alpha_2[s] * _matrix_probability(commit_matrix, s, receiver)
                  for s in node_ids if s != receiver), config)
        deficits[("commit", receiver)] = expected_quorum_shortfall(commit_in, eq)
    global_in = _fault_filtered_values(tuple(alpha_3.values()), config)
    global_deficit = expected_quorum_shortfall(global_in, config.total_quorum)
    return aggregate_deficits(deficits, global_deficit=global_deficit, cvar_alpha=cvar_alpha)


def expected_initiator_quorum_deficit(
    node_ids: tuple[str, ...], fault_tolerance: int, *,
    pre_prepare_matrix: Mapping[tuple[str, str], float],
    prepare_matrix: Mapping[tuple[str, str], float],
    commit_matrix: Mapping[tuple[str, str], float],
    fault_filter_mode: str = FAULT_FILTER_REMOVE_LARGEST,
    cvar_alpha: float = 0.9,
) -> dict:
    """Topology-level ``D_quorum``: the uniform-initiator average of the per-primary deficits (mirrors
    :func:`evaluate_expected_initiator_pbft_reliability`, which averages ``C`` over uniform primaries).
    Returns the mean/max/CVaR aggregates + the worst (phase, receiver, primary)."""
    n = len(node_ids)
    per_primary = [
        per_primary_quorum_deficit(
            node_ids, primary, fault_tolerance, pre_prepare_matrix=pre_prepare_matrix,
            prepare_matrix=prepare_matrix, commit_matrix=commit_matrix,
            fault_filter_mode=fault_filter_mode, cvar_alpha=cvar_alpha)
        for primary in node_ids
    ]
    worst = max(per_primary, key=lambda s: s.d_max)
    return {
        "d_quorum_mean": round(math.fsum(s.d_mean for s in per_primary) / max(1, n), 6),
        "d_quorum_max": round(max((s.d_max for s in per_primary), default=0.0), 6),
        "d_quorum_cvar": round(math.fsum(s.d_cvar for s in per_primary) / max(1, n), 6),
        "global_deficit_mean": round(math.fsum(s.global_deficit for s in per_primary) / max(1, n), 6),
        "worst_phase": worst.worst_phase, "worst_receiver": worst.worst_receiver,
        "n_primaries": n, "cvar_alpha": cvar_alpha,
    }


# -- checks -----------------------------------------------------------------------------------------

def _checked_quorum(q: int) -> int:
    if isinstance(q, bool) or not isinstance(q, int):
        raise TypeError("quorum_size must be an integer")
    if q < 0:
        raise ValueError("quorum_size must be nonnegative")
    return q


def _checked_probs(probabilities: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(v) for v in probabilities)
    for v in values:
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError("probabilities must be finite and in [0, 1]")
    return values


# Re-export for the consistency test (the deficit tail bucket == the reliability quorum tail).
__all__ = [
    "poisson_binomial_low_pmf", "expected_quorum_shortfall", "deficit_cvar", "aggregate_deficits",
    "QuorumDeficitSummary", "per_primary_quorum_deficit", "expected_initiator_quorum_deficit",
    "heterogeneous_quorum_tail",
]
