"""Quorum-completion, timeout-aware PBFT phase latency (Phase 4, Technical-Spec S4.10).

The legacy accounting charges ``min(max_all_pairs_latency, phase_budget)``: a single slow
link dominates, the value is topology-insensitive (it saturates near a constant), and a
FAILED topology pays the same small clipped latency as a success. The spec charges the time
to reach GLOBAL QUORUM, with a failed topology paying the full phase budget (a timeout).

For a phase with per-message arrival latency ``L_ij`` and deadline-delivery ``M_ij``, model
the by-time-``t`` delivery as a step ``M_ij`` once ``t >= L_ij``. Then::

    F_j(t)  = Q_qext({ M_ij if L_ij <= t else 0 }_{i != j})      # receiver j has quorum by t
    F_T(t)  = Q_qglobal({ F_j(t) }_j)                            # phase has global quorum by t
    E[min(T, B)] = integral_0^B (1 - F_T(t)) dt

``F_T`` is a right-continuous step function with jumps at the distinct arrival times, so the
integral is an exact finite sum over the breakpoints. A topology that reaches quorum fast
pays ~that time; one that never reaches it pays ~``B`` (the timeout). Also reports P50, P95,
CVaR and the timeout rate. Uses the same closed-form quorum tail as the reliability metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Mapping

from .quorum_tail import heterogeneous_quorum_tail


QUORUM_COMPLETION_LATENCY_MODEL_ID = "quorum_completion_timeout_aware_latency_v1"
MessageMap = Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class QuorumCompletionLatencyResult:
    """Timeout-aware quorum-completion latency for one PBFT phase."""

    model_id: str
    expected_s: float
    p50_s: float
    p95_s: float
    cvar_s: float
    timeout_rate: float
    phase_budget_s: float
    cvar_alpha: float

    def __post_init__(self) -> None:
        for name, value in (
            ("expected_s", self.expected_s),
            ("p50_s", self.p50_s),
            ("p95_s", self.p95_s),
            ("cvar_s", self.cvar_s),
            ("phase_budget_s", self.phase_budget_s),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if value > self.phase_budget_s + 1e-12:
                raise ValueError(f"{name} cannot exceed the phase budget")
        if not 0.0 <= self.timeout_rate <= 1.0:
            raise ValueError("timeout_rate must be in [0, 1]")


def quorum_completion_latency(
    node_ids: tuple[str, ...],
    *,
    arrival_latencies: MessageMap,
    deliveries: MessageMap,
    external_quorum: int,
    global_quorum: int,
    phase_budget_s: float,
    cvar_alpha: float = 0.95,
) -> QuorumCompletionLatencyResult:
    """``E[min(T, B)]`` and tail metrics for one PBFT phase (Spec S4.10)."""

    if not isfinite(phase_budget_s) or phase_budget_s <= 0.0:
        raise ValueError("phase_budget_s must be positive and finite")
    if external_quorum < 0 or global_quorum < 0:
        raise ValueError("quorum sizes must be nonnegative")
    if not 0.0 < cvar_alpha < 1.0:
        raise ValueError("cvar_alpha must be in (0, 1)")

    def reached_global_quorum_by(t: float) -> float:
        per_receiver: list[float] = []
        for receiver in node_ids:
            incoming = [
                (deliveries.get((sender, receiver), 0.0)
                 if arrival_latencies.get((sender, receiver), inf) <= t else 0.0)
                for sender in node_ids
                if sender != receiver
            ]
            per_receiver.append(heterogeneous_quorum_tail(incoming, external_quorum))
        return heterogeneous_quorum_tail(per_receiver, global_quorum)

    # Breakpoints: distinct arrival times within the budget (F_T is constant between them).
    breakpoints = sorted({lat for lat in arrival_latencies.values() if 0.0 < lat <= phase_budget_s})
    grid = sorted({0.0, *breakpoints, phase_budget_s})
    reached = {t: reached_global_quorum_by(t) for t in grid}

    # E[min(T, B)] = integral_0^B (1 - F_T(t)) dt, F_T constant on [grid[k], grid[k+1]).
    expected_s = 0.0
    for left, right in zip(grid, grid[1:]):
        expected_s += (1.0 - reached[left]) * (right - left)

    timeout_rate = max(0.0, 1.0 - reached[phase_budget_s])

    def percentile(p: float) -> float:
        for t in grid:
            if reached[t] >= p:
                return t
        return phase_budget_s

    p50_s = percentile(0.5)
    p95_s = percentile(0.95)

    # CVaR_alpha(min(T,B)) = VaR_alpha + 1/(1-alpha) * integral_{VaR}^{B} (1 - F_T(t)) dt.
    var_alpha = percentile(cvar_alpha)
    tail = 0.0
    for left, right in zip(grid, grid[1:]):
        lower = max(left, var_alpha)
        if right > lower:
            tail += (1.0 - reached[left]) * (right - lower)
    cvar_s = min(var_alpha + tail / (1.0 - cvar_alpha), phase_budget_s)

    return QuorumCompletionLatencyResult(
        model_id=QUORUM_COMPLETION_LATENCY_MODEL_ID,
        expected_s=min(expected_s, phase_budget_s),
        p50_s=p50_s,
        p95_s=p95_s,
        cvar_s=cvar_s,
        timeout_rate=timeout_rate,
        phase_budget_s=phase_budget_s,
        cvar_alpha=cvar_alpha,
    )
