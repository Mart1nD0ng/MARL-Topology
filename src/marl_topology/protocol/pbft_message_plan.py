"""Phase-specific PBFT message plan + accounting (Phase 4, Technical-Spec S4.8-4.10).

The three PBFT phases carry DISTINCT message sets:

  - **pre-prepare**: the primary ``p`` broadcasts to the backups -- ``{p -> j : j in V_val, j != p}``.
  - **prepare**: every validator votes to every other validator -- ``{i -> j : i,j in V_val, i != j}``.
  - **commit**: same validator<->validator vote set.

Coverage-gated CLIENTS (non-validators) emit NO PBFT votes; they may only RELAY (forward) and incur
only the forwarding cost. The legacy accounting reused ONE all-pairs record set for all three phases
(``phase_records = {pre: records, prepare: records, commit: records}``), which over-counts protocol
energy and mis-attributes the pre-prepare round. This module is the corrected, phase-specific plan.

Energy (S4.9): protocol messages are summed PER-PHASE over the plan (each phase is a separate
transmission round); relay / policy-communication (control) / reconfiguration / view-change are
per-topology-DECISION costs counted ONCE (not multiplied by the three phases). Latency (S4.10): each
phase's completion latency is the timeout-aware quorum-completion time over THAT phase's plan maps
(a failed phase pays the full budget) -- delegated to ``quorum_completion_latency``.

Pure protocol-math (Torch-free), a verified primitive; wiring it into the production evaluator
(replacing the all-pairs-x3 reuse) is a separate opt-in activation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .quorum_completion_latency import QuorumCompletionLatencyResult, quorum_completion_latency

PBFT_MESSAGE_PLAN_MODEL_ID = "phase_specific_pbft_message_plan_v1"
PBFT_PHASES = ("pre_prepare", "prepare", "commit")
DirectedPair = tuple[str, str]
MessageMap = Mapping[DirectedPair, float]


@dataclass(frozen=True, slots=True)
class PBFTMessagePlan:
    """The distinct directed message set of each PBFT phase (Spec S4.8)."""

    model_id: str
    primary: str
    validators: tuple[str, ...]
    clients: tuple[str, ...]
    pre_prepare: frozenset[DirectedPair]
    prepare: frozenset[DirectedPair]
    commit: frozenset[DirectedPair]

    def phase_messages(self, phase: str) -> frozenset[DirectedPair]:
        if phase not in PBFT_PHASES:
            raise ValueError(f"unknown PBFT phase {phase!r}; must be one of {PBFT_PHASES}")
        return getattr(self, phase)

    def senders(self, phase: str) -> set[str]:
        return {s for s, _r in self.phase_messages(phase)}


def build_pbft_message_plan(
    validators: tuple[str, ...],
    primary: str,
    *,
    clients: tuple[str, ...] = (),
) -> PBFTMessagePlan:
    """Build the phase-specific message plan (Spec S4.8).

    ``validators`` vote; ``clients`` (non-validators) emit no votes. ``primary`` must be a validator.
    """
    val = tuple(validators)
    val_set = set(val)
    if len(val_set) != len(val):
        raise ValueError("validators must be unique")
    if primary not in val_set:
        raise ValueError(f"primary {primary!r} must be a validator")
    if val_set & set(clients):
        raise ValueError("validators and clients must be disjoint")

    pre_prepare = frozenset((primary, j) for j in val if j != primary)
    vote_pairs = frozenset((i, j) for i in val for j in val if i != j)
    return PBFTMessagePlan(
        model_id=PBFT_MESSAGE_PLAN_MODEL_ID,
        primary=primary,
        validators=val,
        clients=tuple(clients),
        pre_prepare=pre_prepare,
        prepare=vote_pairs,
        commit=vote_pairs,
    )


def pbft_protocol_energy(
    plan: PBFTMessagePlan,
    link_energy_j: MessageMap,
    *,
    relay_energy_j: float = 0.0,
    control_energy_j: float = 0.0,
    reconfig_energy_j: float = 0.0,
    view_change_energy_j: float = 0.0,
) -> dict[str, float]:
    """Total PBFT energy (Spec S4.9), phase-specific.

    ``protocol`` = sum over the three phases of each plan message's per-link transmission energy
    (a link used in multiple phases is counted once PER PHASE -- each phase is a transmission round).
    ``relay`` / ``control`` (policy-communication) / ``reconfig`` / ``view_change`` are per-topology-
    DECISION costs counted ONCE (not multiplied by the three phases). Returns the per-term breakdown
    and the total.
    """
    protocol = 0.0
    for phase in PBFT_PHASES:
        for pair in plan.phase_messages(phase):
            protocol += float(link_energy_j.get(pair, 0.0))
    once = relay_energy_j + control_energy_j + reconfig_energy_j + view_change_energy_j
    return {
        "protocol": protocol,
        "relay": relay_energy_j,
        "control": control_energy_j,
        "reconfig": reconfig_energy_j,
        "view_change": view_change_energy_j,
        "total": protocol + once,
    }


def phase_completion_latency(
    plan: PBFTMessagePlan,
    phase: str,
    *,
    deliveries: MessageMap,
    arrival_latencies: MessageMap,
    external_quorum: int,
    global_quorum: int,
    phase_budget_s: float,
    cvar_alpha: float = 0.95,
) -> QuorumCompletionLatencyResult:
    """Timeout-aware completion latency of one phase over THAT phase's plan messages (Spec S4.10).

    The delivery/latency maps are restricted to the phase's directed message set, so a failed phase
    (its plan messages never reach quorum) pays the full budget (a timeout). The committee for the
    quorum tail is the validator set (clients do not vote).
    """
    messages = plan.phase_messages(phase)
    deliv = {m: float(deliveries.get(m, 0.0)) for m in messages}
    lat = {m: float(arrival_latencies.get(m, phase_budget_s)) for m in messages}
    return quorum_completion_latency(
        plan.validators,
        arrival_latencies=lat,
        deliveries=deliv,
        external_quorum=external_quorum,
        global_quorum=global_quorum,
        phase_budget_s=phase_budget_s,
        cvar_alpha=cvar_alpha,
    )
