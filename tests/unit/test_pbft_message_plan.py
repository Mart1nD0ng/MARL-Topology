"""R4 (v2 Engineering-Plan §R4, Spec §4.8-4.10): phase-specific PBFT message plan + accounting.

The three PBFT phases must use DISTINCT message sets: pre-prepare = primary->backups; prepare/
commit = validator<->validator votes; coverage-gated clients emit NO votes (relay only). A failed
phase pays a timeout (§4.10); per-decision costs (control/reconfig/view-change) are counted ONCE,
protocol messages per-phase (§4.9).
"""

from __future__ import annotations

import pytest

from marl_topology.protocol.pbft_message_plan import (
    PBFT_PHASES,
    build_pbft_message_plan,
    phase_completion_latency,
    pbft_protocol_energy,
)

VALIDATORS = ("v0", "v1", "v2", "v3")
CLIENTS = ("c0", "c1")


def test_pre_prepare_has_primary_to_backup_messages_only() -> None:
    plan = build_pbft_message_plan(VALIDATORS, primary="v0", clients=CLIENTS)
    assert plan.pre_prepare == frozenset(("v0", j) for j in VALIDATORS if j != "v0")
    assert plan.senders("pre_prepare") == {"v0"}  # only the primary broadcasts the pre-prepare


def test_clients_do_not_emit_pbft_votes() -> None:
    plan = build_pbft_message_plan(VALIDATORS, primary="v0", clients=CLIENTS)
    for phase in PBFT_PHASES:
        senders = plan.senders(phase)
        receivers = {r for _s, r in plan.phase_messages(phase)}
        assert not (senders & set(CLIENTS))      # a client never SENDS a vote
        assert not (receivers & set(CLIENTS))    # and never RECEIVES one (it is not in the committee)


def test_prepare_commit_message_plans_are_phase_specific() -> None:
    plan = build_pbft_message_plan(VALIDATORS, primary="v0", clients=CLIENTS)
    val_all_pairs = frozenset((i, j) for i in VALIDATORS for j in VALIDATORS if i != j)
    assert plan.prepare == val_all_pairs
    assert plan.commit == val_all_pairs
    assert plan.pre_prepare != plan.prepare              # phase-specific, not the same set 3x
    assert plan.pre_prepare < plan.prepare               # pre-prepare is a strict subset (primary row)


def test_failed_phase_pays_timeout() -> None:
    plan = build_pbft_message_plan(VALIDATORS, primary="v0", clients=CLIENTS)
    budget = 0.01
    # a prepare phase whose plan messages NEVER deliver -> never reaches quorum -> pays the timeout
    dead = phase_completion_latency(
        plan, "prepare",
        deliveries={m: 0.0 for m in plan.prepare},
        arrival_latencies={m: budget * 0.5 for m in plan.prepare},
        external_quorum=3, global_quorum=3, phase_budget_s=budget,
    )
    assert dead.timeout_rate == pytest.approx(1.0)
    assert dead.expected_s == pytest.approx(budget)
    # a delivering prepare phase reaches quorum quickly -> pays much less than the timeout
    live = phase_completion_latency(
        plan, "prepare",
        deliveries={m: 1.0 for m in plan.prepare},
        arrival_latencies={m: budget * 0.1 for m in plan.prepare},
        external_quorum=3, global_quorum=3, phase_budget_s=budget,
    )
    assert live.expected_s < dead.expected_s
    assert live.timeout_rate == pytest.approx(0.0)


def test_control_and_relay_energy_are_counted_once() -> None:
    plan = build_pbft_message_plan(VALIDATORS, primary="v0", clients=CLIENTS)
    # per-message protocol energy = 1.0 J on every directed link
    link_energy = {(i, j): 1.0 for i in VALIDATORS + CLIENTS for j in VALIDATORS + CLIENTS if i != j}
    e = pbft_protocol_energy(
        plan, link_energy,
        relay_energy_j=5.0, control_energy_j=7.0, reconfig_energy_j=3.0, view_change_energy_j=2.0,
    )
    # protocol is summed PER-PHASE over the plan (pre_prepare + prepare + commit message counts)
    expected_protocol = len(plan.pre_prepare) + len(plan.prepare) + len(plan.commit)
    assert e["protocol"] == pytest.approx(expected_protocol)
    # control/relay/reconfig/view-change are per-DECISION costs -> counted ONCE, NOT x3 phases
    assert e["relay"] == pytest.approx(5.0)
    assert e["control"] == pytest.approx(7.0)
    assert e["total"] == pytest.approx(expected_protocol + 5.0 + 7.0 + 3.0 + 2.0)


def test_primary_must_be_a_validator() -> None:
    with pytest.raises(ValueError):
        build_pbft_message_plan(VALIDATORS, primary="c0", clients=CLIENTS)  # a client cannot be primary
