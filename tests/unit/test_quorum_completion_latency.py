"""Phase 4: quorum-completion, timeout-aware latency (Technical-Spec S4.10).

The legacy accounting uses ``min(max_all_pairs_latency, phase_budget)`` -- a single slow
link dominates, it is topology-insensitive (saturates ~constant), and a FAILED topology pays
the same small clipped latency as a success (no timeout). The spec wants the time to reach
GLOBAL QUORUM, with a failed topology paying the full phase budget::

    F_T(t) = Q_qglobal({ Q_qext({ M_ij if L_ij <= t else 0 }_i) }_j)     # reached-quorum-by-t
    E[min(T, B)] = integral_0^B (1 - F_T(t)) dt

So a topology that reaches quorum fast pays ~that time; one that never reaches it pays ~B.
"""

from __future__ import annotations

import pytest

from marl_topology.protocol.quorum_completion_latency import quorum_completion_latency


NODES = ("n0", "n1", "n2", "n3")
B = 0.01


def _uniform(nodes, latency, delivery):
    arr = {(i, j): latency for i in nodes for j in nodes if i != j}
    dlv = {(i, j): delivery for i in nodes for j in nodes if i != j}
    return arr, dlv


def _call(arr, dlv, *, ext=2, glob=3, budget=B, alpha=0.95):
    return quorum_completion_latency(
        NODES, arrival_latencies=arr, deliveries=dlv,
        external_quorum=ext, global_quorum=glob, phase_budget_s=budget, cvar_alpha=alpha,
    )


def test_failed_topology_pays_full_timeout() -> None:
    arr, dlv = _uniform(NODES, 0.001, 0.0)   # nothing ever delivers
    r = _call(arr, dlv)
    assert r.expected_s == pytest.approx(B, abs=1e-9)
    assert r.timeout_rate == pytest.approx(1.0)
    assert r.p50_s == pytest.approx(B)
    assert r.p95_s == pytest.approx(B)


def test_fast_reliable_topology_is_quick() -> None:
    arr, dlv = _uniform(NODES, 0.001, 1.0)   # everything delivered at t=0.001
    r = _call(arr, dlv)
    assert r.expected_s == pytest.approx(0.001, abs=1e-9)   # quorum reached at 0.001
    assert r.timeout_rate == pytest.approx(0.0, abs=1e-9)
    assert r.p50_s == pytest.approx(0.001)


def test_unreachable_does_not_get_zero_latency() -> None:
    # the legacy max-all-pairs would give a SMALL latency here; the fix charges ~B.
    arr, dlv = _uniform(NODES, 0.001, 0.1)   # 10% per-link -> quorum rarely reached
    r = _call(arr, dlv)
    assert r.expected_s > 0.5 * B            # not near zero
    assert r.timeout_rate > 0.5


def test_expected_equals_step_integral_hand_case() -> None:
    # 4 nodes, all links deliver perfectly but at staggered times; quorum (ext=2 per receiver,
    # glob=3) is reached once enough links have arrived. expected is the step integral.
    arr = {(i, j): 0.002 for i in NODES for j in NODES if i != j}
    dlv = {(i, j): 1.0 for i in NODES for j in NODES if i != j}
    r = _call(arr, dlv)
    # all arrive at 0.002 -> F_T jumps 0->1 at 0.002 -> E[min(T,B)] = 0.002.
    assert r.expected_s == pytest.approx(0.002, abs=1e-9)


def test_percentiles_ordered_and_cvar_geq_expected() -> None:
    arr, dlv = _uniform(NODES, 0.003, 0.6)
    r = _call(arr, dlv)
    assert 0.0 <= r.p50_s <= r.p95_s <= B + 1e-12
    assert r.cvar_s >= r.expected_s - 1e-12
    assert 0.0 <= r.timeout_rate <= 1.0


def test_faster_links_lower_latency() -> None:
    slow_arr, dlv = _uniform(NODES, 0.006, 1.0)
    fast_arr, _ = _uniform(NODES, 0.001, 1.0)
    assert _call(fast_arr, dlv).expected_s < _call(slow_arr, dlv).expected_s


def test_latency_is_topology_sensitive() -> None:
    # two profiles the legacy max-all-pairs would barely distinguish give different latencies.
    arr_a, dlv_a = _uniform(NODES, 0.002, 0.95)
    arr_b, dlv_b = _uniform(NODES, 0.002, 0.55)
    assert _call(arr_a, dlv_a).expected_s != pytest.approx(_call(arr_b, dlv_b).expected_s)
