"""R2b (v2 Engineering-Plan §R2 items 4-5, Spec §4.2/§4.10): latency-aware relay.

A relayed PBFT message that arrives after the phase deadline must NOT contribute delivery.
``_multi_hop_reach`` must maximize the delivery product over paths whose CUMULATIVE link
latency is within the phase budget -- not merely the delivery product (which lets a path whose
links each pass the per-link filter but SUM past the deadline still deliver). These tests pin
the constrained optimum directly (independent of the implementation's internal DP).
"""

from __future__ import annotations

import pytest

from marl_topology.protocol.message_matrix_adapter import _multi_hop_reach

NODES = ("A", "B", "C", "D")


def test_relay_delivery_only_is_unchanged_without_latency() -> None:
    # latency_matrix=None -> the legacy delivery-only relay, byte-identical.
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.8}
    assert _multi_hop_reach(("A", "B", "C"), one_hop, relay_hops=2) == pytest.approx(
        {("A", "B"): 0.9, ("B", "C"): 0.8, ("A", "C"): 0.72}
    )


def test_relay_keeps_path_within_budget() -> None:
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.8}
    lat = {("A", "B"): 0.4, ("B", "C"): 0.4}  # sum 0.8 <= budget 1.0
    reached = _multi_hop_reach(("A", "B", "C"), one_hop, relay_hops=2,
                               latency_matrix=lat, phase_budget_s=1.0)
    assert reached[("A", "C")] == pytest.approx(0.72)


def test_relay_rejects_path_exceeding_phase_budget() -> None:
    # each link <= budget (0.6 <= 1.0) but the 2-hop SUM 1.2 > 1.0 -> A->C must NOT be reachable.
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.8}
    lat = {("A", "B"): 0.6, ("B", "C"): 0.6}  # sum 1.2 > budget 1.0
    reached = _multi_hop_reach(("A", "B", "C"), one_hop, relay_hops=2,
                               latency_matrix=lat, phase_budget_s=1.0)
    assert ("A", "C") not in reached
    # the direct links survive (each within budget)
    assert reached[("A", "B")] == pytest.approx(0.9)
    assert reached[("B", "C")] == pytest.approx(0.8)


def test_relay_prefers_feasible_lower_delivery_path() -> None:
    # A->C via B: delivery 0.9*0.9=0.81 but latency 0.6+0.6=1.2 > budget (INFEASIBLE).
    # A->C via D: delivery 0.5*0.5=0.25 and latency 0.3+0.3=0.6 <= budget (FEASIBLE).
    # The latency-aware relay must return the FEASIBLE 0.25, not the infeasible 0.81 -- proving it
    # is a constrained optimum, not max-delivery with a post-hoc check on the best path.
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.9, ("A", "D"): 0.5, ("D", "C"): 0.5}
    lat = {("A", "B"): 0.6, ("B", "C"): 0.6, ("A", "D"): 0.3, ("D", "C"): 0.3}
    reached = _multi_hop_reach(NODES, one_hop, relay_hops=2,
                               latency_matrix=lat, phase_budget_s=1.0)
    assert reached[("A", "C")] == pytest.approx(0.25)


def test_relay_three_hop_chain_latency_budget() -> None:
    chain = {("A", "B"): 0.9, ("B", "C"): 0.8, ("C", "D"): 0.7}
    lat = {("A", "B"): 0.3, ("B", "C"): 0.3, ("C", "D"): 0.3}  # sum 0.9 <= 1.0
    ok = _multi_hop_reach(NODES, chain, relay_hops=3, latency_matrix=lat, phase_budget_s=1.0)
    assert ok[("A", "D")] == pytest.approx(0.9 * 0.8 * 0.7)
    # tighten the budget below the 3-hop sum -> A->D drops (B->C and C->D still reachable as 2-hop)
    tight = _multi_hop_reach(NODES, chain, relay_hops=3, latency_matrix=lat, phase_budget_s=0.8)
    assert ("A", "D") not in tight
