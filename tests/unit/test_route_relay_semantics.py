"""Phase 2: single, correct route/relay semantics (Technical-Spec S4.2).

The spec requires ONE multi-hop layer: a true one-hop (direct-link) delivery matrix, then
a relay DP of at most ``relay_hops`` links (max-product). The canonical regression::

    A -- B -- C
    relay_hops = 1:  P(A -> C) = 0   (no DIRECT link)
    relay_hops = 2:  P(A -> C) > 0   (one relay hop through B)

The production path violates this: the evaluator routes every ordered pair via a BFS
shortest path, so ``record.network_delivery_probability`` is ALREADY an end-to-end
multi-hop probability; ``_multi_hop_reach`` then relays it AGAIN at ``relay_hops > 1``
(double counting, hard-constraint #10). These tests pin the correct relay DP, the known
default-mode bug (xfail, deferred to the recalibration), and the opt-in one-hop fix.
"""

from __future__ import annotations

import pytest

from marl_topology.network import NetworkCommunicationRecord
from marl_topology.protocol import PBFTPhaseBudgets, build_pbft_message_matrices_from_network_records
from marl_topology.protocol.message_matrix_adapter import _multi_hop_reach


# --- the relay DP is correct GIVEN a one-hop matrix (passes today) ---

def test_relay_dp_one_hop_only_at_relay_hops_1() -> None:
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.8}
    reached = _multi_hop_reach(("A", "B", "C"), one_hop, relay_hops=1)
    assert reached == one_hop                      # direct links only
    assert ("A", "C") not in reached               # P(A->C) = 0


def test_relay_dp_two_hops_reaches_c() -> None:
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.8}
    reached = _multi_hop_reach(("A", "B", "C"), one_hop, relay_hops=2)
    assert reached[("A", "C")] == pytest.approx(0.9 * 0.8)   # P(A->C) > 0 via B


def test_relay_dp_needs_enough_hops_for_a_chain() -> None:
    chain = {("A", "B"): 0.9, ("B", "C"): 0.8, ("C", "D"): 0.7}
    assert ("A", "D") not in _multi_hop_reach(("A", "B", "C", "D"), chain, relay_hops=2)
    reached3 = _multi_hop_reach(("A", "B", "C", "D"), chain, relay_hops=3)
    assert reached3[("A", "D")] == pytest.approx(0.9 * 0.8 * 0.7)


def test_relay_dp_takes_max_product_over_paths() -> None:
    # A->C via B (0.9*0.8=0.72) and via D (0.5*0.5=0.25): keep the most reliable.
    one_hop = {("A", "B"): 0.9, ("B", "C"): 0.8, ("A", "D"): 0.5, ("D", "C"): 0.5}
    reached = _multi_hop_reach(("A", "B", "C", "D"), one_hop, relay_hops=2)
    assert reached[("A", "C")] == pytest.approx(0.72)


def test_relay_dp_disconnected_never_reaches() -> None:
    one_hop = {("A", "B"): 0.9}     # C isolated
    for hops in (1, 2, 3):
        assert ("A", "C") not in _multi_hop_reach(("A", "B", "C"), one_hop, relay_hops=hops)


# --- record helpers for the A-B-C topology (direct A-B, direct B-C, ROUTED A-C) ---

def _route_record(source, target, prob, latency, route_nodes) -> NetworkCommunicationRecord:
    return NetworkCommunicationRecord(
        scenario_id="phase2_abc",
        network_model_id="stage3_network_communication_v1",
        primitive="route",
        source_id=source,
        target_ids=(target,),
        selected_edge_ids=(),
        active_transmission_ids=(),
        interference_group_ids=(),
        reachable_node_ids=tuple(route_nodes),
        route_node_ids=tuple(route_nodes),
        route_edge_ids=(),
        hop_records=(),
        network_delivery_probability=prob,
        network_latency_s=latency,
        network_scheduled_latency_s=latency,
        network_successful_delivery_latency_s=latency,
        network_energy_j=0.0,
        hop_count=0,  # synthetic record carries no hop_records; the direct/multi-hop
        is_full_graph_baseline=False,  # distinction is via route_node_ids length (Spec S4.2).
    )


def _abc_records():
    # direct A-B and B-C (route length 2); A-C is a BFS ROUTE through B (route length 3).
    return (
        _route_record("A", "B", 0.9, 0.001, ("A", "B")),
        _route_record("B", "A", 0.9, 0.001, ("B", "A")),
        _route_record("B", "C", 0.8, 0.001, ("B", "C")),
        _route_record("C", "B", 0.8, 0.001, ("C", "B")),
        _route_record("A", "C", 0.72, 0.002, ("A", "B", "C")),  # multi-hop route record
        _route_record("C", "A", 0.72, 0.002, ("C", "B", "A")),
    )


_BUDGETS = PBFTPhaseBudgets(pre_prepare_budget_s=1.0, prepare_budget_s=1.0, commit_budget_s=1.0)


@pytest.mark.xfail(reason="Phase 2: production feeds multi-hop route records, so the default "
                          "mode double-counts; the one-hop fix is opt-in and its activation is "
                          "deferred to the scenario recalibration.", strict=True)
def test_default_mode_violates_abc_regression_known_bug() -> None:
    matrices = build_pbft_message_matrices_from_network_records(
        ("A", "B", "C"), {"pre_prepare": _abc_records(), "prepare": _abc_records(), "commit": _abc_records()},
        _BUDGETS, relay_hops=1,
    )
    # Spec S4.2: with relay_hops=1 there is no DIRECT A-C link, so P(A->C) must be 0.
    assert ("A", "C") not in matrices.pre_prepare_matrix


def test_one_hop_relay_satisfies_abc_regression() -> None:
    # relay_hops=1 -> direct only -> no A-C.
    m1 = build_pbft_message_matrices_from_network_records(
        ("A", "B", "C"), {"pre_prepare": _abc_records(), "prepare": _abc_records(), "commit": _abc_records()},
        _BUDGETS, relay_hops=1, one_hop_relay=True,
    )
    assert ("A", "C") not in m1.pre_prepare_matrix
    assert m1.pre_prepare_matrix[("A", "B")] == pytest.approx(0.9)
    # relay_hops=2 -> exactly one relay through B -> A-C = 0.9*0.8 (NOT double-counted).
    m2 = build_pbft_message_matrices_from_network_records(
        ("A", "B", "C"), {"pre_prepare": _abc_records(), "prepare": _abc_records(), "commit": _abc_records()},
        _BUDGETS, relay_hops=2, one_hop_relay=True,
    )
    assert m2.pre_prepare_matrix[("A", "C")] == pytest.approx(0.9 * 0.8)


def test_default_mode_is_byte_unchanged() -> None:
    # Production default (one_hop_relay=False) keeps the current behavior exactly.
    m = build_pbft_message_matrices_from_network_records(
        ("A", "B", "C"), {"pre_prepare": _abc_records(), "prepare": _abc_records(), "commit": _abc_records()},
        _BUDGETS, relay_hops=1,
    )
    assert m.pre_prepare_matrix[("A", "C")] == pytest.approx(0.72)  # the route record's multi-hop value
