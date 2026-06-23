"""Phase 2: single, correct route/relay semantics (Technical-Spec S4.2).

The spec requires ONE multi-hop layer: a true one-hop (direct-link) delivery matrix, then
a relay DP of at most ``relay_hops`` links (max-product). The canonical regression::

    A -- B -- C
    relay_hops = 1:  P(A -> C) = 0   (no DIRECT link)
    relay_hops = 2:  P(A -> C) > 0   (one relay hop through B)

The legacy primitive path double-counts: the evaluator routes every ordered pair via a BFS
shortest path, so ``record.network_delivery_probability`` is ALREADY an end-to-end
multi-hop probability; ``_multi_hop_reach`` then relays it AGAIN at ``relay_hops > 1``
(double counting, hard-constraint #10). R2 (v2 Spec S4.2): the corrected one-hop-relay
semantics are now the PRODUCTION DEFAULT (``PhysicsRegime`` / ``Stage21ObjectiveStackConfig``
default ``one_hop_relay=True, relay_hops=2``, measured feasibility-neutral). The legacy
double-count remains available only as an explicit opt-in (``one_hop_relay=False``) for
byte-reproducing pre-R2 datasets. These tests pin the relay DP, the corrected default, and
the explicit-legacy primitive behaviour.
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


def test_legacy_double_count_is_opt_in_only() -> None:
    # The legacy double-count (the BFS-route record's end-to-end prob, relayed again) is reachable
    # ONLY by explicitly requesting one_hop_relay=False -- it is no longer any production default.
    m = build_pbft_message_matrices_from_network_records(
        ("A", "B", "C"), {"pre_prepare": _abc_records(), "prepare": _abc_records(), "commit": _abc_records()},
        _BUDGETS, relay_hops=1, one_hop_relay=False,
    )
    assert m.pre_prepare_matrix[("A", "C")] == pytest.approx(0.72)  # the route record's multi-hop value


def test_production_config_defaults_to_corrected_relay() -> None:
    # R2: the corrected single-relay-layer semantics are the PRODUCTION DEFAULT, not opt-in.
    from marl_topology.data.stage21_objective_stack_evidence import Stage21ObjectiveStackConfig
    from marl_topology.data.stage31_scenario_generator import PhysicsRegime

    regime = PhysicsRegime()
    assert regime.one_hop_relay is True
    assert regime.relay_hops >= 2  # paired so reachability does not collapse

    # the objective-stack config default likewise carries the corrected pair
    fields = {f.name: f.default for f in __import__("dataclasses").fields(Stage21ObjectiveStackConfig)}
    assert fields["one_hop_relay"] is True
    assert fields["relay_hops"] >= 2
