"""Multi-hop relaying in the PBFT message matrices (Direction 1 enabler). With relay_hops>1
a validator's message reaches peers it has no direct link to, via the most-reliable relay
path (max-product). relay_hops=1 is byte-identical to the original single-hop matrix.
"""

from marl_topology.protocol.message_matrix_adapter import (
    _multi_hop_reach,
    build_pbft_message_matrices_from_network_records,
)


def test_relay_hops_one_is_identity() -> None:
    direct = {("a", "b"): 0.9, ("b", "c"): 0.8}
    assert _multi_hop_reach(("a", "b", "c"), direct, 1) == direct


def test_two_hop_relay_adds_indirect_reach() -> None:
    # a--b--c line, no direct a->c. relay_hops=2 connects a->c via b (max-product).
    direct = {("a", "b"): 0.9, ("b", "c"): 0.8}
    reach = _multi_hop_reach(("a", "b", "c"), direct, 2)
    assert ("a", "c") not in direct
    assert abs(reach[("a", "c")] - 0.72) < 1e-9  # 0.9 * 0.8
    # direct edges are preserved.
    assert reach[("a", "b")] == 0.9


def test_relay_picks_most_reliable_path() -> None:
    # two paths a->c: a-b-c (0.9*0.9=0.81) vs a-d-c (0.5*0.5=0.25). pick the best.
    direct = {("a", "b"): 0.9, ("b", "c"): 0.9, ("a", "d"): 0.5, ("d", "c"): 0.5}
    reach = _multi_hop_reach(("a", "b", "c", "d"), direct, 2)
    assert abs(reach[("a", "c")] - 0.81) < 1e-9


def test_hop_cap_limits_path_length() -> None:
    # a-b-c-d chain. relay_hops=2 cannot reach a->d (needs 3 hops); relay_hops=3 can.
    direct = {("a", "b"): 0.9, ("b", "c"): 0.9, ("c", "d"): 0.9}
    nodes = ("a", "b", "c", "d")
    assert ("a", "d") not in _multi_hop_reach(nodes, direct, 2)
    reach3 = _multi_hop_reach(nodes, direct, 3)
    assert abs(reach3[("a", "d")] - 0.729) < 1e-9  # 0.9^3


def test_build_matrices_relay_hops_defaults_to_single_hop() -> None:
    # the public builder defaults relay_hops=1 (unchanged); a bad value is rejected.
    import pytest

    with pytest.raises(ValueError):
        build_pbft_message_matrices_from_network_records(("a", "b"), {}, _budgets(), relay_hops=0)


def _budgets():
    from marl_topology.protocol.message_matrix_adapter import PBFTPhaseBudgets

    return PBFTPhaseBudgets(0.003, 0.003, 0.003)
