"""Q10 (POMDP-QP-FAR): local edge handshake.

Pins: the shared edge score is symmetric; the decode uses only a node's incident edges (no global sort,
neighbour-only); the handshake recovers a critical edge that an INDEPENDENT directed decode loses to
two-end desync; and the control-message cost is recorded. Fails on HEAD (the module is new).
"""

from __future__ import annotations

import pytest

from marl_topology.training.edge_handshake import (
    correlated_sample_key, handshake_decode, mismatch_metrics, shared_edge_scores)

_EDGE_IDS = ["A--B", "A--C", "B--D"]
_EDGES = {"A--B": ("A", "B"), "A--C": ("A", "C"), "B--D": ("B", "D")}
_BUDGETS = {"A": 1, "B": 1, "C": 1, "D": 1}
_PSUCC = [0.9, 0.3, 0.9]


def test_shared_edge_score_symmetric() -> None:
    a = shared_edge_scores([0.0] * 3, _EDGE_IDS, _EDGES, _PSUCC, [], alpha=1.0, beta=0.5,
                           directed_scores={"A--B": (2.0, 0.0)})
    b = shared_edge_scores([0.0] * 3, _EDGE_IDS, _EDGES, _PSUCC, [], alpha=1.0, beta=0.5,
                           directed_scores={"A--B": (0.0, 2.0)})       # swapped direction
    assert a["A--B"] == pytest.approx(b["A--B"], abs=1e-12)             # the shared score is symmetric


def test_handshake_uses_only_neighbor_messages() -> None:
    z1 = [1.0, 0.5, 1.0]
    z2 = list(z1); z2[2] = -9.0                                         # change B--D (not incident to A or C)
    a1, _t1, _i1 = handshake_decode(z1, _EDGE_IDS, _EDGES, _BUDGETS, _PSUCC, [], alpha=0.0)
    a2, _t2, _i2 = handshake_decode(z2, _EDGE_IDS, _EDGES, _BUDGETS, _PSUCC, [], alpha=0.0)
    assert a1["A"] == a2["A"] and a1["C"] == a2["C"]                    # nodes not incident to B--D unchanged


def test_no_global_sort() -> None:
    z = [1.0, 1.1, 1.0]
    accept, _topo, info = handshake_decode(z, _EDGE_IDS, _EDGES, _BUDGETS, _PSUCC, [], alpha=0.0)
    shared = info["shared_scores"]
    # each node's accept is the top-b of ITS OWN incident edges by the shared score (no global selection)
    for node in ("A", "B", "C", "D"):
        incident = [eid for eid in _EDGE_IDS if node in _EDGES[eid]]
        ranked = sorted((i for i in range(len(_EDGE_IDS)) if _EDGE_IDS[i] in incident),
                        key=lambda i: (-shared[_EDGE_IDS[i]], _EDGE_IDS[i]))
        assert accept[node] == set(ranked[: _BUDGETS[node]])


def test_handshake_recovers_critical_edge_vs_independent_directed() -> None:
    # A strongly wants A--B (s=2.0) but B is indifferent (s=0.0). An INDEPENDENT directed decode (each end
    # ranks by its OWN directed score) loses A--B (B prefers B--D); the handshake's shared score (1.0)
    # makes B prefer A--B -> mutual. Spec §12.
    directed = {"A--B": (2.0, 0.0), "A--C": (0.5, 0.5), "B--D": (0.5, 0.5)}
    _acc, topo, _info = handshake_decode([0.0] * 3, _EDGE_IDS, _EDGES, _BUDGETS, _PSUCC, [],
                                         alpha=0.0, beta=0.0, directed_scores=directed)
    assert "A--B" in topo                                              # handshake recovers the critical edge

    # independent directed decode: each node ranks its incident edges by ITS OWN directed score
    def _own(node, eid):
        u, v = _EDGES[eid]
        return directed[eid][0] if node == u else directed[eid][1]
    ind_accept = {}
    for node in ("A", "B", "C", "D"):
        inc = [eid for eid in _EDGE_IDS if node in _EDGES[eid]]
        ind_accept[node] = set(sorted(inc, key=lambda e: -_own(node, e))[: _BUDGETS[node]])
    ind_topo = [eid for eid in _EDGE_IDS
                if eid in ind_accept[_EDGES[eid][0]] and eid in ind_accept[_EDGES[eid][1]]]
    assert "A--B" not in ind_topo                                      # the independent directed decode loses it


def test_control_message_cost_recorded() -> None:
    _acc, _topo, info = handshake_decode([1.0] * 3, _EDGE_IDS, _EDGES, _BUDGETS, _PSUCC, [], alpha=0.0)
    assert info["control_messages"] == 2 * len(_EDGE_IDS)              # one s_{i->j} per directed edge
    m = mismatch_metrics(_acc, _EDGE_IDS, _EDGES, _PSUCC)
    assert 0.0 <= m["one_sided_proposal_rate"] <= 1.0 and m["mutual_acceptance_rate"] >= 0.0


def test_correlated_sample_key_stable() -> None:
    assert correlated_sample_key("A--B", 3, 7) == correlated_sample_key("A--B", 3, 7)   # reproducible
    assert 0.0 <= correlated_sample_key("A--B", 3, 7) < 1.0
    assert correlated_sample_key("A--B", 3, 7) != correlated_sample_key("A--B", 4, 7)   # frame-sensitive
