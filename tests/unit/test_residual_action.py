"""Q6 (POMDP-QP-FAR): the decentralized residual action space (anchor + residual edit).

Pins the structural invariants: zero residual == anchor (all modes), add-only never removes anchor edges
(even under the budget cap), remove-only never adds, swap preserves the budget, the decode is LOCAL (a
non-incident logit cannot change a node's accept), and every node respects its budget in every mode.
Fails on HEAD (the module is new).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

from marl_topology.training.dynamic_baselines import local_hysteresis_action
from marl_topology.training.dynamic_frames import sample_dynamic_scenes
from marl_topology.training.dynamic_rl import _budgets_edges
from marl_topology.training.residual_action import residual_decode
from marl_topology.training.two_timescale_env import ReconfigCost

_KT, _AT = 0.4, 0.6


def _setup(nodes=(12,)):
    from build_operating_point_dataset import operating_point_regime
    scene = sample_dynamic_scenes(
        seed=5, count=1, node_count_choices=nodes, regime=operating_point_regime(20.0),
        num_frames=1, dt_s=2.0, speed_min_mps=15.0, speed_max_mps=30.0,
        reconfig=ReconfigCost(), hold_interval=1, gamma=0.95)[0]
    obs = scene.observation(0, [])
    budgets, edges = _budgets_edges(obs["context"])
    anchor = local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, [],
                                     keep_threshold=_KT, add_threshold=_AT)
    return obs, anchor, budgets, edges


def _anchor_incident_count(anchor, edges):
    deg: dict = {}
    for eid in anchor:
        u, v = edges[eid]
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def test_residual_zero_equals_anchor() -> None:
    obs, anchor, _b, _e = _setup()
    zeros = [0.0] * len(obs["edge_ids"])
    for mode in ("add", "remove", "swap", "full"):
        _accept, topo = residual_decode(anchor, zeros, obs["edge_ids"], obs["context"], mode=mode)
        assert set(topo) == set(anchor), mode                # zero residual reproduces the anchor exactly


def test_add_only_never_removes_anchor_edges() -> None:
    obs, anchor, _b, _e = _setup()
    aset = set(anchor)
    z = [(-9.0 if e in aset else 9.0) for e in obs["edge_ids"]]   # try to drop anchor + add everything
    _accept, topo = residual_decode(anchor, z, obs["edge_ids"], obs["context"], mode="add")
    assert aset.issubset(set(topo))                          # no anchor edge removed (incl. under budget cap)


def test_remove_only_never_adds_edges() -> None:
    obs, anchor, _b, _e = _setup()
    aset = set(anchor)
    z = [(-9.0 if e in aset else 9.0) for e in obs["edge_ids"]]   # remove anchor, "add" ignored in remove
    _accept, topo = residual_decode(anchor, z, obs["edge_ids"], obs["context"], mode="remove")
    assert set(topo).issubset(aset)                          # final is a subset of the anchor (no new edge)


def test_swap_preserves_budget() -> None:
    obs, anchor, budgets, edges = _setup()
    aset = set(anchor)
    z = [(-1.0 if (e in aset and i % 2 == 0) else (3.0 if e not in aset else 1.0))
         for i, e in enumerate(obs["edge_ids"])]
    accept, _topo = residual_decode(anchor, z, obs["edge_ids"], obs["context"], mode="swap")
    anchor_deg = _anchor_incident_count(anchor, edges)
    for node, s in accept.items():
        assert len(s) <= anchor_deg.get(node, 0)             # swap never grows a node's proposal
        assert len(s) <= int(budgets.get(node, 0))           # ... and respects the budget


def test_residual_decoder_is_local() -> None:
    obs, anchor, _b, _e = _setup()
    eids = obs["edge_ids"]
    ends = {e.edge_id: (e.node_u, e.node_v) for e in obs["context"].graph.edges}
    fu, fv = ends[eids[0]]
    w = next(n for n in obs["context"].graph.node_ids if n not in (fu, fv))
    z1 = [0.5] * len(eids)
    z2 = list(z1); z2[0] = -10.0                             # change a logit for an edge NOT incident to w
    a1, _ = residual_decode(anchor, z1, eids, obs["context"], mode="full")
    a2, _ = residual_decode(anchor, z2, eids, obs["context"], mode="full")
    assert a1.get(w, set()) == a2.get(w, set())              # w's accept is unaffected -> decode is local


def test_residual_respects_budget_all_modes() -> None:
    obs, anchor, budgets, _e = _setup()
    z = [((i % 5) - 2.0) for i in range(len(obs["edge_ids"]))]
    for mode in ("add", "remove", "swap", "full"):
        accept, _topo = residual_decode(anchor, z, obs["edge_ids"], obs["context"], mode=mode)
        for node, s in accept.items():
            assert len(s) <= int(budgets.get(node, 0)), mode
