"""Q5 (POMDP-QP-FAR): local_hysteresis imitation actor.

Pins: (a) the proposals refactor is byte-identical to the deployable action and respects the budget;
(b) the hysteresis teacher trajectory IS the deployable anchor (recon == local_hysteresis_action, 0
evaluator calls by construction); (c) the topology-F1 helper; (d) BC toward the hysteresis proposals
reduces the teacher-subset NLL (the actor can learn to imitate). Fails on HEAD (the symbols are new).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "diagnostics"))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor
from marl_topology.training.decentralized_distillation import feature_standardization
from marl_topology.training.dynamic_baselines import (
    local_hysteresis_action, local_hysteresis_proposals)
from marl_topology.training.dynamic_frames import sample_dynamic_scenes
from marl_topology.training.dynamic_rl import _budgets_edges, _hysteresis_teacher_trajectory
from marl_topology.training.two_timescale_env import ReconfigCost

import anchor_imitation as ai  # noqa: E402

_KT, _AT = 0.4, 0.6


def _scenes(count=2, frames=4, nodes=(8,)):
    from build_operating_point_dataset import operating_point_regime
    return sample_dynamic_scenes(
        seed=11, count=count, node_count_choices=nodes, regime=operating_point_regime(20.0),
        num_frames=frames, dt_s=2.0, speed_min_mps=15.0, speed_max_mps=30.0,
        reconfig=ReconfigCost(), hold_interval=4, gamma=0.95)


def test_hysteresis_proposals_decode_to_action() -> None:
    obs = _scenes()[0].observation(0, [])
    budgets, edges = _budgets_edges(obs["context"])
    accept, mutual = local_hysteresis_proposals(
        obs["ef"], obs["edge_ids"], edges, budgets, [], keep_threshold=_KT, add_threshold=_AT)
    action = local_hysteresis_action(
        obs["ef"], obs["edge_ids"], edges, budgets, [], keep_threshold=_KT, add_threshold=_AT)
    assert set(mutual) == set(action)                        # refactor byte-identical
    for node, idxs in accept.items():
        assert len(idxs) <= int(budgets.get(node, 0))       # proposals respect the budget cap


def test_hysteresis_teacher_is_the_deployable_anchor() -> None:
    scene = _scenes(count=1)[0]
    traj = _hysteresis_teacher_trajectory(scene, keep_threshold=_KT, add_threshold=_AT)
    assert len(traj) == scene.n_frames
    prev: list = []
    for fr in traj:
        # the teacher's reconstructed topology IS the deployable anchor's action at the same state
        obs = scene.observation(fr["obs"]["time_index"], prev)
        budgets, edges = _budgets_edges(obs["context"])
        action = local_hysteresis_action(
            obs["ef"], obs["edge_ids"], edges, budgets, prev, keep_threshold=_KT, add_threshold=_AT)
        assert set(fr["recon"]) == set(action)
        # proposals are valid BCSP subsets (within budget) -- they decode to recon, 0 evaluator calls
        for (_node, inc, acc, b) in fr["proposals"]:
            assert len(acc) <= b and all(0 <= k < len(inc) for k in acc)
        prev = list(fr["recon"])


def test_f1_metric() -> None:
    assert ai.topology_f1(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert ai.topology_f1([], []) == 1.0
    assert ai.topology_f1(["a"], ["b"]) == 0.0
    assert ai.topology_f1(["a", "b", "c"], ["a", "b"]) == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))


def test_warmstart_reduces_teacher_nll() -> None:
    scenes = _scenes(count=2, frames=4)
    teachers = [_hysteresis_teacher_trajectory(s, keep_threshold=_KT, add_threshold=_AT) for s in scenes]
    stat = [s.observation(0, []) for s in scenes]
    mean, std = feature_standardization(stat)
    nd, ed = stat[0]["nf"].shape[1], stat[0]["ef"].shape[1]
    import torch
    torch.manual_seed(0)
    actor = DynamicRecurrentActor(nd, ed, hidden=32)
    init, final = ai.warmstart(actor, teachers, mean, std, epochs=40, lr=0.05, temp=1.0)
    assert final < init - 1e-3                               # BC toward the anchor reduces the NLL
