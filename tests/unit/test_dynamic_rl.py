"""Dynamic (T>1) episode rollout + recurrent PPO mechanics (Spec S3.3-3.6).

Pins (invariant to the stochastic BCSP sampling): per-frame records == n_frames; the discounted
return recursion G_t = r_t + gamma*G_{t+1}; reward = base_objective - reconfig where reconfig =
(e_edge+l_edge)*|E_t triangle E_{t-1}| with no charge at t=0; the held eval metrics are in range; and
the recurrent re-roll produces BPTT gradients that reach the GRU (cross-frame recurrence is trained).
"""

from __future__ import annotations

import torch

from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    _sample_vehicle_motions,
    measure_reliable_range_m,
)
from marl_topology.geometry3d import Point3D
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor
from marl_topology.scenario.scene import Node3D, NodeKind, NodeMotion, Scene3D
from marl_topology.training.decentralized_distillation import feature_standardization
from marl_topology.training.dynamic_frames import dynamic_scene_from_motion
from marl_topology.training.dynamic_rl import _reroll_logp, dynamic_eval, episode_rollout
from marl_topology.training.two_timescale_env import ReconfigCost


def _scene(num_frames=4, e_edge=0.05):
    regime = PhysicsRegime()
    r = measure_reliable_range_m(regime)
    base = Scene3D(
        scenario_id="dynrl_ctrl",
        nodes=(
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(0.2 * r, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 0.2 * r, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(0.2 * r, 0.2 * r, 1.5)),
        ),
    )
    motions = (NodeMotion("veh_2", (0.4 * r, 0.0, 0.0)),)
    return dynamic_scene_from_motion(base, motions, regime, quorum_size=3, num_frames=num_frames,
                                     dt_s=1.0, reliable_range_m=r,
                                     reconfig=ReconfigCost(e_edge=e_edge, l_edge=0.0), gamma=0.9)


def _reward_fn(obs, edges, e_ref, lam_c, lam_b, beta, reward_mode):
    c = float(obs["context"].evaluator.evaluate(set(edges)).metrics["consensus_success_probability"])
    ok = c >= 0.9
    return (c - 0.9), max(0.0, 0.9 - c), 0.0, ok


def _ref_energy(obs):
    return 1.0


def _models(scene):
    s0 = scene.observation(0, [])
    mean, std = feature_standardization([s0])
    nd, ed = s0["nf"].shape[1], s0["ef"].shape[1]
    torch.manual_seed(0)
    actor = DynamicRecurrentActor(nd, ed, hidden=16)
    critic = CentralizedGraphCritic(nd, ed, hidden=16, rounds=2)
    return actor, critic, mean, std


def test_rollout_record_count_and_return_recursion() -> None:
    scene = _scene(num_frames=4)
    actor, critic, mean, std = _models(scene)
    recs, _eref = episode_rollout(actor, critic, scene, mean, std, recurrent=True, temp=1.0,
                                  reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                                  beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(1))
    assert len(recs) == scene.n_frames
    # discounted return recursion
    g = 0.0
    for rec in reversed(recs):
        g = rec.reward + scene.gamma * g
        assert abs(rec.ret - g) < 1e-6


def test_reward_decomposes_into_base_minus_reconfig() -> None:
    scene = _scene(num_frames=4, e_edge=0.05)
    actor, critic, mean, std = _models(scene)
    recs, _ = episode_rollout(actor, critic, scene, mean, std, recurrent=False, temp=1.0,
                              reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                              beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(2))
    prev = []
    for t, rec in enumerate(recs):
        switches = len(frozenset(prev) ^ frozenset(rec.topo)) if t > 0 else 0
        assert rec.switches == switches
        assert abs(rec.reconfig - 0.05 * switches) < 1e-9
        assert abs(rec.reward - (rec.base_reward - rec.reconfig)) < 1e-9
        if t == 0:
            assert rec.reconfig == 0.0          # no reconfiguration charged at the first frame
        prev = rec.topo


def test_recurrent_reroll_bptt_reaches_gru() -> None:
    scene = _scene(num_frames=4)
    actor, critic, mean, std = _models(scene)
    recs, _ = episode_rollout(actor, critic, scene, mean, std, recurrent=True, temp=1.0,
                              reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                              beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(3))
    lp, en, vp = _reroll_logp(actor, critic, scene, recs, mean, std, recurrent=True, temp=1.0)
    torch.stack(lp).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in actor.gru.parameters())


def test_dynamic_eval_metrics_in_range() -> None:
    scene = _scene(num_frames=4)
    actor, critic, mean, std = _models(scene)
    out = dynamic_eval(actor, [scene], mean, std, recurrent=True, temp=1.0, reward_of=_reward_fn,
                       ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")
    assert 0.0 <= out["per_frame_feasibility"] <= 1.0
    assert out["n_frames_total"] == scene.n_frames
    assert out["mean_switches_per_frame"] >= 0.0
    assert len(out["traces"]) == 1 and len(out["traces"][0]["frames"]) == scene.n_frames
