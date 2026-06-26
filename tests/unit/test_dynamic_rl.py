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
from marl_topology.training.dynamic_rl import (
    _frame_teacher_trajectory,
    _reroll_logp,
    dynamic_eval,
    episode_rollout,
    warmstart_actor,
)
from marl_topology.training.two_timescale_env import ReconfigCost


def _scene(num_frames=4, e_edge=0.05, hold_interval=1, gamma=0.9):
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
                                     reconfig=ReconfigCost(e_edge=e_edge, l_edge=0.0),
                                     hold_interval=hold_interval, gamma=gamma)


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


def test_reward_decomposes_into_hold_times_base_minus_reconfig() -> None:
    # The dynamic per-step reward (Contract v3 §3.2): r_t = H * base_t - reconfig_t, reconfig once,
    # nothing at t=0. (At H=1 this reduces to base - reconfig.)
    scene = _scene(num_frames=4, e_edge=0.05, hold_interval=3)
    actor, critic, mean, std = _models(scene)
    recs, _ = episode_rollout(actor, critic, scene, mean, std, recurrent=False, temp=1.0,
                              reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                              beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(2))
    prev = []
    for t, rec in enumerate(recs):
        switches = len(frozenset(prev) ^ frozenset(rec.topo)) if t > 0 else 0
        assert rec.switches == switches
        assert abs(rec.reconfig - 0.05 * switches) < 1e-9
        assert abs(rec.reward - (scene.hold_interval * rec.base_reward - rec.reconfig)) < 1e-9
        if t == 0:
            assert rec.reconfig == 0.0          # no reconfiguration charged at the first frame
        prev = rec.topo


def test_dynamic_reward_multiplies_base_by_hold_interval() -> None:
    # FAILS on HEAD (reward = base - reconfig, hold_interval ignored). The macro topology is held for
    # H PBFT micro-rounds -> the base objective is reaped H times before the one-time switch cost.
    H = 4
    scene = _scene(num_frames=4, e_edge=0.05, hold_interval=H)
    actor, critic, mean, std = _models(scene)
    recs, _ = episode_rollout(actor, critic, scene, mean, std, recurrent=False, temp=1.0,
                              reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                              beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(7))
    # at least one frame has a nonzero base reward, so H*base - reconfig != base - reconfig
    assert any(abs(rec.base_reward) > 1e-6 for rec in recs)
    for rec in recs:
        assert abs(rec.reward - (H * rec.base_reward - rec.reconfig)) < 1e-9


def test_temporal_value_and_rl_use_same_reward() -> None:
    # The RL per-step reward must equal the Temporal-Value-Test env step reward frame-by-frame
    # (same objective: H*base - reconfig, reconfig once). cost_fn = -base maps TVT cost <-> RL reward.
    # FAILS on HEAD at H>1 (RL drops the H factor).
    from marl_topology.training.two_timescale_env import TwoTimescaleTopologyEnv
    scene = _scene(num_frames=4, e_edge=0.05, hold_interval=4)
    actor, critic, mean, std = _models(scene)
    recs, _ = episode_rollout(actor, critic, scene, mean, std, recurrent=False, temp=1.0,
                              reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                              beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(11))
    env = TwoTimescaleTopologyEnv(frames=recs, cost_fn=lambda rec, _topo: -rec.base_reward,
                                  reconfig=scene.reconfig, hold_interval=scene.hold_interval,
                                  gamma=scene.gamma)
    state = env.reset()
    for rec in recs:
        res = env.step(state, frozenset(rec.topo))
        assert abs(res.reward - rec.reward) < 1e-6
        state = res.next_state


def test_dynamic_eval_uses_discounted_return() -> None:
    # dynamic_eval's episode_return must be the discounted, H-scaled return G_0 = sum gamma^t (H*base
    # - reconfig). FAILS on HEAD (undiscounted sum of base - reconfig). Recompute from the per-frame
    # traces and compare.
    scene = _scene(num_frames=4, e_edge=0.05, hold_interval=4, gamma=0.9)
    actor, critic, mean, std = _models(scene)
    out = dynamic_eval(actor, [scene], mean, std, recurrent=False, temp=1.0, reward_of=_reward_fn,
                       ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")
    frames = out["traces"][0]["frames"]
    discounted = sum((scene.gamma ** t) * (scene.hold_interval * f["base_reward"] - f["reconfig"])
                     for t, f in enumerate(frames))
    undiscounted_noH = sum(f["base_reward"] - f["reconfig"] for f in frames)
    reported = out["traces"][0]["episode_return"]
    assert abs(reported - discounted) < 5e-3, f"reported {reported} != discounted {discounted}"
    # guard the test is discriminating: the two definitions actually differ here
    assert abs(discounted - undiscounted_noH) > 1e-2


def test_recurrent_reroll_bptt_reaches_gru() -> None:
    scene = _scene(num_frames=4)
    actor, critic, mean, std = _models(scene)
    recs, _ = episode_rollout(actor, critic, scene, mean, std, recurrent=True, temp=1.0,
                              reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                              beta=0.1, reward_mode="dense", generator=torch.Generator().manual_seed(3))
    lp, en, vp = _reroll_logp(actor, critic, scene, recs, mean, std, recurrent=True, temp=1.0)
    torch.stack(lp).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in actor.gru.parameters())


def test_frame_teacher_trajectory_is_binary_per_frame() -> None:
    scene = _scene(num_frames=4)
    _a, _c, mean, std = _models(scene)
    traj = _frame_teacher_trajectory(scene, mean, std, reward_of=_reward_fn, ref_energy=_ref_energy,
                                     lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")
    assert len(traj) == scene.n_frames
    for obs, tgt in traj:
        assert tgt.shape[0] == len(obs["edge_ids"])
        assert set(float(x) for x in tgt) <= {0.0, 1.0}     # a 0/1 edge indicator (the teacher topology)


def test_warmstart_moves_actor_toward_teacher() -> None:
    # more warm-start epochs -> lower final BCE toward the per-frame teacher (the warm-start converges).
    scene = _scene(num_frames=3)
    a20, _c, mean, std = _models(scene)
    bce20 = warmstart_actor(a20, [scene], mean, std, recurrent=True, epochs=20, lr=5e-3,
                            reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                            beta=0.1, reward_mode="dense")
    a1, _c2, _m, _s = _models(scene)
    bce1 = warmstart_actor(a1, [scene], mean, std, recurrent=True, epochs=1, lr=5e-3,
                           reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0,
                           beta=0.1, reward_mode="dense")
    assert bce20 < bce1, f"warm-start should reduce BCE toward the teacher (20ep {bce20} vs 1ep {bce1})"


def test_dynamic_eval_metrics_in_range() -> None:
    scene = _scene(num_frames=4)
    actor, critic, mean, std = _models(scene)
    out = dynamic_eval(actor, [scene], mean, std, recurrent=True, temp=1.0, reward_of=_reward_fn,
                       ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")
    assert 0.0 <= out["per_frame_feasibility"] <= 1.0
    assert out["n_frames_total"] == scene.n_frames
    assert out["mean_switches_per_frame"] >= 0.0
    assert len(out["traces"]) == 1 and len(out["traces"][0]["frames"]) == scene.n_frames
