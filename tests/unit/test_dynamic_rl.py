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


# --------------------------------------------------------------------------- #
# D3: independent train / val / held split; checkpoint selected by VALIDATION only.
# --------------------------------------------------------------------------- #
def _split(seed_off, tag, count=4, n=8):
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    return sample_dynamic_scenes(
        seed=7 * 1000 + seed_off, count=count, node_count_choices=(n,), regime=PhysicsRegime(),
        num_frames=3, dt_s=1.0, speed_min_mps=15.0, speed_max_mps=30.0,
        reconfig=ReconfigCost(e_edge=0.1, l_edge=0.0), hold_interval=4, gamma=0.9, tag=tag)


def test_dynamic_train_val_held_seeds_disjoint() -> None:
    # FAILS on HEAD: sample_dynamic_scenes has no `tag`, so the three splits share name-by-index ids
    # (N5 nit). With tags train_/val_/held_ the sequence_ids are disjoint, and the distinct seeds give
    # distinct geometry at the same index.
    tr, va, he = _split(1, "train_"), _split(333, "val_"), _split(777, "held_")
    ids = lambda lst: {s.sequence_id for s in lst}
    assert ids(tr).isdisjoint(ids(va))
    assert ids(tr).isdisjoint(ids(he))
    assert ids(va).isdisjoint(ids(he))
    # distinct rng seeds -> distinct geometry (a moved vehicle's frame-0 position differs across splits)
    p_tr = tr[0].scenes[0].nodes[1].position
    p_va = va[0].scenes[0].nodes[1].position
    assert (p_tr.x_m, p_tr.y_m) != (p_va.x_m, p_va.y_m)


def test_split_manifest_records_all_three() -> None:
    # FAILS on HEAD: build_split_manifest does not exist. The manifest records all three splits with
    # disjoint ids, the checkpoint-selection split (val), and that held is NOT used for checkpoint.
    from marl_topology.training.dynamic_rl import build_split_manifest
    tr, va, he = _split(1, "train_"), _split(333, "val_"), _split(777, "held_")
    man = build_split_manifest(seed=7, train=tr, val=va, held=he)
    assert set(man["splits"]) == {"train", "val", "held"}
    assert man["splits"]["train"]["seed"] == 7 * 1000 + 1
    assert man["splits"]["val"]["seed"] == 7 * 1000 + 333
    assert man["splits"]["held"]["seed"] == 7 * 1000 + 777
    for k in ("train", "val", "held"):
        assert man["splits"][k]["count"] == len(_split(1, "train_"))
    assert man["checkpoint_selection_split"] == "val"
    assert man["checkpoint_selection_metric"] == "val_discounted_episode_return"
    assert man["held_used_for_checkpoint"] is False
    # disjointness asserted in the manifest itself
    assert man["splits_disjoint"] is True


def test_split_manifest_pilot_fallback_when_no_validation() -> None:
    # No validation split (dyn_val=0) -> the manifest must label the run pilot-only and record that the
    # checkpoint falls back to train (Contract v3 §3.4: a no-validation run is NOT headline-eligible).
    from marl_topology.training.dynamic_rl import build_split_manifest
    tr, he = _split(1, "train_"), _split(777, "held_")
    man = build_split_manifest(seed=7, train=tr, val=[], held=he)
    assert man["validation_split_present"] is False
    assert man["pilot_only_no_validation"] is True
    assert man["checkpoint_selection_split"] == "train"   # fallback, clearly labeled pilot-only
    assert man["held_used_for_checkpoint"] is False        # held STILL never selects the checkpoint
    assert man["splits"]["val"]["count"] == 0


def test_run_dynamic_training_uses_val_split_for_checkpoint(tmp_path) -> None:
    # Integration: a tiny in-process run must build train/val/held, select the checkpoint on VAL, and
    # leave held for final reporting only. FAILS on HEAD (no --dyn-val arg; no split_manifest.json;
    # keep-best on train).
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "trunk_d3", root / "scripts" / "train" / "train_decentralized_rl.py")
    trunk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trunk)
    from marl_topology.training.dynamic_rl import run_dynamic_training

    run = tmp_path / "run"
    args = trunk.parse_args([
        "--dynamic", "--cold-start", "--reward-mode", "dense", "--dynamic-actor", "memoryless",
        "--dyn-train", "2", "--dyn-val", "2", "--dyn-held", "2", "--frames", "2",
        "--updates", "2", "--ppo-epochs", "1", "--dyn-eval-every", "1", "--dyn-nodes", "8",
        "--seed", "3", "--out-dir", str(run)])
    args._root = str(root)
    run_dynamic_training(args, reward_of=trunk.reward_of, _evaluate=trunk._evaluate,
                         _budgets=trunk._budgets, _ref_energy=trunk._ref_energy, TAU=trunk.TAU)

    man = json.loads((run / "split_manifest.json").read_text())
    assert set(man["splits"]) == {"train", "val", "held"}
    assert man["checkpoint_selection_split"] == "val"
    assert man["held_used_for_checkpoint"] is False
    assert man["splits_disjoint"] is True
    result = json.loads((run / "dynamic_result.json").read_text())
    assert result["checkpoint_selection"]["split"] == "val"
    assert "held_per_frame_feasibility" in result        # held still reported (final only)


# --------------------------------------------------------------------------- #
# D6: decoder-aware (BCSP-subset) warm-start + anti-drift teacher anchor + critic warm-start.
# --------------------------------------------------------------------------- #
_RW = dict(reward_of=_reward_fn, ref_energy=_ref_energy, lam_c=1.0, lam_b=1.0, beta=0.1, reward_mode="dense")


def test_decoder_aware_teacher_reconstructs_topology() -> None:
    # The decoder-aware teacher's per-agent BCSP proposals must reconstruct (via local mutual
    # acceptance) the recorded teacher topology, which is a budget-feasible subset of the myopic best.
    from marl_topology.training.dynamic_rl import _bcsp_teacher_trajectory
    scene = _scene(num_frames=2)
    _a, _c, mean, std = _models(scene)
    traj = _bcsp_teacher_trajectory(scene, mean, std, **_RW)
    assert len(traj) == scene.n_frames
    for fr in traj:
        edges = {e.edge_id: (e.node_u, e.node_v) for e in fr["obs"]["context"].graph.edges}
        edge_ids = fr["obs"]["edge_ids"]
        accept = {node: {inc[k] for k in acc} for (node, inc, acc, _b) in fr["proposals"]}
        mutual = {edge_ids[i] for i, eid in enumerate(edge_ids)
                  if i in accept.get(edges[eid][0], ()) and i in accept.get(edges[eid][1], ())}
        assert mutual == set(fr["recon"])                # recon IS the mutual decode of the proposals
        assert set(fr["recon"]).issubset(set(fr["teacher"]))   # a budget-feasible subset of the teacher
        for (_node, inc, acc, b) in fr["proposals"]:
            assert len(acc) <= b                         # every proposal respects the node budget (BCSP support)


def test_bcsp_warmstart_increases_teacher_subset_logp() -> None:
    # more decoder-aware warm-start epochs -> lower teacher-subset NLL (higher BCSP logp).
    from marl_topology.training.dynamic_rl import warmstart_actor_bcsp
    scene = _scene(num_frames=3)
    a20, _c, mean, std = _models(scene)
    nll20 = warmstart_actor_bcsp(a20, [scene], mean, std, recurrent=True, epochs=20, lr=5e-3, temp=1.0, **_RW)
    a1, _c2, _m, _s = _models(scene)
    nll1 = warmstart_actor_bcsp(a1, [scene], mean, std, recurrent=True, epochs=1, lr=5e-3, temp=1.0, **_RW)
    assert nll20 < nll1, f"decoder-aware warm-start should lower teacher NLL (20ep {nll20} vs 1ep {nll1})"


def test_ppo_kl_anchor_limits_drift_from_teacher() -> None:
    # under an identical drift-inducing (entropy-maximizing) update, the BC anchor (lambda>0) keeps the
    # actor closer to the teacher (higher teacher subset logp) than no anchor (lambda=0).
    from marl_topology.training.dynamic_rl import (
        _bcsp_teacher_trajectory, bcsp_teacher_anchor_loss, warmstart_actor_bcsp)

    def _warm():
        a, _c, mean, std = _models(scene)
        warmstart_actor_bcsp(a, [scene], mean, std, recurrent=False, epochs=15, lr=5e-3, temp=1.0, **_RW)
        return a, mean, std

    scene = _scene(num_frames=3)
    traj_holder = {}

    def _drift(actor, mean, std, lam):
        traj = traj_holder.setdefault("t", _bcsp_teacher_trajectory(scene, mean, std, **_RW))
        opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
        for _ in range(12):
            ent, anchor = [], bcsp_teacher_anchor_loss(actor, scene, traj, mean, std, recurrent=False, temp=1.0)
            for fr in traj:
                nf_s, ef_s = _standardize_obs(actor, fr["obs"], mean, std)
                logits, _h = actor(nf_s, ef_s, fr["obs"]["ei"], hidden=None)
                from marl_topology.training.decentralized_action import recompute_bcsp_entropy
                for (node, inc, acc, b) in fr["proposals"]:
                    if inc:
                        ent.append(recompute_bcsp_entropy(logits, inc, 1.0, b))
            drift = -torch.stack(ent).mean()             # push toward uniform (away from the peaked teacher)
            loss = drift + lam * anchor
            opt.zero_grad(); loss.backward(); opt.step()
        return float(-bcsp_teacher_anchor_loss(actor, scene, traj, mean, std, recurrent=False, temp=1.0))

    a_free, m, s = _warm()
    free_logp = _drift(a_free, m, s, lam=0.0)
    a_anchor, m2, s2 = _warm()
    anchor_logp = _drift(a_anchor, m2, s2, lam=3.0)
    assert anchor_logp > free_logp, f"anchor should limit drift (anchor logp {anchor_logp} > free {free_logp})"


def test_critic_pretrain_tracks_teacher_return() -> None:
    # warm-starting the critic toward the teacher's discounted returns lowers its MSE.
    from marl_topology.training.dynamic_rl import warmstart_critic
    scene = _scene(num_frames=3)
    _a, c30, mean, std = _models(scene)
    mse30 = warmstart_critic(c30, [scene], mean, std, epochs=30, lr=5e-3, **_RW)
    _a2, c1, _m, _s = _models(scene)
    mse1 = warmstart_critic(c1, [scene], mean, std, epochs=1, lr=5e-3, **_RW)
    assert mse30 < mse1, f"critic warm-start should lower MSE to teacher returns (30ep {mse30} vs 1ep {mse1})"


def _standardize_obs(actor, obs, mean, std):
    from marl_topology.training.dynamic_rl import _standardize
    return _standardize(obs["nf"], obs["ef"], mean, std)
