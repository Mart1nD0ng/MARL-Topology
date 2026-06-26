"""Dynamic (two-timescale, T>1) episode RL: rollout + recurrent PPO over moving-vehicle trajectories.

Wires ``two_timescale_env`` semantics into a genuine sequential decision loop on top of the SAME
decentralized components as the static trunk (BCSP per-agent sampler, the torch-free
``local_mutual_assemble`` decoder, the per-agent PPO ratio, the centralized graph critic). The static
``T=1`` trunk passes its ``reward_of`` + objective helpers IN, so the constrained objective has ONE
definition (no fork); the dynamic arm is reached only via ``--dynamic`` and leaves the ``T=1`` path
byte-identical.

The MDP (Spec S3.3-3.6): at each of ``T`` frames the actor observes the current channel + the
previous topology + the step index, samples a topology (BCSP -> mutual decoder), holds it for
``hold_interval`` PBFT micro-rounds, and receives ``reward = base_objective - reconfig_cost`` where
``reconfig_cost = (e_edge + l_edge) * |E_t triangle E_{t-1}|``. The return is the gamma-discounted
sum to the end of the episode; the advantage is ``A_t = G_t - V(s_t)``.

Two arms differ ONLY in whether the actor's per-node hidden state carries across frames (recurrent)
or is reset each frame (memoryless) -- a controlled ablation of cross-frame memory. Training-only
(critic + standardization stats). The HELD-EVAL decoder is the deployed torch-free
``local_mutual_assemble``; the ROLLOUT uses the matched stochastic BCSP sampler whose deterministic
(temperature->0 / MAP) limit IS that decoder -- so train == deploy up to a measure-zero tie set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from marl_topology.budgets import is_budget_feasible, node_budgets_for_scene
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor
from marl_topology.policies.decentralized_mutual_acceptance import local_mutual_assemble
from marl_topology.training.decentralized_action import (
    recompute_bcsp_entropy,
    recompute_bcsp_logp,
    sample_decentralized_bcsp_action,
)
from marl_topology.training.graph_mappo import critic_scene_value, ppo_clip_actor_loss

DYNAMIC_RL_MODEL_ID = "two_timescale_dynamic_rl_v1"


def _standardize(nf, ef, mean, std):
    return (nf - mean[0]) / std[0], (ef - mean[1]) / std[1]


def _budgets_edges(ctx):
    budgets = dict(node_budgets_for_scene(ctx.evaluator.scene))
    edges = {e.edge_id: (e.node_u, e.node_v) for e in ctx.graph.edges}
    return budgets, edges


@dataclass
class FrameRecord:
    obs: dict
    per_agent: list           # (incident, accepted, budget, logp_old)
    active: tuple
    topo: list
    reward: float
    base_reward: float
    reconfig: float
    switches: int
    feasible: bool
    value: float
    ret: float = 0.0          # G_t, filled after the episode


def episode_rollout(actor, critic, scene, mean, std, *, recurrent, temp, reward_of, ref_energy,
                    lam_c, lam_b, beta, reward_mode, generator=None):
    """One no-grad episode: carry hidden across frames (recurrent) or reset (memoryless)."""
    actor.eval()
    records: list[FrameRecord] = []
    prev_topo: list[str] = []
    hidden = None
    e_ref = None
    for t in range(scene.n_frames):
        obs = scene.observation(t, prev_topo)
        ctx = obs["context"]
        if e_ref is None:
            e_ref = ref_energy(obs)                      # fixed per-scene normalizer (frame 0 full graph)
        budgets, edges = _budgets_edges(ctx)
        nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
        with torch.no_grad():
            logits, h_next = actor(nf_s, ef_s, obs["ei"], hidden=hidden)
            act = sample_decentralized_bcsp_action(logits, obs["edge_ids"], edges=edges,
                                                   budgets=budgets, temperature=temp,
                                                   compute_entropy=False, generator=generator)
            v = float(critic_scene_value(critic, obs["nf"], obs["ef"], obs["ei"],
                                         node_mean=mean[0], node_std=std[0],
                                         edge_mean=mean[1], edge_std=std[1]))
        topo = [obs["edge_ids"][j] for j in act.active_edge_indices]
        base_r, _gc, _gb, ok = reward_of(obs, topo, e_ref, lam_c, lam_b, beta, reward_mode)
        switches = len(frozenset(prev_topo) ^ frozenset(topo)) if t > 0 else 0
        reconfig = (scene.reconfig.e_edge + scene.reconfig.l_edge) * switches
        # Contract v3 §3.2 / two_timescale_env: a macro topology is held for H_PBFT micro-rounds, so
        # the per-frame base objective is reaped H times before the one-time switch cost.
        reward = scene.hold_interval * base_r - reconfig
        per_agent = [(pa.incident_edge_indices, pa.accepted_local_indices, pa.budget, float(pa.logp))
                     for pa in act.per_agent if pa.incident_edge_indices]
        records.append(FrameRecord(obs, per_agent, act.active_edge_indices, topo, reward, base_r,
                                   reconfig, switches, bool(ok), v))
        prev_topo = topo
        hidden = h_next if recurrent else None           # the ONLY difference between the two arms
    # discounted returns to episode end
    g = 0.0
    for rec in reversed(records):
        g = rec.reward + scene.gamma * g
        rec.ret = g
    return records, e_ref


def _reroll_logp(actor, critic, scene, records, mean, std, *, recurrent, temp,
                 need_logp=True, need_value=True):
    """Re-roll the episode (grad on). need_logp -> re-roll the actor with BPTT hidden carry and
    recompute per-(frame,agent) logp + entropy (the actor PPO loop); need_value -> forward the
    per-frame critic V_pred (the critic loop). Splitting them avoids the 2x waste of computing both
    in each loop. The decoded action is the one RECORDED at rollout (off-policy re-scoring)."""
    hidden = None
    logp_new, ent_new, v_pred = [], [], []
    for t, rec in enumerate(records):
        if need_logp:
            nf_s, ef_s = _standardize(rec.obs["nf"], rec.obs["ef"], mean, std)
            logits, h_next = actor(nf_s, ef_s, rec.obs["ei"], hidden=hidden)
            for (incident, accepted, bud, _lo) in rec.per_agent:
                logp_new.append(recompute_bcsp_logp(logits, incident, accepted, temp, bud))
                ent_new.append(recompute_bcsp_entropy(logits, incident, temp, bud))
            hidden = h_next if recurrent else None
        if need_value:                                          # critic is per-frame (no actor hidden)
            v_pred.append(critic_scene_value(critic, rec.obs["nf"], rec.obs["ef"], rec.obs["ei"],
                                             node_mean=mean[0], node_std=std[0],
                                             edge_mean=mean[1], edge_std=std[1]))
    return logp_new, ent_new, v_pred


def dynamic_eval(actor, scenes, mean, std, *, recurrent, temp, reward_of, ref_energy,
                 lam_c, lam_b, beta, reward_mode):
    """Deterministic-ish held eval: per-frame feasibility, episode return, switching, energy."""
    actor.eval()
    n_frames_total = feas_frames = 0
    ret_sum = switch_sum = 0.0
    per_scene = []
    for scene in scenes:
        prev_topo: list[str] = []
        hidden = None
        e_ref = None
        frames = []
        ep_ret = 0.0
        discount = 1.0
        for t in range(scene.n_frames):
            obs = scene.observation(t, prev_topo)
            ctx = obs["context"]
            if e_ref is None:
                e_ref = ref_energy(obs)
            budgets, _edges = _budgets_edges(ctx)
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            with torch.no_grad():
                logits, h_next = actor(nf_s, ef_s, obs["ei"], hidden=hidden)
            topo = local_mutual_assemble(logits, obs["edge_ids"], ctx)   # the DEPLOYED decoder
            base_r, _gc, _gb, ok = reward_of(obs, list(topo), e_ref, lam_c, lam_b, beta, reward_mode)
            switches = len(frozenset(prev_topo) ^ frozenset(topo)) if t > 0 else 0
            reconfig = (scene.reconfig.e_edge + scene.reconfig.l_edge) * switches
            # SAME objective as training (Contract v3 §3.2/§3.3): discounted, H-scaled episode return.
            ep_ret += discount * (scene.hold_interval * base_r - reconfig)
            discount *= scene.gamma
            n_frames_total += 1; feas_frames += int(ok); switch_sum += switches
            frames.append({"t": t, "topo_size": len(topo), "switches": switches,
                           "feasible": bool(ok), "base_reward": round(float(base_r), 4),
                           "reconfig": round(float(reconfig), 4)})
            prev_topo = list(topo)
            hidden = h_next if recurrent else None
        ret_sum += ep_ret
        per_scene.append({"sequence_id": scene.sequence_id, "n_frames": scene.n_frames,
                          "episode_return": round(float(ep_ret), 4), "frames": frames})
    return {
        "per_frame_feasibility": feas_frames / max(1, n_frames_total),
        "mean_episode_return": ret_sum / max(1, len(scenes)),
        "mean_switches_per_frame": switch_sum / max(1, n_frames_total),
        "n_scenes": len(scenes), "n_frames_total": n_frames_total,
        "traces": per_scene,
    }


def _frame_teacher_trajectory(scene, mean, std, *, reward_of, ref_energy, lam_c, lam_b, beta, reward_mode):
    """Per-frame myopic-greedy best topology over the canonical candidate variants (reward-greedy,
    feasible-preferring; reconfiguration-blind). The warm-start teacher -- it gives the cold-start
    actor a FEASIBLE target so it can escape the dead ~2-edge region (cold-start RL alone cannot find
    the feasible backbone at N<=16; the static campaign needed the same BC warm-start). Training-only;
    the deployed actor still uses only local info. Returns [(obs_at_frame_t_under_teacher_prev, target)]."""
    out = []
    prev = []
    for t in range(scene.n_frames):
        obs = scene.observation(t, prev)
        ctx = obs["context"]
        tv = getattr(ctx, "topology_variants", None) or {}
        values = tv.values() if isinstance(tv, dict) else tv
        cands = [tuple(str(e) for e in (v.edges if hasattr(v, "edges") else v)) for v in values]
        cands = cands or [tuple(obs["edge_ids"])]
        e_ref = ref_energy(obs)
        best, best_r = cands[0], -1e30
        for cand in cands:
            r, _gc, _gb, _ok = reward_of(obs, list(cand), e_ref, lam_c, lam_b, beta, reward_mode)
            if r > best_r:
                best, best_r = cand, r
        tset = set(best)
        tgt = obs["nf"].new_tensor([1.0 if e in tset else 0.0 for e in obs["edge_ids"]])
        out.append((obs, tgt))
        prev = list(best)
    return out


def warmstart_actor(actor, scenes, mean, std, *, recurrent, epochs, lr, reward_of, ref_energy,
                    lam_c, lam_b, beta, reward_mode):
    """Supervised warm-start: push the actor's per-edge logits toward the per-frame teacher topology
    (BCE), carrying the hidden state across frames (recurrent) or resetting it (memoryless) to match
    the arm. Returns the mean final-epoch BCE loss (a convergence signal)."""
    teachers = [_frame_teacher_trajectory(s, mean, std, reward_of=reward_of, ref_energy=ref_energy,
                                          lam_c=lam_c, lam_b=lam_b, beta=beta, reward_mode=reward_mode)
                for s in scenes]
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    last = 0.0
    for _ep in range(epochs):
        tot, n = 0.0, 0
        for traj in teachers:
            hidden = None
            losses = []
            for obs, tgt in traj:
                nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
                logits, h_next = actor(nf_s, ef_s, obs["ei"], hidden=hidden)
                losses.append(bce(logits, tgt))
                hidden = h_next if recurrent else None
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
        last = tot / max(1, n)
    return last


def build_split_manifest(*, seed, train, val, held):
    """Record the train / val / held three-split (Contract v3 §3.4 + §14 split_manifest.json).

    The checkpoint is selected ONLY on the validation split's discounted episode return; held is
    final-reporting-only and never enters checkpoint selection. Disjoint seeds (+1 / +333 / +777) +
    per-split id tags make the three sequence-id sets literally disjoint (also fixes the name-by-index
    id collision). ``val=[]`` -> pilot-only (no independent validation; not headline-eligible)."""
    def _ids(scenes):
        return [s.sequence_id for s in scenes]
    splits = {
        "train": {"seed": seed * 1000 + 1, "count": len(train), "sequence_ids": _ids(train)},
        "val": {"seed": seed * 1000 + 333, "count": len(val), "sequence_ids": _ids(val)},
        "held": {"seed": seed * 1000 + 777, "count": len(held), "sequence_ids": _ids(held)},
    }
    s = {k: set(v["sequence_ids"]) for k, v in splits.items()}
    disjoint = (s["train"].isdisjoint(s["val"]) and s["train"].isdisjoint(s["held"])
                and s["val"].isdisjoint(s["held"]))
    has_val = len(val) > 0
    return {
        "splits": splits,
        "splits_disjoint": bool(disjoint),
        "checkpoint_selection_split": "val" if has_val else "train",
        "checkpoint_selection_metric": "val_discounted_episode_return",
        "held_used_for_checkpoint": False,
        "validation_split_present": has_val,
        "pilot_only_no_validation": (not has_val),
    }


def run_dynamic_training(args, *, reward_of, _evaluate, _budgets, _ref_energy, TAU):
    """The --dynamic arm: build moving-vehicle scenes, train the episode-recurrent (or memoryless)
    actor with recurrent PPO + a per-frame critic, eval on a held trajectory set, and emit the full
    dynamic-task instrumentation set. Reached only from the trunk when args.dynamic is set."""
    import sys

    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost

    sys.path.insert(0, str(Path(args.__dict__.get("_root", ".")) / "scripts" / "train"))
    from build_operating_point_dataset import operating_point_regime

    torch.manual_seed(args.seed)
    recurrent = (args.dynamic_actor == "recurrent")
    regime = operating_point_regime(args.tx_power)
    reconfig = ReconfigCost(e_edge=float(args.reconfig_e), l_edge=float(args.reconfig_l))

    # deterministic train / val / held split by disjoint seeds (Contract v3 §3.4): keep-best selects
    # the checkpoint ONLY on val; held is FINAL-reporting-only and never enters checkpoint selection.
    def _mk(seed_off, count, tag):
        if count <= 0:
            return []
        return sample_dynamic_scenes(
            seed=args.seed * 1000 + seed_off, count=count, node_count_choices=tuple(args.dyn_nodes),
            regime=regime, num_frames=args.frames, dt_s=args.dt, speed_min_mps=args.speed_min,
            speed_max_mps=args.speed_max, reconfig=reconfig, hold_interval=args.hold_interval,
            gamma=args.gamma, tag=tag)
    n_val = int(getattr(args, "dyn_val", 0))
    train_scenes = _mk(1, args.dyn_train, "train_")
    val_scenes = _mk(333, n_val, "val_")
    held_scenes = _mk(777, args.dyn_held, "held_")
    # the keep-best eval set is the VALIDATION split (never held); fall back to train ONLY when no val
    # split was requested -> that run is pilot-only (not headline-eligible), recorded in the manifest.
    sel_scenes = val_scenes if val_scenes else train_scenes
    sel_split = "val" if val_scenes else "train"

    # standardization from train observations (frame 0 of each scene); features are already realized
    from marl_topology.training.decentralized_distillation import feature_standardization
    stat_samples = [s.observation(0, []) for s in train_scenes]
    mean, std = feature_standardization(stat_samples)
    node_dim, edge_dim = stat_samples[0]["nf"].shape[1], stat_samples[0]["ef"].shape[1]

    actor = DynamicRecurrentActor(node_dim, edge_dim, hidden=args.hidden)
    critic = CentralizedGraphCritic(node_dim, edge_dim, hidden=args.critic_hidden, rounds=args.critic_rounds)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    def ref_energy(obs):
        try:
            _c, e, _l = _evaluate(obs, obs["edge_ids"])
            return max(e, 1e-9)
        except Exception:
            return 1e-9

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = build_split_manifest(seed=args.seed, train=train_scenes, val=val_scenes,
                                          held=held_scenes)
    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    history = []
    best_val = -1e30
    best_state = None
    temp = args.temp

    warmstart_bce = None
    n_warm = int(getattr(args, "dyn_warmstart", 0))
    if n_warm > 0:
        # Supervised warm-start toward the per-frame myopic-greedy teacher BEFORE RL -- gives the
        # cold-start actor a feasible target (both arms get the SAME warm-start; the only difference
        # stays the cross-frame hidden carry). Critic stays cold (RL trains it).
        wlr = getattr(args, "dyn_warmstart_lr", 0.0) or args.lr
        warmstart_bce = warmstart_actor(
            actor, train_scenes, mean, std, recurrent=recurrent, epochs=n_warm, lr=wlr,
            reward_of=reward_of, ref_energy=ref_energy,
            lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta, reward_mode=args.reward_mode)
        print(f"[dynamic:{args.dynamic_actor}] warm-start {n_warm} epochs -> final BCE {warmstart_bce:.4f}",
              flush=True)

    # measure the warm-start init's HELD quality BEFORE RL (measurement only -> no checkpoint leakage),
    # so the report can honestly compare warm-start-alone vs warm-start+RL.
    warmstart_held = None
    if n_warm > 0:
        ws = dynamic_eval(actor, held_scenes, mean, std, recurrent=recurrent, temp=temp,
                          reward_of=reward_of, ref_energy=ref_energy, lam_c=args.lam_c, lam_b=args.lam_b,
                          beta=args.beta, reward_mode=args.reward_mode)
        warmstart_held = {"per_frame_feasibility": round(ws["per_frame_feasibility"], 5),
                          "mean_episode_return": round(ws["mean_episode_return"], 5)}

    for update in range(args.updates):
        # ---- rollout all train episodes (no grad) ----
        episodes = []
        for scene in train_scenes:
            recs, _eref = episode_rollout(
                actor, critic, scene, mean, std, recurrent=recurrent, temp=temp, reward_of=reward_of,
                ref_energy=ref_energy, lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta,
                reward_mode=args.reward_mode)
            if any(r.per_agent for r in recs):
                episodes.append((scene, recs))
        if not episodes:
            continue
        # advantages A_t = G_t - V_t (rollout values), flattened per (scene, frame, agent)
        adv_flat, logp_old_flat = [], []
        for _scene, recs in episodes:
            for rec in recs:
                a = rec.ret - rec.value
                for _pa in rec.per_agent:
                    adv_flat.append(a)
                    logp_old_flat.append(_pa[3])
        adv_flat = torch.tensor(adv_flat)
        logp_old_flat = torch.tensor(logp_old_flat)
        if args.normalize_adv and adv_flat.numel() > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-6)

        # ---- recurrent PPO inner epochs (actor) ----
        last = {"kl": 0.0, "clip": 0.0, "actor_loss": 0.0, "actor_gnorm": 0.0, "entropy": 0.0}
        for _epoch in range(args.ppo_epochs):
            lp_new, ent_new = [], []
            for scene, recs in episodes:
                lp, en, _vp = _reroll_logp(actor, critic, scene, recs, mean, std,
                                           recurrent=recurrent, temp=temp, need_value=False)
                lp_new.extend(lp); ent_new.extend(en)
            if not lp_new:
                break
            ppo_loss, info = ppo_clip_actor_loss(torch.stack(lp_new), logp_old_flat,
                                                 adv_flat.detach(), clip_eps=args.clip_epsilon)
            loss = ppo_loss - args.entropy_coef * torch.stack(ent_new).mean()
            opt.zero_grad(); loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
            opt.step()
            last = {"kl": float(info["approx_kl"]), "clip": float(info["clip_fraction"]),
                    "actor_loss": float(ppo_loss), "actor_gnorm": gnorm,
                    "entropy": float(torch.stack(ent_new).mean())}
            if last["kl"] > args.target_kl * 1.5:
                break

        # ---- critic regression V(s_t) -> G_t ----
        c0 = torch.nn.utils.parameters_to_vector(critic.parameters()).detach().clone()
        rets = torch.tensor([rec.ret for _s, recs in episodes for rec in recs])
        last_closs, last_cgn, ev = 0.0, 0.0, 0.0
        for _ in range(args.ppo_epochs):
            v_all = []
            for scene, recs in episodes:
                _lp, _en, vp = _reroll_logp(actor, critic, scene, recs, mean, std,
                                            recurrent=recurrent, temp=temp, need_logp=False)
                v_all.extend(vp)
            v_pred = torch.stack(v_all)
            v_loss = args.critic_coef * (rets - v_pred).pow(2).mean()
            opt_c.zero_grad(); v_loss.backward()
            cgn = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0))
            opt_c.step()
            last_closs, last_cgn = float(v_loss), cgn
            resid = rets - v_pred.detach()
            ev = float(1.0 - resid.var() / (rets.var() + 1e-9))
        c1 = torch.nn.utils.parameters_to_vector(critic.parameters()).detach()
        cdelta = float((c1 - c0).abs().sum())

        # ---- periodic keep-best eval on the VALIDATION split (sel_scenes; train only in the pilot
        #      fallback). The heavy evaluator makes per-update eval pathological, so eval every
        #      --dyn-eval-every updates + always on the last one. ----
        eval_every = max(1, int(getattr(args, "dyn_eval_every", 5)))
        do_val = (update % eval_every == 0) or (update == args.updates - 1)
        if do_val:
            # keep-best eval on the VALIDATION split (sel_scenes = val; train only in pilot fallback).
            val = dynamic_eval(actor, sel_scenes, mean, std, recurrent=recurrent, temp=temp,
                               reward_of=reward_of, ref_energy=ref_energy, lam_c=args.lam_c,
                               lam_b=args.lam_b, beta=args.beta, reward_mode=args.reward_mode)
            val_score = val["mean_episode_return"]
        else:
            val = {"per_frame_feasibility": float("nan"), "mean_episode_return": float("nan")}
            val_score = None
        mean_switch = sum(rec.switches for _s, recs in episodes for rec in recs) / max(
            1, sum(len(recs) for _s, recs in episodes))
        mean_card = sum(len(pa[1]) for _s, recs in episodes for rec in recs for pa in rec.per_agent) / max(
            1, sum(len(rec.per_agent) for _s, recs in episodes for rec in recs))
        rec_log = {"update": update, "actor_loss": last["actor_loss"], "critic_loss": last_closs,
                   "critic_explained_variance": round(ev, 4), "actor_grad_norm": last["actor_gnorm"],
                   "critic_grad_norm": last_cgn, "critic_parameter_delta": round(cdelta, 5),
                   "per_agent_kl": last["kl"], "clip_fraction": last["clip"], "entropy": last["entropy"],
                   "mean_subset_cardinality": round(mean_card, 3), "mean_switches_per_frame": round(mean_switch, 3),
                   "val_per_frame_feasibility": (round(val["per_frame_feasibility"], 4) if do_val else None),
                   "val_mean_episode_return": (round(val_score, 4) if val_score is not None else None)}
        history.append(rec_log)
        if val_score is not None and val_score > best_val:   # keep-best on the VAL split (never train/held)
            best_val = val_score
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}

    # ---- final held eval with the kept-best actor ----
    if best_state is not None:
        actor.load_state_dict(best_state)
    held = dynamic_eval(actor, held_scenes, mean, std, recurrent=recurrent, temp=temp,
                        reward_of=reward_of, ref_energy=ref_energy, lam_c=args.lam_c,
                        lam_b=args.lam_b, beta=args.beta, reward_mode=args.reward_mode)

    activation = {
        "dynamic_task": {"enabled": True, "episode_length": int(args.frames),
                         "hold_interval": int(args.hold_interval), "gamma": float(args.gamma),
                         "reconfig_e_edge": float(args.reconfig_e), "reconfig_l_edge": float(args.reconfig_l),
                         "reconfiguration_cost_nonzero": bool(args.reconfig_e or args.reconfig_l),
                         "reward_definition": "hold_interval*base - reconfig",
                         "reward_uses_hold_interval": True,
                         "return_definition": "discounted episode return G_0 = sum gamma^t r_t (train==eval==myopic==TVT)",
                         "splits": {"train": len(train_scenes), "val": len(val_scenes), "held": len(held_scenes),
                                    "train_seed": args.seed * 1000 + 1, "val_seed": args.seed * 1000 + 333,
                                    "held_seed": args.seed * 1000 + 777},
                         "checkpoint_selection_split": sel_split, "held_used_for_checkpoint": False,
                         "validation_split_present": bool(val_scenes),
                         "mobility_speed_mps": [args.speed_min, args.speed_max], "dt_s": float(args.dt)},
        "actor": {"model_id": actor.model_id, "cross_frame_recurrence": bool(recurrent),
                  "arm": args.dynamic_actor, "warmstart_epochs": n_warm,
                  "warmstart_final_bce": warmstart_bce},
        "critic": {"enabled": True, "per_frame_state_value": True},
        "action": {"distribution": "bcsp",
                   "rollout_sampler": "sample_decentralized_bcsp_action (stochastic; MAP == deploy)",
                   "eval_decoder": "local_mutual_assemble (deployed, torch-free)",
                   "per_agent_ratio": True},
        "regime": "operating_point urban v2x_37885 shadowing nlosv relay-3 backhaul coverage-gated",
    }
    result = {
        "arm": args.dynamic_actor, "seed": args.seed,
        "warmstart_epochs": n_warm, "warmstart_held": warmstart_held,
        "held_per_frame_feasibility": round(held["per_frame_feasibility"], 5),
        "held_mean_episode_return": round(held["mean_episode_return"], 5),
        "held_mean_switches_per_frame": round(held["mean_switches_per_frame"], 5),
        "best_val_episode_return": round(best_val, 5),
        "checkpoint_selection": {"split": sel_split, "metric": "val_discounted_episode_return",
                                 "held_used_for_checkpoint": False,
                                 "validation_split_present": bool(val_scenes)},
        "n_train_scenes": len(train_scenes), "n_val_scenes": len(val_scenes),
        "n_held_scenes": len(held_scenes),
        "updates": args.updates, "config": {"frames": args.frames, "hold_interval": args.hold_interval,
                                            "gamma": args.gamma, "reconfig_e": args.reconfig_e}}

    (out_dir / "mechanism_activation.json").write_text(json.dumps(activation, indent=2), encoding="utf-8")
    (out_dir / "dynamic_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "held_traces.json").write_text(json.dumps(held["traces"], indent=2), encoding="utf-8")
    print(f"[dynamic:{args.dynamic_actor}] seed={args.seed} held per-frame feasibility="
          f"{result['held_per_frame_feasibility']:.4f} episode_return="
          f"{result['held_mean_episode_return']:.4f} switches/frame="
          f"{result['held_mean_switches_per_frame']:.3f}")
    return result
