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
    incident_index,
    recompute_bcsp_entropy,
    recompute_bcsp_logp,
    sample_decentralized_bcsp_action,
)
from marl_topology.training.counterfactual_credit import _active_onehot, counterfactual_advantages
from marl_topology.training.graph_mappo import critic_q_value, critic_scene_value, ppo_clip_actor_loss
from marl_topology.training.scq_supervision import scq_counterfactual_targets

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
    cf_adv: list | None = None  # D9: per-agent COMA counterfactual advantages (aligned with per_agent)
    scq_ctx: dict | None = None  # D10: {per_agent (full BCSP actions), logits} for SCQ target build


def _critic_value(critic, obs, active_indices, mean, std):
    """V(s_t) or, for an action-conditioned (counterfactual) critic, Q(s_t, S_t) over the recorded
    active-edge one-hot. Dispatches on ``critic.critic_sees_action`` (D9)."""
    if getattr(critic, "critic_sees_action", False):
        oh = _active_onehot(active_indices, obs["ef"].shape[0], device=obs["ef"].device,
                            dtype=obs["ef"].dtype)
        return critic_q_value(critic, obs["nf"], obs["ef"], obs["ei"], oh, node_mean=mean[0],
                              node_std=std[0], edge_mean=mean[1], edge_std=std[1])
    return critic_scene_value(critic, obs["nf"], obs["ef"], obs["ei"], node_mean=mean[0],
                              node_std=std[0], edge_mean=mean[1], edge_std=std[1])


def episode_rollout(actor, critic, scene, mean, std, *, recurrent, temp, reward_of, ref_energy,
                    lam_c, lam_b, beta, reward_mode, generator=None, counterfactual=False, k_cf=4,
                    scq=False):
    """One no-grad episode: carry hidden across frames (recurrent) or reset (memoryless). With
    ``counterfactual`` (D9), the action-conditioned Q critic also produces per-agent COMA advantages
    A_{i,t}=Q(s_t,S_t)-E_{S~_i}Q(s_t,S~_i,S_{-i}) (budget-neutral: critic forwards, no evaluator call)."""
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
            v = float(_critic_value(critic, obs, act.active_edge_indices, mean, std))
            cf_adv = None
            if counterfactual:
                cc = counterfactual_advantages(
                    critic, node_features=obs["nf"], edge_features=obs["ef"], edge_index=obs["ei"],
                    edge_ids=obs["edge_ids"], edges=edges, per_agent_actions=act.per_agent,
                    logits=logits, temperature=temp, k_cf=k_cf, node_mean=mean[0], node_std=std[0],
                    edge_mean=mean[1], edge_std=std[1], generator=generator)
                cf_adv = [cc.advantages[pa.node_id] for pa in act.per_agent if pa.incident_edge_indices]
            scq_ctx = {"per_agent": act.per_agent, "logits": logits.detach()} if scq else None
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
                                   reconfig, switches, bool(ok), v, cf_adv=cf_adv, scq_ctx=scq_ctx))
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
            v_pred.append(_critic_value(critic, rec.obs, rec.active, mean, std))   # V or action-cond Q (D9)
    return logp_new, ent_new, v_pred


def dynamic_scq_targets(records, scene, critic, mean, std, *, reward_of, ref_energy, e_ref,
                        lam_c, lam_b, beta, reward_mode, temp, scq_m, scq_select, generator=None):
    """D10: build the dynamic one-step SCQ targets Δy_i = ΔR_i + γ(V(s_{t+1}) − V(s̃_{t+1})) (Plan §12).

    The immediate ΔR_i = r_t − r̃_t uses the per-frame reward oracle (the EXACT evaluator difference per
    UNIQUE counterfactual -- the SCQ budget, NOT budget-neutral). The bootstrap term re-decodes the
    counterfactual into the NEXT frame's prev-topology, rebuilds the next obs, and re-forwards the Q
    critic -- a critic forward, FREE (no evaluator). This is what makes it a DYNAMIC (one-step return)
    SCQ, not the static single-step ΔR primitive. Returns ((t, actual_active, active_cf, Δy)...,
    scq_evaluator_calls, duplicate_topology_count). The targets are detached (fixed supervision)."""
    targets, calls, dups = [], 0, 0
    n = len(records)
    for t, rec in enumerate(records):
        if rec.scq_ctx is None:
            continue
        obs = rec.obs
        _budgets, edges = _budgets_edges(obs["context"])
        prev_topo = records[t - 1].topo if t > 0 else []

        def rdyn(active_indices, _obs=obs, _prev=prev_topo, _t=t):
            topo = [_obs["edge_ids"][i] for i in active_indices]
            base_r, _gc, _gb, _ok = reward_of(_obs, topo, e_ref, lam_c, lam_b, beta, reward_mode)
            switches = len(frozenset(_prev) ^ frozenset(topo)) if _t > 0 else 0
            reconfig = (scene.reconfig.e_edge + scene.reconfig.l_edge) * switches
            return scene.hold_interval * base_r - reconfig

        scq = scq_counterfactual_targets(
            rdyn, per_agent_actions=rec.scq_ctx["per_agent"], edge_ids=obs["edge_ids"], edges=edges,
            logits=rec.scq_ctx["logits"], temperature=temp, scq_m=scq_m, r_actual=rec.reward,
            selection=scq_select, generator=generator)
        calls += scq.counterfactual_calls
        dups += scq.duplicate_topology_count
        with torch.no_grad():
            for active_cf, delta_R in scq.targets:
                if t < n - 1:    # bootstrap: V(s̃_{t+1}) = Q(next obs with prev=cf, next actual action)
                    cf_topo = [obs["edge_ids"][i] for i in active_cf]
                    next_obs_cf = scene.observation(t + 1, cf_topo)
                    v_cf_next = float(_critic_value(critic, next_obs_cf, records[t + 1].active, mean, std))
                    dy = float(delta_R) + scene.gamma * (records[t + 1].value - v_cf_next)
                else:
                    dy = float(delta_R)
                targets.append((t, scq.actual_active, active_cf, dy))
    return targets, calls, dups


def dynamic_scq_loss(critic, records, targets, mean, std):
    """L_SCQ = mean[(Q(s_t,S_t) − Q(s_t,S̃_i,S_{-i})) − Δy_i]^2 over the cached targets, recomputed
    grad-on each critic epoch (the Δy supervision is fixed). Enters ONLY the critic loss (D10)."""
    if not targets:
        return torch.zeros(()), 0.0
    res = []
    for (t, actual_active, active_cf, dy) in targets:
        obs = records[t].obs
        q_act = _critic_value(critic, obs, actual_active, mean, std)
        q_cf = _critic_value(critic, obs, active_cf, mean, std)
        res.append((q_act - q_cf) - dy)
    r = torch.stack(res)
    return (r ** 2).mean(), float(r.abs().mean().detach())


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


def _bcsp_teacher_trajectory(scene, mean, std, *, reward_of, ref_energy, lam_c, lam_b, beta, reward_mode):
    """Decoder-aware teacher (D6): per frame, the per-agent BCSP proposal subsets S_i^teacher (each
    node's incident edges in the myopic-greedy teacher topology, CAPPED at its budget b_i so |S_i|<=b_i
    is in the BCSP support -- subset_logp is -inf otherwise) and the topology the local mutual decoder
    RECONSTRUCTS from them. The teacher prev = its own reconstructed topology. Returns per-frame dicts
    {obs, proposals=[(node, incident_idxs, accepted_local, budget)], recon, teacher}. Training-only;
    the teacher uses the candidate evaluator exactly like the BCE teacher."""
    out = []
    prev: list[str] = []
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
        edge_ids = obs["edge_ids"]
        budgets, edges = _budgets_edges(ctx)
        incident = incident_index(edge_ids, edges)
        tset = set(best)
        proposals = []
        accept: dict = {}
        for node, idxs in incident.items():
            b = int(budgets.get(node, 0))
            idxs_t = tuple(idxs)
            teacher_local = tuple(k for k, gi in enumerate(idxs_t) if edge_ids[gi] in tset)[:b]
            proposals.append((node, idxs_t, teacher_local, b))
            accept[node] = {idxs_t[k] for k in teacher_local}
        recon = [edge_ids[i] for i, eid in enumerate(edge_ids)
                 if i in accept.get(edges[eid][0], ()) and i in accept.get(edges[eid][1], ())]
        out.append({"obs": obs, "proposals": proposals, "recon": recon, "teacher": list(best)})
        prev = list(recon)
    return out


def _teacher_subset_nll(actor, traj, mean, std, *, recurrent, temp):
    """Mean decoder-aware teacher-subset NLL over a trajectory under the actor's CURRENT logits."""
    hidden = None
    losses = []
    for fr in traj:
        nf_s, ef_s = _standardize(fr["obs"]["nf"], fr["obs"]["ef"], mean, std)
        logits, h_next = actor(nf_s, ef_s, fr["obs"]["ei"], hidden=hidden)
        node_nll = [-recompute_bcsp_logp(logits, inc, acc, temp, b)
                    for (_node, inc, acc, b) in fr["proposals"] if inc]
        if node_nll:
            losses.append(torch.stack(node_nll).mean())
        hidden = h_next if recurrent else None
    if not losses:
        return torch.zeros(())
    return torch.stack(losses).mean()


def bcsp_teacher_anchor_loss(actor, scene, traj, mean, std, *, recurrent, temp):
    """The annealed BC anchor added to the PPO actor loss (and, negated, the teacher subset logp): the
    mean decoder-aware teacher-subset NLL. Keeps the actor near the feasible warm-start so PPO does not
    destroy it (Plan §8.2). ``scene`` is accepted for API symmetry (the trajectory carries the obs)."""
    return _teacher_subset_nll(actor, traj, mean, std, recurrent=recurrent, temp=temp)


def warmstart_actor_bcsp(actor, scenes, mean, std, *, recurrent, epochs, lr, temp, reward_of, ref_energy,
                         lam_c, lam_b, beta, reward_mode):
    """Decoder-aware warm-start (D6): maximize the BCSP likelihood of the teacher's per-agent proposal
    subsets (replaces the per-edge BCE, which ignored the budget cap + mutual-acceptance the deployed
    decoder uses). Returns the mean final-epoch teacher-subset NLL (a convergence signal)."""
    teachers = [_bcsp_teacher_trajectory(s, mean, std, reward_of=reward_of, ref_energy=ref_energy,
                                         lam_c=lam_c, lam_b=lam_b, beta=beta, reward_mode=reward_mode)
                for s in scenes]
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    last = 0.0
    for _ep in range(max(1, epochs)):
        tot, n = 0.0, 0
        for traj in teachers:
            loss = _teacher_subset_nll(actor, traj, mean, std, recurrent=recurrent, temp=temp)
            if not loss.requires_grad:
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
        last = tot / max(1, n)
    return last


def warmstart_critic(critic, scenes, mean, std, *, epochs, lr, reward_of, ref_energy,
                     lam_c, lam_b, beta, reward_mode):
    """Pretrain the per-frame critic to the teacher's discounted returns G_t^teacher (a value
    warm-start; training-only). Returns the mean final-epoch MSE to the teacher returns."""
    data = []
    for s in scenes:
        traj = _bcsp_teacher_trajectory(s, mean, std, reward_of=reward_of, ref_energy=ref_energy,
                                        lam_c=lam_c, lam_b=lam_b, beta=beta, reward_mode=reward_mode)
        e_ref, rewards, prev = None, [], []
        for fr in traj:
            obs, recon = fr["obs"], fr["recon"]
            if e_ref is None:
                e_ref = ref_energy(obs)
            base_r, _gc, _gb, _ok = reward_of(obs, list(recon), e_ref, lam_c, lam_b, beta, reward_mode)
            switches = len(frozenset(prev) ^ frozenset(recon)) if rewards else 0
            reconfig = (s.reconfig.e_edge + s.reconfig.l_edge) * switches
            rewards.append(s.hold_interval * base_r - reconfig)
            prev = recon
        rets = [0.0] * len(rewards)
        g = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            g = rewards[i] + s.gamma * g
            rets[i] = g
        data.append([(fr["obs"], rets[i]) for i, fr in enumerate(traj)])
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    last = 0.0
    for _ep in range(max(1, epochs)):
        tot, n = 0.0, 0
        for traj_data in data:
            preds = [critic_scene_value(critic, obs["nf"], obs["ef"], obs["ei"],
                                        node_mean=mean[0], node_std=std[0], edge_mean=mean[1], edge_std=std[1])
                     for obs, _gt in traj_data]
            targets = torch.tensor([gt for _obs, gt in traj_data])
            loss = (torch.stack(preds) - targets).pow(2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            tot += float(loss); n += 1
        last = tot / max(1, n)
    return last


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
    motion_features = bool(getattr(args, "motion_features", False))
    counterfactual = bool(getattr(args, "counterfactual", False))   # D9: per-agent COMA credit
    k_cf = int(getattr(args, "k_cf", 4))
    scq = bool(getattr(args, "scq", False))                         # D10: closed-form Q supervision
    scq_m = int(getattr(args, "scq_m", 2))
    scq_coef = float(getattr(args, "scq_coef", 0.5))
    scq_select = str(getattr(args, "scq_select", "simple"))
    scq_gen = torch.Generator().manual_seed(args.seed + 7) if scq else None

    def _mk(seed_off, count, tag):
        if count <= 0:
            return []
        return sample_dynamic_scenes(
            seed=args.seed * 1000 + seed_off, count=count, node_count_choices=tuple(args.dyn_nodes),
            regime=regime, num_frames=args.frames, dt_s=args.dt, speed_min_mps=args.speed_min,
            speed_max_mps=args.speed_max, reconfig=reconfig, hold_interval=args.hold_interval,
            gamma=args.gamma, tag=tag, motion_features=motion_features)
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
    critic = CentralizedGraphCritic(node_dim, edge_dim, hidden=args.critic_hidden,
                                    rounds=args.critic_rounds, critic_sees_action=counterfactual)
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
    last_scq_calls = 0
    temp = args.temp

    warmstart_bce = None
    warmstart_final = None
    warmstart_mode = str(getattr(args, "dyn_warmstart_mode", "bce"))   # bce (legacy) | bcsp (decoder-aware)
    bc_anchor = float(getattr(args, "dyn_bc_anchor", 0.0))             # annealed teacher-BC anchor in PPO
    n_critic_warm = int(getattr(args, "dyn_critic_warmstart", 0))
    n_warm = int(getattr(args, "dyn_warmstart", 0))
    # decoder-aware teacher trajectories (built once): used by the BCSP warm-start AND the BC anchor.
    train_teachers = None
    if (n_warm > 0 and warmstart_mode == "bcsp") or bc_anchor > 0.0:
        train_teachers = [_bcsp_teacher_trajectory(s, mean, std, reward_of=reward_of, ref_energy=ref_energy,
                                                   lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta,
                                                   reward_mode=args.reward_mode) for s in train_scenes]
    if n_warm > 0:
        # Supervised warm-start toward the per-frame myopic-greedy teacher BEFORE RL -- gives the
        # cold-start actor a feasible target (both arms get the SAME warm-start). Critic stays cold
        # unless --dyn-critic-warmstart. The teacher uses the candidate evaluator (training-only).
        wlr = getattr(args, "dyn_warmstart_lr", 0.0) or args.lr
        if warmstart_mode == "bcsp":
            warmstart_final = warmstart_actor_bcsp(
                actor, train_scenes, mean, std, recurrent=recurrent, epochs=n_warm, lr=wlr, temp=temp,
                reward_of=reward_of, ref_energy=ref_energy, lam_c=args.lam_c, lam_b=args.lam_b,
                beta=args.beta, reward_mode=args.reward_mode)
            print(f"[dynamic:{args.dynamic_actor}] decoder-aware warm-start {n_warm} ep -> teacher NLL "
                  f"{warmstart_final:.4f}", flush=True)
        else:
            warmstart_bce = warmstart_actor(
                actor, train_scenes, mean, std, recurrent=recurrent, epochs=n_warm, lr=wlr,
                reward_of=reward_of, ref_energy=ref_energy,
                lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta, reward_mode=args.reward_mode)
            warmstart_final = warmstart_bce
            print(f"[dynamic:{args.dynamic_actor}] warm-start {n_warm} ep -> final BCE {warmstart_bce:.4f}",
                  flush=True)
    critic_warm_mse = None
    if n_critic_warm > 0:
        critic_warm_mse = warmstart_critic(
            critic, train_scenes, mean, std, epochs=n_critic_warm, lr=args.critic_lr,
            reward_of=reward_of, ref_energy=ref_energy, lam_c=args.lam_c, lam_b=args.lam_b,
            beta=args.beta, reward_mode=args.reward_mode)
        print(f"[dynamic:{args.dynamic_actor}] critic warm-start {n_critic_warm} ep -> teacher-return MSE "
              f"{critic_warm_mse:.4f}", flush=True)

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
        episode_erefs = []
        for scene in train_scenes:
            recs, _eref = episode_rollout(
                actor, critic, scene, mean, std, recurrent=recurrent, temp=temp, reward_of=reward_of,
                ref_energy=ref_energy, lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta,
                reward_mode=args.reward_mode, counterfactual=counterfactual, k_cf=k_cf, scq=scq)
            if any(r.per_agent for r in recs):
                episodes.append((scene, recs))
                episode_erefs.append(_eref)
        if not episodes:
            continue
        # advantage per (scene, frame, agent): COMA per-agent A_{i,t} (D9) when counterfactual, else the
        # shared scene advantage A_t = G_t - V_t.
        adv_flat, logp_old_flat = [], []
        for _scene, recs in episodes:
            for rec in recs:
                a_shared = rec.ret - rec.value
                for k, _pa in enumerate(rec.per_agent):
                    adv_flat.append(rec.cf_adv[k] if (counterfactual and rec.cf_adv is not None) else a_shared)
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
            anchor_val = 0.0
            if bc_anchor > 0.0 and train_teachers is not None:
                # annealed decoder-aware BC anchor: keep the actor near the feasible warm-start so PPO
                # does not destroy it (Plan §8.2). lambda decays linearly to 0 over training.
                lam_bc = bc_anchor * max(0.0, 1.0 - update / max(1, args.updates))
                if lam_bc > 0.0:
                    anchor = torch.stack([_teacher_subset_nll(actor, tr, mean, std, recurrent=recurrent,
                                                              temp=temp) for tr in train_teachers]).mean()
                    loss = loss + lam_bc * anchor
                    anchor_val = float(anchor)
            opt.zero_grad(); loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
            opt.step()
            last = {"kl": float(info["approx_kl"]), "clip": float(info["clip_fraction"]),
                    "actor_loss": float(ppo_loss), "actor_gnorm": gnorm,
                    "entropy": float(torch.stack(ent_new).mean())}
            if last["kl"] > args.target_kl * 1.5:
                break

        # ---- critic regression Q/V(s_t) -> G_t (+ D10 SCQ supervision) ----
        c0 = torch.nn.utils.parameters_to_vector(critic.parameters()).detach().clone()
        rets = torch.tensor([rec.ret for _s, recs in episodes for rec in recs])
        # D10: build the dynamic SCQ targets ONCE per update (pays the evaluator budget once -- NOT
        # budget-neutral); re-applied each critic epoch. Δy = ΔR + γ(V(s_{t+1}) − V(s̃_{t+1})).
        scq_episodes, scq_calls = [], 0
        if scq:
            for (scene, recs), e_ref in zip(episodes, episode_erefs):
                tgts, calls, _dups = dynamic_scq_targets(
                    recs, scene, critic, mean, std, reward_of=reward_of, ref_energy=ref_energy,
                    e_ref=e_ref, lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta,
                    reward_mode=args.reward_mode, temp=temp, scq_m=scq_m, scq_select=scq_select,
                    generator=scq_gen)
                scq_episodes.append((recs, tgts)); scq_calls += calls
        last_closs, last_cgn, ev, last_scq = 0.0, 0.0, 0.0, 0.0
        for _ in range(args.ppo_epochs):
            v_all = []
            for scene, recs in episodes:
                _lp, _en, vp = _reroll_logp(actor, critic, scene, recs, mean, std,
                                            recurrent=recurrent, temp=temp, need_logp=False)
                v_all.extend(vp)
            v_pred = torch.stack(v_all)
            v_loss = args.critic_coef * (rets - v_pred).pow(2).mean()
            if scq and scq_episodes:
                scq_terms = [dynamic_scq_loss(critic, recs, tgts, mean, std)[0] for recs, tgts in scq_episodes]
                scq_l = torch.stack(scq_terms).mean()
                v_loss = v_loss + scq_coef * scq_l
                last_scq = float(scq_l)
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
                   "scq_loss": round(last_scq, 6), "scq_evaluator_calls": scq_calls if scq else 0,
                   "val_per_frame_feasibility": (round(val["per_frame_feasibility"], 4) if do_val else None),
                   "val_mean_episode_return": (round(val_score, 4) if val_score is not None else None)}
        history.append(rec_log)
        if scq:
            last_scq_calls = scq_calls
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
                         "motion_features": motion_features,
                         "motion_node_features": (["velocity_x", "velocity_y", "speed", "heading_sin",
                                                   "heading_cos"] if motion_features else []),
                         "motion_edge_features": (["relative_velocity_along_link", "distance_delta",
                                                   "csi_delta", "csi_age"] if motion_features else []),
                         "mobility_speed_mps": [args.speed_min, args.speed_max], "dt_s": float(args.dt)},
        "actor": {"model_id": actor.model_id, "cross_frame_recurrence": bool(recurrent),
                  "arm": args.dynamic_actor, "warmstart_epochs": n_warm,
                  "warmstart_mode": warmstart_mode,            # bce (legacy) | bcsp (decoder-aware, D6)
                  "warmstart_final_bce": warmstart_bce,
                  "warmstart_final_metric": warmstart_final,
                  "bc_anchor_lambda": bc_anchor,               # annealed teacher-BC anchor in PPO (D6)
                  "teacher_source": "myopic-greedy over canonical candidate variants",
                  "teacher_uses_evaluator": True, "teacher_uses_held": False},
        "critic": {"enabled": True, "per_frame_state_value": not counterfactual,
                   "critic_sees_action": counterfactual,        # D9: action-conditioned Q critic
                   "counterfactual": counterfactual, "k_cf": (k_cf if counterfactual else None),
                   "counterfactual_budget_neutral": True,       # CF Q-evals are critic forwards, no evaluator call
                   "scq": scq, "scq_m": (scq_m if scq else None), "scq_coef": (scq_coef if scq else None),
                   "scq_select": (scq_select if scq else None),
                   "scq_budget_neutral": False,                 # D10: SCQ spends extra evaluator calls/scene
                   "scq_evaluator_calls_per_update": (last_scq_calls if scq else 0),
                   "warmstart_epochs": n_critic_warm, "warmstart_teacher_return_mse": critic_warm_mse},
        "action": {"distribution": "bcsp",
                   "rollout_sampler": "sample_decentralized_bcsp_action (stochastic; MAP == deploy)",
                   "eval_decoder": "local_mutual_assemble (deployed, torch-free)",
                   "per_agent_ratio": True},
        "regime": "operating_point urban v2x_37885 shadowing nlosv relay-3 backhaul coverage-gated",
    }
    # post-RL drift (Contract §10.3): how much RL moved the warm-start-alone HELD return.
    post_rl_drift = None
    if warmstart_held is not None:
        post_rl_drift = round(float(held["mean_episode_return"]) - float(warmstart_held["mean_episode_return"]), 5)
    result = {
        "arm": args.dynamic_actor, "seed": args.seed,
        "warmstart_epochs": n_warm, "warmstart_mode": warmstart_mode, "bc_anchor_lambda": bc_anchor,
        "critic_warmstart_epochs": n_critic_warm,
        "counterfactual": counterfactual, "scq": scq,
        "scq_evaluator_calls_per_update": (last_scq_calls if scq else 0),
        "teacher": {"source": "myopic-greedy over canonical candidate variants",
                    "uses_evaluator": True, "uses_held": False},
        "warmstart_held": warmstart_held, "warmstart_alone_return": (
            warmstart_held["mean_episode_return"] if warmstart_held is not None else None),
        "post_rl_drift_held_return": post_rl_drift,   # held_return - warmstart_alone_return (RL effect)
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
