"""R3 (Belief-Guided Residual PPO): the residual PPO trainer + CTDE value critic (the training-stability chain).

Replaces the Q9/Q14 REINFORCE residual trainer's optimizer with GENUINE residual PPO: per-edge (per-agent)
PPO clip via graph_mappo.ppo_clip_actor_loss (NOT the trunk's PPO -- this trainer calls it on the residual
path), approx_kl / clip_fraction logging, target_kl early-stop, a CTDE value critic (A_t = G_t - V_phi; the
critic reads a training-only global state summary incl. a true-CSI summary, NEVER used at deployment),
entropy bonus, the R1 raw-logit L2 on the unsaturated residual head, and an ablatable L_CSI (R2 no-op,
interface kept). Eval (deployed-style MAP decode) reports edit_rate / retention / residual vs anchor feas.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: E402
from marl_topology.training import graph_mappo  # noqa: E402  (ppo_clip_actor_loss lives here; spy target)
from marl_topology.training.dynamic_baselines import local_hysteresis_action  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges, _standardize  # noqa: E402
from marl_topology.training.residual_action import (residual_decode_from_flips,  # noqa: E402
                                                    residual_logp_per_edge, sample_residual)
from marl_topology.training.residual_saturation import (feature_standardization_all_frames,  # noqa: E402
                                                        raw_logit_l2_penalty)
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

_KT, _AT = 0.4, 0.6
_LAM_C, _LAM_B, _BETA = 1.0, 1.0, 0.1


class ResidualValueCritic(nn.Module):
    """CTDE value critic V_phi(s_t) over a global state summary (training-only; not used at deployment)."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64) -> None:
        super().__init__()
        # LayerNorm bounds the pooled global state (a near-constant standardized feature can hit the std-
        # clamp and explode under mean-pooling) so the value head stays O(1) -> well-conditioned EV.
        self.net = nn.Sequential(nn.LayerNorm(node_dim + edge_dim + 1),
                                 nn.Linear(node_dim + edge_dim + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _anchor(obs, prev):
    budgets, edges = _budgets_edges(obs["context"])
    return local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                   keep_threshold=_KT, add_threshold=_AT)


def _global_state(sc, t, obs, nf_s, ef_s):
    """CTDE global summary: pooled standardized node+edge features + a TRUE-current-CSI summary (training-
    only)."""
    recs = sc.context(t).link_records
    true_p = torch.tensor([float(recs[eid].link_success_probability) for eid in obs["edge_ids"]]).mean()
    return torch.cat([nf_s.mean(0), ef_s.mean(0), true_p.reshape(1)])


def _entropy(z, mask):
    p = torch.sigmoid(z)
    ent = -(p * F.logsigmoid(z) + (1.0 - p) * F.logsigmoid(-z))
    return (ent * mask).sum() / mask.sum().clamp_min(1.0)


def _build(data, seed, count, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(20.0), num_frames=args.frames, dt_s=2.0, speed_min_mps=15.0,
                  speed_max_mps=30.0, reconfig=ReconfigCost(), hold_interval=4, gamma=args.gamma)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def train_residual_ppo(train_scenes, *, epochs=15, ppo_epochs=4, clip_eps=0.2, target_kl=0.05,
                       entropy_coef=0.01, raw_l2=0.02, lr=0.02, critic_lr=0.005, hidden=64, seed=0,
                       residual_prior=-1.0, gamma=0.95, T=None) -> dict:
    if T is None:
        T = _load_trunk()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden, residual_logit_scale=3.0)
    critic = ResidualValueCritic(nd, ed, hidden=hidden)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    approx_kl_hist, clip_fracs, inner_ran, ent_log, vloss_log, ev_log = [], [], [], [], [], []
    ppo_calls = 0
    critic_delta = 0.0
    diverged = False

    for _ep in range(epochs):
        # ---- rollout (no grad) ----
        episodes = []
        for sc in train_scenes:
            prev, e_ref, frames = [], None, []
            for t in range(sc.n_frames):
                obs = sc.observation(t, prev)
                if e_ref is None:
                    e_ref = T._ref_energy(obs)
                nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
                with torch.no_grad():
                    z_raw, _raw, _h = actor(nf_s, ef_s, obs["ei"], hidden=None)
                z = z_raw + float(residual_prior)
                anchor = _anchor(obs, prev)
                s = sample_residual(z, anchor, obs["edge_ids"], obs["context"], mode="full", generator=gen)
                r, _gc, _gb, _ok = T.reward_of(obs, list(s["topology"]), e_ref, _LAM_C, _LAM_B, _BETA, "dense")
                state = _global_state(sc, t, obs, nf_s, ef_s)
                frames.append({"nf_s": nf_s, "ef_s": ef_s, "ei": obs["ei"], "decisions": s["decisions"],
                               "mask": s["candidate_mask"], "logp_old": residual_logp_per_edge(
                                   z, s["decisions"], s["candidate_mask"]).detach(), "reward": float(r),
                               "state": state})
                prev = list(s["topology"])
            G, acc = [0.0] * len(frames), 0.0
            for t in reversed(range(len(frames))):
                acc = frames[t]["reward"] + gamma * acc
                G[t] = acc
            for t, fr in enumerate(frames):
                fr["G"] = G[t]
            episodes.append(frames)
        flat = [fr for ep in episodes for fr in ep]
        # standardize the critic TARGET -> O(1) value targets -> well-conditioned EV (raw returns are large
        # and low-variance on infeasible-heavy random, which makes EV ill-conditioned).
        G_all = torch.tensor([fr["G"] for fr in flat])
        G_norm = (G_all - G_all.mean()) / (G_all.std() + 1e-8)
        for i, fr in enumerate(flat):
            fr["G_norm"] = float(G_norm[i])
            with torch.no_grad():
                fr["v_old"] = float(critic(fr["state"].unsqueeze(0)))
            fr["adv"] = fr["G_norm"] - fr["v_old"]
        adv_all = torch.tensor([fr["adv"] for fr in flat])
        adv_norm = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        for i, fr in enumerate(flat):
            fr["adv_n"] = float(adv_norm[i])

        # ---- PPO inner epochs (target_kl early-stop) ----
        c_before = [p.detach().clone() for p in critic.parameters()]
        ran = 0
        for _pe in range(ppo_epochs):
            lp_new, lp_old, adv_rep, ent_terms, raw_terms, vpred, vtgt = [], [], [], [], [], [], []
            for fr in flat:
                z_new = actor(fr["nf_s"], fr["ef_s"], fr["ei"], hidden=None)[0] + float(residual_prior)
                pe = residual_logp_per_edge(z_new, fr["decisions"], fr["mask"])
                cand = fr["mask"] > 0
                if cand.any():
                    lp_new.append(pe[cand]); lp_old.append(fr["logp_old"][cand])
                    adv_rep.append(torch.full((int(cand.sum()),), fr["adv_n"]))
                ent_terms.append(_entropy(z_new, fr["mask"]))
                raw_terms.append(raw_logit_l2_penalty(actor(fr["nf_s"], fr["ef_s"], fr["ei"], hidden=None)[1]))
                vpred.append(critic(fr["state"].unsqueeze(0))); vtgt.append(fr["G_norm"])
            if not lp_new:
                break
            logp_new = torch.cat(lp_new); logp_old = torch.cat(lp_old); adv = torch.cat(adv_rep)
            ppo_loss, info = graph_mappo.ppo_clip_actor_loss(logp_new, logp_old, adv, clip_eps=clip_eps)
            ppo_calls += 1
            entropy = torch.stack(ent_terms).mean()
            raw_pen = torch.stack(raw_terms).mean()
            actor_loss = ppo_loss - float(entropy_coef) * entropy + float(raw_l2) * raw_pen
            opt_a.zero_grad(); actor_loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
            opt_a.step()
            vp = torch.cat([v.reshape(1) for v in vpred]); vt = torch.tensor(vtgt)
            value_loss = F.smooth_l1_loss(vp, vt)              # Huber: robust to large target outliers
            opt_c.zero_grad(); value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_c.step()
            ak = float(info["approx_kl"])
            approx_kl_hist.append(ak); clip_fracs.append(float(info["clip_fraction"]))
            ent_log.append(float(entropy)); vloss_log.append(float(value_loss))
            ev_log.append(1.0 - float((vt - vp.detach()).var() / (vt.var() + 1e-8)))
            ran += 1
            if not (ak == ak) or not (gn == gn):
                diverged = True; break
            if ak > float(target_kl):
                break
        inner_ran.append(ran)
        critic_delta += float(sum((p.detach() - b).pow(2).sum() for p, b in zip(critic.parameters(), c_before)) ** 0.5)
        if diverged:
            break

    ev = _eval_residual(actor, train_scenes, mean, std, T, residual_prior=residual_prior)
    return {"ppo_calls": ppo_calls, "approx_kl": approx_kl_hist[-1] if approx_kl_hist else 0.0,
            "approx_kl_history": approx_kl_hist, "clip_fraction": clip_fracs[-1] if clip_fracs else 0.0,
            "inner_epochs_ran": inner_ran, "critic_parameter_delta": round(critic_delta, 6),
            "value_loss": round(vloss_log[-1], 6) if vloss_log else 0.0,
            "explained_variance": round(ev_log[-1], 4) if ev_log else 0.0,
            "entropy": round(ent_log[-1], 6) if ent_log else 0.0, "entropy_coef": entropy_coef,
            "diverged": diverged, **ev}


def train_residual_reinforce(train_scenes, *, epochs=15, lr=0.02, hidden=64, seed=0, residual_prior=-1.0,
                             gamma=0.95, T=None) -> dict:
    """A/B control: the SAME BeliefResidualActor + rollout but a FREE REINFORCE optimizer (joint logp, scalar
    moving baseline, NO PPO/critic/KL/flip-penalty) -- isolates the TRAINER as the variable vs PPO. Expected
    to clamp-or-collapse (the Q14/R2 free-REINFORCE failure mode)."""
    if T is None:
        T = _load_trunk()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden, residual_logit_scale=3.0)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    baseline = 0.0
    diverged = False
    for _ep in range(epochs):
        losses, returns = [], []
        for sc in train_scenes:
            prev, e_ref, logps, rewards = [], None, [], []
            for t in range(sc.n_frames):
                obs = sc.observation(t, prev)
                if e_ref is None:
                    e_ref = T._ref_energy(obs)
                nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
                z = actor(nf_s, ef_s, obs["ei"], hidden=None)[0] + float(residual_prior)
                anchor = _anchor(obs, prev)
                s = sample_residual(z, anchor, obs["edge_ids"], obs["context"], mode="full", generator=gen)
                r, _gc, _gb, _ok = T.reward_of(obs, list(s["topology"]), e_ref, _LAM_C, _LAM_B, _BETA, "dense")
                logps.append(s["logp"]); rewards.append(float(r)); prev = list(s["topology"])
            G, acc = [0.0] * len(rewards), 0.0
            for t in reversed(range(len(rewards))):
                acc = rewards[t] + gamma * acc; G[t] = acc
            returns.append(G[0])
            for t in range(len(rewards)):
                losses.append(-logps[t] * (G[t] - baseline))
        if not losses:
            break
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
        opt.step()
        baseline = 0.9 * baseline + 0.1 * (sum(returns) / len(returns))
        if not (float(loss) == float(loss)) or not (gn == gn):
            diverged = True; break
    ev = _eval_residual(actor, train_scenes, mean, std, T, residual_prior=residual_prior)
    return {"trainer": "reinforce_free", "diverged": diverged, **ev}


def _eval_residual(actor, scenes, mean, std, T, *, residual_prior, tau=0.9):
    """Deployed-style MAP decode (flip iff z>0): residual vs anchor feasibility + edit_rate / retention."""
    from marl_topology.training.quorum_deficit_bridge import topology_reliability
    feas = afeas = nfr = 0
    ret = edits = zero_ep = all_ep = sw = 0.0
    for sc in scenes:
        prev, a_prev = [], []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            anchor = _anchor(obs, prev)
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            with torch.no_grad():
                z = actor(nf_s, ef_s, obs["ei"], hidden=None)[0] + float(residual_prior)
            flipped = [obs["edge_ids"][i] for i in range(z.numel()) if float(z[i]) > 0.0]
            _a, topo = residual_decode_from_flips(anchor, flipped, z.tolist(), obs["edge_ids"],
                                                  obs["context"], mode="full")
            m = topology_reliability(obs["context"].evaluator, topo)
            feas += int(m["consensus"] >= tau)
            ret += len(set(anchor) & set(topo)) / max(1, len(anchor))
            ed_frac = len(frozenset(anchor) ^ frozenset(topo)) / max(1, len(obs["edge_ids"]))
            edits += ed_frac; zero_ep += int(ed_frac == 0.0)
            obs_a = sc.observation(t, a_prev); anchor_a = _anchor(obs_a, a_prev)
            ma = topology_reliability(obs_a["context"].evaluator, anchor_a)
            afeas += int(ma["consensus"] >= tau)
            nfr += 1; prev = list(topo); a_prev = list(anchor_a)
    n = max(1, nfr)
    return {"residual_feasibility": round(feas / n, 4), "anchor_feasibility": round(afeas / n, 4),
            "edit_rate": round(edits / n, 4), "zero_edit_rate": round(zero_ep / n, 4),
            "retention": round(ret / n, 4), "n_frames": nfr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "belief_residual" / "R3" / "residual_ppo.json"))
    args = ap.parse_args()
    T = _load_trunk()
    report = {"scope": "R3 residual PPO + CTDE critic; eval MAP decode true PBFT feas", "by_data": {}}
    for data in args.data:
        train = _build(data, args.seed * 1000 + 1, args.train, args)
        m = train_residual_ppo(train, epochs=args.epochs, ppo_epochs=args.ppo_epochs,
                               target_kl=args.target_kl, seed=args.seed, T=T)
        report["by_data"][data] = m
        print(f"[{data}] ppo_calls={m['ppo_calls']} approx_kl={m['approx_kl']:.4f} clip_frac={m['clip_fraction']:.3f} "
              f"critic_dParam={m['critic_parameter_delta']:.4f} EV={m['explained_variance']:.3f} "
              f"entropy={m['entropy']:.3f} resid_feas={m['residual_feasibility']} anchor_feas={m['anchor_feasibility']} "
              f"edit_rate={m['edit_rate']} retention={m['retention']} diverged={m['diverged']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
