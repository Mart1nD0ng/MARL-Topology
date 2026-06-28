"""Q9 PART 2 (POMDP-QP-FAR): residual + PBRS end-to-end training (the residual arm's trainer).

Trains a residual policy (Q6 action space, Q9-PART1 sampler) by REINFORCE with the PBRS shaping
F_t = lam(gamma*Phi_{t+1} - Phi_t), Phi(s_t) = -D_quorum(x_{t-1}, g_t), terminal Phi_T = 0 (Spec S9). The
shaped reward is used ONLY in training; the EVALUATION uses NO shaping (true closed-form PBFT C/E/L). The
deployed residual action is the MAP decode around the deployable local_hysteresis anchor (0 eval calls).
Reports the honest A/B vs the anchor: retention / feasibility / energy / latency / switches. A deployable
WIN is NOT assumed (bounded by the Q7/Q8 central ceilings; heed the Q5 RL-instability).
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

from marl_topology.models.dynamic_pna_actor import DynamicPNAActor  # noqa: E402
from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor  # noqa: E402
from marl_topology.training.decentralized_distillation import feature_standardization  # noqa: E402
from marl_topology.training.dynamic_baselines import local_hysteresis_action  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges, _standardize  # noqa: E402
from marl_topology.training.potential_shaping import dquorum_potential, episode_pbrs  # noqa: E402
from marl_topology.training.quorum_deficit_bridge import topology_reliability  # noqa: E402
from marl_topology.training.residual_action import residual_decode_from_flips, sample_residual  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

_KT, _AT = 0.4, 0.6
_LAM_C, _LAM_B, _BETA = 1.0, 1.0, 0.1


def make_actor(arch, node_dim, edge_dim, hidden):
    """Q11: the residual actor architecture -- mlp (DynamicRecurrentActor) or pna (DynamicPNAActor,
    directional message passing + PNA aggregation + omega-preference; a signature-compatible drop-in)."""
    if arch == "pna":
        return DynamicPNAActor(node_dim, edge_dim, hidden=hidden)
    return DynamicRecurrentActor(node_dim, edge_dim, hidden=hidden)


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


def rollout_residual_train(actor, scene, mean, std, T, *, mode, residual_prior, lam_pbrs, gamma,
                           use_pbrs, generator):
    """One TRAINING episode of the residual policy. Returns (logps, shaped_rewards, info). The shaped
    reward s_t = r_t + F_t (PBRS, training-only when use_pbrs)."""
    prev: list = []
    logps, rewards, potentials = [], [], []
    flip_pen = torch.zeros(())          # anchor trust-region: expected #flips (differentiable)
    e_ref = None
    for t in range(scene.n_frames):
        obs = scene.observation(t, prev)
        ev = obs["context"].evaluator
        if e_ref is None:
            e_ref = T._ref_energy(obs)
        potentials.append(dquorum_potential(ev, prev) if use_pbrs else 0.0)   # Phi(s_t) = -D_quorum(x_{t-1})
        nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
        logits, _ = actor(nf_s, ef_s, obs["ei"], hidden=None)
        z = logits.reshape(-1) + float(residual_prior)
        anchor = _anchor(obs, prev)
        s = sample_residual(z, anchor, obs["edge_ids"], obs["context"], mode=mode, generator=generator)
        flip_pen = flip_pen + (torch.sigmoid(z) * s["candidate_mask"]).sum()   # penalize deviating from anchor
        r, _gc, _gb, _ok = T.reward_of(obs, list(s["topology"]), e_ref, _LAM_C, _LAM_B, _BETA, "dense")
        logps.append(s["logp"]); rewards.append(float(r))
        prev = list(s["topology"])
    shaping = episode_pbrs(potentials, gamma, lam_pbrs) if use_pbrs else [0.0] * scene.n_frames
    shaped = [rewards[t] + shaping[t] for t in range(scene.n_frames)]
    return logps, shaped, {"rewards": rewards, "shaping": shaping, "flip_penalty": flip_pen}


def reinforce_update(actor, opt, scenes, mean, std, T, *, mode, residual_prior, lam_pbrs, gamma,
                     use_pbrs, baseline, generator, anchor_reg=0.0):
    """One REINFORCE update: loss = -sum_t logp_t*(G_t - baseline) + anchor_reg*mean(expected #flips).
    ``anchor_reg`` is the anchor trust-region that keeps the residual policy near the anchor (prevents the
    Q5-style instability of walking off the manifold)."""
    losses = []
    flip_pens = []
    returns = []
    for sc in scenes:
        logps, shaped, info = rollout_residual_train(
            actor, sc, mean, std, T, mode=mode, residual_prior=residual_prior, lam_pbrs=lam_pbrs,
            gamma=gamma, use_pbrs=use_pbrs, generator=generator)
        G = [0.0] * len(shaped)
        acc = 0.0
        for t in reversed(range(len(shaped))):
            acc = shaped[t] + gamma * acc
            G[t] = acc
        returns.append(G[0])
        for t in range(len(shaped)):
            losses.append(-logps[t] * (G[t] - baseline))
        flip_pens.append(info["flip_penalty"])
    if not losses:
        return baseline, 0.0
    loss = torch.stack(losses).mean()
    if anchor_reg > 0.0 and flip_pens:
        loss = loss + float(anchor_reg) * torch.stack(flip_pens).mean()
    opt.zero_grad(); loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))   # pre-clip total norm
    opt.step()
    new_baseline = 0.9 * baseline + 0.1 * (sum(returns) / len(returns))
    return new_baseline, float(loss), grad_norm


def eval_residual(actor, scenes, mean, std, T, *, mode, residual_prior, tau=0.9):
    """Deployed-style eval (NO shaping): MAP residual decode around the anchor; report TRUE C/E/L,
    retention (anchor edges kept), switches, vs the anchor's own metrics."""
    feas = anchor_feas = nfr = 0
    energy = latency = a_energy = a_latency = ret = switch = 0.0
    for sc in scenes:
        prev: list = []
        a_prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            ev = obs["context"].evaluator
            anchor = _anchor(obs, prev)
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            with torch.no_grad():
                logits, _ = actor(nf_s, ef_s, obs["ei"], hidden=None)
            z = (logits.reshape(-1) + float(residual_prior))
            # MAP flip decision (same convention as the sampler: flip iff sigmoid(z) > 0.5 <=> z > 0).
            flipped = [obs["edge_ids"][i] for i in range(z.numel()) if float(z[i]) > 0.0]
            _acc, topo = residual_decode_from_flips(anchor, flipped, z.tolist(), obs["edge_ids"],
                                                    obs["context"], mode=mode)
            m = topology_reliability(ev, topo)                                   # TRUE C/E/L, NO shaping
            feas += int(m["consensus"] >= tau); energy += m["energy"]; latency += m["latency"]
            ret += (len(set(anchor) & set(topo)) / max(1, len(anchor)))
            if t > 0:
                switch += len(frozenset(prev) ^ frozenset(topo))
            # the anchor's own deployed metrics (rolled with its own prev)
            obs_a = sc.observation(t, a_prev)
            anchor_a = _anchor(obs_a, a_prev)
            ma = topology_reliability(obs_a["context"].evaluator, anchor_a)
            anchor_feas += int(ma["consensus"] >= tau); a_energy += ma["energy"]; a_latency += ma["latency"]
            nfr += 1
            prev = list(topo); a_prev = list(anchor_a)
    n = max(1, nfr)
    return {"residual_feasibility": round(feas / n, 4), "anchor_feasibility": round(anchor_feas / n, 4),
            "residual_energy": round(energy / n, 6), "anchor_energy": round(a_energy / n, 6),
            "residual_latency": round(latency / n, 6), "anchor_latency": round(a_latency / n, 6),
            "retention": round(ret / n, 4), "switches_per_frame": round(switch / n, 4), "n_frames": nfr}


def _build(data, seed, count, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=2.0,
                  speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                  hold_interval=4, gamma=args.gamma)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--actor", choices=["mlp", "pna"], default="mlp")
    ap.add_argument("--mode", choices=["add", "full"], default="full")
    ap.add_argument("--residual-prior", type=float, default=-3.0)
    ap.add_argument("--lam-pbrs", type=float, default=0.5)
    ap.add_argument("--anchor-reg", type=float, default=0.0, help="anchor trust-region (penalize flips)")
    ap.add_argument("--pbrs", action="store_true", help="enable PBRS shaping (training-only)")
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4909)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "residual_pbrs_train.json"))
    args = ap.parse_args()

    T = _load_trunk()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)
    report = {"scope": "Q9 PART 2 residual+PBRS end-to-end training (eval NO shaping; final metrics true C/E/L)",
              "activation": {"actor": args.actor, "residual_mode": args.mode, "residual_prior": args.residual_prior,
                             "pbrs_enabled": bool(args.pbrs), "potential": "-D_quorum(x_{t-1}, g_t)",
                             "terminal_potential_zero": True, "lam_pbrs": args.lam_pbrs,
                             "anchor_reg": args.anchor_reg,
                             "eval_uses_shaping": False, "final_metric": "true closed-form PBFT C/E/L"},
              "config": {"train": args.train, "held": args.held, "frames": args.frames,
                         "updates": args.updates, "seed": args.seed}, "by_data": {}}
    for data in args.data:
        train = _build(data, args.seed * 1000 + 1, args.train, args)
        held = _build(data, args.seed * 1000 + 777, args.held, args)
        stat = [s.observation(0, []) for s in train]
        mean, std = feature_standardization(stat)
        nd, ed = stat[0]["nf"].shape[1], stat[0]["ef"].shape[1]
        actor = make_actor(args.actor, nd, ed, args.hidden)
        n_params = sum(p.numel() for p in actor.parameters())
        opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
        baseline = 0.0
        last_loss = 0.0
        grad_norms = []
        diverged = False
        for _u in range(args.updates):
            baseline, last_loss, gn = reinforce_update(
                actor, opt, train, mean, std, T, mode=args.mode, residual_prior=args.residual_prior,
                lam_pbrs=args.lam_pbrs, gamma=args.gamma, use_pbrs=bool(args.pbrs), baseline=baseline,
                generator=gen, anchor_reg=args.anchor_reg)
            grad_norms.append(gn)
            if not (last_loss == last_loss) or not (gn == gn):    # NaN -> diverged (the Q5 failure mode)
                diverged = True
                break
        ev = eval_residual(actor, held, mean, std, T, mode=args.mode, residual_prior=args.residual_prior)
        row = {"actor": args.actor, "n_params": n_params, "diverged": diverged,
               "final_loss": round(last_loss, 4), "baseline_return": round(baseline, 4),
               "grad_norm_mean": round(sum(grad_norms) / max(1, len(grad_norms)), 4),
               "grad_norm_max": round(max(grad_norms) if grad_norms else 0.0, 4), **ev}
        report["by_data"][data] = row
        print(f"[{data}] actor={args.actor} params={row['n_params']} diverged={diverged} "
              f"resid_feas={ev['residual_feasibility']} anchor_feas={ev['anchor_feasibility']} "
              f"retention={ev['retention']} grad_norm_mean={row['grad_norm_mean']} "
              f"grad_norm_max={row['grad_norm_max']} sw={ev['switches_per_frame']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
