"""Joint experiment: stale-CSI POMDP + feasible-anchored residual learning (closing the open loop).

The campaign validated stale CSI only at the PERCEPTION layer (Q2) and ran the residual headline (Q9-Q12)
under CURRENT CSI. This script closes that loop by training/evaluating the residual policy UNDER stale CSI,
with a RECURRENT actor that carries hidden state across frames (so it can exploit temporal structure),
contrasted with the memoryless actor and the deployable anchor.

Two reported sections:
  1. ANCHOR SENSITIVITY (training-free): does stale CSI hurt the deployable local_hysteresis anchor?
     Roll the anchor under {current, delay-1, delay-2, partial} and report TRUE PBFT feasibility/C/E/L.
     The anchor acts on the STALE observed channel g_hat_t; the metric uses the TRUE current channel.
     If the anchor barely drops, there is nothing for temporal modeling to recover.
  2. RESIDUAL RECOVERY: under a stale mode, train the residual policy {memoryless, recurrent} (light
     anchor trust-region so it CAN deviate) and compare residual feasibility to the anchor's. If the
     recurrent residual beats the anchor under stale CSI, temporal modeling buys control gain.

Hard constraints preserved: deployed actor sees ONLY g_hat_t + age + motion + prev topo (never true
current CSI / critic / evaluator); the final metric is the true closed-form PBFT C/E/L on the current
channel; PBRS is training-only; eval uses NO shaping; the anchor is the deployable reference.
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
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import torch  # noqa: E402

import residual_pbrs_train as rp  # noqa: E402  (reuse the verified residual machinery)
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402
from marl_topology.training.decentralized_distillation import feature_standardization  # noqa: E402
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.potential_shaping import dquorum_potential, episode_pbrs  # noqa: E402
from marl_topology.training.quorum_deficit_bridge import topology_reliability  # noqa: E402
from marl_topology.training.residual_action import residual_decode_from_flips, sample_residual  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

TAU = 0.9


def _model_for(mode, seed):
    if mode == "current":
        return CsiObservationModel(mode="current")
    if mode == "delay1":
        return CsiObservationModel(mode="delay", delay_frames=1)
    if mode == "delay2":
        return CsiObservationModel(mode="delay", delay_frames=2)
    if mode == "partial":
        return CsiObservationModel(mode="partial", probe_probability=0.5, seed=seed)
    raise ValueError(mode)


def _build_stale(data, seed, count, csi_model, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(20.0), num_frames=args.frames, dt_s=2.0,
                  speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                  hold_interval=4, gamma=args.gamma, csi_observation_model=csi_model)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def anchor_only_eval(scenes, T, tau=TAU):
    """Training-free: roll the deployable anchor (acting on the stale observed CSI) and score with the
    TRUE PBFT metric on the current channel."""
    feas = nfr = 0
    energy = latency = 0.0
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            anchor = rp._anchor(obs, prev)
            m = topology_reliability(obs["context"].evaluator, anchor)
            feas += int(m["consensus"] >= tau); energy += m["energy"]; latency += m["latency"]
            nfr += 1
            prev = list(anchor)
    n = max(1, nfr)
    return {"anchor_feasibility": round(feas / n, 4), "anchor_energy": round(energy / n, 6),
            "anchor_latency": round(latency / n, 6), "n_frames": nfr}


# ---- recurrent variants of rp.rollout / rp.eval that CARRY hidden across frames -------------------

def rollout_rec(actor, scene, mean, std, T, *, mode, residual_prior, lam_pbrs, gamma, use_pbrs, generator,
                recurrent):
    prev: list = []
    logps, rewards, potentials = [], [], []
    flip_pen = torch.zeros(())
    e_ref = None
    h = None
    for t in range(scene.n_frames):
        obs = scene.observation(t, prev)
        ev = obs["context"].evaluator
        if e_ref is None:
            e_ref = T._ref_energy(obs)
        potentials.append(dquorum_potential(ev, prev) if use_pbrs else 0.0)
        nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
        logits, h_next = actor(nf_s, ef_s, obs["ei"], hidden=(h if recurrent else None))
        h = h_next if recurrent else None                        # carry hidden ONLY in the recurrent arm
        z = logits.reshape(-1) + float(residual_prior)
        anchor = rp._anchor(obs, prev)
        s = sample_residual(z, anchor, obs["edge_ids"], obs["context"], mode=mode, generator=generator)
        flip_pen = flip_pen + (torch.sigmoid(z) * s["candidate_mask"]).sum()
        r, _gc, _gb, _ok = T.reward_of(obs, list(s["topology"]), e_ref, rp._LAM_C, rp._LAM_B, rp._BETA, "dense")
        logps.append(s["logp"]); rewards.append(float(r))
        prev = list(s["topology"])
    shaping = episode_pbrs(potentials, gamma, lam_pbrs) if use_pbrs else [0.0] * scene.n_frames
    shaped = [rewards[t] + shaping[t] for t in range(scene.n_frames)]
    return logps, shaped, {"flip_penalty": flip_pen}


def update_rec(actor, opt, scenes, mean, std, T, *, mode, residual_prior, lam_pbrs, gamma, use_pbrs,
               baseline, generator, anchor_reg, recurrent):
    losses, flip_pens, returns = [], [], []
    for sc in scenes:
        logps, shaped, info = rollout_rec(actor, sc, mean, std, T, mode=mode, residual_prior=residual_prior,
                                          lam_pbrs=lam_pbrs, gamma=gamma, use_pbrs=use_pbrs,
                                          generator=generator, recurrent=recurrent)
        G = [0.0] * len(shaped); acc = 0.0
        for t in reversed(range(len(shaped))):
            acc = shaped[t] + gamma * acc; G[t] = acc
        returns.append(G[0])
        for t in range(len(shaped)):
            losses.append(-logps[t] * (G[t] - baseline))
        flip_pens.append(info["flip_penalty"])
    if not losses:
        return baseline, 0.0, 0.0
    loss = torch.stack(losses).mean()
    if anchor_reg > 0.0 and flip_pens:
        loss = loss + float(anchor_reg) * torch.stack(flip_pens).mean()
    opt.zero_grad(); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
    opt.step()
    return 0.9 * baseline + 0.1 * (sum(returns) / len(returns)), float(loss), gn


def eval_rec(actor, scenes, mean, std, T, *, mode, residual_prior, recurrent, tau=TAU):
    feas = anchor_feas = nfr = 0
    energy = a_energy = ret = switch = 0.0
    for sc in scenes:
        prev: list = []; a_prev: list = []; h = None
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            ev = obs["context"].evaluator
            anchor = rp._anchor(obs, prev)
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            with torch.no_grad():
                logits, h_next = actor(nf_s, ef_s, obs["ei"], hidden=(h if recurrent else None))
            h = h_next if recurrent else None
            z = logits.reshape(-1) + float(residual_prior)
            flipped = [obs["edge_ids"][i] for i in range(z.numel()) if float(z[i]) > 0.0]
            _acc, topo = residual_decode_from_flips(anchor, flipped, z.tolist(), obs["edge_ids"],
                                                    obs["context"], mode=mode)
            m = topology_reliability(ev, topo)
            feas += int(m["consensus"] >= tau); energy += m["energy"]
            ret += (len(set(anchor) & set(topo)) / max(1, len(anchor)))
            if t > 0:
                switch += len(frozenset(prev) ^ frozenset(topo))
            obs_a = sc.observation(t, a_prev); anchor_a = rp._anchor(obs_a, a_prev)
            ma = topology_reliability(obs_a["context"].evaluator, anchor_a)
            anchor_feas += int(ma["consensus"] >= tau); a_energy += ma["energy"]
            nfr += 1; prev = list(topo); a_prev = list(anchor_a)
    n = max(1, nfr)
    return {"residual_feasibility": round(feas / n, 4), "anchor_feasibility": round(anchor_feas / n, 4),
            "residual_energy": round(energy / n, 6), "anchor_energy": round(a_energy / n, 6),
            "retention": round(ret / n, 4), "switches_per_frame": round(switch / n, 4), "n_frames": nfr}


def train_one(data, csi_mode, recurrent, anchor_reg, seed, args, T):
    csi_train = _model_for(csi_mode, seed * 1000 + 1)
    csi_held = _model_for(csi_mode, seed * 1000 + 777)
    train = _build_stale(data, seed * 1000 + 1, args.train, csi_train, args)
    held = _build_stale(data, seed * 1000 + 777, args.held, csi_held, args)
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1)
    stat = [s.observation(0, []) for s in train]
    mean, std = feature_standardization(stat)
    nd, ed = stat[0]["nf"].shape[1], stat[0]["ef"].shape[1]
    actor = rp.make_actor("mlp", nd, ed, args.hidden)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    baseline = 0.0; diverged = False; gns = []
    for _u in range(args.updates):
        baseline, loss, gn = update_rec(actor, opt, train, mean, std, T, mode="full",
                                        residual_prior=args.residual_prior, lam_pbrs=args.lam_pbrs,
                                        gamma=args.gamma, use_pbrs=True, baseline=baseline, generator=gen,
                                        anchor_reg=anchor_reg, recurrent=recurrent)
        gns.append(gn)
        if not (loss == loss) or not (gn == gn):
            diverged = True; break
    ev = eval_rec(actor, held, mean, std, T, mode="full", residual_prior=args.residual_prior,
                  recurrent=recurrent)
    ev["diverged"] = diverged
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--section", choices=["sensitivity", "recovery", "both"], default="both")
    ap.add_argument("--recovery-modes", nargs="+", default=["current", "delay1"])
    ap.add_argument("--anchor-reg", type=float, default=0.1, help="LIGHT leash so the residual CAN deviate")
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--residual-prior", type=float, default=-3.0)
    ap.add_argument("--lam-pbrs", type=float, default=0.5)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "stale_csi_residual_joint.json"))
    args = ap.parse_args()
    T = rp._load_trunk()

    def agg(xs):
        xs = [x for x in xs if x == x]
        if not xs:
            return None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return {"mean": round(m, 4), "lo": round(m, 4), "hi": round(m, 4), "n": len(xs)}
        sd = (sum((v - m) ** 2 for v in xs) / (len(xs) - 1)) ** 0.5
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(xs), 2.776)
        h = t * sd / len(xs) ** 0.5
        return {"mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4), "n": len(xs)}

    report = {"scope": "joint stale-CSI + residual; deployed obs = g_hat_t only; metric = true PBFT C/E/L",
              "config": {"seeds": args.seeds, "anchor_reg": args.anchor_reg, "train": args.train,
                         "held": args.held, "updates": args.updates}, "sensitivity": {}, "recovery": {}}

    if args.section in ("sensitivity", "both"):
        print("=== SECTION 1: anchor sensitivity to stale CSI (training-free) ===")
        for data in args.data:
            report["sensitivity"][data] = {}
            for mode in ["current", "delay1", "delay2", "partial"]:
                feas, en, lat = [], [], []
                for seed in args.seeds:
                    sc = _build_stale(data, seed * 1000 + 777, args.held, _model_for(mode, seed * 1000 + 777), args)
                    r = anchor_only_eval(sc, T)
                    feas.append(r["anchor_feasibility"]); en.append(r["anchor_energy"]); lat.append(r["anchor_latency"])
                report["sensitivity"][data][mode] = {"anchor_feasibility": agg(feas),
                                                     "anchor_energy": agg(en), "anchor_latency": agg(lat)}
                f = report["sensitivity"][data][mode]["anchor_feasibility"]
                print(f"  [{data}] {mode:<8} anchor_feas={f['mean']:.3f} CI[{f['lo']:.3f},{f['hi']:.3f}]")

    if args.section in ("recovery", "both"):
        print("\n=== SECTION 2: residual recovery under stale CSI (memoryless vs recurrent) ===")
        for data in args.data:
            report["recovery"][data] = {}
            for mode in args.recovery_modes:
                for rec in [False, True]:
                    rfeas, afeas, rete, dvg = [], [], [], 0
                    for seed in args.seeds:
                        ev = train_one(data, mode, rec, args.anchor_reg, seed, args, T)
                        rfeas.append(ev["residual_feasibility"]); afeas.append(ev["anchor_feasibility"])
                        rete.append(ev["retention"]); dvg += int(ev["diverged"])
                    key = f"{mode}/{'recurrent' if rec else 'memoryless'}"
                    paired = [rfeas[i] - afeas[i] for i in range(len(rfeas))]
                    report["recovery"][data][key] = {
                        "residual_feasibility": agg(rfeas), "anchor_feasibility": agg(afeas),
                        "paired_residual_minus_anchor": agg(paired), "retention": agg(rete),
                        "diverged_seeds": dvg, "n_seeds": len(args.seeds)}
                    rr = report["recovery"][data][key]
                    print(f"  [{data}] {key:<20} resid={rr['residual_feasibility']['mean']:.3f} "
                          f"anchor={rr['anchor_feasibility']['mean']:.3f} "
                          f"paired={rr['paired_residual_minus_anchor']['mean']:+.3f} "
                          f"CI[{rr['paired_residual_minus_anchor']['lo']:+.3f},{rr['paired_residual_minus_anchor']['hi']:+.3f}] "
                          f"retention={rr['retention']['mean']:.2f} diverged={dvg}/{len(args.seeds)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
