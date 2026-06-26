"""D4 full-model dynamic headline: recurrent vs memoryless (+ a myopic-greedy reference).

Runs the --dynamic trunk arm (subprocess, clean stdout/stderr per run) for each (arm, seed), reads the
per-run dynamic_result.json, and reports per-seed held metrics + paired CIs. Also computes a MYOPIC-
GREEDY reference: at each held frame pick the per-frame best candidate topology (reward-greedy, RECON-
BLIND) and pay the reconfiguration cost -- the reactive baseline the recurrent actor must beat to show
temporal value. Pre-registered expectation (from D2): with full per-frame CSI the task is ~Markov in
(channel, prev topology) -> recurrent ~= memoryless; both should approach the myopic reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def _ci(xs):
    if not xs:
        return None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"mean": round(m, 5), "ci95": [round(m, 5), round(m, 5)], "n": len(xs), "seeds": xs}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    se = (var / len(xs)) ** 0.5
    t = _T975.get(len(xs) - 1, 1.96)
    return {"mean": round(m, 5), "ci95": [round(m - t * se, 5), round(m + t * se, 5)],
            "n": len(xs), "seeds": [round(x, 5) for x in xs]}


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _myopic_reference(seed, args, T):
    """Per-frame reward-greedy (reconfiguration-blind) over the candidate variants, paying reconfig."""
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost
    from build_operating_point_dataset import operating_point_regime

    regime = operating_point_regime(args.tx_power)
    scenes = sample_dynamic_scenes(
        seed=seed * 1000 + 777, count=args.dyn_held, node_count_choices=tuple(args.dyn_nodes),
        regime=regime, num_frames=args.frames, dt_s=args.dt, speed_min_mps=args.speed_min,
        speed_max_mps=args.speed_max, reconfig=ReconfigCost(e_edge=args.reconfig_e, l_edge=0.0),
        hold_interval=args.hold_interval, gamma=args.gamma)
    n_frames_total = feas = 0
    ret_sum = 0.0
    for sc in scenes:
        ctx0 = sc.context(0)
        tv = getattr(ctx0, "topology_variants", None) or {}
        values = tv.values() if isinstance(tv, dict) else tv
        cands = [frozenset(str(e) for e in (v.edges if hasattr(v, "edges") else v)) for v in values]
        cands = cands or [frozenset(sc.edge_ids)]
        prev = []
        ep = 0.0
        discount = 1.0
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            try:
                _c, e_full, _l = T._evaluate(obs, sc.edge_ids)
                e_ref = max(e_full, 1e-9)
            except Exception:
                e_ref = 1e-9
            best, best_r, best_ok = None, -1e30, False
            for cand in cands:
                r, _gc, _gb, ok = T.reward_of(obs, list(cand), e_ref, args.lam_c, args.lam_b,
                                              args.beta, "dense")
                if r > best_r:
                    best, best_r, best_ok = cand, r, ok
            switches = len(frozenset(prev) ^ best) if t > 0 else 0
            reconfig = (sc.reconfig.e_edge + sc.reconfig.l_edge) * switches
            # SAME discounted, H-scaled objective as the learned arms (Contract v3 §3.3).
            ep += discount * (sc.hold_interval * best_r - reconfig)
            discount *= sc.gamma
            n_frames_total += 1; feas += int(best_ok)
            prev = list(best)
        ret_sum += ep
    return {"per_frame_feasibility": feas / max(1, n_frames_total),
            "mean_episode_return": ret_sum / max(1, len(scenes))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--dyn-train", type=int, default=24)
    ap.add_argument("--dyn-val", type=int, default=24,
                    help="validation trajectories (seed*1000+333); checkpoint selected on val only")
    ap.add_argument("--dyn-held", type=int, default=24)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--hold-interval", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--reconfig-e", type=float, default=0.1)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--lam-c", type=float, default=2.0)
    ap.add_argument("--lam-b", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--dyn-eval-every", type=int, default=8)
    ap.add_argument("--dyn-warmstart", type=int, default=25,
                    help="supervised warm-start epochs toward the per-frame myopic teacher (both arms); "
                         "0=cold-start. Cold-start RL collapses on most seeds at N<=16.")
    ap.add_argument("--dyn-warmstart-lr", type=float, default=5e-4)
    ap.add_argument("--normalize-adv", action="store_true", default=True)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "dynamic_headline.json"))
    ap.add_argument("--run-dir", default=str(ROOT / "result_save" / "_dyn_headline"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    arms = ["recurrent", "memoryless"]
    per_arm = {a: {"feas": [], "ret": [], "ws_feas": []} for a in arms}
    paired_feas, paired_ret = [], []

    for seed in args.seeds:
        seed_res = {}
        for arm in arms:
            out = run_dir / f"{arm}_seed{seed}"
            cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_decentralized_rl.py"),
                   "--dynamic", "--cold-start", "--reward-mode", "dense", "--dynamic-actor", arm,
                   "--reconfig-e", str(args.reconfig_e), "--updates", str(args.updates),
                   "--dyn-train", str(args.dyn_train), "--dyn-val", str(args.dyn_val),
                   "--dyn-held", str(args.dyn_held),
                   "--frames", str(args.frames), "--dt", str(args.dt),
                   "--speed-min", str(args.speed_min), "--speed-max", str(args.speed_max),
                   "--hold-interval", str(args.hold_interval), "--gamma", str(args.gamma),
                   "--lr", str(args.lr), "--entropy-coef", str(args.entropy_coef),
                   "--ppo-epochs", str(args.ppo_epochs), "--dyn-eval-every", str(args.dyn_eval_every),
                   "--dyn-warmstart", str(args.dyn_warmstart), "--dyn-warmstart-lr", str(args.dyn_warmstart_lr),
                   "--dyn-nodes", *[str(n) for n in args.dyn_nodes], "--seed", str(seed),
                   "--out-dir", str(out)]
            if args.normalize_adv:
                cmd.append("--normalize-adv")
            env = {"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONIOENCODING": "utf-8"}
            import os
            full_env = {**os.environ, **env}
            print(f"[run] {arm} seed={seed} ...", flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), env=full_env, capture_output=True, text=True)
            (out / "stdout.txt").parent.mkdir(parents=True, exist_ok=True)
            (out / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
            res = json.loads((out / "dynamic_result.json").read_text())
            per_arm[arm]["feas"].append(res["held_per_frame_feasibility"])
            per_arm[arm]["ret"].append(res["held_mean_episode_return"])
            if res.get("warmstart_held"):
                per_arm[arm]["ws_feas"].append(res["warmstart_held"]["per_frame_feasibility"])
            seed_res[arm] = res
        paired_feas.append(seed_res["recurrent"]["held_per_frame_feasibility"]
                           - seed_res["memoryless"]["held_per_frame_feasibility"])
        paired_ret.append(seed_res["recurrent"]["held_mean_episode_return"]
                          - seed_res["memoryless"]["held_mean_episode_return"])

    T = _load_trunk()
    myopic = {"feas": [], "ret": []}
    for seed in args.seeds:
        ref = _myopic_reference(seed, args, T)
        myopic["feas"].append(ref["per_frame_feasibility"])
        myopic["ret"].append(ref["mean_episode_return"])

    report = {
        "scope": "dynamic T>1 headline: recurrent vs memoryless (+ myopic-greedy reference)",
        "config": {"seeds": args.seeds, "updates": args.updates, "frames": args.frames,
                   "dyn_train": args.dyn_train, "dyn_held": args.dyn_held, "reconfig_e": args.reconfig_e,
                   "hold_interval": args.hold_interval, "gamma": args.gamma,
                   "mobility_speed_mps": [args.speed_min, args.speed_max], "dt_s": args.dt},
        "recurrent": {"per_frame_feasibility": _ci(per_arm["recurrent"]["feas"]),
                      "mean_episode_return": _ci(per_arm["recurrent"]["ret"]),
                      "warmstart_alone_feasibility": _ci(per_arm["recurrent"]["ws_feas"])},
        "memoryless": {"per_frame_feasibility": _ci(per_arm["memoryless"]["feas"]),
                       "mean_episode_return": _ci(per_arm["memoryless"]["ret"]),
                       "warmstart_alone_feasibility": _ci(per_arm["memoryless"]["ws_feas"])},
        "myopic_greedy_reference": {"per_frame_feasibility": _ci(myopic["feas"]),
                                    "mean_episode_return": _ci(myopic["ret"])},
        "paired_recurrent_minus_memoryless": {"per_frame_feasibility": _ci(paired_feas),
                                              "mean_episode_return": _ci(paired_ret)},
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== DYNAMIC HEADLINE ===")
    for arm in ("recurrent", "memoryless", "myopic_greedy_reference"):
        f = report[arm]["per_frame_feasibility"]; r = report[arm]["mean_episode_return"]
        print(f"{arm:26s} feas={f['mean']:.4f} CI{f['ci95']}  return={r['mean']:.4f} CI{r['ci95']}")
    p = report["paired_recurrent_minus_memoryless"]
    print(f"paired (rec-mem) feas={p['per_frame_feasibility']['mean']:+.4f} "
          f"CI{p['per_frame_feasibility']['ci95']}  return={p['mean_episode_return']['mean']:+.4f} "
          f"CI{p['mean_episode_return']['ci95']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
