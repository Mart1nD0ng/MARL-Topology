"""D8: the 2x2 {motion x recurrence} dynamic retest on the fully-corrected pipeline.

Runs the four arms -- memoryless/recurrent x current-CSI/velocity -- via the ``--dynamic`` subprocess
(all sharing the corrected D2/D3/D4 objective + the decoder-aware D6 warm-start), reads each run's
held metrics, evaluates the D7 DEPLOYABLE baselines + the CENTRAL myopic-greedy reference on the SAME
held set per seed, and emits a grouped report (learned_arms / deployable_policies / central_references)
with per-seed values + 95% CIs + the paired (velocity-csi) and (recurrent-memoryless) diffs.

Honest scope: single-RSU random geometry (D1 urban data not yet built) -> the conclusion does NOT
extrapolate to urban. PILOT params (few seeds / small N) are NOT a headline.
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

# the 2x2 ablation: actor in {memoryless, recurrent} x motion in {False (current-CSI), True (velocity)}.
D8_ARMS = [
    {"label": "memoryless_csi", "actor": "memoryless", "motion": False},
    {"label": "memoryless_velocity", "actor": "memoryless", "motion": True},
    {"label": "recurrent_csi", "actor": "recurrent", "motion": False},
    {"label": "recurrent_velocity", "actor": "recurrent", "motion": True},
]

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _ci(xs):
    xs = [float(x) for x in xs if x is not None]
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


def _agg_arm(seeds):
    """Aggregate a list of per-seed metric dicts -> {metric: CI} over the numeric keys + n_seeds."""
    if not seeds:
        return {"n_seeds": 0}
    keys = [k for k, v in seeds[0].items() if isinstance(v, (int, float))]
    out = {k: _ci([s.get(k) for s in seeds]) for k in keys}
    out["n_seeds"] = len(seeds)
    return out


def build_report(arm_results, deployable_results, central_references, paired=None):
    """Group the learned arms (per-seed -> CI) and the D7 baseline classes; never mix them
    (Contract v3 §10.1). ``arm_results`` = {arm_label: [per_seed_metric_dict]}."""
    return {
        "scope": ("dynamic 2x2 {motion x recurrence} retest on the corrected pipeline; SINGLE-RSU "
                  "random geometry (NOT urban -- D1 pending); conclusion does not extrapolate to urban"),
        "learned_arms": {label: _agg_arm(seeds) for label, seeds in arm_results.items()},
        "deployable_policies": deployable_results,
        "central_references": central_references,
        "paired": paired or {},
    }


def _load_trunk():
    spec = importlib.util.spec_from_file_location("trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run_arm(arm, seed, args, run_dir):
    out = run_dir / f"{arm['label']}_seed{seed}"
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_decentralized_rl.py"),
           "--dynamic", "--cold-start", "--reward-mode", "dense", "--dynamic-actor", arm["actor"],
           "--reconfig-e", str(args.reconfig_e), "--updates", str(args.updates),
           "--dyn-train", str(args.dyn_train), "--dyn-val", str(args.dyn_val), "--dyn-held", str(args.dyn_held),
           "--frames", str(args.frames), "--dt", str(args.dt), "--hold-interval", str(args.hold_interval),
           "--gamma", str(args.gamma), "--lr", str(args.lr), "--ppo-epochs", str(args.ppo_epochs),
           "--dyn-eval-every", str(args.dyn_eval_every), "--dyn-warmstart", str(args.dyn_warmstart),
           "--dyn-warmstart-mode", args.dyn_warmstart_mode, "--dyn-bc-anchor", str(args.dyn_bc_anchor),
           "--dyn-nodes", *[str(n) for n in args.dyn_nodes], "--seed", str(seed), "--out-dir", str(out)]
    if arm["motion"]:
        cmd.append("--motion-features")
    if args.normalize_adv:
        cmd.append("--normalize-adv")
    import os
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    print(f"[run] {arm['label']} seed={seed} ...", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    (out / "stdout.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    res = json.loads((out / "dynamic_result.json").read_text())
    hist = json.loads((out / "training_history.json").read_text())
    last_val = next((h for h in reversed(hist) if h.get("val_mean_episode_return") is not None), hist[-1] if hist else {})
    return {
        "held_feas": res["held_per_frame_feasibility"], "held_return": res["held_mean_episode_return"],
        "held_switches": res["held_mean_switches_per_frame"], "best_val_return": res.get("best_val_episode_return"),
        "warmstart_alone_return": res.get("warmstart_alone_return"),
        "post_rl_drift": res.get("post_rl_drift_held_return"),
        "critic_ev": last_val.get("critic_explained_variance"), "actor_kl": last_val.get("per_agent_kl"),
    }


def _baselines_for_seed(seed, args, T):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_baselines import (
        baseline_budget_report, evaluate_central_reference, evaluate_deployable_baseline,
        local_hysteresis_action, local_threshold_action)
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost

    scenes = sample_dynamic_scenes(
        seed=seed * 1000 + 777, count=args.dyn_held, node_count_choices=tuple(args.dyn_nodes),
        regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=args.dt,
        speed_min_mps=args.speed_min, speed_max_mps=args.speed_max,
        reconfig=ReconfigCost(e_edge=args.reconfig_e, l_edge=0.0), hold_interval=args.hold_interval,
        gamma=args.gamma, tag="held_")

    def rof(obs, edges, e_ref, lc, lb, beta, rm):
        return T.reward_of(obs, edges, e_ref, lc, lb, beta, rm)

    def rfe(obs):
        try:
            _c, e, _l = T._evaluate(obs, obs["edge_ids"])
            return max(e, 1e-9)
        except Exception:
            return 1e-9

    rw = dict(reward_of=rof, ref_energy=rfe, lam_c=args.lam_c, lam_b=args.lam_b, beta=args.beta, reward_mode="dense")
    thr = evaluate_deployable_baseline(
        lambda ef, e, ed, b, p: local_threshold_action(ef, e, ed, b, p, threshold=0.5), scenes,
        label="local_threshold", **rw)
    hys = evaluate_deployable_baseline(
        lambda ef, e, ed, b, p: local_hysteresis_action(ef, e, ed, b, p, keep_threshold=0.4, add_threshold=0.6),
        scenes, label="local_hysteresis", **rw)
    cen = evaluate_central_reference(scenes, label="myopic_greedy", **rw)
    return thr, hys, cen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--dyn-train", type=int, default=24)
    ap.add_argument("--dyn-val", type=int, default=24)
    ap.add_argument("--dyn-held", type=int, default=24)
    ap.add_argument("--frames", type=int, default=6)
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
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--dyn-eval-every", type=int, default=6)
    ap.add_argument("--dyn-warmstart", type=int, default=25)
    ap.add_argument("--dyn-warmstart-mode", default="bcsp")
    ap.add_argument("--dyn-bc-anchor", type=float, default=0.0)
    ap.add_argument("--normalize-adv", action="store_true", default=True)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "dynamic_d8_matrix.json"))
    ap.add_argument("--run-dir", default=str(ROOT / "result_save" / "_dyn_d8"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    arm_results = {a["label"]: [] for a in D8_ARMS}
    for seed in args.seeds:
        for arm in D8_ARMS:
            arm_results[arm["label"]].append(_run_arm(arm, seed, args, run_dir))

    T = _load_trunk()
    thr_s, hys_s, cen_s = [], [], []
    for seed in args.seeds:
        thr, hys, cen = _baselines_for_seed(seed, args, T)
        thr_s.append(thr); hys_s.append(hys); cen_s.append(cen)

    def _agg_base(rows, label, group):
        return {"label": label, "group": group,
                "action_evaluator_calls": _ci([r["action_evaluator_calls"] for r in rows]),
                "per_frame_feasibility": _ci([r["per_frame_feasibility"] for r in rows]),
                "mean_episode_return": _ci([r["mean_episode_return"] for r in rows]),
                "mean_switches_per_frame": _ci([r["mean_switches_per_frame"] for r in rows])}

    deployable = [_agg_base(thr_s, "local_threshold", "deployable_policy"),
                  _agg_base(hys_s, "local_hysteresis", "deployable_policy")]
    central = [_agg_base(cen_s, "myopic_greedy", "central_reference")]

    def _paired(a, b, key):
        xa = [s[key] for s in arm_results[a]]
        xb = [s[key] for s in arm_results[b]]
        return _ci([va - vb for va, vb in zip(xa, xb) if va is not None and vb is not None])

    paired = {
        "velocity_minus_csi_memoryless_feas": _paired("memoryless_velocity", "memoryless_csi", "held_feas"),
        "velocity_minus_csi_memoryless_return": _paired("memoryless_velocity", "memoryless_csi", "held_return"),
        "recurrent_minus_memoryless_csi_feas": _paired("recurrent_csi", "memoryless_csi", "held_feas"),
        "recurrent_minus_memoryless_velocity_feas": _paired("recurrent_velocity", "memoryless_velocity", "held_feas"),
    }
    report = build_report(arm_results, deployable, central, paired)
    report["config"] = {"seeds": args.seeds, "updates": args.updates, "frames": args.frames,
                        "dyn_train": args.dyn_train, "dyn_val": args.dyn_val, "dyn_held": args.dyn_held,
                        "dyn_nodes": args.dyn_nodes, "warmstart": args.dyn_warmstart,
                        "warmstart_mode": args.dyn_warmstart_mode, "bc_anchor": args.dyn_bc_anchor}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== D8 2x2 MATRIX (held) ===")
    for label, agg in report["learned_arms"].items():
        f, r = agg.get("held_feas"), agg.get("held_return")
        if f and r:
            print(f"{label:22s} feas={f['mean']:.4f} CI{f['ci95']}  return={r['mean']:.4f} CI{r['ci95']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
