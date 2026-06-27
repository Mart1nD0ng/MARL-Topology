"""D13: the full dynamic campaign -- urban-vs-random contrast + the D9-D12 mechanism A/Bs.

Runs a scoped, single-variable arm set on the corrected pipeline: the recommended baseline
(mlp + memoryless + decoder-aware bcsp warm-start) on BOTH the real 4-RSU urban grid (--dyn-data urban)
and the single-RSU random-geometry ablation (--dyn-data random), plus -- on urban -- each Phase-8-11
mechanism toggled ON as a single-variable A/B vs the urban baseline (COMA / SCQ / chance / Pareto / PNA)
and the D8-best temporal arm (recurrent+velocity). Each (arm, seed) is a --dynamic subprocess; the D7
DEPLOYABLE baselines + the CENTRAL myopic reference are evaluated per data source per seed. Emits a
grouped report (per data: learned_arms with per-seed + CI; deployable_policies; central_references) +
the paired (mechanism - baseline) diffs + the urban-vs-random baseline contrast + the per-arm budget
(evaluator calls -- the SCQ/Pareto arms are NOT budget-neutral).

Honest scope: urban NLOS is genuinely harder than single-RSU random; the headline uses only the >=5-seed
run; failed seeds are reported, not hidden; a default-off mechanism is NOT written as "full model tested".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# Each arm is a single-variable change vs the baseline. flags = extra CLI toggles.
D13_ARMS = [
    {"label": "baseline", "data": "urban", "actor": "memoryless", "arch": "mlp", "flags": []},
    {"label": "baseline", "data": "random", "actor": "memoryless", "arch": "mlp", "flags": []},
    {"label": "recurrent_velocity", "data": "urban", "actor": "recurrent", "arch": "mlp",
     "flags": ["--motion-features"]},
    {"label": "coma", "data": "urban", "actor": "memoryless", "arch": "mlp",
     "flags": ["--counterfactual", "--k-cf", "4"]},
    {"label": "scq", "data": "urban", "actor": "memoryless", "arch": "mlp",
     "flags": ["--counterfactual", "--k-cf", "4", "--scq", "--scq-m", "2", "--scq-coef", "0.5"]},
    {"label": "chance", "data": "urban", "actor": "memoryless", "arch": "mlp",
     "flags": ["--chance", "--chance-delta", "0.1", "--chance-lr", "0.2"]},
    {"label": "pareto", "data": "urban", "actor": "memoryless", "arch": "mlp",
     "flags": ["--pareto-archive"]},
    {"label": "pna", "data": "urban", "actor": "recurrent", "arch": "pna", "flags": []},
]


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
    if not seeds:
        return {"n_seeds": 0}
    keys = [k for k, v in seeds[0].items() if isinstance(v, (int, float)) or v is None]
    out = {k: _ci([s.get(k) for s in seeds]) for k in keys}
    out["n_seeds"] = len(seeds)
    return out


def build_report(arm_results, baselines_by_data, paired):
    """Group per data source: learned_arms (per-seed -> CI) + the D7 deployable/central baselines, never
    mixed (Contract §10.1). ``arm_results`` = {(data,label): [per_seed_metric_dict]}."""
    per_data = {}
    for (data, label), seeds in arm_results.items():
        per_data.setdefault(data, {"learned_arms": {}})["learned_arms"][label] = _agg_arm(seeds)
    for data, groups in baselines_by_data.items():
        per_data.setdefault(data, {"learned_arms": {}})
        per_data[data]["deployable_policies"] = groups.get("deployable_policies", [])
        per_data[data]["central_references"] = groups.get("central_references", [])
    return {
        "scope": ("dynamic full campaign: urban (real 4-RSU grid) vs random (single-RSU geometry) on the "
                  "corrected pipeline; >=5 seeds; mechanisms (COMA/SCQ/chance/Pareto/PNA) are single-variable "
                  "A/Bs vs the urban baseline. Urban NLOS is harder than random -- not extrapolated beyond test."),
        "by_data": per_data,
        "paired_vs_urban_baseline": paired,
    }


def _load_trunk():
    spec = importlib.util.spec_from_file_location("trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run_arm(arm, seed, args, run_dir):
    out = run_dir / f"{arm['data']}_{arm['label']}_seed{seed}"
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_decentralized_rl.py"),
           "--dynamic", "--cold-start", "--reward-mode", "dense",
           "--dynamic-actor", arm["actor"], "--dynamic-actor-arch", arm["arch"],
           "--dyn-data", arm["data"], "--reconfig-e", str(args.reconfig_e), "--updates", str(args.updates),
           "--dyn-train", str(args.dyn_train), "--dyn-val", str(args.dyn_val), "--dyn-held", str(args.dyn_held),
           "--frames", str(args.frames), "--dt", str(args.dt), "--hold-interval", str(args.hold_interval),
           "--gamma", str(args.gamma), "--lr", str(args.lr), "--ppo-epochs", str(args.ppo_epochs),
           "--dyn-eval-every", str(args.dyn_eval_every), "--dyn-warmstart", str(args.dyn_warmstart),
           "--dyn-warmstart-mode", "bcsp", "--dyn-nodes", *[str(n) for n in args.dyn_nodes],
           "--seed", str(seed), "--out-dir", str(out), "--normalize-adv", *arm["flags"]]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    print(f"[run] {arm['data']}/{arm['label']} seed={seed} ...", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    (out / "stdout.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    res = json.loads((out / "dynamic_result.json").read_text())
    act = json.loads((out / "mechanism_activation.json").read_text())
    return {
        "held_feas": res["held_per_frame_feasibility"], "held_return": res["held_mean_episode_return"],
        "held_switches": res["held_mean_switches_per_frame"],
        "held_cvar": res.get("held_cvar_shortfall"), "chance_lambda": res.get("chance_lambda"),
        "post_rl_drift": res.get("post_rl_drift_held_return"),
        "scq_evaluator_calls": act["critic"].get("scq_evaluator_calls_per_update", 0),
        "pareto_evaluator_calls": act["reliability"].get("pareto_evaluator_calls_held", 0),
    }


def _baselines_for_seed(seed, args, T, data):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_baselines import (
        evaluate_central_reference, evaluate_deployable_baseline,
        local_hysteresis_action, local_threshold_action)
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost

    common = dict(seed=seed * 1000 + 777, count=args.dyn_held, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=args.dt,
                  speed_min_mps=args.speed_min, speed_max_mps=args.speed_max,
                  reconfig=ReconfigCost(e_edge=args.reconfig_e, l_edge=0.0), hold_interval=args.hold_interval,
                  gamma=args.gamma, tag="held_")
    scenes = (sample_dynamic_urban_scenes(**common) if data == "urban" else sample_dynamic_scenes(**common))

    def rof(obs, edges, e_ref, lc, lb, beta, rm):
        return T.reward_of(obs, edges, e_ref, lc, lb, beta, rm)

    def rfe(obs):
        try:
            _c, e, _l = T._evaluate(obs, obs["edge_ids"]); return max(e, 1e-9)
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
    ap.add_argument("--arms", type=str, nargs="+", default=None,
                    help="restrict to these arm labels (for a pilot); default all D13_ARMS")
    ap.add_argument("--out", default=str(ROOT / "result_save" / "dynamic_d13_campaign.json"))
    ap.add_argument("--run-dir", default=str(ROOT / "result_save" / "_dyn_d13"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in D13_ARMS if (args.arms is None or a["label"] in args.arms)]
    arm_results = {}
    for seed in args.seeds:
        for arm in arms:
            arm_results.setdefault((arm["data"], arm["label"]), []).append(_run_arm(arm, seed, args, run_dir))

    T = _load_trunk()
    baselines_by_data = {}
    for data in sorted({a["data"] for a in arms}):
        thr_s, hys_s, cen_s = [], [], []
        for seed in args.seeds:
            thr, hys, cen = _baselines_for_seed(seed, args, T, data)
            thr_s.append(thr); hys_s.append(hys); cen_s.append(cen)

        def _agg_base(rows, label, group):
            return {"label": label, "group": group,
                    "action_evaluator_calls": _ci([r["action_evaluator_calls"] for r in rows]),
                    "per_frame_feasibility": _ci([r["per_frame_feasibility"] for r in rows]),
                    "mean_episode_return": _ci([r["mean_episode_return"] for r in rows])}
        baselines_by_data[data] = {
            "deployable_policies": [_agg_base(thr_s, "local_threshold", "deployable_policy"),
                                    _agg_base(hys_s, "local_hysteresis", "deployable_policy")],
            "central_references": [_agg_base(cen_s, "myopic_greedy", "central_reference")]}

    # paired (mechanism - urban baseline) feasibility, per seed
    base_key = ("urban", "baseline")
    paired = {}
    if base_key in arm_results:
        base_feas = [s["held_feas"] for s in arm_results[base_key]]
        for (data, label), seeds in arm_results.items():
            if data == "urban" and label != "baseline":
                diffs = [s["held_feas"] - b for s, b in zip(seeds, base_feas)]
                paired[f"{label}_minus_baseline_feas"] = _ci(diffs)

    report = build_report(arm_results, baselines_by_data, paired)
    report["config"] = {"seeds": args.seeds, "updates": args.updates, "dyn_nodes": args.dyn_nodes,
                        "dyn_train": args.dyn_train, "frames": args.frames, "warmstart": args.dyn_warmstart}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
