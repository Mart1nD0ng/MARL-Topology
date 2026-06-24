"""Phase-8 RE-REVIEW headline: 8b (COMA per-agent counterfactual credit) vs R7 (shared Graph-MAPPO
advantage) on the CORRECTED-env-math dataset, at EQUAL evaluator budget (both 1 evaluator call/scene --
8b's counterfactuals are critic forwards). Owner step 2 (2026-06-24): re-review Phase 8 against the
rebuilt data.

Runs N seeds x {r7, 8b} as independent trunk subprocesses (cold-start, identical shards/updates/split),
then aggregates the keep-best held raw per seed into a PAIRED comparison (8b - r7 per seed) with a small-
n t-interval. Honest: per-seed values + CI are reported; nothing is a headline unless the CI excludes 0.
No smoke configs; the dataset is the corrected op_corrected shards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# t_{0.975, df} for small-n paired CI
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _run(spec):
    seed, arm, shards, updates, val_scenes, out_root = spec
    out_dir = Path(out_root) / f"{arm}_seed{seed}"
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_decentralized_rl.py"),
           "--baseline", "graph-mappo", "--cold-start", "--seed", str(seed),
           "--updates", str(updates), "--eval-every", str(max(1, updates // 4)),
           "--val-scenes", str(val_scenes), "--shards", *shards, "--out-dir", str(out_dir)]
    if arm == "8b":
        cmd.append("--counterfactual")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src"),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    res_path = out_dir / "rl_result.json"
    if proc.returncode != 0 or not res_path.exists():
        return {"seed": seed, "arm": arm, "ok": False,
                "err": (proc.stdout[-800:] + proc.stderr[-800:])}
    r = json.loads(res_path.read_text(encoding="utf-8"))
    hist = r.get("history", [])
    return {"seed": seed, "arm": arm, "ok": True, "raw": r["rl"]["raw"],
            "bc_raw": r["warm_start_bc"]["raw"], "rl_final_raw": r["rl_final"]["raw"],
            "val_curve": [h.get("val_raw") for h in hist]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=80)
    ap.add_argument("--val-scenes", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out-root", default=str(ROOT / "result_save" / "_phase8_headline"))
    ap.add_argument("--out", default=str(ROOT / "result_save" / "phase8_corrected_headline.json"))
    args = ap.parse_args()

    specs = [(s, arm, args.shards, args.updates, args.val_scenes, args.out_root)
             for s in args.seeds for arm in ("r7", "8b")]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(_run, specs))

    by = {"r7": {}, "8b": {}}
    for r in results:
        if r["ok"]:
            by[r["arm"]][r["seed"]] = r
    paired = []
    for s in args.seeds:
        if s in by["r7"] and s in by["8b"]:
            paired.append({"seed": s, "r7": by["r7"][s]["raw"], "8b": by["8b"][s]["raw"],
                           "diff_8b_minus_r7": by["8b"][s]["raw"] - by["r7"][s]["raw"]})
    diffs = [p["diff_8b_minus_r7"] for p in paired]
    report = {
        "scope": "Phase-8 re-review on CORRECTED dataset; equal evaluator budget (1/scene both arms)",
        "shards_n": len(args.shards), "seeds": args.seeds, "updates": args.updates,
        "failures": [r for r in results if not r["ok"]],
        "r7_raw_per_seed": {s: by["r7"][s]["raw"] for s in by["r7"]},
        "8b_raw_per_seed": {s: by["8b"][s]["raw"] for s in by["8b"]},
        "r7_mean_raw": _mean([by["r7"][s]["raw"] for s in by["r7"]]),
        "8b_mean_raw": _mean([by["8b"][s]["raw"] for s in by["8b"]]),
        "paired": paired,
    }
    if len(diffs) >= 2:
        md = _mean(diffs)
        var = sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1)
        se = (var / len(diffs)) ** 0.5
        t = _T975.get(len(diffs) - 1, 1.96)
        report["paired_diff_8b_minus_r7"] = {
            "mean": md, "ci95": [md - t * se, md + t * se], "n": len(diffs),
            "verdict": ("8b > r7 (CI excludes 0)" if md - t * se > 0 else
                        "r7 > 8b (CI excludes 0)" if md + t * se < 0 else
                        "MATCHES: CI spans 0 (no significant credit/sample-efficiency edge on corrected data)"),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if report["failures"]:
        print(f"\n[WARN] {len(report['failures'])} run(s) failed", file=sys.stderr)


if __name__ == "__main__":
    main()
