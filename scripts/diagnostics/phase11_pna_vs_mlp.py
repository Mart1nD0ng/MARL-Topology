"""Phase 11 honest headline: the directional PNA actor (--actor pna) vs the current MLP actor
(--actor mlp, the default) on the CORRECTED dataset, at matched config (same baseline, shards, updates,
split, seeds; differ ONLY by --actor). Both cold-start. Paired per-seed comparison of the keep-best held
raw with a small-n t-interval. Honest: per-seed values + CI reported; a win is claimed only if the CI
excludes 0 (else KEEP-as-opt-in / deferred, per the Phase 8/9 standard).
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
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _run(spec):
    seed, actor, shards, updates, val_scenes, baseline, out_root = spec
    out_dir = Path(out_root) / f"{actor}_seed{seed}"
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "train_decentralized_rl.py"),
           "--baseline", baseline, "--cold-start", "--actor", actor, "--seed", str(seed),
           "--updates", str(updates), "--eval-every", str(max(1, updates // 4)),
           "--val-scenes", str(val_scenes), "--shards", *shards, "--out-dir", str(out_dir)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src"),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    res = out_dir / "rl_result.json"
    if proc.returncode != 0 or not res.exists():
        return {"seed": seed, "actor": actor, "ok": False, "err": (proc.stdout[-800:] + proc.stderr[-800:])}
    r = json.loads(res.read_text(encoding="utf-8"))
    return {"seed": seed, "actor": actor, "ok": True, "raw": r["rl"]["raw"],
            "bc_raw": r["warm_start_bc"]["raw"], "rl_final_raw": r["rl_final"]["raw"]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=80)
    ap.add_argument("--val-scenes", type=int, default=15)
    ap.add_argument("--baseline", default="ema")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out-root", default=str(ROOT / "result_save" / "_phase11_pna"))
    ap.add_argument("--out", default=str(ROOT / "result_save" / "phase11_pna_vs_mlp.json"))
    args = ap.parse_args()

    specs = [(s, a, args.shards, args.updates, args.val_scenes, args.baseline, args.out_root)
             for s in args.seeds for a in ("mlp", "pna")]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(_run, specs))

    by = {"mlp": {}, "pna": {}}
    for r in results:
        if r["ok"]:
            by[r["actor"]][r["seed"]] = r
    paired = [{"seed": s, "mlp": by["mlp"][s]["raw"], "pna": by["pna"][s]["raw"],
               "diff_pna_minus_mlp": by["pna"][s]["raw"] - by["mlp"][s]["raw"]}
              for s in args.seeds if s in by["mlp"] and s in by["pna"]]
    diffs = [p["diff_pna_minus_mlp"] for p in paired]
    report = {
        "scope": "Phase-11 PNA-vs-MLP actor on CORRECTED dataset; matched config (only --actor differs)",
        "baseline": args.baseline, "shards_n": len(args.shards), "seeds": args.seeds,
        "updates": args.updates, "failures": [r for r in results if not r["ok"]],
        "mlp_raw_per_seed": {s: by["mlp"][s]["raw"] for s in by["mlp"]},
        "pna_raw_per_seed": {s: by["pna"][s]["raw"] for s in by["pna"]},
        "mlp_mean_raw": _mean([by["mlp"][s]["raw"] for s in by["mlp"]]),
        "pna_mean_raw": _mean([by["pna"][s]["raw"] for s in by["pna"]]),
        "paired": paired,
    }
    if len(diffs) >= 2:
        md = _mean(diffs)
        var = sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1)
        se = (var / len(diffs)) ** 0.5
        t = _T975.get(len(diffs) - 1, 1.96)
        report["paired_diff_pna_minus_mlp"] = {
            "mean": md, "ci95": [md - t * se, md + t * se], "n": len(diffs),
            "verdict": ("PNA > MLP (CI excludes 0)" if md - t * se > 0 else
                        "MLP > PNA (CI excludes 0)" if md + t * se < 0 else
                        "MATCHES: CI spans 0 (no significant actor-architecture edge; PNA stays opt-in)"),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if report["failures"]:
        print(f"\n[WARN] {len(report['failures'])} run(s) failed", file=sys.stderr)


if __name__ == "__main__":
    main()
