"""4090 campaign orchestrator -- the packed experiment battery for the decentralized MARL trunk.

Runs the tiered campaign (`docs/FOUR090_CAMPAIGN_PLAN.md`) end-to-end and unattended. Each block
writes its result JSON to ``result_save/campaign/<id>.json`` THE MOMENT it finishes, so a crash /
preemption never loses completed work -- rerun and finished blocks are skipped (resume). Datasets
are built once per regime (CPU-heavy SA teacher) and cached; training arms then reuse the pool.

Blocks (see the plan doc for the science):
  T1  E1_op_headline   full retrain at the regulatory operating point (raw + conditional + cost)
      E2b_rich26       denser realistic deployment (26 dBm)        -> Route B raw >= 0.9
      E2b_rich6r       denser realistic deployment (6 RSU)         -> Route B raw >= 0.9
  T2  E4_K{0,2,4}      K-rounds actor ablation (message passing vs bandit)
      E5_baselines     actor vs SA-teacher / full-graph / empty / budget-random
      E8_sa_teacher    planner-target vs raw-SA-teacher-target lift
  T3  E6_genmix        mixed-N {8,12,16,20} generalization
      E6_genlo_train   train N {8,12} ...
      E6_zeroshot_hi   ... eval zero-shot on N {16,20}
      E7_ood_zeroshot  operating-point actor on a -3 dB (17 dBm) regime
      E7_ood_retrain   retrain at 17 dBm (recovery)
      E9_rich30 / E9_sparse2   deployment-design sensitivity arms (with E1/E2b: a tx x RSU surface)

Usage::

    # one-shot: build any missing datasets, then run everything (hours; bank-as-you-go)
    python scripts/train/run_4090_campaign.py --build-missing

    # local push-button smoke (tiny configs) to prove the pipeline before shipping
    python scripts/train/run_4090_campaign.py --smoke --build-missing --camp-dir result_save/_smoke

    # run a subset / a single tier / force a rerun
    python scripts/train/run_4090_campaign.py --build-missing --tiers T1
    python scripts/train/run_4090_campaign.py --build-missing --only E1_op_headline,E5_baselines
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from random import Random
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from train_recovered_decentralized import load_pool, run_campaign_arm  # noqa: E402
from evaluate_actor_on_dataset import _budget_random, fixed_topology_breakdown  # noqa: E402

from marl_topology.training.decentralized_distillation import (  # noqa: E402
    build_samples,
    ci95,
    feasible_breakdown,
    global_argsort_assemble,
    load_actor_from_state,
    local_mutual_assemble,
)

# ----------------------------------------------------------------- regimes & recipes
REGIMES = {
    "op":      dict(rsu=4, tx=20, nodes=[8, 12, 16],     feas=0.6, near=0.1, infeas=0.3),
    "rich26":  dict(rsu=4, tx=26, nodes=[8, 12, 16],     feas=0.8, near=0.1, infeas=0.1),
    "rich6r":  dict(rsu=6, tx=20, nodes=[8, 12, 16],     feas=0.8, near=0.1, infeas=0.1),
    "rich30":  dict(rsu=4, tx=30, nodes=[8, 12, 16],     feas=0.8, near=0.1, infeas=0.1),
    "sparse2": dict(rsu=2, tx=20, nodes=[8, 12, 16],     feas=0.4, near=0.1, infeas=0.5),
    "ood17":   dict(rsu=4, tx=17, nodes=[8, 12, 16],     feas=0.6, near=0.1, infeas=0.3),
    "gen_lo":  dict(rsu=4, tx=20, nodes=[8, 12],         feas=0.6, near=0.1, infeas=0.3),
    "gen_hi":  dict(rsu=4, tx=20, nodes=[16, 20],        feas=0.6, near=0.1, infeas=0.3),
    "genmix":  dict(rsu=4, tx=20, nodes=[8, 12, 16, 20], feas=0.6, near=0.1, infeas=0.3),
}

FULL = dict(critic_seeds=[0, 1, 2], dagger_iters=2, dagger_scenes=40, val_scenes=12,
            critic_epochs=40, beam_width=6, verify_k=20, actor_seeds=[0, 1, 2, 3, 4], train_size=0)
MED = dict(critic_seeds=[0], dagger_iters=1, dagger_scenes=40, val_scenes=12,
           critic_epochs=40, beam_width=6, verify_k=20, actor_seeds=[0, 1, 2], train_size=0)
SMOKE = dict(critic_seeds=[0], dagger_iters=1, dagger_scenes=8, val_scenes=4,
             critic_epochs=6, beam_width=3, verify_k=6, actor_seeds=[0], train_size=9999)

# ----------------------------------------------------------------- the job battery (ordered)
JOBS = [
    dict(id="E1_op_headline", tier="T1", kind="train", regime="op", recipe="FULL"),
    dict(id="E2b_rich26",     tier="T1", kind="train", regime="rich26", recipe="FULL"),
    dict(id="E2b_rich6r",     tier="T1", kind="train", regime="rich6r", recipe="FULL"),

    dict(id="E4_K0", tier="T2", kind="train", regime="op", recipe="MED", rounds=0),
    dict(id="E4_K2", tier="T2", kind="train", regime="op", recipe="MED", rounds=2),
    dict(id="E4_K4", tier="T2", kind="train", regime="op", recipe="MED", rounds=4),
    dict(id="E5_baselines", tier="T2", kind="eval", artifacts="E1_op_headline", regime="op",
         eval_split="held", baselines=True),
    dict(id="E8_sa_teacher", tier="T2", kind="train", regime="op", recipe="MED", use_planner=False),

    dict(id="E6_genmix",      tier="T3", kind="train", regime="genmix", recipe="FULL"),
    dict(id="E6_genlo_train", tier="T3", kind="train", regime="gen_lo", recipe="FULL"),
    dict(id="E6_zeroshot_hi", tier="T3", kind="eval", artifacts="E6_genlo_train", regime="gen_hi",
         eval_split="all"),
    dict(id="E7_ood_zeroshot", tier="T3", kind="eval", artifacts="E1_op_headline", regime="ood17",
         eval_split="all"),
    dict(id="E7_ood_retrain", tier="T3", kind="train", regime="ood17", recipe="FULL"),
    dict(id="E9_rich30",  tier="T3", kind="train", regime="rich30", recipe="MED"),
    dict(id="E9_sparse2", tier="T3", kind="train", regime="sparse2", recipe="MED"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camp-dir", default=str(ROOT / "result_save" / "campaign"))
    p.add_argument("--build-missing", action="store_true", help="build any absent regime dataset")
    p.add_argument("--force", action="store_true", help="rerun blocks even if their JSON exists")
    p.add_argument("--only", default="", help="comma-separated job ids to run (subset)")
    p.add_argument("--tiers", default="T1,T2,T3", help="comma-separated tiers to run")
    p.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003, 3004])
    p.add_argument("--count", type=int, default=60, help="scenes per dataset shard")
    p.add_argument("--smoke", action="store_true", help="tiny configs for a local dry-run")
    return p.parse_args()


def shard_paths(data_dir: Path, regime_id: str, seeds) -> list[Path]:
    return [data_dir / regime_id / f"_op_shard_{s}.pkl" for s in seeds]


def ensure_dataset(data_dir, regime_id, r, seeds, count, build_missing) -> list[Path] | None:
    paths = shard_paths(data_dir, regime_id, seeds)
    if all(p.exists() for p in paths):
        return paths
    if not build_missing:
        return None
    cmd = [sys.executable, str(ROOT / "scripts" / "train" / "build_operating_point_dataset.py"),
           "--seeds", *[str(s) for s in seeds], "--count", str(count),
           "--node-choices", *[str(n) for n in r["nodes"]], "--rsu-count", str(r["rsu"]),
           "--tx-power", str(r["tx"]), "--blocks-per-side", "3",
           "--feasible-frac", str(r["feas"]), "--near-frac", str(r["near"]),
           "--infeasible-frac", str(r["infeas"]), "--out-dir", str(data_dir / regime_id)]
    print(f"  [build] {regime_id}: {' '.join(cmd[3:])}", flush=True)
    subprocess.run(cmd, check=True)
    paths = shard_paths(data_dir, regime_id, seeds)
    return paths if all(p.exists() for p in paths) else None


def run_train_job(job, pool, recipe):
    rounds = int(job.get("rounds", 4))
    use_planner = bool(job.get("use_planner", True))
    arm = run_campaign_arm(pool, rounds=rounds, use_planner=use_planner,
                           log=lambda m: print(f"    {m}", flush=True), **recipe)
    return arm


def run_eval_job(job, pool, camp_dir, seeds):
    art_path = Path(camp_dir) / job["artifacts"] / "_artifacts.pt"
    art = torch.load(art_path, map_location="cpu", weights_only=False)
    rounds, hidden = int(art.get("rounds", 4)), int(art.get("hidden", 64))
    if job.get("eval_split", "held") == "held":
        pool = list(pool)
        Random(7).shuffle(pool)
        items = pool[int(0.6 * len(pool)):]
    else:
        items = list(pool)
    samples = build_samples(items)
    ceiling = fmean(float(bool(l["feasible_exists"])) for _r, _c, l in items)
    nd, ed = samples[0]["nf"].shape[1], samples[0]["ef"].shape[1]
    dec, cond, glob, per_actor = [], [], [], []
    for i, a in enumerate(art["actors"]):
        actor = load_actor_from_state(a["state"], nd, ed, hidden=hidden, rounds=rounds)
        bd = feasible_breakdown(actor, samples, a["mean"], a["std"], local_mutual_assemble)
        bg = feasible_breakdown(actor, samples, a["mean"], a["std"], global_argsort_assemble)
        dec.append(bd["raw"]); glob.append(bg["raw"])
        if bd["conditional"] is not None:
            cond.append(bd["conditional"])
        per_actor.append({"actor": i, "decentralized": bd, "global_ablation_raw": bg["raw"]})
    dm, dh = ci95(dec)
    gm, gh = ci95(glob)
    cm, ch = ci95(cond) if cond else (None, 0.0)
    out = {"scenes": len(samples), "ceiling": ceiling, "eval_split": job.get("eval_split", "held"),
           "from_artifacts": job["artifacts"], "rounds": rounds, "hidden": hidden,
           "decentralized_raw_mean": dm, "decentralized_raw_ci95": dh,
           "conditional_mean": cm, "conditional_ci95": ch,
           "global_raw_mean": gm, "global_raw_ci95": gh, "decentralization_cost": gm - dm,
           "per_actor": per_actor}
    if job.get("baselines"):
        builders = {
            "sa_teacher": lambda s: set(str(e) for e in s["label"]["selected_physical_edges"]),
            "full_graph": lambda s: set(s["edge_ids"]),
            "empty": lambda s: set(),
            "budget_random": lambda s: _budget_random(s, 7),
        }
        out["baselines"] = {name: fixed_topology_breakdown(samples, fn) for name, fn in builders.items()}
    return out


def main() -> None:
    args = parse_args()
    camp_dir = Path(args.camp_dir)
    data_dir = camp_dir / "data"
    camp_dir.mkdir(parents=True, exist_ok=True)
    # Smoke validates plumbing (build -> train -> eval -> JSON -> resume), not the science, so it
    # uses tiny N {6,8} (the dominant SA-teacher cost is N) and few scenes/seeds -- seconds-to-minutes
    # per shard instead of the ~18 min/shard the full N {8,12,16} build takes.
    count = 8 if args.smoke else args.count
    seeds = args.seeds[:2] if args.smoke else args.seeds
    only = {s for s in args.only.split(",") if s}
    tiers = {t for t in args.tiers.split(",") if t}

    def regime_spec(regime_id):
        r = dict(REGIMES[regime_id])
        if args.smoke:
            r["nodes"] = [6, 8]
        return r

    pool_cache: dict[str, object] = {}

    def pool_for(regime_id):
        if regime_id not in pool_cache:
            paths = ensure_dataset(data_dir, regime_id, regime_spec(regime_id), seeds, count,
                                   args.build_missing)
            if paths is None:
                return None
            pool_cache[regime_id] = load_pool([str(p) for p in paths])
        return pool_cache[regime_id]

    summary = {}
    for job in JOBS:
        if job["tier"] not in tiers:
            continue
        if only and job["id"] not in only:
            continue
        out_json = camp_dir / f"{job['id']}.json"
        if out_json.exists() and not args.force:
            print(f"[skip] {job['id']} (exists)", flush=True)
            summary[job["id"]] = json.loads(out_json.read_text(encoding="utf-8")).get("results", {})
            continue

        t0 = time.time()
        print(f"\n[run] {job['id']} ({job['tier']} / {job['kind']} / regime={job['regime']})", flush=True)
        pool = pool_for(job["regime"])
        if pool is None:
            print(f"[miss] {job['id']}: dataset '{job['regime']}' absent "
                  f"(pass --build-missing to build it) -- skipped", flush=True)
            continue

        if job["kind"] == "train":
            recipe = dict(SMOKE if args.smoke else {"FULL": FULL, "MED": MED}[job["recipe"]])
            arm = run_train_job(job, pool, recipe)
            (camp_dir / job["id"]).mkdir(parents=True, exist_ok=True)
            torch.save({"actors": arm["actor_states"], "critic_state": arm["critic_state"],
                        "critic_tag": arm["critic_tag"], "norm_state": arm["norm_state"],
                        "rounds": int(job.get("rounds", 4)), "hidden": 64},
                       camp_dir / job["id"] / "_artifacts.pt")
            record = {"id": job["id"], "tier": job["tier"], "kind": "train",
                      "regime": job["regime"], "results": arm["results"]}
            summary[job["id"]] = arm["results"]
        else:
            res = run_eval_job(job, pool, camp_dir, seeds)
            record = {"id": job["id"], "tier": job["tier"], "kind": "eval",
                      "regime": job["regime"], "results": res}
            summary[job["id"]] = res

        out_json.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"[done] {job['id']} in {time.time() - t0:.0f}s -> {out_json.name}", flush=True)

    (camp_dir / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[campaign] wrote summary for {len(summary)} blocks -> "
          f"{camp_dir / 'campaign_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
