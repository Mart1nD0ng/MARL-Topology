"""Evaluate a frozen decentralized actor on an arbitrary operating-point dataset.

Eval-only (no training). One reusable tool for several campaign blocks:

  * cross-N generalization  -- train on N{8,12}, eval zero-shot on a N{16,20} dataset
  * cross-regime / OOD       -- eval a 20 dBm-trained actor on a 17 dBm (-3 dB) dataset
  * decentralization cost    -- local mutual acceptance vs centralized global-argsort decode
  * baselines table          -- actor vs SA-teacher / full-graph / empty / budget-random

Reports raw and solvable-conditional feasibility (overall + per-N), mean +/- 95% CI across the
actors stored in the artifacts file. The deployed decode is decentralized; the global-argsort
number is the centralized ablation only.

Usage::

    python scripts/train/evaluate_actor_on_dataset.py \
        --artifacts result_save/decentralized_retrain/_artifacts_retrain.pt \
        --shards result_save/ood_tx17/_op_shard_3001.pkl ... \
        --eval-split all --baselines --out result_save/campaign/E7_ood.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from random import Random
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from marl_topology.budgets import is_budget_feasible, node_budgets_for_scene  # noqa: E402
from marl_topology.training.decentralized_distillation import (  # noqa: E402
    TAU,
    build_samples,
    ci95,
    feasible_breakdown,
    global_argsort_assemble,
    load_actor_from_state,
    local_mutual_assemble,
)
from marl_topology.training.production_mappo_adapter import (  # noqa: E402
    Stage33ProductionMappoAdapter,
)


def load_pool(shard_paths):
    adapter = Stage33ProductionMappoAdapter()
    pool = []
    for shard_path in shard_paths:
        with open(shard_path, "rb") as handle:
            dataset = pickle.load(handle)
        labels = dataset.source_dataset.teacher_labels
        for split in ("train", "eval", "test"):
            for row, context in adapter.build_row_contexts(dataset, split):
                pool.append((row, context, labels[context.fixture.fixture_id]))
    return pool


def _eval_topology(sample, topo):
    ctx = sample["context"]
    p = float(ctx.evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
    return p >= TAU and is_budget_feasible(tuple(topo), node_budgets_for_scene(ctx.evaluator.scene))


def fixed_topology_breakdown(samples, build_fn):
    """Raw + solvable-conditional feasibility of a FIXED (non-learned) topology rule."""

    solved = total = solved_solv = solvable = 0
    for sample in samples:
        ok = _eval_topology(sample, build_fn(sample))
        total += 1
        solved += int(ok)
        if bool(sample["label"]["feasible_exists"]):
            solvable += 1
            solved_solv += int(ok)
    return {"raw": solved / max(1, total),
            "conditional": (solved_solv / solvable) if solvable else None,
            "solved": solved, "total": total, "solvable": solvable}


def _budget_random(sample, seed):
    rng = Random(seed)
    edges = list(sample["edge_ids"])
    rng.shuffle(edges)
    budgets = dict(node_budgets_for_scene(sample["context"].evaluator.scene))
    graph = sample["context"].graph
    deg, chosen = {}, []
    for e in edges:
        edge = graph.get_edge(e)
        a, b = edge.node_u, edge.node_v
        if deg.get(a, 0) >= budgets.get(a, 0) or deg.get(b, 0) >= budgets.get(b, 0):
            continue
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
        chosen.append(e)
    return set(chosen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", required=True)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--eval-split", choices=["held", "all"], default="held",
                   help="'held' reconstructs the training 60/40 split (same-dataset reproduction); "
                        "'all' evaluates every scene (zero-shot on a different dataset)")
    p.add_argument("--held-frac", type=float, default=0.4)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--baselines", action="store_true", help="also score SA-teacher/full/empty/random")
    p.add_argument("--out", default="", help="optional JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pool = load_pool(args.shards)
    if args.eval_split == "held":
        Random(args.split_seed).shuffle(pool)
        cut = int((1.0 - args.held_frac) * len(pool))
        items = pool[cut:]
    else:
        items = pool
    samples = build_samples(items)
    ceiling = fmean(float(bool(l["feasible_exists"])) for _r, _c, l in items)
    nd, ed = samples[0]["nf"].shape[1], samples[0]["ef"].shape[1]
    art = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    print(f"[eval] scenes={len(samples)} ceiling={ceiling:.3f} node_dim={nd} edge_dim={ed} "
          f"actors={len(art['actors'])} split={args.eval_split}")

    dec, cond, glob, per_actor = [], [], [], []
    for i, a in enumerate(art["actors"]):
        actor = load_actor_from_state(a["state"], nd, ed, hidden=args.hidden, rounds=args.rounds)
        bd = feasible_breakdown(actor, samples, a["mean"], a["std"], local_mutual_assemble)
        bg = feasible_breakdown(actor, samples, a["mean"], a["std"], global_argsort_assemble)
        dec.append(bd["raw"]); glob.append(bg["raw"])
        if bd["conditional"] is not None:
            cond.append(bd["conditional"])
        per_actor.append({"actor": i, "decentralized": bd, "global_ablation_raw": bg["raw"]})
        c = bd["conditional"]
        print(f"[eval] actor {i}: DEC raw={bd['raw']:.3f} conditional={c if c is None else round(c,3)} "
              f"by_N={bd['raw_by_n']} global={bg['raw']:.3f}")

    dm, dh = ci95(dec)
    gm, gh = ci95(glob)
    cm, ch = ci95(cond) if cond else (None, 0.0)
    summary = {"scenes": len(samples), "ceiling": ceiling, "eval_split": args.eval_split,
               "rounds": args.rounds, "hidden": args.hidden,
               "decentralized_raw_mean": dm, "decentralized_raw_ci95": dh,
               "conditional_mean": cm, "conditional_ci95": ch,
               "global_raw_mean": gm, "global_raw_ci95": gh, "decentralization_cost": gm - dm,
               "per_actor": per_actor}
    print("=" * 72)
    print(f"[eval] DECENTRALIZED raw = {dm:.3f} +/- {dh:.3f}  (teacher ceiling {ceiling:.3f})")
    if cm is not None:
        print(f"[eval] DECENTRALIZED conditional (solved/solvable) = {cm:.3f} +/- {ch:.3f}")
    print(f"[eval] global ablation raw = {gm:.3f} +/- {gh:.3f}  (decentralization cost {gm - dm:+.3f})")

    if args.baselines:
        builders = {
            "sa_teacher": lambda s: set(str(e) for e in s["label"]["selected_physical_edges"]),
            "full_graph": lambda s: set(s["edge_ids"]),
            "empty": lambda s: set(),
            "budget_random": lambda s: _budget_random(s, args.split_seed),
        }
        baselines = {name: fixed_topology_breakdown(samples, fn) for name, fn in builders.items()}
        summary["baselines"] = baselines
        print("-" * 72)
        for name, bd in baselines.items():
            c = bd["conditional"]
            print(f"[baseline] {name:>14}: raw={bd['raw']:.3f} "
                  f"conditional={c if c is None else round(c, 3)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
