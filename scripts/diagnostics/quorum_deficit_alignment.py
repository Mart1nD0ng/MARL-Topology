"""Q4 (POMDP-QP-FAR): D_quorum <-> true C alignment test (Spec S8.3) -- the gate before reward.

Samples real --dyn-data topologies, applies LOCAL single-edge edits (add / remove one edge), and for
each edit records the TRUE reliability change dC = C(x') - C(x) (fixed-set robust path), the energy
change, and the deficit IMPROVEMENT dD = D_quorum(x) - D_quorum(x'). It reports Spearman(dD, dC), the
top-k repair hit rate, and the false-improvement rates (dD>0 but C worsens / energy explodes).

EXIT CONDITION (Spec Q4): if D_quorum aligns with the true C (strong positive Spearman + low
false-improvement), D_quorum MAY enter the reward at Q9 (as the PBRS potential). If not, it MUST NOT --
an honest gate result. C/energy and D_quorum come from the SAME reference evaluator (Contract D5.5).
Eval-only: no training, no checkpoint, no reward.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from marl_topology.training.quorum_deficit_bridge import (  # noqa: E402
    topology_quorum_deficit, topology_reliability)
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402


# -- metric core (pure; testable) -------------------------------------------------------------------

def spearman(x, y) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    def _rank(z):
        order = sorted(range(n), key=lambda i: z[i])
        r = [0.0] * n
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def alignment_metrics(dD, dC, dE_rel, *, top_k_frac: float = 0.2, energy_explode: float = 0.5) -> dict:
    """dD = deficit improvement (D_base - D_edit), dC = reliability improvement (C_edit - C_base),
    dE_rel = relative energy change of the edit. Reports the alignment + the false-improvement rates."""
    n = len(dD)
    if n == 0:
        return {"n": 0}
    k = max(1, int(round(top_k_frac * n)))
    top_d = set(sorted(range(n), key=lambda i: -dD[i])[:k])
    top_c = set(sorted(range(n), key=lambda i: -dC[i])[:k])
    improving = [i for i in range(n) if dD[i] > 1e-9]                 # edits the proxy calls "better"
    worsens = sum(1 for i in improving if dC[i] < -1e-9)
    explodes = sum(1 for i in improving if dE_rel[i] > energy_explode)
    # On a deep infeasible plateau the true C is FLAT (dC ~ 0) -> the rank correlation is dominated by
    # ties and understates D's usefulness. Report the flat fraction + the alignment CONDITIONED on C
    # actually moving (the honest signal where C carries information).
    moving = [i for i in range(n) if abs(dC[i]) > 1e-9]
    return {
        "n": n, "n_improving": len(improving),
        "spearman_dD_dC": round(spearman(dD, dC), 4),
        "frac_C_flat": round((n - len(moving)) / n, 4),
        "n_C_moving": len(moving),
        "spearman_on_C_moving": round(spearman([dD[i] for i in moving], [dC[i] for i in moving]), 4)
        if len(moving) >= 3 else None,
        "top_k_repair_hit_rate": round(len(top_d & top_c) / k, 4), "top_k": k,
        "dD_pos_but_C_worsens_rate": round(worsens / max(1, len(improving)), 4),
        "dD_pos_but_energy_explodes_rate": round(explodes / max(1, len(improving)), 4),
        "mean_dC_of_improving": round(sum(dC[i] for i in improving) / max(1, len(improving)), 5),
    }


# -- sweep ------------------------------------------------------------------------------------------

def _edits(rng, base: set, candidates: list, n_add: int, n_remove: int):
    absent = [e for e in candidates if e not in base]
    present = list(base)
    rng.shuffle(absent); rng.shuffle(present)
    out = []
    for e in absent[:n_add]:
        out.append(base | {e})
    for e in present[:n_remove]:
        out.append(base - {e})
    return out


def _collect_for_scene(ev, candidates, rng, args):
    dD, dC, dE_rel = [], [], []
    for keep in args.base_keep:
        k = max(4, int(round(keep * len(candidates))))
        base = set(rng.sample(candidates, min(k, len(candidates))))
        d_base = topology_quorum_deficit(ev, base)
        r_base = topology_reliability(ev, base)
        if d_base is None:
            continue
        for edited in _edits(rng, base, candidates, args.n_add, args.n_remove):
            d_e = topology_quorum_deficit(ev, edited)
            if d_e is None:
                continue
            r_e = topology_reliability(ev, edited)
            dD.append(d_base["d_quorum_mean"] - d_e["d_quorum_mean"])      # deficit improvement
            dC.append(r_e["consensus"] - r_base["consensus"])              # reliability improvement
            base_e = max(r_base["energy"], 1e-9)
            dE_rel.append((r_e["energy"] - r_base["energy"]) / base_e)
    return dD, dC, dE_rel


def _build(data, seed, count, args):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import (
        sample_dynamic_scenes, sample_dynamic_urban_scenes)
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(args.tx_power), num_frames=1, dt_s=1.0,
                  speed_min_mps=args.speed_min, speed_max_mps=args.speed_max,
                  reconfig=ReconfigCost(), hold_interval=1, gamma=0.95)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["random", "urban"])
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--base-keep", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--n-add", type=int, default=4)
    ap.add_argument("--n-remove", type=int, default=4)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=4404)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "quorum_deficit_alignment.json"))
    args = ap.parse_args()

    report = {"scope": "Q4 D_quorum<->C alignment under local edits (eval-only; fixed_set robust C)",
              "config": {"scenes": args.scenes, "base_keep": args.base_keep, "n_add": args.n_add,
                         "n_remove": args.n_remove, "dyn_nodes": args.dyn_nodes, "seed": args.seed},
              "by_data": {}}
    for data in args.data:
        rng = random.Random(args.seed)
        scenes = _build(data, args.seed * 1000 + 1, args.scenes, args)
        dD, dC, dE = [], [], []
        for sc in scenes:
            ev = sc.context(0).evaluator
            cand = list(sc.edge_ids_at(0))
            a, b, c = _collect_for_scene(ev, cand, rng, args)
            dD += a; dC += b; dE += c
        m = alignment_metrics(dD, dC, dE)
        report["by_data"][data] = m
        print(f"[{data}] n={m.get('n')} spearman={m.get('spearman_dD_dC')} "
              f"flat_C={m.get('frac_C_flat')} spearman|C-moving={m.get('spearman_on_C_moving')} "
              f"(n={m.get('n_C_moving')}) top-k_hit={m.get('top_k_repair_hit_rate')} "
              f"dD>0_C_worse={m.get('dD_pos_but_C_worsens_rate')} "
              f"dD>0_E_explode={m.get('dD_pos_but_energy_explodes_rate')}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
