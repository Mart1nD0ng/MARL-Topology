"""Phase 12 (Technical-Spec "泛化"): honest cross-N generalization on the CORRECTED held set.

Trains on N in {8,12,16} (Spec); here we EVALUATE the per-N held raw (raw_by_n) of the already-trained
arms -- the R7 default (graph-mappo) + the opt-in mechanisms (8b counterfactual, MLP-vs-PNA actor) --
by EVAL ONLY (load each run's saved actor artifact, decode with the torch-free local_mutual_assemble on
held 3017-3024, no retraining). Reports per-(arm, N) mean +/- t-CI95 over seeds so per-N degradation /
which mechanism helps which N is visible. NO held-checkpoint selection (rl.raw was keep-best on VAL);
NO p-hacking. N=24 out-of-range is handled separately (see --n24-probe).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.models.pna_directional_actor import PNADirectionalActor  # noqa: E402
from marl_topology.models.pna_aggregation import training_degree_delta  # noqa: E402

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _load_trunk():
    spec = importlib.util.spec_from_file_location("trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pna_delta(T, train_shards):
    # replicate the trunk's PNA delta EXACTLY: shuffle(split_seed=7), cut at (1-held_frac=0.4), then
    # the fit set is train minus the 15 val scenes (delta is computed from fit_items, not incl. val).
    pool = T.load_pool(train_shards)
    from random import Random
    Random(7).shuffle(pool)
    cut = int(0.6 * len(pool))
    fit = pool[:cut][:-15] if cut > 15 else pool[:cut]
    train = T.build_samples(fit)
    degs = []
    for s in train:
        ei = s["ei"]; deg = [0] * s["nf"].shape[0]
        for e in range(ei.shape[0]):
            deg[int(ei[e, 0])] += 1; deg[int(ei[e, 1])] += 1
        degs.extend(deg)
    return training_degree_delta(degs)


def _agg(xs):
    if not xs:
        return None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"mean": m, "ci95": [m, m], "n": len(xs)}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    se = (var / len(xs)) ** 0.5
    t = _T975.get(len(xs) - 1, 1.96)
    return {"mean": m, "ci95": [m - t * se, m + t * se], "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-shards", nargs="+", default=None)   # default: op_corrected 3017-3024
    ap.add_argument("--train-shards", nargs="+", default=None)  # for the PNA delta (3001-3016)
    ap.add_argument("--out", default=str(ROOT / "result_save" / "phase12_generalization.json"))
    args = ap.parse_args()

    T = _load_trunk()
    corr = sorted(glob.glob(str(ROOT / "result_save" / "campaign" / "data" / "op_corrected" / "_op_shard_30*.pkl")))
    held_shards = args.held_shards or corr[16:24]
    train_shards = args.train_shards or corr[:16]
    held_s = T.build_samples(T.load_pool(held_shards))
    node_dim, edge_dim = held_s[0]["nf"].shape[1], held_s[0]["ef"].shape[1]

    arms = {
        "R7_default (graph-mappo, mlp)": ("result_save/_phase8_headline/r7_seed*", "mlp"),
        "8b_counterfactual (mlp)": ("result_save/_phase8_headline/8b_seed*", "mlp"),
        "ema_mlp": ("result_save/_phase11_pna/mlp_seed*", "mlp"),
        "ema_pna (preference PNA actor)": ("result_save/_phase11_pna/pna_seed*", "pna"),
    }
    pna_delta = _pna_delta(T, train_shards)

    report = {"scope": "cross-N generalization on op_corrected held 3017-3024 (eval-only; raw_by_n)",
              "held_shards_n": len(held_shards), "pna_delta": pna_delta, "arms": {}}
    for arm, (pattern, kind) in arms.items():
        per_n = {}        # N -> [seed raw]
        overall = []
        dirs = sorted(glob.glob(str(ROOT / pattern)))
        for d in dirs:
            art = Path(d) / "_rl_artifacts.pt"
            if not art.exists():
                continue
            a = torch.load(art, map_location="cpu", weights_only=False)["actors"][0]
            mean, std = a["mean"], a["std"]
            if kind == "pna":
                actor = PNADirectionalActor(node_dim, edge_dim, hidden=64, rounds=4, delta=pna_delta)
                actor.load_state_dict(a["state"])
            else:
                actor = T.load_actor_from_state(a["state"], node_dim, edge_dim, hidden=64, rounds=4)
            ve = T.eval_held(actor, held_s, mean, std)
            overall.append(ve["raw"])
            for n, r in ve["raw_by_n"].items():
                per_n.setdefault(int(n), []).append(r)
        report["arms"][arm] = {
            "n_seeds": len(overall),
            "overall": _agg(overall),
            "by_n": {n: _agg(per_n[n]) for n in sorted(per_n)},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    # compact table
    ns = sorted({n for arm in report["arms"].values() for n in arm["by_n"]})
    print(f"{'arm':40s} " + " ".join(f"N={n:>2}" for n in ns) + "  overall")
    for arm, d in report["arms"].items():
        cells = " ".join(f"{d['by_n'].get(n, {}).get('mean', float('nan')):.3f}" if n in d["by_n"]
                         else "  -  " for n in ns)
        ov = d["overall"]["mean"] if d["overall"] else float("nan")
        print(f"{arm:40s} {cells}  {ov:.3f} (n={d['n_seeds']})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
