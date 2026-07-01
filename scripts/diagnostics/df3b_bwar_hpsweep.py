"""DF3b (Decision-Focused, goal 1): BWAR hyperparameter robustness sweep (the pre-registered gate on the negative).

DF3's DEFAULT BWAR config lost to plain MSE on anchor feasibility. Per DF0 validation_design B.3 (and the
DF0-verify MINOR-3), a negative is honest ONLY IF a val-selected BEST BWAR config still fails to beat MSE on held
-- otherwise the negative could be a mis-tuned config. This sweep trains a grid of BWAR configs (INCLUDING an
MSE-equivalent config, beta=gamma=kappa=0), selects the config with the best VAL feasibility per seed, and reports
it on HELD. Three disjoint scene sets (train/val/held). If the val-best decision-focus config does not beat MSE on
held, decision-focus provably does not help at the anchor boundary.

Reuses DF3 (BWAR loss, collect, eval). Fast R8 mobility; true CSI = training label only; deployed 0-eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import df3_decision_focused_gen as df3  # noqa: E402
import t1_oracle_recovery_gen as t1  # noqa: E402
from r8_stale_csi_gen import _ci95, build_csi_scenes  # noqa: E402
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402

# BWAR config grid. mse_equiv (beta=gamma=kappa=0) is plain weighted MSE (decision-focus OFF) -- the in-grid
# control that any 'decision-focus helps' claim must beat.
CONFIGS = [
    {"name": "mse_equiv", "beta": 0.0, "gamma": 0.0, "kappa": 0.0},
    {"name": "bwar_default", "beta": 1.0, "gamma": 0.3, "kappa": 4.0, "c_fd": 1.0, "c_fk": 0.5},
    {"name": "bwar_beta0p5", "beta": 0.5, "gamma": 0.3, "kappa": 4.0},
    {"name": "bwar_beta2", "beta": 2.0, "gamma": 0.3, "kappa": 4.0},
    {"name": "bwar_kappa8", "beta": 1.0, "gamma": 0.3, "kappa": 8.0},
    {"name": "bwar_kappa0", "beta": 1.0, "gamma": 0.3, "kappa": 0.0},
    {"name": "bwar_cfd2", "beta": 1.0, "gamma": 0.3, "kappa": 4.0, "c_fd": 2.0, "c_fk": 0.5},
    {"name": "bwar_sym", "beta": 1.0, "gamma": 0.3, "kappa": 4.0, "c_fd": 1.0, "c_fk": 1.0},
    {"name": "bwar_gamma1", "beta": 1.0, "gamma": 1.0, "kappa": 4.0},
    {"name": "bwar_aux_only", "beta": 0.0, "gamma": 1.0, "kappa": 0.0},
]


def _train_cfg(X, Y, S, M, cfg, seed, epochs=400):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = t1._MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    Xn = (X - mu) / sd
    kw = {k: v for k, v in cfg.items() if k != "name"}
    for _ in range(epochs):
        opt.zero_grad()
        pred = (S + net(Xn)).clamp(0.0, 1.0)
        df3.bwar_loss(pred, Y, M, **kw).backward()
        opt.step()
    return net, mu, sd


def _feas(scenes, net):
    return sum(df3._eval_feas(scenes, lambda sc, t, obs, feats, stale: df3._predict(*net, feats, stale))) \
        / max(1, len(scenes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="urban", choices=["urban", "random"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-count", type=int, default=6)
    ap.add_argument("--val-count", type=int, default=10)
    ap.add_argument("--held-count", type=int, default=14)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF3" / "df3b_hpsweep_metrics.json"))
    args = ap.parse_args()

    csi = CsiObservationModel(mode="delay", delay_frames=1)
    mse_held, valbest_held, valbest_names = [], [], []
    per_cfg_held = {c["name"]: [] for c in CONFIGS}
    for s in range(args.seeds):
        tr = build_csi_scenes(args.data, s * 1000 + 1, args.train_count, df3._ARGS, csi)
        va = build_csi_scenes(args.data, s * 1000 + 333, args.val_count, df3._ARGS, csi)
        hd = build_csi_scenes(args.data, s * 1000 + 777, args.held_count, df3._ARGS, csi)
        X, Y, S, M = df3._collect(tr)
        mse_net = df3._train(X, Y, S, M, "mse", s)
        mse_held.append(_feas(hd, mse_net))
        best_val, best = -1.0, None
        for cfg in CONFIGS:
            net = _train_cfg(X, Y, S, M, cfg, s)
            vf, hf = _feas(va, net), _feas(hd, net)
            per_cfg_held[cfg["name"]].append(hf)
            if vf > best_val:
                best_val, best = vf, (cfg["name"], hf)
        valbest_held.append(best[1]); valbest_names.append(best[0])
        print(f"seed {s}: mse_held {mse_held[-1]:.3f} | val-best cfg {best[0]} held {best[1]:.3f}")

    report = {
        "scope": "DF3b BWAR hyperparameter robustness. Grid incl. mse_equiv (decision-focus OFF). Select best VAL "
                 "feasibility per seed, report HELD. If val-best decision-focus config does not beat MSE on held, "
                 "decision-focus does not help (honest negative). Fast R8 mobility.",
        "config": vars(args), "grid": [c["name"] for c in CONFIGS],
        "mse_held": _ci95(mse_held),
        "valbest_held": _ci95(valbest_held),
        "valbest_minus_mse": _ci95([valbest_held[i] - mse_held[i] for i in range(len(mse_held))]),
        "valbest_config_per_seed": valbest_names,
        "per_config_held": {k: _ci95(v) for k, v in per_cfg_held.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    vm = report["valbest_minus_mse"]
    print(f"\nmse_held {report['mse_held']['mean']} | valbest_held {report['valbest_held']['mean']} | "
          f"valbest-mse {vm['mean']} CI[{vm['lo']},{vm['hi']}] | val-best cfgs {valbest_names}")
    print("per-config held mean: " + " ".join(f"{k} {report['per_config_held'][k]['mean']}" for k in per_cfg_held))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
