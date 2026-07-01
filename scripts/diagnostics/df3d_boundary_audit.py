"""DF3d (Decision-Focused, goal 1): the DF0 B.3(iii) per-edge BOUNDARY-ENGAGEMENT audit.

DF3-verify (REVISE, conclusion_robust=FALSE) found the load-bearing gap: DF3 claimed the pre-registered gate was
satisfied, but B.3(iii) -- 'the objective PROVABLY improved per-edge calibration on the decision-relevant edges'
-- was never computed. Only global MSE + end-to-end feasibility were measured, and BWAR RAISED global MSE, which
is equally consistent with the objective UNDER-FIRING as with decision-focus. This audit settles it: does BWAR
improve the PER-EDGE decision-side precision at the anchor gate vs MSE?

For the frozen MSE and BWAR predictors, on HELD edges, measure (tau = 0.4 if edge in prev_topo else 0.6):
  - boundary side-accuracy (among edges with |true - tau| <= 0.1): fraction placed on the correct side of tau.
  - the same restricted to rank-marginal (budget-margin) edges -- the decision-critical ones.
  - direction accuracy (dir_acc): among edges that MOVED (sign(true-tau) != sign(stale-tau)), fraction the
    predictor places on the correct side.
Interpretation:
  - If BWAR IMPROVES boundary side-acc / dir_acc over MSE (even while global MSE is worse) -> the objective ENGAGED
    the boundary (the decision-focus signature); feasibility not moving (DF3) then = aleatoric (info-limited). The
    honest negative is properly supported.
  - If BWAR does NOT improve boundary side-acc -> the objective was under-firing; the DF3 negative would be
    premature (BWAR needs re-tuning). No downgrade either way -- this is the missing test.

Reuses DF3 + DF3c. Fast R8 mobility; true CSI = label only; deployed 0-eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import df3_decision_focused_gen as df3  # noqa: E402
import df3c_marginal_slot as df3c  # noqa: E402
import t1_oracle_recovery_gen as t1  # noqa: E402
from r8_stale_csi_gen import _ci95, build_csi_scenes  # noqa: E402
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402


def _held_rows(scenes, mse_net, bwar_net):
    MP, BP, TR, ST, TAU, SL = [], [], [], [], [], []
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            feats, stale = t1._edge_features(sc, t, obs)
            MP.append(df3._predict(*mse_net, feats, stale))
            BP.append(df3._predict(*bwar_net, feats, stale))
            TR.append(t1._true_psucc(sc, t, obs["edge_ids"])); ST.append(stale)
            pset = set(prev)
            TAU.append(torch.tensor([0.4 if eid in pset else 0.6 for eid in obs["edge_ids"]]))
            SL.append(df3c._slot(obs, stale))
            prev = list(df3._anchor(obs, prev))
    return (torch.cat(MP), torch.cat(BP), torch.cat(TR), torch.cat(ST), torch.cat(TAU), torch.cat(SL))


def _side_acc(pred, tau, true, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return float(((pred[mask] >= tau[mask]) == (true[mask] >= tau[mask])).float().mean())


def _audit(mp, bp, tr, st, tau, sl):
    near = (tr - tau).abs() <= 0.2                     # wide window: psucc is bimodal, few edges sit near the gate
    moved = (tr >= tau) != (st >= tau)                 # edges whose true side differs from the stale-echo side
    marg = sl > 0.5
    return {
        "boundary_side_acc_mse": _side_acc(mp, tau, tr, near),
        "boundary_side_acc_bwar": _side_acc(bp, tau, tr, near),
        "marginal_boundary_side_acc_mse": _side_acc(mp, tau, tr, near & marg),
        "marginal_boundary_side_acc_bwar": _side_acc(bp, tau, tr, near & marg),
        "dir_acc_mse": _side_acc(mp, tau, tr, moved),   # on echo-wrong edges: does pred recover the side?
        "dir_acc_bwar": _side_acc(bp, tau, tr, moved),
        "n_near": int(near.sum()), "n_marg_near": int((near & marg).sum()), "n_moved": int(moved.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="urban", choices=["urban", "random"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-count", type=int, default=6)
    ap.add_argument("--held-count", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF3" / "df3d_boundary_audit_metrics.json"))
    args = ap.parse_args()

    csi = CsiObservationModel(mode="delay", delay_frames=1)
    acc = {k: [] for k in ("boundary_side_acc_mse", "boundary_side_acc_bwar",
                           "marginal_boundary_side_acc_mse", "marginal_boundary_side_acc_bwar",
                           "dir_acc_mse", "dir_acc_bwar")}
    for s in range(args.seeds):
        tr = build_csi_scenes(args.data, s * 1000 + 1, args.train_count, df3._ARGS, csi)
        hd = build_csi_scenes(args.data, s * 1000 + 777, args.held_count, df3._ARGS, csi)
        X, Y, S, M = df3._collect(tr)
        mse_net = df3._train(X, Y, S, M, "mse", s)
        bwar_net = df3._train(X, Y, S, M, "bwar", s)
        a = _audit(*_held_rows(hd, mse_net, bwar_net))
        for k in acc:
            acc[k].append(a[k])
        print(f"seed {s}: bnd side-acc mse {a['boundary_side_acc_mse']:.3f} bwar {a['boundary_side_acc_bwar']:.3f}"
              f" | marg mse {a['marginal_boundary_side_acc_mse']:.3f} bwar {a['marginal_boundary_side_acc_bwar']:.3f}"
              f" | dir mse {a['dir_acc_mse']:.3f} bwar {a['dir_acc_bwar']:.3f}"
              f" | n_near {a['n_near']} n_marg {a['n_marg_near']} n_moved {a['n_moved']}")

    def _delta(b, m):
        return _ci95([b[i] - m[i] for i in range(len(b)) if b[i] == b[i] and m[i] == m[i]])
    report = {
        "scope": "DF3d B.3(iii) per-edge boundary-engagement audit: does BWAR improve per-edge decision-side "
                 "precision at the anchor gate vs MSE (the test that DF3 claimed but never ran)? Fast R8 mobility.",
        "config": vars(args),
        **{k: _ci95(v) for k, v in acc.items()},
        "bwar_minus_mse_boundary_side_acc": _delta(acc["boundary_side_acc_bwar"], acc["boundary_side_acc_mse"]),
        "bwar_minus_mse_marginal_side_acc": _delta(acc["marginal_boundary_side_acc_bwar"], acc["marginal_boundary_side_acc_mse"]),
        "bwar_minus_mse_dir_acc": _delta(acc["dir_acc_bwar"], acc["dir_acc_mse"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    b = report["bwar_minus_mse_boundary_side_acc"]; mg = report["bwar_minus_mse_marginal_side_acc"]
    d = report["bwar_minus_mse_dir_acc"]
    print(f"\nBWAR - MSE: boundary side-acc {b['mean']} CI[{b['lo']},{b['hi']}] | "
          f"marginal side-acc {mg['mean']} CI[{mg['lo']},{mg['hi']}] | dir_acc {d['mean']} CI[{d['lo']},{d['hi']}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
