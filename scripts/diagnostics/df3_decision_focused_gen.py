"""DF3 (Decision-Focused, goal 1): decision-focused CSI prediction (BWAR) vs MSE, on ANCHOR FEASIBILITY.

The owner's headline goal: optimizing MSE is ineffective (predicting the average channel better does not help the
decision-critical edges); solve DECISION-relevant per-edge CSI prediction. DF2 showed MSE-trained recovery does
NOT convert. DF3 asks: does a DECISION-FOCUSED objective beat MSE at the anchor's keep(0.4)/add(0.6) boundary?

BWAR (Boundary-Weighted Asymmetric Regret-surrogate, DF0 validation_design B.1): boundary-weighted regression +
an asymmetric wrong-side hinge around the active gate + a side-classification aux. Trained on the true current
psucc LABEL (training-only); inference reads ONLY leak-free features (stale psucc/lat/en, current distance,
rel-vel, dist-delta, csi_age). The active gate tau_e = 0.4 if the edge is in prev_topo (keep) else 0.6 (add),
built from the SAME anchor code path as deployment (train==deploy).

Arms fed to the SAME anchor, scored on the TRUE evaluator (0 eval at decision):
  echo        stale psucc (the deployed floor)
  mse         MLP trained with plain MSE (physics-residual) -- the DF2 realizable
  recal_mse   the MSE predictor + a global monotone transform g(p)=sigmoid(a*logit(p)+b) fit on TRAIN to MAXIMIZE
              anchor feasibility (the CONTROL for the fake-positive from BWAR's asymmetric hinge: if BWAR merely
              shifts predictions globally, a 2-param recalibration of the MSE predictor captures the same gain --
              DF0-verify MAJOR-1). HEADLINE = Feas(BWAR) - Feas(recal_mse).
  bwar        MLP trained with the BWAR loss (physics-residual)
  true        true current psucc (the oracle ceiling)

Reports per-seed + CI: feasibility per arm; PRIMARY Feas(bwar)-Feas(recal_mse); Feas(bwar)-Feas(mse); the MSE
landscape (decision-focus may WORSEN MSE while improving feasibility -- the expected signature). Fast R8 mobility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import t1_oracle_recovery_gen as t1  # noqa: E402
import r6_evidence_gate_gen as g6  # noqa: E402
from r8_stale_csi_gen import _ci95, _TAU_FEAS, build_csi_scenes  # noqa: E402
from marl_topology.training.edit_head_training import _anchor  # noqa: E402
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402

_KEEP, _ADD = 0.4, 0.6
ARMS = ("echo", "mse", "recal_mse", "bwar", "true")


def bwar_loss(pred, true, m, *, kappa=4.0, h=0.05, T=0.05, alpha=1.0, beta=1.0, gamma=0.3, c_fd=1.0, c_fk=0.5):
    """Boundary-Weighted Asymmetric Regret-surrogate. m = 1 if edge in prev_topo (gate 0.4) else 0 (gate 0.6)."""
    tau = torch.where(m > 0.5, torch.full_like(pred, _KEEP), torch.full_like(pred, _ADD))
    w = 1.0 + kappa * torch.exp(-(true - tau) ** 2 / (2 * h * h))
    above = (true >= tau).float()
    fd = c_fd * torch.clamp(tau - pred, min=0.0) * above          # false-DROP: true above, predicted below
    fk = c_fk * torch.clamp(pred - tau, min=0.0) * (1.0 - above)  # false-KEEP: true below, predicted above
    reg = w * (pred - true) ** 2
    hinge = w * (fd + fk)
    aux = F.binary_cross_entropy_with_logits((pred - tau) / T, above)
    return (alpha * reg + beta * hinge).mean() + gamma * aux


def _collect(scenes):
    """Roll the anchor (deployment distribution); collect (features, true psucc, stale, membership-in-prev)."""
    X, Y, S, M = [], [], [], []
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            feats, stale = t1._edge_features(sc, t, obs)
            X.append(feats); Y.append(t1._true_psucc(sc, t, obs["edge_ids"])); S.append(stale)
            pset = set(prev)
            M.append(torch.tensor([1.0 if eid in pset else 0.0 for eid in obs["edge_ids"]]))
            prev = list(_anchor(obs, prev))
    return torch.cat(X), torch.cat(Y), torch.cat(S), torch.cat(M)


def _train(X, Y, S, M, mode, seed, epochs=400):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = t1._MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    Xn = (X - mu) / sd
    for _ in range(epochs):
        opt.zero_grad()
        pred = (S + net(Xn)).clamp(0.0, 1.0)                      # physics-residual on the stale echo
        loss = F.mse_loss(pred, Y) if mode == "mse" else bwar_loss(pred, Y, M)
        loss.backward(); opt.step()
    return net, mu, sd


def _predict(net, mu, sd, X, S):
    net.eval()
    with torch.no_grad():
        return (S + net((X - mu) / sd)).clamp(0.0, 1.0)


def _recal(p, a, b):
    p = p.clamp(1e-4, 1.0 - 1e-4)
    return torch.sigmoid(a * torch.log(p / (1.0 - p)) + b)


def _psucc_fn(arm, mse_net, bwar_net, ab):
    if arm == "echo":
        return lambda sc, t, obs, feats, stale: stale
    if arm == "true":
        return lambda sc, t, obs, feats, stale: t1._true_psucc(sc, t, obs["edge_ids"])
    if arm == "mse":
        return lambda sc, t, obs, feats, stale: _predict(*mse_net, feats, stale)
    if arm == "bwar":
        return lambda sc, t, obs, feats, stale: _predict(*bwar_net, feats, stale)
    return lambda sc, t, obs, feats, stale: _recal(_predict(*mse_net, feats, stale), ab[0], ab[1])


def _eval_feas(scenes, psucc_fn):
    """Per-scene feasibility (fraction of frames feasible) rolling the anchor with the arm's psucc."""
    per = []
    for sc in scenes:
        prev, fr = [], []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            feats, stale = t1._edge_features(sc, t, obs)
            topo = t1._anchor_with_psucc(obs, prev, psucc_fn(sc, t, obs, feats, stale))
            fr.append(1.0 if g6._cons(obs["context"].evaluator, topo) >= _TAU_FEAS else 0.0)
            prev = list(topo)
        per.append(sum(fr) / len(fr))
    return per


def _fit_recal(mse_net, scenes):
    """Fit g(p)=sigmoid(a*logit(p)+b) on TRAIN to MAXIMIZE anchor feasibility (the global-transform control)."""
    best, best_ab = -1.0, (1.0, 0.0)
    for a in (0.8, 1.0, 1.4):
        for b in (-0.1, 0.0, 0.1):
            fn = lambda sc, t, obs, feats, stale, _a=a, _b=b: _recal(_predict(*mse_net, feats, stale), _a, _b)
            feas = sum(_eval_feas(scenes, fn)) / max(1, len(scenes))
            if feas > best:
                best, best_ab = feas, (a, b)
    return best_ab


def _mse_landscape(scenes, mse_net, bwar_net):
    se = {"echo": 0.0, "mse": 0.0, "bwar": 0.0}
    n = 0
    for sc in scenes:
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            feats, stale = t1._edge_features(sc, t, obs)
            true = t1._true_psucc(sc, t, obs["edge_ids"])
            se["echo"] += float(((stale - true) ** 2).sum())
            se["mse"] += float(((_predict(*mse_net, feats, stale) - true) ** 2).sum())
            se["bwar"] += float(((_predict(*bwar_net, feats, stale) - true) ** 2).sum())
            n += true.numel()
    return {k: v / n for k, v in se.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="urban", choices=["urban", "random"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-count", type=int, default=6)
    ap.add_argument("--held-count", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF3" / "df3_bwar_metrics.json"))
    args = ap.parse_args()

    csi = CsiObservationModel(mode="delay", delay_frames=1)
    feas = {a: [] for a in ARMS}
    prim_bwar_recal, sec_bwar_mse, sec_bwar_echo = [], [], []
    mse_land = {"echo": [], "mse": [], "bwar": []}
    recal_ab = []
    for s in range(args.seeds):
        tr = build_csi_scenes(args.data, s * 1000 + 1, args.train_count, _ARGS, csi)
        hd = build_csi_scenes(args.data, s * 1000 + 777, args.held_count, _ARGS, csi)
        X, Y, S, M = _collect(tr)
        mse_net = _train(X, Y, S, M, "mse", s)
        bwar_net = _train(X, Y, S, M, "bwar", s)
        ab = _fit_recal(mse_net, tr)
        recal_ab.append(list(ab))
        per = {a: _eval_feas(hd, _psucc_fn(a, mse_net, bwar_net, ab)) for a in ARMS}
        fmean = {a: sum(per[a]) / len(per[a]) for a in ARMS}
        for a in ARMS:
            feas[a].append(fmean[a])
        prim_bwar_recal.append(fmean["bwar"] - fmean["recal_mse"])
        sec_bwar_mse.append(fmean["bwar"] - fmean["mse"])
        sec_bwar_echo.append(fmean["bwar"] - fmean["echo"])
        land = _mse_landscape(hd, mse_net, bwar_net)
        for k in mse_land:
            mse_land[k].append(land[k])
        print(f"seed {s}: echo {fmean['echo']:.3f} mse {fmean['mse']:.3f} recal {fmean['recal_mse']:.3f} "
              f"bwar {fmean['bwar']:.3f} true {fmean['true']:.3f} | ab={ab} | "
              f"MSE echo {land['echo']:.4f} mse {land['mse']:.4f} bwar {land['bwar']:.4f}")

    report = {
        "scope": "DF3 decision-focused BWAR vs MSE on anchor feasibility; recal_mse = global-transform control "
                 "(catches BWAR global-shift fake positive); PRIMARY = Feas(bwar)-Feas(recal_mse); fast R8 "
                 "mobility; true CSI = training label only; deployed 0-eval; true evaluator scores.",
        "config": vars(args), "csi": csi.manifest(),
        "feas": {a: _ci95(feas[a]) for a in ARMS},
        "PRIMARY_bwar_minus_recal": _ci95(prim_bwar_recal),
        "bwar_minus_mse": _ci95(sec_bwar_mse),
        "bwar_minus_echo": _ci95(sec_bwar_echo),
        "mse_landscape": {k: _ci95(v) for k, v in mse_land.items()},
        "recal_ab_per_seed": recal_ab,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    p = report["PRIMARY_bwar_minus_recal"]
    print(f"\nPRIMARY Feas(bwar)-Feas(recal_mse) mean {p['mean']} CI[{p['lo']},{p['hi']}] | "
          f"bwar-mse {report['bwar_minus_mse']['mean']} bwar-echo {report['bwar_minus_echo']['mean']}")
    print(f"feas: " + " ".join(f"{a} {report['feas'][a]['mean']}" for a in ARMS))
    print(f"wrote {args.out}")


class _ARGS:
    dyn_nodes = [8, 12, 16]
    frames = 4
    gamma = 0.95


if __name__ == "__main__":
    main()
