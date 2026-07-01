"""DF3c (Decision-Focused, goal 1): the BWAR MARGINAL-SLOT variant -- the last pre-registered gate on the negative.

DF3 (default BWAR) and DF3b (10-config val sweep) both showed no decision-focus config beats plain MSE on anchor
feasibility. DF0 validation_design B.3 pre-registered ONE more requirement before an honest negative: the
budget/mutual-aware MARGINAL-SLOT variant must ALSO fail to beat MSE. The marginal-slot replaces BWAR's
boundary-distance weight with a weight that up-weights exactly the edges whose inclusion is on the node's BUDGET
MARGIN under the stale ordering (ranks b-1, b, b+1) -- the edges whose flip actually changes the top-k anchor
decision. This is the most decision-relevant weighting; if even it does not beat MSE, decision-focus provably does
not help (the limit is information-in-features).

Reuses DF3. Fast R8 mobility; true CSI = training label only; deployed 0-eval.
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
from marl_topology.training.decentralized_action import incident_index  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges  # noqa: E402

_KEEP, _ADD = 0.4, 0.6


def _slot(obs, stale):
    """Per-edge indicator: 1 if the edge sits at a node's BUDGET-MARGIN rank (b-1,b,b+1) under the stale-psucc
    ordering for either endpoint -- the edges whose inclusion is on the bubble of the top-k anchor decision."""
    edge_ids = obs["edge_ids"]
    budgets, edges = _budgets_edges(obs["context"])
    incident = incident_index(edge_ids, edges)
    slot = [0.0] * len(edge_ids)
    for node, idxs in incident.items():
        b = int(budgets.get(node, 0))
        ranked = sorted(idxs, key=lambda i: -float(stale[i]))
        for r in (b - 1, b, b + 1):
            if 0 <= r < len(ranked):
                slot[ranked[r]] = 1.0
    return torch.tensor(slot)


def _collect_slot(scenes):
    X, Y, S, M, SL = [], [], [], [], []
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            feats, stale = t1._edge_features(sc, t, obs)
            X.append(feats); Y.append(t1._true_psucc(sc, t, obs["edge_ids"])); S.append(stale)
            pset = set(prev)
            M.append(torch.tensor([1.0 if eid in pset else 0.0 for eid in obs["edge_ids"]]))
            SL.append(_slot(obs, stale))
            prev = list(df3._anchor(obs, prev))
    return torch.cat(X), torch.cat(Y), torch.cat(S), torch.cat(M), torch.cat(SL)


def bwar_marginal_loss(pred, true, m, slot, *, kappa=4.0, T=0.05, alpha=1.0, beta=1.0, gamma=0.3,
                       c_fd=1.0, c_fk=0.5):
    tau = torch.where(m > 0.5, torch.full_like(pred, _KEEP), torch.full_like(pred, _ADD))
    w = 1.0 + kappa * slot                                    # MARGINAL-SLOT weight (vs boundary-distance)
    above = (true >= tau).float()
    fd = c_fd * torch.clamp(tau - pred, min=0.0) * above
    fk = c_fk * torch.clamp(pred - tau, min=0.0) * (1.0 - above)
    reg = w * (pred - true) ** 2
    hinge = w * (fd + fk)
    aux = F.binary_cross_entropy_with_logits((pred - tau) / T, above)
    return (alpha * reg + beta * hinge).mean() + gamma * aux


def _train_marginal(X, Y, S, M, SL, seed, epochs=400):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = t1._MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    Xn = (X - mu) / sd
    for _ in range(epochs):
        opt.zero_grad()
        pred = (S + net(Xn)).clamp(0.0, 1.0)
        bwar_marginal_loss(pred, Y, M, SL).backward()
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
    ap.add_argument("--held-count", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF3" / "df3c_marginal_slot_metrics.json"))
    args = ap.parse_args()

    csi = CsiObservationModel(mode="delay", delay_frames=1)
    mse_h, marg_h, echo_h, true_h = [], [], [], []
    slot_frac = []
    for s in range(args.seeds):
        tr = build_csi_scenes(args.data, s * 1000 + 1, args.train_count, df3._ARGS, csi)
        hd = build_csi_scenes(args.data, s * 1000 + 777, args.held_count, df3._ARGS, csi)
        X, Y, S, M, SL = _collect_slot(tr)
        slot_frac.append(float(SL.mean()))
        mse_net = df3._train(X, Y, S, M, "mse", s)
        marg_net = _train_marginal(X, Y, S, M, SL, s)
        mse_h.append(_feas(hd, mse_net)); marg_h.append(_feas(hd, marg_net))
        echo_h.append(sum(df3._eval_feas(hd, lambda sc, t, obs, feats, stale: stale)) / len(hd))
        true_h.append(sum(df3._eval_feas(hd, lambda sc, t, obs, feats, stale: t1._true_psucc(sc, t, obs["edge_ids"]))) / len(hd))
        print(f"seed {s}: echo {echo_h[-1]:.3f} mse {mse_h[-1]:.3f} bwar_marginal {marg_h[-1]:.3f} "
              f"true {true_h[-1]:.3f} | slot_frac {slot_frac[-1]:.2f}")

    report = {
        "scope": "DF3c BWAR marginal-slot variant (budget-margin weighting) vs MSE on anchor feasibility -- the "
                 "last pre-registered gate on the goal-1 negative (DF0 B.3). Fast R8 mobility.",
        "config": vars(args), "csi": csi.manifest(),
        "echo": _ci95(echo_h), "mse": _ci95(mse_h), "bwar_marginal": _ci95(marg_h), "true": _ci95(true_h),
        "marginal_minus_mse": _ci95([marg_h[i] - mse_h[i] for i in range(len(mse_h))]),
        "mean_slot_fraction": sum(slot_frac) / len(slot_frac),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    mm = report["marginal_minus_mse"]
    print(f"\necho {report['echo']['mean']} mse {report['mse']['mean']} bwar_marginal {report['bwar_marginal']['mean']} "
          f"true {report['true']['mean']} | marginal-mse {mm['mean']} CI[{mm['lo']},{mm['hi']}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
