"""DF2b (Decision-Focused, goal 2): persistent-link decision-recovery attribution probe.

DF2's oracle re-gate showed feas(realizable) == feas(stale) (a real per-link MSE gain that produces ZERO
feasibility movement). This probe attributes the failure on IDEALIZED persistent links (fixed tx-rx pairs,
boundary tx power) where DF1 proved rho sweeps with d_corr. It separates GEOMETRY recovery (current distance,
leak-free) from PURE-TEMPORAL recovery (stale psucc lags only) with NESTED feature-subset arms, so the
geometry-vs-temporal claim is NOT tautological (v2, after DF2-verify REVISE hole 3):

  echo       : baseline = stale psucc at t-1 (no predictor).
  temporal2  : residual predictor from TWO stale psucc lags {p[t-1], p[t-2]} -- NO distance. Can AR-extrapolate a
               smooth shadow trend; if its edge over echo RISES with rho, temporal recovery is genuinely possible.
  geometry   : residual predictor from {p[t-1], current distance d[t]} -- the "current geometry observed" arm.
  full       : {p[t-1], p[t-2], d[t]}.

For each arm we report MSE, decision-side accuracy at gate tau, flip_recovery (fraction of echo-WRONG decision
cases the arm gets RIGHT; floor 0 = copies the wrong echo, ~0.5 = independent guess), and each metric's ADVANTAGE
over the echo, as a function of the measured lag-1 psucc autocorrelation rho. NLOSv off (isolate the shadow knob).
Reuses the DF1 probe (persistent moving links, leak-free). Deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import df1_decorr_env_probe as df1  # noqa: E402
from marl_topology.channel import ChannelModelConfig, evaluate_channel  # noqa: E402
from marl_topology.channel.model import PATH_LOSS_MODEL_V2X  # noqa: E402
from marl_topology.link import LinkTransmissionConfig, evaluate_link_transmission  # noqa: E402

DECORRS = (10.0, 25.0, 50.0, 100.0, 200.0)
# arm -> the columns of X = [p_lag1, p_lag2, dist_norm] it may use (echo uses none; it is the raw stale value)
ARMS = {"temporal2": [0, 1], "geometry": [0, 2], "full": [0, 1, 2]}


def _series(link, d_corr, field_seed, n_frames, dt, link_cfg, tx):
    cfg = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True, nlosv_37885=False,
                             shadow_decorrelation_distance_m=d_corr, default_tx_power_dbm=tx)
    ps, ds = [], []
    for f in range(n_frames):
        rec = evaluate_channel(df1._scene_at(link, f, dt), "tx", "rx", config=cfg, shadowing_seed=field_seed)
        ps.append(float(evaluate_link_transmission(rec, link_cfg).packet_success_probability))
        ds.append(float(rec.distance_3d_m))
    return ps, ds


class _MLP(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _samples(links, d_corr, field_seeds, n_frames, dt, link_cfg, tx):
    """X = [p[t-1], p[t-2], dist[t]/100] for t>=2; S = echo = p[t-1]; Y = p[t]; plus mean per-link lag-1 rho."""
    Xs, St, Y, rhos = [], [], [], []
    for link in links:
        for fs in field_seeds:
            ps, ds = _series(link, d_corr, fs, n_frames, dt, link_cfg, tx)
            r = df1._lag1_autocorr(ps)
            if r is not None:
                rhos.append(r)
            for t in range(2, n_frames):
                Xs.append([ps[t - 1], ps[t - 2], ds[t] / 100.0])
                St.append(ps[t - 1])
                Y.append(ps[t])
    return (torch.tensor(Xs), torch.tensor(St), torch.tensor(Y),
            (sum(rhos) / len(rhos) if rhos else float("nan")))


def _train(X, Y, S, seed, epochs=300):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = _MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        pred = (S + net((X - mu) / sd)).clamp(0.0, 1.0)
        F.mse_loss(pred, Y).backward()
        opt.step()
    return net, mu, sd


def _predict(net, mu, sd, X, S):
    net.eval()
    with torch.no_grad():
        return (S + net((X - mu) / sd)).clamp(0.0, 1.0)


def _recovery(stale, pred, true, tau):
    es, rs, ts = (stale >= tau), (pred >= tau), (true >= tau)
    wrong = (es != ts)
    n_flip = int(wrong.sum())
    flip_rec = float((rs[wrong] == ts[wrong]).float().mean()) if n_flip else float("nan")
    return float((rs == ts).float().mean()), flip_rec, n_flip


def _spearman(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pts) < 3:
        return float("nan")

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank([p[0] for p in pts]), rank([p[1] for p in pts])
    n = len(pts)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decorrs", type=float, nargs="+", default=list(DECORRS))
    ap.add_argument("--links", type=int, default=200)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--field-seeds", type=int, default=4)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--speed-min", type=float, default=5.0)
    ap.add_argument("--speed-max", type=float, default=15.0)
    ap.add_argument("--dist-min", type=float, default=60.0)
    ap.add_argument("--dist-max", type=float, default=170.0)
    ap.add_argument("--tx-power", type=float, default=-8.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF2" / "df2b_persistent_recovery.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_links = df1._gen_links(rng, args.links, args.speed_min, args.speed_max, args.dist_min, args.dist_max)
    split = int(0.6 * len(all_links))
    tr_links, hd_links = all_links[:split], all_links[split:]
    fseeds = list(range(args.field_seeds))
    link_cfg = LinkTransmissionConfig()

    by_decorr, rho_list = {}, []
    arm_adv = {a: {"mse": [], "side": [], "flip": []} for a in ARMS}   # arm advantage-over-echo vs rho
    for d_corr in args.decorrs:
        Xtr, Str, Ytr, _ = _samples(tr_links, d_corr, fseeds, args.frames, args.dt, link_cfg, args.tx_power)
        Xhd, Shd, Yhd, rho = _samples(hd_links, d_corr, fseeds, args.frames, args.dt, link_cfg, args.tx_power)
        rho_list.append(rho)
        mse_echo = float(((Shd - Yhd) ** 2).mean())
        echo_side, echo_flip, n_flip = _recovery(Shd, Shd, Yhd, args.tau)
        entry = {"rho": rho, "n_held": int(Yhd.numel()), "n_flip": n_flip,
                 "echo": {"mse": mse_echo, "side_acc": echo_side, "flip_recovery": echo_flip}}
        for a, cols in ARMS.items():
            net = _train(Xtr[:, cols], Ytr, Str, args.seed)
            pred = _predict(*net, Xhd[:, cols], Shd)
            mse = float(((pred - Yhd) ** 2).mean())
            side, flip, _ = _recovery(Shd, pred, Yhd, args.tau)
            entry[a] = {"mse": mse, "side_acc": side, "flip_recovery": flip,
                        "mse_adv_vs_echo": mse_echo - mse, "side_adv_vs_echo": side - echo_side}
            arm_adv[a]["mse"].append(mse_echo - mse)
            arm_adv[a]["side"].append(side - echo_side)
            arm_adv[a]["flip"].append(flip)
        by_decorr[str(d_corr)] = entry
        print(f"d_corr={d_corr:6.0f} rho={rho:+.3f} n_flip={n_flip:4d} | MSE echo {mse_echo:.4f} "
              f"temporal2 {entry['temporal2']['mse']:.4f} geometry {entry['geometry']['mse']:.4f} "
              f"full {entry['full']['mse']:.4f} | side_adv T {entry['temporal2']['side_adv_vs_echo']:+.3f} "
              f"G {entry['geometry']['side_adv_vs_echo']:+.3f}")

    # Does each arm's advantage over echo RISE with rho? Spearman over the sweep (rho-ordered, not endpoint diff).
    trend = {a: {"mse_adv_vs_rho_spearman": _spearman(rho_list, arm_adv[a]["mse"]),
                 "side_adv_vs_rho_spearman": _spearman(rho_list, arm_adv[a]["side"])}
             for a in ARMS}
    report = {
        "scope": "DF2b v2 persistent-link decision-recovery. Idealized boundary-tx links, clean rho sweep. NESTED "
                 "arms {echo, temporal2=2 stale lags no distance, geometry=echo+current distance, full}. Tests "
                 "geometry-vs-temporal NON-tautologically: does the PURE-TEMPORAL arm's edge over echo rise with "
                 "rho? Metric trends use Spearman over rho, not endpoint diffs (DF2-verify hole 4).",
        "config": vars(args), "gate_tau": args.tau, "rho_by_decorr": {str(d): by_decorr[str(d)]["rho"] for d in args.decorrs},
        "by_decorr": by_decorr, "advantage_vs_rho_spearman": trend,
        "note_flip_recovery_floor": "0 = arm copies the WRONG echo call; ~0.5 = independent guess; higher = recovery.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nADVANTAGE-vs-rho Spearman (does the arm's edge over echo rise with rho?):")
    for a in ARMS:
        print(f"  {a:10s} MSE-adv rho-corr {trend[a]['mse_adv_vs_rho_spearman']:+.3f} | "
              f"side-adv rho-corr {trend[a]['side_adv_vs_rho_spearman']:+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
