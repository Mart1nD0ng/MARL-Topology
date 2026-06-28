"""Q2 (POMDP-QP-FAR): CSI-prediction + Temporal-Value health check under stale CSI.

Stage A of the spec (S13): BEFORE training any control policy, verify that the Q1 stale/partial CSI
observation actually creates a POMDP whose hidden current channel is recoverable from HISTORY and/or
VELOCITY. We train tiny predictors of the TRUE current link success probability g_t from the OBSERVED
(lagged/held) channel and compare:

    identity            -> predict the stale observed value as-is (the naive baseline)
    memoryless          -> MLP on [obs_psucc, age]                       (no velocity)
    memoryless+velocity -> MLP on [obs_psucc, age, rel_vel, dist_delta]
    recurrent           -> GRU over the observed [obs_psucc, age] history (no velocity)
    recurrent+velocity  -> GRU over [obs_psucc, age, rel_vel, dist_delta]

EXIT CONDITION (Spec Q2): recurrent OR velocity must beat memoryless/identity at predicting the true
current CSI. If not, the stale parameters are too weak (or the observation still too rich) and the
temporal machinery is not yet justified -- recorded honestly as a gate result, NOT spun as a general
failure of recurrence.

HARD INVARIANT: the TRUE current psucc is a training-only TARGET for the diagnostic (the deployed actor
never receives it). The OBSERVED input is exactly what the Q1 model exposes -- a real value from an
earlier true frame via ``CsiObservationModel.edge_plan`` (no fabrication, no leak). Eval-only: no
checkpoint, no reward, no change to any production path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import torch  # noqa: E402

from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402
from marl_topology.training.dynamic_frames import sample_dynamic_scenes  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402


# -- data extraction (pure; the testable core) ------------------------------------------------------

def _edge_endpoints(scene) -> dict:
    """edge_id -> (node_u, node_v), from the frame-invariant candidate graph."""
    return {e.edge_id: (e.node_u, e.node_v) for e in scene.context(0).graph.edges}


def _rel_velocity(scene, u, v, t):
    """Current local relative velocity along the link + distance delta (mirrors _motion_edge_block).
    Velocity/geometry are sensed FRESH each frame (not staleified) -- the signal that can extrapolate
    the lagged CSI forward."""
    pos = {n.node_id: n.position for n in scene.scenes[t].nodes}
    vux, vuy, _u = scene.velocities.get(u, (0.0, 0.0, 0.0))
    vvx, vvy, _w = scene.velocities.get(v, (0.0, 0.0, 0.0))
    pu, pv = pos[u], pos[v]
    dx, dy = pu.x_m - pv.x_m, pu.y_m - pv.y_m
    dist = math.hypot(dx, dy) or 1e-9
    rel_vel = ((vux - vvx) * dx + (vuy - vvy) * dy) / dist
    return rel_vel, rel_vel * scene.dt_s


def extract_series(scenes, csi_model: CsiObservationModel):
    """Return one sequence per edge: a list over frames of
    ``(obs_psucc, age, rel_vel, distance_delta, true_psucc)``.

    ``true_psucc`` is the TRUE current channel (target). ``obs_psucc`` is the Q1 OBSERVED value at the
    plan's source frame (``edge_plan``). Distances/velocities are the current (fresh) local motion.
    """
    out = []
    for scene in scenes:
        endpoints = _edge_endpoints(scene)
        # cache per-frame link records (true channel) once
        recs = [scene.context(t).link_records for t in range(scene.n_frames)]
        for eid, (u, v) in endpoints.items():
            seq = []
            for t in range(scene.n_frames):
                src, age, _obs_now = csi_model.edge_plan(eid, t)
                true_p = float(recs[t][eid].link_success_probability)
                obs_p = float(recs[src][eid].link_success_probability)
                rel_vel, dist_delta = _rel_velocity(scene, u, v, t)
                seq.append((obs_p, float(age), rel_vel, dist_delta, true_p))
            out.append(seq)
    return out


def _feature_dim(use_velocity: bool) -> int:
    return 4 if use_velocity else 2


def _seq_tensors(series, use_velocity: bool):
    """Pad to a (n_seq, T, F) input tensor + (n_seq, T) target + (n_seq, T) mask."""
    T = max(len(s) for s in series)
    F = _feature_dim(use_velocity)
    x = torch.zeros((len(series), T, F))
    y = torch.zeros((len(series), T))
    m = torch.zeros((len(series), T))
    for i, seq in enumerate(series):
        for t, (obs_p, age, rel_vel, dist_delta, true_p) in enumerate(seq):
            x[i, t, 0] = obs_p
            x[i, t, 1] = age
            if use_velocity:
                x[i, t, 2] = rel_vel
                x[i, t, 3] = dist_delta
            y[i, t] = true_p
            m[i, t] = 1.0
    return x, y, m


# -- metrics ----------------------------------------------------------------------------------------

def _mse(pred, true, mask):
    se = ((pred - true) ** 2 * mask).sum()
    return float(se / mask.sum().clamp_min(1.0))


def _spearman(pred, true, mask):
    """Spearman rank correlation over the masked flat samples (no scipy)."""
    sel = mask.bool()
    p = pred[sel].detach().flatten()
    t = true[sel].detach().flatten()
    if p.numel() < 3:
        return 0.0
    def _rank(z):
        order = torch.argsort(torch.argsort(z)).float()
        return order
    rp, rt = _rank(p), _rank(t)
    rp = rp - rp.mean(); rt = rt - rt.mean()
    denom = (rp.norm() * rt.norm()).clamp_min(1e-9)
    return float((rp @ rt) / denom)


def identity_metrics(series):
    """The naive baseline: predict obs_psucc as the current estimate."""
    x, y, m = _seq_tensors(series, use_velocity=False)
    pred = x[:, :, 0]                       # obs_psucc
    return {"mse": _mse(pred, y, m), "spearman": _spearman(pred, y, m)}


# -- predictors -----------------------------------------------------------------------------------

class _MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, 1))

    def forward(self, x):                   # x: (n, T, F) -> (n, T)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class _GRU(torch.nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.gru = torch.nn.GRU(in_dim, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):                   # x: (n, T, F) -> (n, T) (causal: hidden_t sees 0..t)
        h, _ = self.gru(x)
        return torch.sigmoid(self.head(h)).squeeze(-1)


def train_eval(train_series, held_series, *, arch: str, use_velocity: bool, epochs=300, lr=0.02, seed=0):
    torch.manual_seed(seed)
    xt, yt, mt = _seq_tensors(train_series, use_velocity)
    xh, yh, mh = _seq_tensors(held_series, use_velocity)
    in_dim = _feature_dim(use_velocity)
    model = _GRU(in_dim) if arch == "recurrent" else _MLP(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xt)
        loss = ((pred - yt) ** 2 * mt).sum() / mt.sum().clamp_min(1.0)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        ph = model(xh)
    return {"mse": _mse(ph, yh, mh), "spearman": _spearman(ph, yh, mh),
            "train_mse": _mse(model(xt), yt, mt)}


# -- driver -----------------------------------------------------------------------------------------

def _build(seed, count, tag, args, csi_model):
    from build_operating_point_dataset import operating_point_regime
    return sample_dynamic_scenes(
        seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
        regime=operating_point_regime(args.tx_power), num_frames=args.frames, dt_s=args.dt,
        speed_min_mps=args.speed_min, speed_max_mps=args.speed_max, reconfig=ReconfigCost(),
        hold_interval=args.hold_interval, gamma=args.gamma, tag=tag, csi_observation_model=csi_model)


def _model_for(mode, args):
    if mode == "current":
        return CsiObservationModel(mode="current")
    if mode == "delay1":
        return CsiObservationModel(mode="delay", delay_frames=1)
    if mode == "delay2":
        return CsiObservationModel(mode="delay", delay_frames=2)
    if mode == "partial":
        return CsiObservationModel(mode="partial", probe_probability=0.5, seed=args.seed)
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--hold-interval", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--tx-power", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4202)
    ap.add_argument("--modes", nargs="+", default=["current", "delay1", "delay2", "partial"])
    ap.add_argument("--out", default=str(ROOT / "result_save" / "csi_prediction_health.json"))
    args = ap.parse_args()

    arms = [("identity", None, None), ("memoryless", "memoryless", False),
            ("memoryless+velocity", "memoryless", True), ("recurrent", "recurrent", False),
            ("recurrent+velocity", "recurrent", True)]
    report = {"scope": "Q2 CSI-prediction health check under stale CSI (eval-only diagnostic)",
              "config": {"train": args.train, "held": args.held, "frames": args.frames, "dt_s": args.dt,
                         "speed_mps": [args.speed_min, args.speed_max], "epochs": args.epochs,
                         "seed": args.seed, "dyn_nodes": args.dyn_nodes}, "by_mode": {}}

    for mode in args.modes:
        csi_model = _model_for(mode, args)
        train_series = extract_series(_build(args.seed * 1000 + 1, args.train, "train_", args, csi_model), csi_model)
        held_series = extract_series(_build(args.seed * 1000 + 777, args.held, "held_", args, csi_model), csi_model)
        ident = identity_metrics(held_series)
        row = {"identity": ident, "n_held_edges": len(held_series)}
        for label, arch, vel in arms:
            if arch is None:
                continue
            row[label] = train_eval(train_series, held_series, arch=arch, use_velocity=vel,
                                     epochs=args.epochs, seed=args.seed)
        # verdict: does recurrent OR velocity beat memoryless and identity at held MSE?
        base = min(ident["mse"], row["memoryless"]["mse"])
        best_temporal = min(row["memoryless+velocity"]["mse"], row["recurrent"]["mse"],
                            row["recurrent+velocity"]["mse"])
        row["verdict"] = {"identity_mse": round(ident["mse"], 6),
                          "memoryless_mse": round(row["memoryless"]["mse"], 6),
                          "best_temporal_mse": round(best_temporal, 6),
                          "temporal_beats_baseline": bool(best_temporal < base - 1e-6),
                          "relative_improvement": round((base - best_temporal) / max(base, 1e-9), 4)}
        report["by_mode"][mode] = row
        print(f"[{mode}] identity_mse={ident['mse']:.5f} memoryless={row['memoryless']['mse']:.5f} "
              f"mem+vel={row['memoryless+velocity']['mse']:.5f} rec={row['recurrent']['mse']:.5f} "
              f"rec+vel={row['recurrent+velocity']['mse']:.5f} "
              f"-> temporal_beats_baseline={row['verdict']['temporal_beats_baseline']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
