"""T6 (Temporal Recovery): env temporal-structure diagnostic (task-1 fallback) -- WHY recovery fails, and what
env change would fix it.

The model-side levers (T2-T5) are exhausted: no realizable predictor recovers the decision-critical magnitude.
T6 asks the ENV-level question (task 1 "improve the env temporal hidden features"): does the STALE history carry
recoverable information about the current channel BEYOND what current geometry already gives? Two parts:

Part A (real env, robust variance-explained decomposition): fit small predictors of the true current psucc and
report held R^2:
  geo       : current geometry only [current distance, rel_vel, distance_delta]  (the leak-free recoverable part)
  temporal  : geometry + stale psucc                                              (adds the temporal history)
  echo      : predict the stale psucc (the trivial baseline)
The TEMPORAL CONTRIBUTION = R2_temporal - R2_geo is how much the stale history adds beyond geometry -- the
recoverable temporal structure. If it is ~0, the env's decision-critical component has no recoverable temporal
structure (the root of the T1-T5 negatives, at the env level).

Part B (synthetic AR sweep, method validation): generate a channel whose shadowing is AR(1) with correlation
rho, and show a predictor recovers a fraction of the variance that INCREASES with rho. This proves the method is
sound (it recovers temporally-autocorrelated structure when present) and quantifies the env autocorrelation that
task-1's "temporal hidden features" would need to add to make recovery convert. Self-contained; no env change."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from r8_stale_csi_gen import _ci95, build_csi_scenes  # noqa: E402
from marl_topology.training.csi_belief import belief_target_logits, leak_free_motion_features  # noqa: E402


class _A6:
    dyn_nodes = [8, 12, 16]
    frames = 8
    gamma = 0.95


class _MLP(nn.Module):
    def __init__(self, fin: int, hidden: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_r2(Xtr, ytr, Xhd, yhd, seed, epochs=300):
    """Train a small MLP; return held R^2 = 1 - MSE/Var(y). R^2<=0 -> no better than the mean."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    net = _MLP(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    Xn, Xhn = (Xtr - mu) / sd, (Xhd - mu) / sd
    for _e in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(net(Xn), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xhn)
    var = yhd.var().clamp_min(1e-9)
    return float(1.0 - ((pred - yhd) ** 2).mean() / var)


def _collect_real(scenes):
    """Pool per-edge (stale_psucc, current_distance, rel_vel, distance_delta, true_current_psucc) over frames
    t>=1 (a stale observation exists). Stale psucc = the actor's observed ef col 0 (frame t-1)."""
    S, GEO, Y = [], [], []
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            eids = obs["edge_ids"]
            if t == 0 or len(eids) == 0:
                prev = list(eids)[: max(1, len(eids) // 2)]
                continue
            ef = obs["ef"]
            motion = leak_free_motion_features(sc, t, eids)                       # [E,2] rel_vel, dist_delta
            # LEAK-FREE: col 0 is the STALE psucc (in csi_columns, staleified to t-1); col 5 (distance) is NOT in
            # csi_columns=(0,1,2,3) -> it stays CURRENT geometry (leak-free); the true current psucc is Y (label).
            S.append(ef[:, 0])                                                    # stale psucc (t-1)
            GEO.append(torch.stack([ef[:, 5], motion[:, 0], motion[:, 1]], dim=1))  # current geometry (leak-free)
            Y.append(torch.sigmoid(belief_target_logits(sc, t, eids)))           # true current psucc (label)
            prev = list(eids)[: max(1, len(eids) // 2)]
    return torch.cat(S), torch.cat(GEO), torch.cat(Y)


def _real_part(data, seed):
    tr = build_csi_scenes(data, seed * 1000 + 1, 4, _A6(), _CSI)
    hd = build_csi_scenes(data, seed * 1000 + 777, 4, _A6(), _CSI)
    s_tr, g_tr, y_tr = _collect_real(tr)
    s_hd, g_hd, y_hd = _collect_real(hd)
    r2_geo = _fit_r2(g_tr, y_tr, g_hd, y_hd, seed)
    r2_tmp = _fit_r2(torch.cat([g_tr, s_tr[:, None]], 1), y_tr,
                     torch.cat([g_hd, s_hd[:, None]], 1), y_hd, seed)
    var = y_hd.var().clamp_min(1e-9)
    r2_echo = float(1.0 - ((s_hd - y_hd) ** 2).mean() / var)                       # predict stale (echo)
    return {"r2_geo": r2_geo, "r2_temporal": r2_tmp, "temporal_contribution": r2_tmp - r2_geo,
            "r2_echo": r2_echo}


def _synth_ar(rho, seed, n=4000, hidden_std=1.5):
    """AR(1) shadow s_t = rho*s_{t-1} + sqrt(1-rho^2) eps; psucc_t = sigmoid(hidden_std*s_t) (the shadow is the
    ONLY structure). Predict the current psucc from the stale psucc; ``r2_pred`` (held R^2) is the ABSOLUTE
    recoverability -- it rises monotonically from ~0 at rho=0 (IID, unrecoverable) to ~1 at rho=1. This is the
    knob task-1's env temporal-hidden-features would add to the decision-critical channel component."""
    torch.manual_seed(1000 + seed)
    s_prev = torch.randn(n)
    s_cur = rho * s_prev + (1 - rho ** 2) ** 0.5 * torch.randn(n)
    p_prev = torch.sigmoid(hidden_std * s_prev)                                    # stale psucc
    p_cur = torch.sigmoid(hidden_std * s_cur)                                      # true current psucc
    k = n // 2
    r2 = _fit_r2(p_prev[:k, None], p_cur[:k], p_prev[k:, None], p_cur[k:], seed, epochs=400)
    var = p_cur[k:].var().clamp_min(1e-9)
    r2_echo = float(1.0 - ((p_prev[k:] - p_cur[k:]) ** 2).mean() / var)            # echo (predict stale)
    return {"rho": rho, "r2_pred": r2, "r2_echo_synth": r2_echo, "recovery_over_echo": r2 - r2_echo}


_CSI = None


def main() -> None:
    global _CSI
    from marl_topology.training.csi_observation_model import CsiObservationModel
    _CSI = CsiObservationModel(mode="delay", delay_frames=1)
    seeds = [0, 1, 2, 3, 4]
    report = {"scope": "T6 env temporal-structure diagnostic (delay-1): does the stale history add recoverable "
                       "info about the current psucc BEYOND geometry? real-env R^2 decomposition + synthetic AR "
                       "sweep. 5 seeds.", "real": {}, "synthetic_ar_sweep": {}}
    for data in ["random", "urban"]:
        per = {"r2_geo": [], "r2_temporal": [], "temporal_contribution": [], "r2_echo": []}
        for s in seeds:
            r = _real_part(data, s)
            for k in per:
                per[k].append(r[k])
        d = {k: _ci95(v) for k, v in per.items()}
        d["temporal_adds_beyond_geometry"] = (d["temporal_contribution"]["lo"] is not None
                                              and d["temporal_contribution"]["lo"] > 0.02)
        report["real"][data] = d
        print(f"[real {data}] R2_geo {d['r2_geo']['mean']} R2_temporal {d['r2_temporal']['mean']} "
              f"temporal_contribution {d['temporal_contribution']['mean']} "
              f"CI[{d['temporal_contribution']['lo']},{d['temporal_contribution']['hi']}] "
              f"adds={d['temporal_adds_beyond_geometry']} | R2_echo {d['r2_echo']['mean']}")
    for rho in [0.0, 0.3, 0.6, 0.9]:
        rr = [_synth_ar(rho, s) for s in seeds]
        d = {k: _ci95([r[k] for r in rr]) for k in ("r2_pred", "r2_echo_synth", "recovery_over_echo")}
        report["synthetic_ar_sweep"][f"rho_{rho}"] = d
        print(f"[synth rho={rho}] r2_pred {d['r2_pred']['mean']} CI[{d['r2_pred']['lo']},{d['r2_pred']['hi']}] "
              f"r2_echo {d['r2_echo_synth']['mean']} recovery_over_echo {d['recovery_over_echo']['mean']}")
    out = ROOT / "result_save" / "temporal_recovery" / "T6" / "env_temporal_structure_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
