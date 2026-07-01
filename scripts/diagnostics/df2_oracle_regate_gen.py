"""DF2 (Decision-Focused, goal 2 GATE): oracle re-gate under the decorrelation sweep.

Question: does the tunable, marginal-invariant temporal structure DF1 created become CONVERTIBLE? Under higher
autocorrelation rho, does a REALIZABLE predictor beat the distance-only geometry null and close a CI-positive,
non-trivial fraction of the oracle feasibility gap at the anchor's N<=16 decision boundary?

Reuses the T1 oracle harness (feed the anchor a psucc vector, score the produced topology with the TRUE
evaluator, 0 eval at decision; train leak-free predictors of true psucc). Adds, per DF0 validation_design v2:
  - the d_corr sweep (dataclasses.replace(operating_point_regime(20.0), shadow_decorrelation_distance_m=d_corr));
  - a SLOW urban mobility regime (dt=1 s, 5-15 m/s ~ 10 m/frame) so the sweep {10,25,50,100,200} m spans
    rho ~ exp(-10/d_corr) ~ 0.37..0.95 (the fast operating-point regime, 30-60 m/frame, has rho~0 even at 100 m);
  - the NESTED-SUBSET arms (M2 fix): geometry_only = best predictor from DISTANCE ALONE; mse_realizable = best
    predictor from the FULL leak-free set; plus stale echo (floor) and true_csi (ceiling);
  - RGF = (Feas(arm)-Feas(stale))/(Feas(true)-Feas(stale)), pooled per seed, CI over seeds (M3: pooled not
    per-frame); PRIMARY endpoint RGF(mse_realizable) - RGF(geometry_only) = incremental value over distance;
  - the measured rho per setting, the MSE landscape, and a boundary-restricted side-accuracy leak/recovery check.

motion_features are OFF (build path leaves them off -> the csi_delta current-psucc leak is closed). NLOSv is ON
here (faithful), unlike the DF1 isolation probe. Deterministic (seeded). Gate-exempt diagnostics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import t1_oracle_recovery_gen as t1  # noqa: E402
import r6_evidence_gate_gen as g6  # noqa: E402
from r8_stale_csi_gen import _ci95, _TAU_FEAS  # noqa: E402
from marl_topology.training.edit_head_training import _anchor  # noqa: E402
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402
from marl_topology.training.csi_belief import belief_target_logits  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

ARMS = ("stale", "geometry", "realizable", "true")
_KEEP, _ADD = 0.4, 0.6


def _build_scenes(data, seed, count, d_corr, dt_s, speed_min, speed_max, csi):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import (
        sample_dynamic_scenes, sample_dynamic_urban_scenes)
    regime = dataclasses.replace(operating_point_regime(20.0), shadow_decorrelation_distance_m=float(d_corr))
    common = dict(seed=seed, count=count, node_count_choices=(8, 12, 16), regime=regime,
                  num_frames=4, dt_s=dt_s, speed_min_mps=speed_min, speed_max_mps=speed_max,
                  reconfig=ReconfigCost(), hold_interval=4, gamma=0.95, csi_observation_model=csi)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def _lag1(xs):
    n = len(xs)
    if n < 3:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    if var < 1e-9:
        return None
    cov = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(n - 1)) / (n - 1)
    return cov / var


def _measure_rho(scenes):
    """Lag-1 autocorr of the DECISION-CRITICAL true psucc over trajectory frames (prev-independent: geometry is
    driven by frame index). Per-edge series (by edge_id), averaged. Also returns psucc-distribution stats and the
    lag-1 autocorr of the stale->true DELTA (the recoverable signal), plus how many edge-series were non-constant
    (defined) -- to diagnose saturation confounds."""
    rhos, all_p, n_series, n_defined = [], [], 0, 0
    for sc in scenes:
        series: dict = {}
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            eids = obs["edge_ids"]
            lg = belief_target_logits(sc, t, eids)          # UNSATURATED decision logit (rho is measured here)
            tp = torch.sigmoid(lg)                           # psucc prob (for saturation stats only)
            for i, eid in enumerate(eids):
                series.setdefault(eid, []).append(float(lg[i]))
                all_p.append(float(tp[i]))
        for s in series.values():
            n_series += 1
            r = _lag1(s)
            if r is not None:
                rhos.append(r); n_defined += 1
    npv = len(all_p) or 1
    stats = {
        "psucc_mean": sum(all_p) / npv,
        "frac_saturated_hi": sum(1 for p in all_p if p >= 0.95) / npv,
        "frac_saturated_lo": sum(1 for p in all_p if p <= 0.05) / npv,
        "frac_boundary": sum(1 for p in all_p if 0.3 <= p <= 0.7) / npv,
        "rho_defined_frac": (n_defined / n_series) if n_series else float("nan"),
    }
    return (sum(rhos) / len(rhos) if rhos else float("nan")), stats


def _eval_per_scene(scenes, geo, real):
    """Per-scene feasibility (fraction of frames feasible) for each arm; each arm keeps its own prev trajectory."""
    per = {a: [] for a in ARMS}
    for sc in scenes:
        prev = {a: [] for a in ARMS}
        fr = {a: [] for a in ARMS}
        for t in range(sc.n_frames):
            for a in ARMS:
                obs = sc.observation(t, prev[a])
                feats, stale = t1._edge_features(sc, t, obs)
                if a == "stale":
                    p = stale
                elif a == "true":
                    p = t1._true_psucc(sc, t, obs["edge_ids"])
                elif a == "geometry":
                    p = t1._predict(*geo, feats[:, 3:4], stale, "direct")
                else:
                    p = t1._predict(*real, feats, stale, "physics")
                topo = t1._anchor_with_psucc(obs, prev[a], p)
                ev = obs["context"].evaluator
                fr[a].append(1.0 if g6._cons(ev, topo) >= _TAU_FEAS else 0.0)
                prev[a] = list(topo)
        for a in ARMS:
            per[a].append(sum(fr[a]) / len(fr[a]))
    return per


def _mse_and_boundary(scenes, geo, real):
    """MSE landscape (echo/geo/realizable) + boundary-restricted side-accuracy (G3): among edges with true psucc
    within +-0.1 of a gate (0.4/0.6), does the predictor call the correct side better than the geometry null?"""
    se_echo = se_geo = se_real = n = 0.0
    b_geo = b_real = b_n = 0
    for sc in scenes:
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            feats, stale = t1._edge_features(sc, t, obs)
            true = t1._true_psucc(sc, t, obs["edge_ids"])
            pg = t1._predict(*geo, feats[:, 3:4], stale, "direct")
            pr = t1._predict(*real, feats, stale, "physics")
            se_echo += float(((stale - true) ** 2).sum()); se_geo += float(((pg - true) ** 2).sum())
            se_real += float(((pr - true) ** 2).sum()); n += true.numel()
            for gate in (_KEEP, _ADD):
                near = (true - gate).abs() <= 0.1
                if not bool(near.any()):
                    continue
                side = (true[near] >= gate)
                b_geo += int(((pg[near] >= gate) == side).sum())
                b_real += int(((pr[near] >= gate) == side).sum())
                b_n += int(near.sum())
    return {"mse_echo": se_echo / n, "mse_geometry": se_geo / n, "mse_realizable": se_real / n,
            "boundary_side_acc_geometry": (b_geo / b_n) if b_n else float("nan"),
            "boundary_side_acc_realizable": (b_real / b_n) if b_n else float("nan"),
            "boundary_n": b_n}


def _rgf_seed(per):
    """Per-seed pooled RGF for each arm (None if the headroom denominator is ~0)."""
    n = len(per["stale"])
    pooled = {a: sum(per[a]) / n for a in ARMS}
    denom = pooled["true"] - pooled["stale"]
    if denom <= 0.02:              # no meaningful headroom this seed -> RGF undefined
        return None, pooled, denom
    return {a: (pooled[a] - pooled["stale"]) / denom for a in ARMS}, pooled, denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="urban", choices=["urban", "random"])
    ap.add_argument("--decorrs", type=float, nargs="+", default=[10.0, 50.0, 100.0, 200.0, 400.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-count", type=int, default=6)
    ap.add_argument("--held-count", type=int, default=12)
    # FAST operating-point mobility (the R8 regime with the big stale-drop / headroom). At v*dt~45 m,
    # rho = exp(-45/d_corr) spans ~0.01/0.41/0.64/0.80/0.89 over d_corr {10,50,100,200,400} -- so raising
    # d_corr lifts the shadow autocorrelation WHILE keeping the (distance-driven) headroom large.
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--speed-min", type=float, default=15.0)
    ap.add_argument("--speed-max", type=float, default=30.0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF2" / "oracle_regate_metrics.json"))
    args = ap.parse_args()

    csi = CsiObservationModel(mode="delay", delay_frames=1)
    by_decorr = {}
    for d_corr in args.decorrs:
        rhos, feas_arm = [], {a: [] for a in ARMS}
        rgf_arm = {a: [] for a in ARMS}
        incr, mse_landscape, bnd = [], {"echo": [], "geometry": [], "realizable": []}, {"geo": [], "real": []}
        psucc_mean, frac_sat, frac_bnd, rho_defined = [], [], [], []
        n_headroom = 0
        for s in range(args.seeds):
            tr = _build_scenes(args.data, s * 1000 + 1, args.train_count, d_corr,
                               args.dt, args.speed_min, args.speed_max, csi)
            hd = _build_scenes(args.data, s * 1000 + 777, args.held_count, d_corr,
                               args.dt, args.speed_min, args.speed_max, csi)
            rho_hd, pstats = _measure_rho(hd)
            rhos.append(rho_hd)
            psucc_mean.append(pstats["psucc_mean"]); frac_sat.append(pstats["frac_saturated_hi"])
            frac_bnd.append(pstats["frac_boundary"]); rho_defined.append(pstats["rho_defined_frac"])
            X, Y, S = t1._collect(tr)
            geo = t1._train_predictor(X[:, 3:4], Y, S, "direct", s)     # distance-only null
            real = t1._train_predictor(X, Y, S, "physics", s)           # full leak-free set
            per = _eval_per_scene(hd, geo, real)
            for a in ARMS:
                feas_arm[a].append(sum(per[a]) / len(per[a]))
            rgf, pooled, denom = _rgf_seed(per)
            if rgf is not None:
                n_headroom += 1
                for a in ARMS:
                    rgf_arm[a].append(rgf[a])
                incr.append(rgf["realizable"] - rgf["geometry"])
            mb = _mse_and_boundary(hd, geo, real)
            mse_landscape["echo"].append(mb["mse_echo"]); mse_landscape["geometry"].append(mb["mse_geometry"])
            mse_landscape["realizable"].append(mb["mse_realizable"])
            bnd["geo"].append(mb["boundary_side_acc_geometry"]); bnd["real"].append(mb["boundary_side_acc_realizable"])
        rho_mean = sum(r for r in rhos if r == r) / max(1, sum(1 for r in rhos if r == r))
        by_decorr[str(d_corr)] = {
            "rho_mean": rho_mean,
            "psucc_mean": sum(psucc_mean) / len(psucc_mean),
            "frac_saturated_hi": sum(frac_sat) / len(frac_sat),
            "frac_boundary": sum(frac_bnd) / len(frac_bnd),
            "rho_defined_frac": sum(rho_defined) / len(rho_defined),
            "feas": {a: _ci95(feas_arm[a]) for a in ARMS},
            "headroom_true_minus_stale": _ci95([feas_arm["true"][i] - feas_arm["stale"][i]
                                                for i in range(len(feas_arm["true"]))]),
            "rgf": {a: _ci95(rgf_arm[a]) for a in ARMS},
            "primary_incremental_real_minus_geo": _ci95(incr),
            "n_seeds_with_headroom": n_headroom,
            "mse": {k: _ci95(v) for k, v in mse_landscape.items()},
            "boundary_side_acc": {"geometry": _ci95(bnd["geo"]), "realizable": _ci95(bnd["real"])},
        }
        d = by_decorr[str(d_corr)]
        inc = d["primary_incremental_real_minus_geo"]
        hr = d["headroom_true_minus_stale"]
        print(f"d_corr={d_corr:6.0f} rho={rho_mean:+.3f} psuccMean={d['psucc_mean']:.2f} "
              f"satHi={d['frac_saturated_hi']:.2f} bnd={d['frac_boundary']:.2f} rhoDef={d['rho_defined_frac']:.2f}")
        print(f"    feas stale {d['feas']['stale']['mean']} geo {d['feas']['geometry']['mean']} "
              f"real {d['feas']['realizable']['mean']} true {d['feas']['true']['mean']} | "
              f"headroom {hr['mean']} | RGF real {d['rgf']['realizable']['mean']} geo {d['rgf']['geometry']['mean']} "
              f"| INCR(real-geo) {inc['mean']} CI[{inc['lo']},{inc['hi']}] | headroom_seeds {n_headroom}/{args.seeds}")

    report = {"scope": "DF2 oracle re-gate under d_corr sweep; FAST operating-point mobility (dt=2, 15-30 m/s; "
                       "the R8 big-headroom regime -- NOTE in-scene psucc-rho is ~0 here, the knob is washed out "
                       "by fast mobility + bimodal saturation, see DF2/decision.md scope caveat); nested-subset "
                       "arms; NLOSv ON (faithful); motion_features OFF. HEADLINE = feas(realizable)==feas(stale) "
                       "point-identity + MSE(real)<MSE(echo), NOT RGF(realizable) (CI too wide) and NOT the "
                       "pre-registered RGF(real)-RGF(geo) (VOID: geometry_only degenerate). See DF2/decision.md.",
              "config": vars(args), "csi": csi.manifest(), "by_decorr": by_decorr}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
