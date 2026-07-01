"""DF1 (Decision-Focused, goal 2) env-property probe.

Proves the decorrelation-distance knob does what we claim, HONESTLY (validation_design v2 A.2c-d):
  1. rho-monotonicity: the lag-1 temporal autocorrelation of the DECISION-CRITICAL psucc,
     rho(psucc_t, psucc_{t-1}), over moving-vehicle trajectory frames, rises monotonically with d_corr
     (field-seed averaged, with a seed-level CI). This is the intended effect.
  2. MARGINAL-INVARIANCE (critic-1 M4, the most dangerous confound): the psucc marginal distribution must be
     ~invariant across d_corr, else the knob moves DIFFICULTY not just autocorrelation and any downstream
     "recovery converts" result is a marginal-shift artifact. Reported as mean/std + a two-sample KS statistic
     vs the d_corr=10 baseline.

NLOSv is held OFF here to ISOLATE the shadow-decorrelation knob (its persistent per-pair loss adds a
d_corr-independent autocorrelation; guard g). Shadowing sigma and geometry are held fixed across the sweep; only
d_corr varies. Field realizations are varied via shadowing_seed and AVERAGED (M4).

Gate-exempt training/diagnostics; no deployed-path change. Deterministic (seeded).
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

from marl_topology.channel import ChannelModelConfig, evaluate_channel  # noqa: E402
from marl_topology.channel.model import PATH_LOSS_MODEL_V2X  # noqa: E402
from marl_topology.link import LinkTransmissionConfig, evaluate_link_transmission  # noqa: E402
from marl_topology.geometry3d import Point3D  # noqa: E402
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D  # noqa: E402

DECORRS = (10.0, 25.0, 50.0, 100.0)


def _scene_at(link, frame: int, dt: float) -> Scene3D:
    (tx0, txv, rx0, rxv) = link
    tx = Point3D(tx0[0] + txv[0] * frame * dt, tx0[1] + txv[1] * frame * dt, 1.5)
    rx = Point3D(rx0[0] + rxv[0] * frame * dt, rx0[1] + rxv[1] * frame * dt, 1.5)
    return Scene3D(scenario_id="df1probe",
                   nodes=(Node3D("tx", NodeKind.VEHICLE, tx), Node3D("rx", NodeKind.VEHICLE, rx)))


def _psucc_series(link, d_corr: float, field_seed: int, n_frames: int, dt: float,
                  link_cfg: LinkTransmissionConfig, tx_power_dbm: float) -> list[float]:
    cfg = ChannelModelConfig(path_loss_model=PATH_LOSS_MODEL_V2X, shadowing_37885=True,
                             nlosv_37885=False, shadow_decorrelation_distance_m=d_corr,
                             default_tx_power_dbm=tx_power_dbm)
    out = []
    for f in range(n_frames):
        rec = evaluate_channel(_scene_at(link, f, dt), "tx", "rx", config=cfg, shadowing_seed=field_seed)
        out.append(float(evaluate_link_transmission(rec, link_cfg).packet_success_probability))
    return out


def _lag1_autocorr(xs: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    if var < 1e-9:
        return None  # constant (saturated) series -> autocorr undefined; excluded
    cov = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(n - 1)) / (n - 1)
    return cov / var


def _gen_links(rng: random.Random, n: int, speed_min: float, speed_max: float,
               dist_min: float, dist_max: float):
    links = []
    for _ in range(n):
        tx0 = (rng.uniform(0.0, 500.0), rng.uniform(0.0, 500.0))
        ang = rng.uniform(0.0, 2 * math.pi)
        d = rng.uniform(dist_min, dist_max)
        rx0 = (tx0[0] + d * math.cos(ang), tx0[1] + d * math.sin(ang))

        def vel():
            sp = rng.uniform(speed_min, speed_max)
            a = rng.uniform(0.0, 2 * math.pi)
            return (sp * math.cos(a), sp * math.sin(a))

        links.append((tx0, vel(), rx0, vel()))
    return links


def _ks(a: list[float], b: list[float]) -> float:
    """Two-sample KS statistic (max abs diff of empirical CDFs)."""
    if not a or not b:
        return float("nan")
    sa, sb = sorted(a), sorted(b)
    grid = sorted(set(sa) | set(sb))

    def cdf(s, v):
        # fraction <= v
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi) // 2
            if s[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(s)

    return max(abs(cdf(sa, v) - cdf(sb, v)) for v in grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=int, default=120)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--field-seeds", type=int, default=6)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--speed-min", type=float, default=5.0)
    ap.add_argument("--speed-max", type=float, default=15.0)
    ap.add_argument("--dist-min", type=float, default=60.0)
    ap.add_argument("--dist-max", type=float, default=170.0)
    ap.add_argument("--tx-power", type=float, default=20.0,
                    help="tx power dBm; lower it to push psucc toward the 0.4/0.6 anchor boundary")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "docs" / "decision_focused" / "DF1" / "env_probe_metrics.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    links = _gen_links(rng, args.links, args.speed_min, args.speed_max, args.dist_min, args.dist_max)
    link_cfg = LinkTransmissionConfig()

    per_decorr = {}
    baseline_marginal: list[float] = []
    for d_corr in DECORRS:
        seed_rhos = []          # per-field-seed mean rho (over links)
        all_psucc: list[float] = []
        defined = total = 0
        for fs in range(args.field_seeds):
            link_rhos = []
            for link in links:
                series = _psucc_series(link, d_corr, fs, args.frames, args.dt, link_cfg, args.tx_power)
                all_psucc.extend(series)
                r = _lag1_autocorr(series)
                total += 1
                if r is not None:
                    link_rhos.append(r)
                    defined += 1
            if link_rhos:
                seed_rhos.append(sum(link_rhos) / len(link_rhos))
        mean_rho = sum(seed_rhos) / len(seed_rhos) if seed_rhos else float("nan")
        # seed-level 95% CI (normal approx)
        if len(seed_rhos) > 1:
            sd = (sum((r - mean_rho) ** 2 for r in seed_rhos) / (len(seed_rhos) - 1)) ** 0.5
            half = 1.96 * sd / (len(seed_rhos) ** 0.5)
        else:
            half = float("nan")
        mmean = sum(all_psucc) / len(all_psucc)
        mstd = (sum((p - mmean) ** 2 for p in all_psucc) / len(all_psucc)) ** 0.5
        frac_mid = sum(1 for p in all_psucc if 0.2 <= p <= 0.8) / len(all_psucc)
        if d_corr == DECORRS[0]:
            baseline_marginal = list(all_psucc)
        per_decorr[str(d_corr)] = {
            "mean_rho": mean_rho, "rho_ci_half": half, "n_seed_means": len(seed_rhos),
            "rho_defined_frac": defined / total if total else float("nan"),
            "psucc_mean": mmean, "psucc_std": mstd, "psucc_frac_in_0p2_0p8": frac_mid,
            "ks_vs_d10": _ks(all_psucc, baseline_marginal), "n_psucc": len(all_psucc),
        }

    rhos = [per_decorr[str(d)]["mean_rho"] for d in DECORRS]
    monotone = all(rhos[i] <= rhos[i + 1] + 1e-9 for i in range(len(rhos) - 1))
    rise = rhos[-1] - rhos[0]
    max_ks = max(per_decorr[str(d)]["ks_vs_d10"] for d in DECORRS)
    result = {
        "config": vars(args),
        "per_decorr": per_decorr,
        "rho_by_decorr": {str(d): per_decorr[str(d)]["mean_rho"] for d in DECORRS},
        "rho_monotone_nondecreasing": monotone,
        "rho_rise_d10_to_d100": rise,
        "knob_gate_pass": bool(monotone and rise > 0.1),
        "max_ks_marginal_vs_d10": max_ks,
        "marginal_invariance_note": "smaller KS = more invariant; large KS => renormalize before the sweep",
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("d_corr | mean_rho (95% CI half) | psucc mean/std | frac_mid | KS_vs_d10 | rho_defined")
    for d in DECORRS:
        r = per_decorr[str(d)]
        print(f"{d:6.0f} | {r['mean_rho']:+.3f} ({r['rho_ci_half']:.3f}) | "
              f"{r['psucc_mean']:.3f}/{r['psucc_std']:.3f} | {r['psucc_frac_in_0p2_0p8']:.2f} | "
              f"{r['ks_vs_d10']:.3f} | {r['rho_defined_frac']:.2f}")
    print(f"\nrho monotone non-decreasing: {monotone} | rise d10->d100: {rise:+.3f} | "
          f"knob_gate_pass: {result['knob_gate_pass']} | max KS vs d10: {max_ks:.3f}")
    print(f"written: {outp}")


if __name__ == "__main__":
    main()
