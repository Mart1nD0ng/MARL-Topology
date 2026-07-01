"""T5 (Temporal Recovery) generator: uncertainty-gated correction belief (task 4.2/4.3).

T3/T4 showed the correction captures direction but not magnitude, and applying the noisy magnitude everywhere
HURTS the anchor. T5 trains a HETEROSCEDASTIC belief (Gaussian NLL): the head predicts a correction mean mu AND
a per-edge log-variance logvar. At inference two recovery rules are compared on the SAME trained model (single
variable = the gating):
  mean  : recovered_psucc = sigmoid(stale_logit + mu)                         (apply the full magnitude, ~T3)
  gated : recovered_psucc = sigmoid(stale_logit + confidence_gate(logvar)*mu) (apply mu only where confident;
          else fall back to stale/echo)
The stale observation is the physical baseline (the last true measurement); the correction is a
confidence-weighted residual on it (task 4.2). Hypothesis: gating removes the magnitude-noise HARM -> the gated
recovered psucc is at worst == stale (feas_gain ~ 0), unlike the mean rule which perturbs the anchor.

Metrics (5 seeds x {random,urban}, delay-1, held; 95% CI): {mean,gated}_beats_floor, {mean,gated}_feas_gain,
gated_minus_mean_feas, uncertainty calibration, mean confidence gate + fraction of edges gated toward stale."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import r6_evidence_gate_gen as g6  # noqa: E402
from csi_belief_train import _extra  # noqa: E402
from r8_stale_csi_gen import _A, _ci95, _TAU_FEAS, build_csi_scenes  # noqa: E402
from t1_oracle_recovery_gen import _anchor_with_psucc  # noqa: E402
from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: E402
from marl_topology.training.csi_belief import (belief_correction_target, belief_mse,  # noqa: E402
                                               belief_target_logits, belief_weights, confidence_gate,
                                               csi_correction_nll, gated_correction, stale_echo_floor_mse,
                                               stale_logit, uncertainty_calibration)
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.residual_saturation import feature_standardization_all_frames  # noqa: E402

_USE_VEL = True


def _mu_logvar(actor, obs, mean, std, extra, h):
    nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
    return actor.belief_with_uncertainty(nf_s, ef_s, obs["ei"], extra=extra, hidden=h)


def _train(train_scenes, seed, *, epochs=60, hidden=32, lr=0.02):
    torch.manual_seed(seed)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden, belief_extra_dim=(2 if _USE_VEL else 0),
                                residual_leak=0.1, belief_uncertainty=True)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    for _e in range(epochs):
        total = torch.zeros(())
        for sc in train_scenes:
            h = None
            for t in range(sc.n_frames):
                obs = sc.observation(t, [])
                eids = obs["edge_ids"]
                mu, logvar, h = _mu_logvar(actor, obs, mean, std, _extra(sc, t, eids, _USE_VEL), h)
                ct = belief_correction_target(sc, t, eids, obs["ef"][:, 0])         # training-only label
                w = belief_weights(eids, prev_topo=[], anchor=list(eids))
                total = total + csi_correction_nll(mu, logvar, ct, w)
        opt.zero_grad(); total.backward(); opt.step()
    return actor, mean, std


def _held(actor, scenes, mean, std):
    m_mse, g_mse, floor, cal, gate, f_mean, f_gated, f_stale = [], [], [], [], [], [], [], []
    for sc in scenes:
        h = None
        prev_m: list = []
        prev_g: list = []
        prev_s: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            eids = obs["edge_ids"]
            if len(eids) == 0:
                continue
            with torch.no_grad():
                mu, logvar, h = _mu_logvar(actor, obs, mean, std, _extra(sc, t, eids, _USE_VEL), h)
            stale_p = obs["ef"][:, 0]
            sl = stale_logit(stale_p)
            # LEAK BOUNDARY: belief_target_logits (true current) + ev are METRIC-ONLY here, never fed to _mu_logvar.
            tgt = belief_target_logits(sc, t, eids)
            ct = belief_correction_target(sc, t, eids, stale_p)
            bl_mean = sl + mu                                                 # apply full magnitude
            bl_gated = sl + gated_correction(mu, logvar)                      # apply mu only where confident
            m_mse.append(belief_mse(bl_mean, tgt)); g_mse.append(belief_mse(bl_gated, tgt))
            floor.append(stale_echo_floor_mse(sc, t, eids, stale_p))
            cal.append(uncertainty_calibration(mu, logvar, ct))
            cg = confidence_gate(logvar)
            gate.append(float(cg.mean()));
            ev = obs["context"].evaluator
            topo_m = _anchor_with_psucc(obs, prev_m, torch.sigmoid(bl_mean))
            topo_g = _anchor_with_psucc(obs, prev_g, torch.sigmoid(bl_gated))
            topo_s = _anchor_with_psucc(obs, prev_s, stale_p)
            f_mean.append(1.0 if g6._cons(ev, topo_m) >= _TAU_FEAS else 0.0)
            f_gated.append(1.0 if g6._cons(ev, topo_g) >= _TAU_FEAS else 0.0)
            f_stale.append(1.0 if g6._cons(ev, topo_s) >= _TAU_FEAS else 0.0)
            prev_m, prev_g, prev_s = list(topo_m), list(topo_g), list(topo_s)

    def _m(x):
        x = [v for v in x if v == v]
        return sum(x) / len(x) if x else float("nan")
    return {"mean_mse": _m(m_mse), "gated_mse": _m(g_mse), "floor_mse": _m(floor),
            "calibration": _m(cal), "mean_gate": _m(gate),
            "feas_mean": _m(f_mean), "feas_gated": _m(f_gated), "feas_stale": _m(f_stale)}


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    seeds = [0, 1, 2, 3, 4]
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    report = {"scope": "T5 uncertainty-gated correction belief (delay-1); heteroscedastic NLL; mean vs "
                       "confidence-gated recovery on the SAME model; in-policy, residual_leak=0.1; true current "
                       "psucc = training-only label; 5 seeds x {random,urban}.", "csi": csi.manifest(), "by_data": {}}
    for data in ["random", "urban"]:
        per = {"mean_beats_floor": [], "gated_beats_floor": [], "gated_vs_mean_mse": [],
               "mean_feas_gain": [], "gated_feas_gain": [], "gated_minus_mean_feas": [],
               "calibration": [], "mean_gate": []}
        for s in seeds:
            tr = build_csi_scenes(data, s * 1000 + 1, 5, _A(), csi)
            hd = build_csi_scenes(data, s * 1000 + 777, 8, _A(), csi)
            actor, m, sd = _train(tr, s)
            r = _held(actor, hd, m, sd)
            per["mean_beats_floor"].append(r["floor_mse"] - r["mean_mse"])
            per["gated_beats_floor"].append(r["floor_mse"] - r["gated_mse"])
            per["gated_vs_mean_mse"].append(r["mean_mse"] - r["gated_mse"])
            per["mean_feas_gain"].append(r["feas_mean"] - r["feas_stale"])
            per["gated_feas_gain"].append(r["feas_gated"] - r["feas_stale"])
            per["gated_minus_mean_feas"].append(r["feas_gated"] - r["feas_mean"])
            per["calibration"].append(r["calibration"]); per["mean_gate"].append(r["mean_gate"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["gated_no_harm"] = d["gated_feas_gain"]["lo"] is not None and d["gated_feas_gain"]["lo"] >= -0.02
        d["gated_beats_mean"] = d["gated_minus_mean_feas"]["lo"] is not None and d["gated_minus_mean_feas"]["lo"] > 0
        report["by_data"][data] = d
        print(f"[{data}] gated_feas_gain {d['gated_feas_gain']['mean']} CI[{d['gated_feas_gain']['lo']},{d['gated_feas_gain']['hi']}] "
              f"mean_feas_gain {d['mean_feas_gain']['mean']} | gated_minus_mean {d['gated_minus_mean_feas']['mean']} "
              f"CI[{d['gated_minus_mean_feas']['lo']},{d['gated_minus_mean_feas']['hi']}] beats_mean={d['gated_beats_mean']} | "
              f"calib {d['calibration']['mean']} gate {d['mean_gate']['mean']} | gated_beats_floor {d['gated_beats_floor']['mean']}")
    out = ROOT / "result_save" / "temporal_recovery" / "T5" / "uncertainty_gated_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
