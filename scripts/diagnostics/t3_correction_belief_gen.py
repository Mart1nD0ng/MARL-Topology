"""T3 (Temporal Recovery) generator: the belief CORRECTION target (task 3.1/3.2) -- the highest-value stage.

R2 trained the belief head to predict the ABSOLUTE current psucc logit and found it does NOT beat the
stale-echo floor (it learned to echo the stale input). T1 explained why: a direct/absolute predictor destroys
the ranking; a CORRECTION on the stale value (echo = zero-baseline) recovers signal. T3 changes ONE variable --
the belief parametrization:
  absolute (R2):   belief_logit = head_output
  correction (T3): belief_logit = stale_logit + head_output   (head learns the SIGNED stale->current delta)
everything else identical (same actor init, GRU, features, loss fn, weights, epochs). The head is trained
IN the policy actor (shares the GRU), residual_leak=0.1 (T2). True current psucc is a training-only label.

Metrics (5 seeds x {random,urban}, delay-1, held; 95% CI):
  beats_floor        = stale_echo_MSE - belief_MSE      (R2 failed this; does T3 recover CSI?)
  correction_vs_abs  = absolute_MSE - correction_MSE    (does the parametrization reduce error?)
  directional_acc    = frac of MOVED edges with correct correction sign (task 3.2; chance 0.5)
  anchor_feas_gain   = feas(recovered psucc -> anchor) - feas(stale -> anchor)   (T1 ranking/decision metric)
The headline is beats_floor for the correction arm (reverses R2) + directional_acc > chance; the anchor
feasibility gain is the T1-style conversion check (expected marginal, per T1 arm C)."""

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
from csi_belief_train import _extra  # noqa: E402  (leak-free velocity extras / _VEL_SCALE)
from r8_stale_csi_gen import _A, _ci95, _TAU_FEAS, build_csi_scenes  # noqa: E402
from t1_oracle_recovery_gen import _anchor_with_psucc  # noqa: E402
from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: E402
from marl_topology.training.csi_belief import (belief_correction_target, belief_mse,  # noqa: E402
                                               belief_target_logits, belief_weights, csi_belief_loss,
                                               directional_accuracy, stale_echo_floor_mse, stale_logit)
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.residual_saturation import feature_standardization_all_frames  # noqa: E402

_USE_VEL = True


def _belief_logit(actor, obs, mean, std, extra, h, mode):
    """Belief current-psucc logit under the chosen parametrization; returns (belief_logit, head_out, new_h)."""
    nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
    head, h = actor.belief(nf_s, ef_s, obs["ei"], extra=extra, hidden=h)
    bl = stale_logit(obs["ef"][:, 0]) + head if mode == "correction" else head
    return bl, head, h


def _train(train_scenes, mode, seed, *, epochs=60, hidden=32, lr=0.02, move_weight=0.0):
    """mode in {absolute, correction}. move_weight>0 upweights the loss on MOVED edges by (1+w*|correction
    target|) -- a TRAINING-ONLY directional/decision emphasis (task 3.2) that counters the magnitude shrinkage
    toward echo (the label shapes the loss weight only; inference stays leak-free)."""
    torch.manual_seed(seed)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden, belief_extra_dim=(2 if _USE_VEL else 0),
                                residual_leak=0.1)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    for _e in range(epochs):
        total = torch.zeros(())
        for sc in train_scenes:
            h = None
            for t in range(sc.n_frames):
                obs = sc.observation(t, [])
                eids = obs["edge_ids"]
                bl, _head, h = _belief_logit(actor, obs, mean, std, _extra(sc, t, eids, _USE_VEL), h, mode)
                w = belief_weights(eids, prev_topo=[], anchor=list(eids))
                if move_weight:
                    ct = belief_correction_target(sc, t, eids, obs["ef"][:, 0])
                    w = w * (1.0 + move_weight * ct.abs())
                total = total + csi_belief_loss(bl, belief_target_logits(sc, t, eids), w)
        opt.zero_grad(); total.backward(); opt.step()
    return actor, mean, std


def _held(actor, scenes, mean, std, mode):
    bmse, floor, dacc, f_rec, f_stale = [], [], [], [], []
    for sc in scenes:
        h = None
        prev_r: list = []
        prev_s: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            eids = obs["edge_ids"]
            if len(eids) == 0:
                continue
            with torch.no_grad():
                bl, _head, h = _belief_logit(actor, obs, mean, std, _extra(sc, t, eids, _USE_VEL), h, mode)
            stale_p = obs["ef"][:, 0]
            # LEAK BOUNDARY: belief_target_logits (true current) and ev below are METRIC-ONLY scorers here --
            # they are NEVER fed into _belief_logit (the belief input is leak-free stale/geometry/GRU only).
            tgt = belief_target_logits(sc, t, eids)
            bmse.append(belief_mse(bl, tgt))
            floor.append(stale_echo_floor_mse(sc, t, eids, stale_p))
            ct = belief_correction_target(sc, t, eids, stale_p)
            dacc.append(directional_accuracy(bl - stale_logit(stale_p), ct))     # implied correction sign
            ev = obs["context"].evaluator
            topo_r = _anchor_with_psucc(obs, prev_r, torch.sigmoid(bl))
            topo_s = _anchor_with_psucc(obs, prev_s, stale_p)
            f_rec.append(1.0 if g6._cons(ev, topo_r) >= _TAU_FEAS else 0.0)
            f_stale.append(1.0 if g6._cons(ev, topo_s) >= _TAU_FEAS else 0.0)
            prev_r, prev_s = list(topo_r), list(topo_s)

    def _m(x):
        x = [v for v in x if v == v]
        return sum(x) / len(x) if x else float("nan")
    return {"belief_mse": _m(bmse), "floor_mse": _m(floor), "dir_acc": _m(dacc),
            "feas_recovered": _m(f_rec), "feas_stale": _m(f_stale)}


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    seeds = [0, 1, 2, 3, 4]
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    report = {"scope": "T3 belief CORRECTION vs ABSOLUTE parametrization (delay-1); in-policy belief head "
                       "(shares GRU, residual_leak=0.1); true current psucc = training-only label; 5 seeds x "
                       "{random,urban}; held belief MSE vs stale-echo floor + directional acc + recovered-psucc "
                       "anchor feasibility.", "csi": csi.manifest(), "by_data": {}}
    for data in ["random", "urban"]:
        per = {"abs_beats_floor": [], "cor_beats_floor": [], "correction_vs_abs_mse": [],
               "cor_dir_acc": [], "abs_dir_acc": [], "cor_feas_gain": [], "abs_feas_gain": [],
               "cor_belief_mse": [], "abs_belief_mse": [], "floor_mse": []}
        for s in seeds:
            tr = build_csi_scenes(data, s * 1000 + 1, 5, _A(), csi)
            hd = build_csi_scenes(data, s * 1000 + 777, 8, _A(), csi)
            a_abs, m_a, s_a = _train(tr, "absolute", s)
            a_cor, m_c, s_c = _train(tr, "correction", s)
            r_abs = _held(a_abs, hd, m_a, s_a, "absolute")
            r_cor = _held(a_cor, hd, m_c, s_c, "correction")
            per["abs_beats_floor"].append(r_abs["floor_mse"] - r_abs["belief_mse"])
            per["cor_beats_floor"].append(r_cor["floor_mse"] - r_cor["belief_mse"])
            per["correction_vs_abs_mse"].append(r_abs["belief_mse"] - r_cor["belief_mse"])
            per["cor_dir_acc"].append(r_cor["dir_acc"]); per["abs_dir_acc"].append(r_abs["dir_acc"])
            per["cor_feas_gain"].append(r_cor["feas_recovered"] - r_cor["feas_stale"])
            per["abs_feas_gain"].append(r_abs["feas_recovered"] - r_abs["feas_stale"])
            per["cor_belief_mse"].append(r_cor["belief_mse"]); per["abs_belief_mse"].append(r_abs["belief_mse"])
            per["floor_mse"].append(r_cor["floor_mse"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["correction_beats_floor"] = d["cor_beats_floor"]["lo"] is not None and d["cor_beats_floor"]["lo"] > 0
        report["by_data"][data] = d
        print(f"[{data}] cor_beats_floor {d['cor_beats_floor']['mean']} CI[{d['cor_beats_floor']['lo']},{d['cor_beats_floor']['hi']}] "
              f"beats={d['correction_beats_floor']} | abs_beats_floor {d['abs_beats_floor']['mean']} | "
              f"cor_vs_abs_mse {d['correction_vs_abs_mse']['mean']} | cor_dir_acc {d['cor_dir_acc']['mean']} "
              f"abs_dir_acc {d['abs_dir_acc']['mean']} | cor_feas_gain {d['cor_feas_gain']['mean']} "
              f"CI[{d['cor_feas_gain']['lo']},{d['cor_feas_gain']['hi']}]")
    out = ROOT / "result_save" / "temporal_recovery" / "T3" / "correction_belief_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
