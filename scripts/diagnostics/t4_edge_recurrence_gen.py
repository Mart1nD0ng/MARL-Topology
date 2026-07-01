"""T4 (Temporal Recovery) generator: EDGE-level vs NODE-level recurrence for the correction belief (task 3.4).

T3 showed the correction belief captures direction but not magnitude; the recurrence was per-NODE while the CSI
dynamics are per-EDGE. T4 changes ONE variable -- the recurrence LOCUS:
  node (T3): belief() -- per-node GRUCell; the belief reads node hu*hv
  edge (T4): belief_edge() -- a per-EDGE GRUCell carrying a per-edge hidden [E,H] across frames; the belief
             reads that edge hidden directly
both with the CORRECTION parametrization (belief_logit = stale_logit + head) + residual_leak=0.1. True current
psucc = training-only label. Question: does an edge-indexed temporal state recover the MAGNITUDE the per-node
state could not (beat the stale-echo floor / improve the recovered-psucc anchor)?

Metrics (5 seeds x {random,urban}, delay-1, held; 95% CI): {node,edge}_beats_floor, edge_vs_node_mse,
{node,edge}_dir_acc, {node,edge}_feas_gain (recovered psucc -> anchor)."""

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
                                               belief_target_logits, belief_weights, csi_belief_loss,
                                               directional_accuracy, stale_echo_floor_mse, stale_logit)
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.residual_saturation import feature_standardization_all_frames  # noqa: E402

_USE_VEL = True


def _belief(actor, obs, mean, std, extra, h, mode):
    """Correction belief_logit under the chosen recurrence locus; returns (belief_logit, new_hidden)."""
    nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
    if mode == "edge":
        head, h = actor.belief_edge(nf_s, ef_s, obs["ei"], extra=extra, edge_hidden=h)
    else:
        head, h = actor.belief(nf_s, ef_s, obs["ei"], extra=extra, hidden=h)
    return stale_logit(obs["ef"][:, 0]) + head, h                     # correction parametrization (both arms)


def _train(train_scenes, mode, seed, *, epochs=60, hidden=32, lr=0.02):
    torch.manual_seed(seed)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden, belief_extra_dim=(2 if _USE_VEL else 0),
                                residual_leak=0.1, edge_recurrent=(mode == "edge"))
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    for _e in range(epochs):
        total = torch.zeros(())
        for sc in train_scenes:
            h = None
            for t in range(sc.n_frames):
                obs = sc.observation(t, [])
                eids = obs["edge_ids"]
                bl, h = _belief(actor, obs, mean, std, _extra(sc, t, eids, _USE_VEL), h, mode)
                w = belief_weights(eids, prev_topo=[], anchor=list(eids))
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
                bl, h = _belief(actor, obs, mean, std, _extra(sc, t, eids, _USE_VEL), h, mode)
            stale_p = obs["ef"][:, 0]
            # LEAK BOUNDARY: belief_target_logits (true current) + ev are METRIC-ONLY scorers, never fed to _belief.
            tgt = belief_target_logits(sc, t, eids)
            bmse.append(belief_mse(bl, tgt))
            floor.append(stale_echo_floor_mse(sc, t, eids, stale_p))
            dacc.append(directional_accuracy(bl - stale_logit(stale_p),
                                             belief_correction_target(sc, t, eids, stale_p)))
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
            "feas_gain": _m(f_rec) - _m(f_stale)}


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    seeds = [0, 1, 2, 3, 4]
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    report = {"scope": "T4 EDGE vs NODE recurrence for the correction belief (delay-1); in-policy belief, "
                       "residual_leak=0.1; true current psucc = training-only label; 5 seeds x {random,urban}.",
              "csi": csi.manifest(), "by_data": {}}
    for data in ["random", "urban"]:
        per = {"node_beats_floor": [], "edge_beats_floor": [], "edge_vs_node_mse": [],
               "node_dir_acc": [], "edge_dir_acc": [], "node_feas_gain": [], "edge_feas_gain": [],
               "node_belief_mse": [], "edge_belief_mse": [], "floor_mse": []}
        for s in seeds:
            tr = build_csi_scenes(data, s * 1000 + 1, 5, _A(), csi)
            hd = build_csi_scenes(data, s * 1000 + 777, 8, _A(), csi)
            a_n, mn, sn = _train(tr, "node", s)
            a_e, me, se = _train(tr, "edge", s)
            r_n = _held(a_n, hd, mn, sn, "node")
            r_e = _held(a_e, hd, me, se, "edge")
            per["node_beats_floor"].append(r_n["floor_mse"] - r_n["belief_mse"])
            per["edge_beats_floor"].append(r_e["floor_mse"] - r_e["belief_mse"])
            per["edge_vs_node_mse"].append(r_n["belief_mse"] - r_e["belief_mse"])
            per["node_dir_acc"].append(r_n["dir_acc"]); per["edge_dir_acc"].append(r_e["dir_acc"])
            per["node_feas_gain"].append(r_n["feas_gain"]); per["edge_feas_gain"].append(r_e["feas_gain"])
            per["node_belief_mse"].append(r_n["belief_mse"]); per["edge_belief_mse"].append(r_e["belief_mse"])
            per["floor_mse"].append(r_e["floor_mse"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["edge_beats_floor_sig"] = d["edge_beats_floor"]["lo"] is not None and d["edge_beats_floor"]["lo"] > 0
        d["edge_beats_node_sig"] = d["edge_vs_node_mse"]["lo"] is not None and d["edge_vs_node_mse"]["lo"] > 0
        report["by_data"][data] = d
        print(f"[{data}] edge_beats_floor {d['edge_beats_floor']['mean']} CI[{d['edge_beats_floor']['lo']},{d['edge_beats_floor']['hi']}] "
              f"sig={d['edge_beats_floor_sig']} | node_beats_floor {d['node_beats_floor']['mean']} | "
              f"edge_vs_node_mse {d['edge_vs_node_mse']['mean']} sig={d['edge_beats_node_sig']} | "
              f"edge_dir_acc {d['edge_dir_acc']['mean']} node {d['node_dir_acc']['mean']} | "
              f"edge_feas_gain {d['edge_feas_gain']['mean']} CI[{d['edge_feas_gain']['lo']},{d['edge_feas_gain']['hi']}] "
              f"node_feas_gain {d['node_feas_gain']['mean']}")
    out = ROOT / "result_save" / "temporal_recovery" / "T4" / "edge_recurrence_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
