"""R8 generator: the stale-CSI PREMISE test -- does the evidence-gated action REPAIR the stale feasibility drop?

The campaign's raison d'etre: stale CSI degrades the deployable anchor (urban ~0.80 -> ~0.66). R1-R7 ran mostly
with the overlay OFF. R8 turns it ON (delay-1): the deployed actor observes a STALE channel; the R5 heads are
RETRAINED on stale features (labels from the TRUE current evaluator = training-only); the evidence-gated action
is deployed under stale CSI and scored by the TRUE evaluator (current real channel + closed-form quorum tail).

Arms (deployed on held, 0 eval at decision; the evaluator scores ONLY the produced topology):
  current_anchor  the anchor on the CURRENT channel (the non-degraded reference / ceiling)
  stale_anchor    the deployable anchor under STALE CSI (the degraded baseline -- what we try to repair)
  stale_gated     the evidence-gated action under STALE CSI (stale-retrained heads) -- does it repair?
Reports per seed + 95% CI on (stale_gated - stale_anchor) feas [the repair], the stale drop
(current_anchor - stale_anchor), edit_rate, unsafe_edit. (stale_gated - stale_anchor) CI > 0 => the method
repairs the stale drop (a genuine positive). Spans/below 0 => it does not (negative under the real regime).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import residual_ppo_train as rp3  # noqa: E402
import r6_evidence_gate_gen as g6  # noqa: E402
from marl_topology.training.edit_head_training import _LAM_B, _LAM_C, _BETA, _anchor, train_edit_heads  # noqa: E402
from marl_topology.training.evidence_gated_action import evidence_gated_residual  # noqa: E402
from marl_topology.training.quorum_deficit_bridge import topology_reliability  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402

_TAU_FEAS = 0.9
_TAUS = {"tau_edit": 0.5, "tau_repair": 0.0, "tau_safety": 0.0}


class _A:
    dyn_nodes = [8, 12, 16]
    frames = 4
    gamma = 0.95


def build_csi_scenes(data, seed, count, args, csi_model):
    """Build dynamic scenes with an optional stale-CSI observation overlay (None = current channel)."""
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes, sample_dynamic_urban_scenes
    common = dict(seed=seed, count=count, node_count_choices=tuple(args.dyn_nodes),
                  regime=operating_point_regime(20.0), num_frames=args.frames, dt_s=2.0, speed_min_mps=15.0,
                  speed_max_mps=30.0, reconfig=ReconfigCost(), hold_interval=4, gamma=args.gamma,
                  csi_observation_model=csi_model)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


def _ci95(xs):
    xs = [x for x in xs if x is not None and x == x]
    n = len(xs)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = sum(xs) / n
    if n < 2:
        return {"mean": round(m, 5), "lo": round(m, 5), "hi": round(m, 5), "n": n}
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (n - 1))
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.776)
    h = t * sd / math.sqrt(n)
    return {"mean": round(m, 5), "lo": round(m - h, 5), "hi": round(m + h, 5), "n": n}


def _rollout(sc, T, kind, actor=None, mean=None, std=None):
    prev, e_ref = [], None
    feas, Js, edits, unsafe = [], [], [], []
    for t in range(sc.n_frames):
        obs = sc.observation(t, prev)
        ev = obs["context"].evaluator
        if e_ref is None:
            e_ref = T._ref_energy(obs)
        anchor = list(_anchor(obs, prev))
        if kind == "anchor":
            topo, er = anchor, 0.0
        else:
            res = evidence_gated_residual(obs, actor, prev, mean, std, **_TAUS)
            topo, er = res["topology"], res["edit_rate"]
        cons = g6._cons(ev, topo)
        feas.append(1.0 if cons >= _TAU_FEAS else 0.0)
        Js.append(float(T.reward_of(obs, list(topo), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]))
        edits.append(er)
        if kind != "anchor":
            unsafe.append(1.0 if (g6._cons(ev, anchor) >= _TAU_FEAS and cons < _TAU_FEAS) else 0.0)
        prev = list(topo)
    return feas, Js, edits, unsafe


def _agg(scenes, T, kind, actor=None, mean=None, std=None):
    f, j, e, u = [], [], [], []
    for sc in scenes:
        ff, jj, ee, uu = _rollout(sc, T, kind, actor, mean, std)
        f += ff; j += jj; e += ee; u += uu
    def _m(x):
        return sum(x) / len(x) if x else 0.0
    return {"feas": _m(f), "ret": _m(j), "edit": _m(e), "unsafe": _m(u)}


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    seeds = [0, 1, 2, 3, 4]
    T = rp3._load_trunk()
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    report = {"scope": "R8 stale-CSI premise test (delay-1): does the evidence-gated action repair the stale "
                       "feasibility drop? 5 seeds x {random,urban}; heads retrained on STALE features, labels "
                       "from TRUE current evaluator (training-only); deployed 0-eval, true evaluator scores.",
              "csi": csi.manifest(), "taus": _TAUS, "by_data": {}}
    for data in ["random", "urban"]:
        per = {"current_anchor_feas": [], "stale_anchor_feas": [], "stale_gated_feas": [], "stale_drop": [],
               "gated_minus_stale_anchor_feas": [], "gated_minus_stale_anchor_ret": [], "stale_gated_edit": [],
               "stale_gated_unsafe": []}
        for s in seeds:
            tr_s = build_csi_scenes(data, s * 1000 + 1, 5, _A(), csi)      # STALE train
            hd_s = build_csi_scenes(data, s * 1000 + 777, 10, _A(), csi)   # STALE held
            hd_c = build_csi_scenes(data, s * 1000 + 777, 10, _A(), None)  # CURRENT held (same seed -> ref)
            r = train_edit_heads(tr_s, hd_s[:1], T, epochs=40, hidden=64, seed=s)
            actor, mean, std = r["actor"], r["mean"], r["std"]
            ca = _agg(hd_c, T, "anchor")                                   # current-channel anchor (reference)
            sa = _agg(hd_s, T, "anchor")                                   # stale anchor (degraded)
            sg = _agg(hd_s, T, "gated", actor, mean, std)                  # stale evidence-gated
            per["current_anchor_feas"].append(ca["feas"]); per["stale_anchor_feas"].append(sa["feas"])
            per["stale_gated_feas"].append(sg["feas"]); per["stale_drop"].append(ca["feas"] - sa["feas"])
            per["gated_minus_stale_anchor_feas"].append(sg["feas"] - sa["feas"])
            per["gated_minus_stale_anchor_ret"].append(sg["ret"] - sa["ret"])
            per["stale_gated_edit"].append(sg["edit"]); per["stale_gated_unsafe"].append(sg["unsafe"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["repairs_stale_drop"] = (d["gated_minus_stale_anchor_feas"]["lo"] is not None
                                   and d["gated_minus_stale_anchor_feas"]["lo"] > 0)
        report["by_data"][data] = d
        print(f"[{data}] current_anchor {d['current_anchor_feas']['mean']} stale_anchor {d['stale_anchor_feas']['mean']} "
              f"(drop {d['stale_drop']['mean']}) stale_gated {d['stale_gated_feas']['mean']} | "
              f"gated-stale_anchor {d['gated_minus_stale_anchor_feas']['mean']} "
              f"CI[{d['gated_minus_stale_anchor_feas']['lo']},{d['gated_minus_stale_anchor_feas']['hi']}] "
              f"repairs={d['repairs_stale_drop']} | edit {d['stale_gated_edit']['mean']} unsafe {d['stale_gated_unsafe']['mean']}")
    out = ROOT / "result_save" / "belief_residual" / "R8" / "stale_csi_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
