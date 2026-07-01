"""R8 stale-CSI threshold sweep: does ANY tau_edit repair the stale drop (the 'room' hypothesis)?

At tau_edit 0.5 the stale-retrained gate == stale_anchor (urban edit 0) / +0.010 spans 0 (random). But the
stale anchor is DEGRADED (there IS room, unlike the near-optimal current-channel anchor of R6). This sweep
REUSES each seed's stale-trained actor and varies only tau_edit, to see whether a LOWER threshold (fire more
edits) REPAIRS the stale drop -- or hurts as on the current channel (R6). Reports, per tau_edit, the 5-seed
(gated - stale_anchor) feasibility/return CI + edit_rate/unsafe_edit, all under STALE CSI (delay-1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import residual_ppo_train as rp3  # noqa: E402
import r6_evidence_gate_gen as g6  # noqa: E402
import r8_stale_csi_gen as g8  # noqa: E402
from marl_topology.training.edit_head_training import _LAM_B, _LAM_C, _BETA, _anchor, train_edit_heads  # noqa: E402
from marl_topology.training.evidence_gated_action import evidence_gated_residual  # noqa: E402

_TAUS_EDIT = [0.5, 0.35, 0.25, 0.15, 0.05]
_TAU_FEAS = 0.9


def _rollout_tau(sc, T, actor, mean, std, tau_edit):
    prev, e_ref = [], None
    feas, Js, edits, unsafe = [], [], [], []
    for t in range(sc.n_frames):
        obs = sc.observation(t, prev)
        ev = obs["context"].evaluator
        if e_ref is None:
            e_ref = T._ref_energy(obs)
        anchor = list(_anchor(obs, prev))
        res = evidence_gated_residual(obs, actor, prev, mean, std, tau_edit=tau_edit,
                                      tau_repair=0.0, tau_safety=0.0)
        topo, er = res["topology"], res["edit_rate"]
        cons = g6._cons(ev, topo)
        feas.append(1.0 if cons >= _TAU_FEAS else 0.0)
        Js.append(float(T.reward_of(obs, list(topo), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]))
        edits.append(er)
        unsafe.append(1.0 if (g6._cons(ev, anchor) >= _TAU_FEAS and cons < _TAU_FEAS) else 0.0)
        prev = list(topo)
    return feas, Js, edits, unsafe


def _stale_anchor(sc, T):
    prev, e_ref = [], None
    feas, Js = [], []
    for t in range(sc.n_frames):
        obs = sc.observation(t, prev)
        ev = obs["context"].evaluator
        if e_ref is None:
            e_ref = T._ref_energy(obs)
        anchor = list(_anchor(obs, prev))
        feas.append(1.0 if g6._cons(ev, anchor) >= _TAU_FEAS else 0.0)
        Js.append(float(T.reward_of(obs, list(anchor), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]))
        prev = list(anchor)
    return feas, Js


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    seeds = [0, 1, 2, 3, 4]
    T = rp3._load_trunk()
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    report = {"scope": "R8 stale-CSI threshold sweep: (gated - stale_anchor) feas/return vs tau_edit; 5 seeds x "
                       "{random,urban}; reuses stale-trained actor per seed", "csi": csi.manifest(),
              "tau_edit_grid": _TAUS_EDIT, "by_data": {}}
    for data in ["random", "urban"]:
        cache = []
        for s in seeds:
            tr = g8.build_csi_scenes(data, s * 1000 + 1, 5, g8._A(), csi)
            hd = g8.build_csi_scenes(data, s * 1000 + 777, 10, g8._A(), csi)
            r = train_edit_heads(tr, hd[:1], T, epochs=40, hidden=64, seed=s)
            a_feas, a_ret = [], []
            for sc in hd:
                f, j = _stale_anchor(sc, T)
                a_feas += f; a_ret += j
            cache.append({"actor": r["actor"], "mean": r["mean"], "std": r["std"], "hd": hd,
                          "A_feas": sum(a_feas) / len(a_feas), "A_ret": sum(a_ret) / len(a_ret)})
        per_tau = {}
        for tau in _TAUS_EDIT:
            ga_f, ga_r, er_l, un_l = [], [], [], []
            for c in cache:
                f, j, e, u = [], [], [], []
                for sc in c["hd"]:
                    ff, jj, ee, uu = _rollout_tau(sc, T, c["actor"], c["mean"], c["std"], tau)
                    f += ff; j += jj; e += ee; u += uu
                ga_f.append(sum(f) / len(f) - c["A_feas"]); ga_r.append(sum(j) / len(j) - c["A_ret"])
                er_l.append(sum(e) / len(e)); un_l.append(sum(u) / len(u))
            per_tau[str(tau)] = {"gated_minus_stale_anchor_feas": g6._ci95(ga_f),
                                 "gated_minus_stale_anchor_ret": g6._ci95(ga_r),
                                 "edit_rate": g6._ci95(er_l), "unsafe_edit": g6._ci95(un_l)}
            d = per_tau[str(tau)]
            print(f"[{data} tau={tau}] gated-stale_anchor feas {d['gated_minus_stale_anchor_feas']['mean']} "
                  f"CI[{d['gated_minus_stale_anchor_feas']['lo']},{d['gated_minus_stale_anchor_feas']['hi']}] | "
                  f"ret {d['gated_minus_stale_anchor_ret']['mean']} | edit {d['edit_rate']['mean']} "
                  f"unsafe {d['unsafe_edit']['mean']}")
        report["by_data"][data] = per_tau
    out = ROOT / "result_save" / "belief_residual" / "R8" / "stale_threshold_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
