"""R6 threshold sweep: characterize the evidence gate's OPERATING CURVE (does ANY tau_edit beat the anchor?).

The default R6 run (tau_edit=0.5) found the well-trained heads almost never pass the gate (edit_rate ~0.001)
-> B == anchor. This sweep REUSES each seed's trained actor (training is the expensive part) and only varies
the deployment threshold tau_edit, so we see whether a LOWER threshold (more edits fire) converts the R5
ranking into a deployed feasibility/return gain over the anchor -- or just degrades toward the random-gate
control. Reports, per tau_edit, the 5-seed (B-A) feasibility/return CI + edit_rate/unsafe_edit_rate.
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
import r6_evidence_gate_gen as g  # noqa: E402
from marl_topology.training.edit_head_training import _LAM_B, _LAM_C, _BETA, _anchor, train_edit_heads  # noqa: E402
from marl_topology.training.evidence_gated_action import evidence_gated_residual  # noqa: E402

_TAUS_EDIT = [0.5, 0.35, 0.25, 0.15, 0.05]


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
        cons = g._cons(ev, topo)
        cons_a = g._cons(ev, anchor)
        feas.append(1.0 if cons >= g._TAU_FEAS else 0.0)
        Js.append(float(T.reward_of(obs, list(topo), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]))
        edits.append(er)
        unsafe.append(1.0 if (cons_a >= g._TAU_FEAS and cons < g._TAU_FEAS) else 0.0)
        prev = list(topo)
    return feas, Js, edits, unsafe


def _anchor_feas_ret(sc, T):
    prev, e_ref = [], None
    feas, Js = [], []
    for t in range(sc.n_frames):
        obs = sc.observation(t, prev)
        ev = obs["context"].evaluator
        if e_ref is None:
            e_ref = T._ref_energy(obs)
        anchor = list(_anchor(obs, prev))
        feas.append(1.0 if g._cons(ev, anchor) >= g._TAU_FEAS else 0.0)
        Js.append(float(T.reward_of(obs, list(anchor), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]))
        prev = list(anchor)
    return feas, Js


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    T = rp3._load_trunk()
    report = {"scope": "R6 threshold sweep: (B-A) feas/return vs tau_edit; 5 seeds x {random,urban}; "
                       "reuses trained actor per seed", "tau_edit_grid": _TAUS_EDIT, "by_data": {}}
    for data in ["random", "urban"]:
        # per seed: train once, cache the trained actor + the held scenes + anchor baselines
        cache = []
        for s in seeds:
            tr = rp3._build(data, s * 1000 + 1, 5, g._A())
            hd = rp3._build(data, s * 1000 + 777, 10, g._A())
            r = train_edit_heads(tr, hd[:1], T, epochs=40, hidden=64, seed=s)
            a_feas, a_ret = [], []
            for sc in hd:
                f, j = _anchor_feas_ret(sc, T)
                a_feas += f; a_ret += j
            cache.append({"actor": r["actor"], "mean": r["mean"], "std": r["std"], "hd": hd,
                          "A_feas": sum(a_feas) / len(a_feas), "A_ret": sum(a_ret) / len(a_ret)})
        per_tau = {}
        for tau in _TAUS_EDIT:
            ba_f, ba_r, er_l, un_l, bf_l = [], [], [], [], []
            for c in cache:
                f, j, e, u = [], [], [], []
                for sc in c["hd"]:
                    ff, jj, ee, uu = _rollout_tau(sc, T, c["actor"], c["mean"], c["std"], tau)
                    f += ff; j += jj; e += ee; u += uu
                b_feas = sum(f) / len(f); b_ret = sum(j) / len(j)
                ba_f.append(b_feas - c["A_feas"]); ba_r.append(b_ret - c["A_ret"])
                er_l.append(sum(e) / len(e)); un_l.append(sum(u) / len(u)); bf_l.append(b_feas)
            per_tau[str(tau)] = {"BminusA_feas": g._ci95(ba_f), "BminusA_ret": g._ci95(ba_r),
                                 "B_feas": g._ci95(bf_l), "edit_rate": g._ci95(er_l),
                                 "unsafe_edit": g._ci95(un_l)}
            d = per_tau[str(tau)]
            print(f"[{data} tau={tau}] B-A feas {d['BminusA_feas']['mean']} "
                  f"CI[{d['BminusA_feas']['lo']},{d['BminusA_feas']['hi']}] | B-A ret "
                  f"{d['BminusA_ret']['mean']} CI[{d['BminusA_ret']['lo']},{d['BminusA_ret']['hi']}] | "
                  f"edit {d['edit_rate']['mean']} unsafe {d['unsafe_edit']['mean']}")
        report["by_data"][data] = per_tau
    out = ROOT / "result_save" / "belief_residual" / "R6" / "threshold_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
