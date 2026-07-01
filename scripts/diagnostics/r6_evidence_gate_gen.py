"""R6 generator: deployable A/B of the evidence-gated residual action vs the anchor (>=5 seeds, CI).

Arms (deployed on HELD scenes, 0 evaluator at decision -- the evaluator scores ONLY the produced topology for
the honest C/feasibility/J metric on the current real channel):
  A  anchor (local_hysteresis; edit_rate 0 by construction)
  B  evidence-gated residual (TRAINED R5 heads, frozen)
  C  evidence-gated residual (UNTRAINED heads) -- random-gate control
Reports per-seed + 95% CI on (B-A) and (B-C) feasibility / return, plus edit_rate / unsafe_edit_rate /
retention / zero_edit_rate. (B-A) feasibility CI > 0 on a regime => the locally-learned ranking (R5) becomes a
DEPLOYED gain (KEEP -> R7). Spans 0 => mechanism correct but no net gain yet (tune at R7, NOT a STOP).
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
from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: E402
from marl_topology.training.edit_head_training import _LAM_B, _LAM_C, _BETA, _anchor, train_edit_heads  # noqa: E402
from marl_topology.training.evidence_gated_action import evidence_gated_residual  # noqa: E402
from marl_topology.training.quorum_deficit_bridge import topology_reliability  # noqa: E402

_TAU_FEAS = 0.9
_TAUS = {"tau_edit": 0.5, "tau_repair": 0.0, "tau_safety": 0.0}


class _A:
    dyn_nodes = [8, 12, 16]
    frames = 4
    gamma = 0.95


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


def _cons(ev, topo):
    r = topology_reliability(ev, topo)
    return float(r["consensus"]) if r else 0.0


def _rollout(sc, T, kind, actor=None, mean=None, std=None):
    """Roll one arm over a scene; each arm tracks its OWN previous topology. Returns per-frame lists."""
    prev, e_ref = [], None
    feas, Js, edits, unsafe, zero = [], [], [], [], []
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
        cons = _cons(ev, topo)
        feas.append(1.0 if cons >= _TAU_FEAS else 0.0)
        Js.append(float(T.reward_of(obs, list(topo), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]))
        edits.append(er)
        if kind != "anchor":
            cons_a = _cons(ev, anchor)
            unsafe.append(1.0 if (cons_a >= _TAU_FEAS and cons < _TAU_FEAS) else 0.0)
            zero.append(1.0 if er == 0.0 else 0.0)
        prev = list(topo)
    return {"feas": feas, "J": Js, "edit": edits, "unsafe": unsafe, "zero": zero}


def _agg(scenes, T, kind, actor=None, mean=None, std=None):
    f, j, e, u, z = [], [], [], [], []
    for sc in scenes:
        r = _rollout(sc, T, kind, actor, mean, std)
        f += r["feas"]; j += r["J"]; e += r["edit"]; u += r["unsafe"]; z += r["zero"]
    def _m(x):
        return sum(x) / len(x) if x else 0.0
    return {"feasibility": _m(f), "return": _m(j), "edit_rate": _m(e),
            "unsafe_edit_rate": _m(u), "zero_edit_rate": _m(z)}


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    T = rp3._load_trunk()
    report = {"scope": "R6 evidence-gated residual action vs anchor; deployed on held, 0 eval at decision; "
                       "5 seeds x {random,urban}; (B-A) and (B-C) feasibility/return CI", "taus": _TAUS,
              "tau_feas": _TAU_FEAS, "by_data": {}}
    for data in ["random", "urban"]:
        per = {"A_feas": [], "B_feas": [], "C_feas": [], "A_ret": [], "B_ret": [], "C_ret": [],
               "BminusA_feas": [], "BminusA_ret": [], "BminusC_feas": [], "BminusC_ret": [],
               "B_edit_rate": [], "B_unsafe_edit": [], "B_zero_edit": []}
        for s in seeds:
            tr = rp3._build(data, s * 1000 + 1, 5, _A())
            hd = rp3._build(data, s * 1000 + 777, 10, _A())
            # train the heads on TRAIN only; pass a 1-scene held for the (unused-here) internal top-k metric
            # so train_edit_heads does NOT rebuild the evaluator-heavy edit examples for all 10 deploy scenes.
            r = train_edit_heads(tr, hd[:1], T, epochs=40, hidden=64, seed=s)
            actor, mean, std = r["actor"], r["mean"], r["std"]
            untrained = BeliefResidualActor(tr[0].observation(0, [])["nf"].shape[1],
                                            tr[0].observation(0, [])["ef"].shape[1], hidden=r["hidden"])
            A = _agg(hd, T, "anchor")
            B = _agg(hd, T, "gated", actor, mean, std)
            C = _agg(hd, T, "gated", untrained, mean, std)
            per["A_feas"].append(A["feasibility"]); per["B_feas"].append(B["feasibility"])
            per["C_feas"].append(C["feasibility"]); per["A_ret"].append(A["return"])
            per["B_ret"].append(B["return"]); per["C_ret"].append(C["return"])
            per["BminusA_feas"].append(B["feasibility"] - A["feasibility"])
            per["BminusA_ret"].append(B["return"] - A["return"])
            per["BminusC_feas"].append(B["feasibility"] - C["feasibility"])
            per["BminusC_ret"].append(B["return"] - C["return"])
            per["B_edit_rate"].append(B["edit_rate"]); per["B_unsafe_edit"].append(B["unsafe_edit_rate"])
            per["B_zero_edit"].append(B["zero_edit_rate"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["beats_anchor_feas"] = d["BminusA_feas"]["lo"] is not None and d["BminusA_feas"]["lo"] > 0
        d["beats_anchor_ret"] = d["BminusA_ret"]["lo"] is not None and d["BminusA_ret"]["lo"] > 0
        report["by_data"][data] = d
        print(f"[{data}] A_feas {d['A_feas']['mean']} B_feas {d['B_feas']['mean']} | "
              f"B-A feas {d['BminusA_feas']['mean']} CI[{d['BminusA_feas']['lo']},{d['BminusA_feas']['hi']}] "
              f"beats={d['beats_anchor_feas']} | B-A ret {d['BminusA_ret']['mean']} "
              f"CI[{d['BminusA_ret']['lo']},{d['BminusA_ret']['hi']}] | edit {d['B_edit_rate']['mean']} "
              f"unsafe {d['B_unsafe_edit']['mean']} zero {d['B_zero_edit']['mean']}")
    out = ROOT / "result_save" / "belief_residual" / "R6" / "evidence_gate_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
