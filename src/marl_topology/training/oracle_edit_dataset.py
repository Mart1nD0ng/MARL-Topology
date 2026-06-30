"""R4 (Belief-Guided Residual PPO): beneficial oracle-EDIT dataset (the missing direction signal).

R3 showed the residual PPO trainer is stable but stays at the anchor for lack of a beneficial-edit direction
signal. This builds the teacher: over the deployable local_hysteresis anchor x_H, enumerate LOCAL edits
(add / remove / bounded swap), score each with the TRAINING evaluator (true current channel + closed-form
whole-network quorum tail) -> ΔC, ΔD_quorum, ΔE, ΔL, ΔJ (ΔJ in the SAME dense reward the RL uses), and label
the POSITIVE-gain safe edits. Stores ONLY (anchor, the single edit descriptor, the Δ's) -- NEVER the full
oracle topology (TechSpec §11). Teacher-only (train scenes; never held). If the positive-edit rate is ~0 the
anchor is near a local optimum -> nothing to learn (an honest campaign-closing negative).
"""

from __future__ import annotations

import hashlib
import json

from marl_topology.budgets import node_budgets_for_scene
from marl_topology.training.dynamic_baselines import local_hysteresis_action
from marl_topology.training.dynamic_rl import _budgets_edges
from marl_topology.training.quorum_deficit_bridge import topology_quorum_deficit, topology_reliability
from marl_topology.training.residual_repair import _degrees, _ends

_KT, _AT = 0.4, 0.6
_LAM_C, _LAM_B, _BETA = 1.0, 1.0, 0.1
_DEFICIT_KEY = "d_quorum_mean"


def _anchor(obs, prev):
    budgets, edges = _budgets_edges(obs["context"])
    return local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev,
                                   keep_threshold=_KT, add_threshold=_AT)


def _budget_ok(topo, budgets, ends) -> bool:
    deg = _degrees(topo, ends)
    return all(deg.get(n, 0) <= budgets.get(n, 0) for n in deg)


def _d_value(ev, topo):
    d = topology_quorum_deficit(ev, topo)
    return d[_DEFICIT_KEY] if d else float("inf")


def _record(ev, anchor, edited, kind, edge, ra, da, ja, obs, e_ref, T, tau):
    re = topology_reliability(ev, edited)
    de = _d_value(ev, edited)
    je = float(T.reward_of(obs, list(edited), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0])
    safe = not (ra["consensus"] >= tau and re["consensus"] < tau)
    return {"anchor": tuple(anchor), "kind": kind, "edge": edge,
            "dC": round(float(re["consensus"] - ra["consensus"]), 5),
            "dD_quorum": (round(float(de - da), 5) if (da != float("inf") and de != float("inf")) else None),
            "dE": round(float(re["energy"] - ra["energy"]), 5),
            "dL": round(float(re["latency"] - ra["latency"]), 5),
            "dJ": round(float(je - ja), 5), "safe": bool(safe),
            "anchor_feasible": bool(ra["consensus"] >= tau),
            "edited_C": round(float(re["consensus"]), 5)}


def build_edit_dataset(train_scenes, T, *, margin: float = 0.0, max_swap: int = 3, tau: float = 0.9):
    """Enumerate local edits over the anchor per frame, score with the true evaluator, return (records,
    metrics). Train scenes only (teacher-only; held never enters). One record per (frame, edit)."""
    records = []
    calls = 0
    frame_feasible, frame_repairable, frame_safe_prunable = [], [], []
    for sc in train_scenes:
        prev, e_ref = [], None
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            ev = obs["context"].evaluator
            if e_ref is None:
                e_ref = T._ref_energy(obs)
            ends = _ends(obs["context"])
            budgets = dict(node_budgets_for_scene(ev.scene))
            anchor = _anchor(obs, prev)
            ra = topology_reliability(ev, anchor); calls += 1
            da = _d_value(ev, anchor); calls += 1
            ja = float(T.reward_of(obs, list(anchor), e_ref, _LAM_C, _LAM_B, _BETA, "dense")[0]); calls += 1
            anchor_set = set(anchor)
            deg = _degrees(anchor, ends)
            add_recs, rem_recs = [], []
            for e in obs["edge_ids"]:                                # ADD (budget-feasible non-anchor edges)
                if e in anchor_set:
                    continue
                u, v = ends[e]
                if deg.get(u, 0) >= budgets.get(u, 0) or deg.get(v, 0) >= budgets.get(v, 0):
                    continue
                add_recs.append(_record(ev, anchor, list(anchor) + [e], "add", e, ra, da, ja, obs, e_ref, T, tau))
                calls += 2
            for e in anchor:                                          # REMOVE (anchor edges)
                rem_recs.append(_record(ev, anchor, [x for x in anchor if x != e], "remove", e, ra, da, ja,
                                        obs, e_ref, T, tau))
                calls += 2
            swap_recs = []                                            # SWAP (bounded: top-k add x top-k remove)
            for ar in sorted(add_recs, key=lambda r: -r["dJ"])[:max_swap]:
                for rr in sorted(rem_recs, key=lambda r: -r["dJ"])[:max_swap]:
                    edited = [x for x in anchor if x != rr["edge"]] + [ar["edge"]]
                    if not _budget_ok(edited, budgets, ends):
                        continue
                    swap_recs.append(_record(ev, anchor, edited, "swap", (rr["edge"], ar["edge"]), ra, da, ja,
                                             obs, e_ref, T, tau))
                    calls += 2
            frame = add_recs + rem_recs + swap_recs
            records.extend(frame)
            feas = ra["consensus"] >= tau
            frame_feasible.append(feas)
            frame_repairable.append((not feas) and any(r["edited_C"] >= tau for r in add_recs + swap_recs))
            frame_safe_prunable.append(feas and any(r["edited_C"] >= tau and r["dE"] < 0 for r in rem_recs))
            prev = list(anchor)
    metrics = _metrics(records, calls, margin, frame_feasible, frame_repairable, frame_safe_prunable)
    return records, metrics


def supervised_records(records, *, margin: float = 0.0):
    """The teacher subset: ONLY positive-gain (ΔJ > margin) AND safe edits (TechSpec §11)."""
    return [r for r in records if r["dJ"] > margin and r["safe"]]


def dataset_hash(records) -> str:
    payload = json.dumps([[r["kind"], r["edge"], r["dJ"], r["dC"]] for r in records], sort_keys=True,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _metrics(records, calls, margin, frame_feasible, frame_repairable, frame_safe_prunable):
    n = len(records)
    pos = [r for r in records if r["dJ"] > margin and r["safe"]]
    n_infeasible = sum(1 for f in frame_feasible if not f)
    n_feasible = sum(1 for f in frame_feasible if f)
    best_gains = sorted((r["dJ"] for r in records), reverse=True)
    return {
        "n_records": n, "n_frames": len(frame_feasible), "margin": margin,
        "positive_edit_rate": round(len(pos) / max(1, n), 5),
        "n_positive_edits": len(pos),
        "best_edit_gain_max": round(best_gains[0], 5) if best_gains else 0.0,
        "best_edit_gain_p99": round(best_gains[max(0, len(best_gains) // 100)], 5) if best_gains else 0.0,
        "anchor_failure_repairable_rate": round(sum(frame_repairable) / max(1, n_infeasible), 5),
        "safe_prune_rate": round(sum(frame_safe_prunable) / max(1, n_feasible), 5),
        "n_anchor_infeasible_frames": n_infeasible, "n_anchor_feasible_frames": n_feasible,
        "evaluator_calls": calls, "dataset_hash": dataset_hash(records),
    }
