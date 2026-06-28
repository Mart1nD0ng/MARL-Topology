"""Q7 (POMDP-QP-FAR): D_quorum-guided add-only repair of the local_hysteresis anchor.

On anchor-FAILURE scenes (true C < tau), ADD edges (residual add-only, Q6) to repair feasibility while
RETAINING every anchor edge. Two groups (Contract S10.1):
  * greedy_dquorum_add_repair -- CENTRAL REFERENCE (training-only): greedily add the non-anchor incident
    edge that most reduces D_quorum, using the evaluator per candidate. Establishes the ceiling: can the
    Q4-aligned proxy guide repair at all? NOT deployable.
  * repair_head_targets -- the per-edge supervised target r_e = D(x) - D(x + e) (positive = adding e
    reduces the deficit) for a DEPLOYABLE repair head trained on LOCAL features (targets are training-only
    via the bridge; the head at deploy uses local features + 0 eval).

The final metric is ALWAYS the true closed-form PBFT C (topology_reliability); D_quorum is the
Q4-authorized AUXILIARY guide only.
"""

from __future__ import annotations

from marl_topology.budgets import node_budgets_for_scene
from marl_topology.training.quorum_deficit_bridge import (
    topology_quorum_deficit, topology_reliability)

_DEFICIT_KEY = "d_quorum_mean"


def _ends(context) -> dict:
    return {e.edge_id: (e.node_u, e.node_v) for e in context.graph.edges}


def _degrees(topology, ends) -> dict:
    deg: dict = {}
    for eid in topology:
        u, v = ends[eid]
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def greedy_dquorum_add_repair(evaluator, anchor_topology, edge_ids, context, *, tau=0.9, max_adds=None):
    """CENTRAL REFERENCE (training-only): greedily ADD the non-anchor incident edge that most reduces
    D_quorum (budget + mutual respected) until C >= tau or no candidate reduces D. Uses the evaluator per
    candidate -> NOT deployable. Returns a trajectory dict (final metric = true C)."""
    ends = _ends(context)
    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    anchor = set(anchor_topology)
    topo = set(anchor)
    added: list = []
    calls = 0
    d_anchor = topology_quorum_deficit(evaluator, topo); calls += 1
    d_init = d_anchor[_DEFICIT_KEY] if d_anchor else float("inf")
    c_anchor = topology_reliability(evaluator, topo)["consensus"]; calls += 1
    d_cur = d_init
    cap = max_adds if max_adds is not None else len(edge_ids)
    while len(added) < cap:
        if topology_reliability(evaluator, topo)["consensus"] >= tau:
            calls += 1
            break
        calls += 1
        deg = _degrees(topo, ends)
        best_e, best_d = None, d_cur
        for eid in edge_ids:
            if eid in topo:
                continue
            u, v = ends[eid]
            if deg.get(u, 0) >= budgets.get(u, 0) or deg.get(v, 0) >= budgets.get(v, 0):
                continue
            d = topology_quorum_deficit(evaluator, topo | {eid}); calls += 1
            if d is not None and d[_DEFICIT_KEY] < best_d - 1e-12:
                best_e, best_d = eid, d[_DEFICIT_KEY]
        if best_e is None:
            break
        topo.add(best_e); added.append(best_e); d_cur = best_d
    c_final = topology_reliability(evaluator, topo)["consensus"]; calls += 1
    return {
        "repaired": sorted(topo), "added": added, "n_added": len(added),
        "anchor_C": round(float(c_anchor), 5), "repaired_C": round(float(c_final), 5),
        "C_improvement": round(float(c_final - c_anchor), 5),
        "d_quorum_init": round(float(d_init), 5), "d_quorum_final": round(float(d_cur), 5),
        "d_quorum_reduction": round(float(d_init - d_cur), 5),
        "anchor_retained": anchor.issubset(topo), "retention": 1.0 if anchor.issubset(topo) else
        (len(anchor & topo) / max(1, len(anchor))),
        "repaired_feasible": bool(c_final >= tau), "anchor_feasible": bool(c_anchor >= tau),
        "evaluator_calls": calls,
    }


def greedy_conservative_prune(evaluator, anchor_topology, edge_ids, context, *, tau=0.9, max_removes=None):
    """CENTRAL REFERENCE (training-only): from a FEASIBLE anchor, greedily REMOVE the lowest-RISK edge
    (``risk_e = D(x∖e) − D(x)``) that KEEPS feasibility (C >= tau), until none can be safely removed.
    Lowers cost (energy/latency) while retaining reliability; NEVER deletes a critical edge (checks true
    C before each removal). Uses the evaluator per candidate -> NOT deployable. Returns a trajectory dict
    (final metrics = true C / energy / latency)."""
    anchor = set(anchor_topology)
    base = topology_reliability(evaluator, anchor)
    calls = 1
    if base["consensus"] < tau:                              # prune only applies to a FEASIBLE anchor
        return {"applicable": False, "anchor_C": round(float(base["consensus"]), 5),
                "removed": [], "n_removed": 0, "retention": None, "evaluator_calls": calls}
    topo = set(anchor)
    removed: list = []
    cap = max_removes if max_removes is not None else len(anchor)
    while len(removed) < cap and len(topo) > 0:
        d_cur = topology_quorum_deficit(evaluator, topo); calls += 1
        d_cur_val = d_cur[_DEFICIT_KEY] if d_cur is not None else 0.0
        best_e, best_risk = None, float("inf")
        for eid in list(topo):
            cand = topo - {eid}
            c_cand = topology_reliability(evaluator, cand)["consensus"]; calls += 1
            if c_cand < tau:                                  # removing eid breaks feasibility -> critical
                continue
            d_cand = topology_quorum_deficit(evaluator, cand); calls += 1
            risk = (d_cand[_DEFICIT_KEY] if d_cand is not None else 0.0) - d_cur_val
            if risk < best_risk:
                best_e, best_risk = eid, risk
        if best_e is None:                                    # no edge can be safely removed
            break
        topo.discard(best_e); removed.append(best_e)
    pruned = topology_reliability(evaluator, topo); calls += 1
    return {
        "applicable": True, "pruned": sorted(topo), "removed": removed, "n_removed": len(removed),
        "anchor_C": round(float(base["consensus"]), 5), "pruned_C": round(float(pruned["consensus"]), 5),
        "anchor_energy": round(float(base["energy"]), 6), "pruned_energy": round(float(pruned["energy"]), 6),
        "energy_reduction": round(float(base["energy"] - pruned["energy"]), 6),
        "anchor_latency": round(float(base["latency"]), 6), "pruned_latency": round(float(pruned["latency"]), 6),
        "latency_reduction": round(float(base["latency"] - pruned["latency"]), 6),
        "feasibility_retained": bool(pruned["consensus"] >= tau),
        "retention": 1.0 if pruned["consensus"] >= tau else 0.0,
        "removed_subset_of_anchor": set(removed).issubset(anchor),
        "critical_edge_deletions": 0,                         # the greedy checks C>=tau before each removal
        "evaluator_calls": calls,
    }


def safety_head_targets(evaluator, anchor_topology, edge_ids, context):
    """Per-edge supervised target for the DEPLOYABLE safety head: ``risk_e = D(x∖e) − D(x)`` (positive =
    removing e increases the deficit = load-bearing; low = safe to remove). Anchor edges only; computed
    via the bridge -> TRAINING-ONLY. Returns ``{edge_id: risk}``."""
    anchor = set(anchor_topology)
    d_anchor = topology_quorum_deficit(evaluator, anchor)
    base = d_anchor[_DEFICIT_KEY] if d_anchor else 0.0
    targets: dict = {}
    for eid in anchor:
        d = topology_quorum_deficit(evaluator, anchor - {eid})
        targets[eid] = round(float((d[_DEFICIT_KEY] if d is not None else base) - base), 6)
    return targets


def repair_head_targets(evaluator, anchor_topology, edge_ids, context):
    """Per-edge supervised target for the DEPLOYABLE repair head: ``r_e = D(x) - D(x + e)`` (positive =
    adding e reduces the deficit). Computed via the bridge -> TRAINING-ONLY. Non-anchor edges only; an
    edge that violates a budget if added still gets a target (the head learns desirability; the decoder
    enforces the budget). Returns ``{edge_id: delta_D}``."""
    anchor = set(anchor_topology)
    d_anchor = topology_quorum_deficit(evaluator, anchor)
    base = d_anchor[_DEFICIT_KEY] if d_anchor else 0.0
    targets: dict = {}
    for eid in edge_ids:
        if eid in anchor:
            continue
        d = topology_quorum_deficit(evaluator, anchor | {eid})
        targets[eid] = round(float(base - (d[_DEFICIT_KEY] if d is not None else base)), 6)
    return targets
