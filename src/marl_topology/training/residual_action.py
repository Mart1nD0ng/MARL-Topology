"""Q6 (POMDP-QP-FAR): the decentralized RESIDUAL action space (anchor + residual edit).

The deployable ``local_hysteresis`` anchor is the DIRECTLY-COMPUTED base; the actor's per-edge residual
logits only ADD / REMOVE / SWAP incident edges around it. The final topology is decoded LOCALLY and
MUTUALLY -- each node edits ONLY its own incident edges, an edge is active iff BOTH endpoints keep it,
the radio budget is respected by construction, and 0 evaluator calls are made (deployable). ``zero
residual`` (no logit crosses a threshold) reproduces the anchor EXACTLY, so the residual policy starts
on the feasible manifold instead of searching the full BCSP space (Spec S10). This sidesteps the Q5
BC-imitation failure: the anchor is reproduced by construction, not learned.

Mirrors ``policies/decentralized_mutual_acceptance.local_mutual_assemble`` (per-node rank, top-b, mutual)
but pivots around the anchor. Torch-free (``float()`` on the logits) -> usable in the deployed decoder.
"""

from __future__ import annotations

from collections import defaultdict

from marl_topology.budgets import node_budgets_for_scene

_MODES = ("add", "remove", "swap", "full")


def residual_decode(anchor_topology, residual_logits, edge_ids, context, *, mode="full",
                    add_threshold=0.0, remove_threshold=0.0):
    """Decode a residual EDIT of the anchor -> ``(accept, topology)``.

    ``accept`` is the per-node set of kept edge-ids (the node's final proposal); ``topology`` is the
    mutual-acceptance edge list. Each node starts from its anchor-incident edges and, from LOCAL residual
    logits, REMOVES anchor edges with logit < ``remove_threshold`` (modes remove/swap/full) and/or ADDS
    non-anchor incident edges with logit > ``add_threshold`` (modes add/swap/full), respecting the budget.
    ``zero residual`` -> ``topology == anchor_topology`` exactly.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    ends = {e.edge_id: (e.node_u, e.node_v) for e in context.graph.edges}
    anchor_set = set(anchor_topology)
    incident: dict = defaultdict(list)                       # node -> [(edge_id, global_index)]
    for i, eid in enumerate(edge_ids):
        a, b = ends[eid]
        incident[a].append((eid, i))
        incident[b].append((eid, i))

    accept: dict = {}
    for node, lst in incident.items():
        cap = int(budgets.get(node, 0))
        keep = {eid for (eid, _i) in lst if eid in anchor_set}      # base = anchor-incident edges (<= cap)
        removed = set()
        if mode in ("remove", "swap", "full"):
            removed = {eid for (eid, i) in lst
                       if eid in keep and float(residual_logits[i]) < remove_threshold}
            keep -= removed
        if mode in ("add", "full"):
            cands = sorted(((float(residual_logits[i]), eid) for (eid, i) in lst
                            if eid not in anchor_set and float(residual_logits[i]) > add_threshold),
                           key=lambda t: (-t[0], t[1]))
            room = max(0, cap - len(keep))                          # keep ALL anchor edges first
            keep |= {eid for _s, eid in cands[:room]}
        elif mode == "swap":
            cands = sorted(((float(residual_logits[i]), eid) for (eid, i) in lst
                            if eid not in anchor_set and float(residual_logits[i]) > add_threshold),
                           key=lambda t: (-t[0], t[1]))
            keep |= {eid for _s, eid in cands[:len(removed)]}       # add at most as many as removed
        if len(keep) > cap:                                        # safety cap (anchor already <= cap)
            idx_of = {eid: i for (eid, i) in lst}
            keep = set(sorted(keep, key=lambda eid: (-float(residual_logits[idx_of[eid]]), eid))[:cap])
        accept[node] = keep

    topology = [eid for eid in edge_ids
                if eid in accept.get(ends[eid][0], set()) and eid in accept.get(ends[eid][1], set())]
    return accept, topology


def deployable_residual_topology(obs, residual_logits, prev_topology, *, mode="full",
                                 keep_threshold=0.4, add_threshold_anchor=0.6,
                                 add_threshold=0.0, remove_threshold=0.0):
    """Convenience deployable decoder: compute the local_hysteresis anchor from ``obs`` (local features +
    own previous topology, 0 eval) then apply the residual. Returns ``(accept, topology)``. The whole
    path is local + mutual -> deployment-decentralized."""
    from marl_topology.training.dynamic_baselines import local_hysteresis_action
    from marl_topology.training.dynamic_rl import _budgets_edges
    budgets, edges = _budgets_edges(obs["context"])
    anchor = local_hysteresis_action(obs["ef"], obs["edge_ids"], edges, budgets, prev_topology,
                                     keep_threshold=keep_threshold, add_threshold=add_threshold_anchor)
    return residual_decode(anchor, residual_logits, obs["edge_ids"], obs["context"],
                           mode=mode, add_threshold=add_threshold, remove_threshold=remove_threshold)
