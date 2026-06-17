"""Decentralized mutual-acceptance topology decoder (the production decoder).

Each node ranks ONLY its own incident edges by its locally computed logits, accepts its top-b
(b = its own radio budget) among those with logit >= 0 (sigmoid >= 0.5); an undirected edge
activates iff BOTH endpoints accept it. The decode is per-node computable, uses zero global
state, and respects each node's radio budget BY CONSTRUCTION -- a genuinely decentralized
end-to-end action. This is the decoder behind the project's validated 0.82 result
(``docs/URBAN_V2X_RESEARCH_LOG.md`` Step-3); recovered verbatim from the logs research path.

``global_argsort_assemble`` is retained ONLY as the centralized-decode ABLATION (the "price of
decentralized assembly", which is ~0 at the multi-RSU operating point), not a deployment path.
"""

from __future__ import annotations

from collections import defaultdict

from marl_topology.budgets import node_budgets_for_scene


def local_mutual_assemble(logits, edge_ids, context):
    """Fully decentralized: each node ranks ONLY its incident edges by the local logit, accepts
    its top-b (b = own radio budget) among those with logit >= 0; an edge activates iff both
    endpoints accept. Per-node computable; budgets respected by construction."""

    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    ends = {e.edge_id: (e.node_u, e.node_v) for e in context.graph.edges}
    incident = defaultdict(list)
    for i, eid in enumerate(edge_ids):
        a, b = ends[eid]
        score = float(logits[i])
        incident[a].append((score, eid))
        incident[b].append((score, eid))
    accept = {}
    for node, lst in incident.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        accept[node] = {eid for s, eid in lst[: budgets.get(node, 0)] if s >= 0.0}
    return [eid for eid in edge_ids
            if eid in accept.get(ends[eid][0], ()) and eid in accept.get(ends[eid][1], ())]


def global_argsort_assemble(logits, edge_ids, context):
    """ABLATION ONLY: the covertly centralized assembly (global sort over all edges)."""

    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    ends = {e.edge_id: (e.node_u, e.node_v) for e in context.graph.edges}
    order = sorted(range(len(edge_ids)), key=lambda i: -float(logits[i]))
    selected, deg = [], defaultdict(int)
    for i in order:
        if float(logits[i]) < 0.0:
            break
        a, b = ends[edge_ids[i]]
        if deg[a] >= budgets.get(a, 0) or deg[b] >= budgets.get(b, 0):
            continue
        selected.append(edge_ids[i]); deg[a] += 1; deg[b] += 1
    return selected
