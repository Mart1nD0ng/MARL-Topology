"""Q10 (POMDP-QP-FAR): local edge handshake -- a shared edge score for symmetric mutual acceptance.

Spec §12. The independent mutual decode (``local_mutual_assemble``) has each node rank its incident edges
by the per-edge logit and accept its top-b; an edge activates iff BOTH endpoints accept. Two endpoints
with different incident competition can disagree (a one-sided proposal -> a lost edge). The HANDSHAKE
adds a two-round neighbour exchange: node i sends its endpoint score ``s_{i->j}`` to neighbour j; both
compute the IDENTICAL shared edge score

    s_ij = (s_{i->j} + s_{j->i}) / 2  +  alpha * log(psucc_ij)  +  beta * 1[e in x_{t-1}]

and both rank their budget-incident edges by ``s_ij``. The score being symmetric (and the psucc/prev bias
shared), both ends are pulled toward the SAME high-quality edges, reducing the mismatch. Per-node
computable, NO global sort, only neighbour scalar exchange -> deployment-decentralized. The control cost
(one scalar ``s_{i->j}`` per directed edge = 2*|E|) is recorded. Torch-free.
"""

from __future__ import annotations

import hashlib
import math

from marl_topology.training.decentralized_action import incident_index

_PSUCC_FLOOR = 1e-6


def shared_edge_scores(logits, edge_ids, edges, observed_psucc, prev_topology, *, alpha=1.0, beta=0.0,
                       directed_scores=None):
    """``{eid: s_ij}`` -- the symmetric shared edge score. ``logits[i]`` is the per-edge base endpoint
    score (symmetric: ``s_{i->j}=s_{j->i}=logits[i]``); pass ``directed_scores={eid:(s_uv,s_vu)}`` for a
    directional actor (then the average is taken). ``observed_psucc[i]`` = the OBSERVED link success prob
    (stale-CSI-safe); ``prev_topology`` = the node's own previous edges."""
    prev = set(prev_topology)
    out: dict = {}
    for i, eid in enumerate(edge_ids):
        if directed_scores is not None and eid in directed_scores:
            s_uv, s_vu = directed_scores[eid]
            base = 0.5 * (float(s_uv) + float(s_vu))
        else:
            base = float(logits[i])
        p = max(float(observed_psucc[i]), _PSUCC_FLOOR)
        out[eid] = base + alpha * math.log(p) + beta * (1.0 if eid in prev else 0.0)
    return out


def handshake_decode(logits, edge_ids, edges, budgets, observed_psucc, prev_topology, *, alpha=1.0,
                     beta=0.0, directed_scores=None, threshold=None):
    """Two-round local handshake decode -> ``(accept, topology, info)``. Each node ranks its incident
    edges by the SHARED score ``s_ij`` (identical at both ends) and accepts its top-b (optionally only
    those with score >= ``threshold``); an edge activates iff BOTH endpoints accept. Per-node computable,
    NO global sort. ``info`` records the control-message cost (2*|E| scalars) and the shared scores."""
    shared = shared_edge_scores(logits, edge_ids, edges, observed_psucc, prev_topology,
                                alpha=alpha, beta=beta, directed_scores=directed_scores)
    incident = incident_index(edge_ids, edges)
    accept: dict = {}
    for node, idxs in incident.items():
        b = int(budgets.get(node, 0))
        ranked = sorted(idxs, key=lambda i: (-shared[edge_ids[i]], edge_ids[i]))   # PER-NODE rank only
        chosen = ranked[:b]
        if threshold is not None:
            chosen = [i for i in chosen if shared[edge_ids[i]] >= threshold]
        accept[node] = set(chosen)
    topology = [edge_ids[i] for i, eid in enumerate(edge_ids)
                if i in accept.get(edges[eid][0], set()) and i in accept.get(edges[eid][1], set())]
    info = {"control_messages": 2 * len(edge_ids),          # one s_{i->j} per directed edge (Spec §12)
            "shared_scores": shared}
    return accept, topology, info


def mismatch_metrics(accept, edge_ids, edges, observed_psucc=None, *, critical_psucc=0.8):
    """Mutual-acceptance diagnostics (Spec §15.3) for a per-node ``accept`` map: one-sided proposal rate,
    mutual acceptance rate, and the critical-edge disagreement rate (a high-psucc edge proposed one-sided)."""
    one_sided = mutual = considered = critical_lost = critical_total = 0
    for i, eid in enumerate(edge_ids):
        u, v = edges[eid]
        au = i in accept.get(u, set())
        av = i in accept.get(v, set())
        if au or av:
            considered += 1
            if au and av:
                mutual += 1
            else:
                one_sided += 1
        if observed_psucc is not None and float(observed_psucc[i]) >= critical_psucc:
            critical_total += 1
            if au != av:                                    # a high-quality edge lost to a one-sided proposal
                critical_lost += 1
    return {"one_sided_proposal_rate": one_sided / max(1, considered),
            "mutual_acceptance_rate": mutual / max(1, considered),
            "critical_edge_disagreement_rate": critical_lost / max(1, critical_total),
            "n_proposed": considered}


def correlated_sample_key(edge_id, frame, seed) -> float:
    """A stable per-edge common random number ``xi_e = hash(e, t, seed)`` in [0,1) both endpoints share
    (Spec §12.3 correlated sampling) -- reduces independent-sampling misalignment."""
    d = hashlib.sha256(f"xi|{seed}|{edge_id}|{frame}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / float(1 << 64)
