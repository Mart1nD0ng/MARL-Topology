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


def residual_decode_from_flips(anchor_topology, flipped_edge_ids, residual_logits, edge_ids, context,
                               *, mode="full"):
    """Decode a SAMPLED residual (the set of FLIPPED candidate edges) -> ``(accept, topology)``. A flipped
    NON-anchor edge -> ADD; a flipped ANCHOR edge -> REMOVE (mode-gated). Budget-capped (anchor edges kept
    first; added ranked by logit), mutual. Same semantics as :func:`residual_decode` but driven by
    explicit flips (for the RL sampler) instead of thresholds. ``flipped = []`` -> the anchor exactly."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    ends = {e.edge_id: (e.node_u, e.node_v) for e in context.graph.edges}
    anchor = set(anchor_topology)
    flipped = set(flipped_edge_ids)
    logit_of = {eid: float(residual_logits[i]) for i, eid in enumerate(edge_ids)}
    incident: dict = defaultdict(list)
    for eid in edge_ids:
        a, b = ends[eid]
        incident[a].append(eid); incident[b].append(eid)
    accept: dict = {}
    for node, eids in incident.items():
        cap = int(budgets.get(node, 0))
        keep = {e for e in eids if e in anchor}
        if mode in ("remove", "swap", "full"):
            keep -= {e for e in eids if e in anchor and e in flipped}           # remove flipped anchor edges
        if mode in ("add", "swap", "full"):
            added = sorted((e for e in eids if e not in anchor and e in flipped),
                           key=lambda e: (-logit_of[e], e))
            room = max(0, cap - len(keep))
            keep |= set(added[:room])
        if len(keep) > cap:
            keep = set(sorted(keep, key=lambda e: (-logit_of[e], e))[:cap])
        accept[node] = keep
    topology = [eid for eid in edge_ids
                if eid in accept.get(ends[eid][0], set()) and eid in accept.get(ends[eid][1], set())]
    return accept, topology


def residual_candidate_mask(anchor_topology, edge_ids, *, mode="full"):
    """Per-edge flip-candidate mask (torch tensor): which edges the residual MAY flip under ``mode`` --
    non-anchor edges for add, anchor edges for remove, all for full/swap."""
    import torch
    anchor = set(anchor_topology)
    m = torch.zeros(len(edge_ids))
    for i, eid in enumerate(edge_ids):
        in_anchor = eid in anchor
        if (mode == "add" and not in_anchor) or (mode == "remove" and in_anchor) or mode in ("full", "swap"):
            m[i] = 1.0
    return m


def residual_logp(residual_logits, decisions, candidate_mask):
    """Per-edge Bernoulli log-prob of a residual: ``sum_candidates [d*logsigmoid(z)+(1-d)*logsigmoid(-z)]``
    (torch, differentiable -> the PPO per-agent ratio). Non-candidate edges contribute 0."""
    import torch
    import torch.nn.functional as F
    z = residual_logits.reshape(-1)
    d = decisions.reshape(-1).to(z.dtype)
    msk = candidate_mask.reshape(-1).to(z.dtype)
    logp_e = d * F.logsigmoid(z) + (1.0 - d) * F.logsigmoid(-z)
    return (logp_e * msk).sum()


def sample_residual(residual_logits, anchor_topology, edge_ids, context, *, mode="full", generator=None):
    """Sample a residual (per-candidate-edge Bernoulli flip ~ sigmoid(logit)) + its log-prob, and decode
    to the final topology. The ACTION is the flip vector (its logp is tractable for PPO); the decode is a
    deterministic, budget-capped, mutual post-process (zero flips -> the anchor)."""
    import torch
    z = residual_logits.reshape(-1)
    mask = residual_candidate_mask(anchor_topology, edge_ids, mode=mode).to(z.dtype)
    probs = (torch.sigmoid(z) * mask).clamp(0.0, 1.0)
    decisions = torch.bernoulli(probs, generator=generator)
    flipped = [edge_ids[i] for i in range(len(edge_ids)) if float(decisions[i]) > 0.5]
    logp = residual_logp(z, decisions, mask)
    accept, topology = residual_decode_from_flips(anchor_topology, flipped, residual_logits, edge_ids,
                                                  context, mode=mode)
    return {"decisions": decisions, "flipped": flipped, "logp": logp, "candidate_mask": mask,
            "accept": accept, "topology": topology}


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
