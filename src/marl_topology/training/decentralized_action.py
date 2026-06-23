"""Decentralized per-agent action API (Engineering-Plan Phase 6).

Formalizes the production mutual-acceptance action as a per-agent stochastic policy. For every
node ``i`` it returns that node's own action ``a_i`` (the ordered set of incident edges it
accepts within its radio budget), the action log-probability ``logp_i`` and the real policy
entropy ``H_i`` -- the quantities MAPPO / COMA / SCQ require. An undirected edge activates iff
BOTH endpoints accept it (mutual acceptance); the joint log-prob is the sum of the per-agent
log-probs and the joint entropy is the sum of the per-agent entropies.

Each node uses ONLY its own incident-edge logits and its own budget -- no global sort, no global
state -- so the deterministic (temperature -> 0) limit is exactly the deployed decoder
``policies.decentralized_mutual_acceptance.local_mutual_assemble`` (train == deploy, D1). This
module is the TRAINING-side sampler (it carries autograd through ``logp``/``entropy``); the
DEPLOYED path uses the torch-free decoder in ``policies/``. It therefore lives under
``training/`` and is exempt from the deployed-path purity gates.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PerAgentAction:
    """One node's directed action under the mutual-acceptance policy."""

    node_id: object
    gated_edge_indices: tuple[int, ...]   # candidate incident edges (logit >= 0), global indices
    accepted_order: tuple[int, ...]       # the sampled ordered acceptance set, global indices
    logp: Tensor                          # scalar; ordered Plackett-Luce log-prob of accepted_order
    entropy: Tensor                       # scalar; exact entropy of this node's action policy
    budget: int


@dataclass(frozen=True)
class DecentralizedAction:
    per_agent: tuple[PerAgentAction, ...]
    active_edge_indices: tuple[int, ...]  # edges active after mutual acceptance
    joint_logp: Tensor                    # = sum_i logp_i
    joint_entropy: Tensor                 # = sum_i entropy_i


def incident_index(edge_ids, edges) -> dict:
    """node -> [global edge index] for every incident edge, in ``edge_ids`` order (u then v)."""
    incident: dict = defaultdict(list)
    for i, eid in enumerate(edge_ids):
        u, v = edges[eid]
        incident[u].append(i)
        incident[v].append(i)
    return incident


def _ordered_topk_entropy(z: Tensor, k: int) -> Tensor:
    """Exact Shannon entropy of the ordered top-k-without-replacement Plackett-Luce policy over
    the ``m`` candidates with (temperature-scaled) scores ``z``: ``H = -sum_o p(o) log p(o)`` over
    ordered k-tuples ``o``, where ``p(o) = prod_t softmax(z over the remaining)``. Computed by
    exact enumeration -- tractable at this project's node degrees and (small) radio budgets.
    Autograd flows to ``z``.
    """
    m = int(z.shape[0])
    k = min(k, m)
    if k == 0:
        return z.new_zeros(())
    entropy = z.new_zeros(())
    universe = list(range(m))
    for perm in itertools.permutations(universe, k):
        logp = z.new_zeros(())
        remaining = universe.copy()
        for j in perm:
            logp = logp + (z[j] - torch.logsumexp(z[remaining], dim=0))
            remaining.remove(j)
        entropy = entropy - torch.exp(logp) * logp
    return entropy


def _gated_incident(logits, idxs) -> list:
    """A node's candidate edges: its incident edges whose logit clears the same >= 0 gate the
    deployed decoder uses."""
    return [i for i in idxs if float(logits[i]) >= 0.0]


def recompute_logp(logits: Tensor, gated_edge_indices, accepted_order, temperature: float) -> Tensor:
    """Differentiable Plackett-Luce log-prob of a RECORDED ordered acceptance set over a FROZEN
    gated candidate set, given fresh logits.

    This is the re-scorer behind the PPO importance ratio: it re-scores the recorded
    ``accepted_order`` over the ``gated_edge_indices`` frozen at sample time, rather than
    re-deriving the ``logit >= 0`` gate (which drifts as logits move across PPO inner epochs and
    would spuriously break ratio == 1 at epoch 0). ``sample_decentralized_action`` scores its own
    sampled order through this same function, so the sampler's ``logp`` and the ratio's
    ``logp_new`` are identical by construction. Summed over a scene's per-agent actions this is the
    joint log-prob.
    """
    gated = list(gated_edge_indices)
    position = {g: p for p, g in enumerate(gated)}
    z = torch.stack([logits[i] for i in gated]) / temperature
    logp = z.new_zeros(())
    chosen = torch.zeros(len(gated), dtype=torch.bool)
    for g in accepted_order:
        p = position[g]
        logp = logp + (z[p] - torch.logsumexp(z[~chosen], dim=0))
        chosen[p] = True
    return logp


def recompute_entropy(logits: Tensor, gated_edge_indices, k: int, temperature: float) -> Tensor:
    """Exact Plackett-Luce action entropy over a FROZEN gated set on fresh logits -- the
    entropy-bonus twin of :func:`recompute_logp`. ``k`` is the number of edges accepted
    (``min(budget, len(gated))`` at sample time). Summed over a scene's per-agent actions this is
    the joint policy entropy the PPO entropy bonus consumes; it is consistent with the ratio because
    both score over the SAME frozen gated set.
    """
    gated = list(gated_edge_indices)
    if not gated or k <= 0:
        return logits.new_zeros(())
    z = torch.stack([logits[i] for i in gated]) / temperature
    return _ordered_topk_entropy(z, k)


def sample_decentralized_action(
    logits: Tensor,
    edge_ids,
    *,
    edges,
    budgets,
    temperature: float,
    compute_entropy: bool = True,
) -> DecentralizedAction:
    """Sample a fully-decentralized joint action.

    Per node: among its incident edges with logit >= 0, sample up to ``budget`` edges without
    replacement via Plackett-Luce / Gumbel-top-b over ``softmax(logit / temperature)``. An edge
    activates iff both endpoints sampled it. Returns the per-agent actions plus the joint
    log-prob / entropy. Budget-feasible by construction; grad flows through ``logp``/``entropy``.
    """

    incident = incident_index(edge_ids, edges)
    per_agent: list[PerAgentAction] = []
    accept: dict = {}
    joint_logp = logits.new_zeros(())
    joint_entropy = logits.new_zeros(())
    for node, idxs in incident.items():
        budget = int(budgets.get(node, 0))
        gated = _gated_incident(logits, idxs)
        if budget <= 0 or not gated:
            per_agent.append(
                PerAgentAction(node, tuple(gated), (), logits.new_zeros(()), logits.new_zeros(()), budget)
            )
            accept[node] = set()
            continue
        k = min(budget, len(gated))
        z = torch.stack([logits[i] for i in gated]) / temperature
        # Gumbel(0,1) = -log(-log U), U ~ Uniform(0,1). Clamp U away from {0,1} so neither log
        # blows up. (NB: the clamp must sit on U, not on -log U -- the trunk's historical
        # `-torch.log(...).clamp_min(1e-12)` clamps the negative inner log to a constant and
        # yields NaN gumbels, collapsing the sampler to a fixed order with no exploration.)
        u = torch.rand_like(z).clamp(min=1e-12, max=1.0 - 1e-12)
        gumbel = -torch.log(-torch.log(u))
        order = torch.argsort((z + gumbel).detach(), descending=True)[:k].tolist()
        accepted_order = tuple(gated[pos] for pos in order)
        # Score the sampled order through the shared re-scorer so the sampler's logp and the PPO
        # importance ratio's logp_new (which re-scores the recorded order over this same frozen
        # gated set) are identical by construction.
        logp = recompute_logp(logits, tuple(gated), accepted_order, temperature)
        entropy = _ordered_topk_entropy(z, k) if compute_entropy else logits.new_zeros(())
        per_agent.append(
            PerAgentAction(node, tuple(gated), accepted_order, logp, entropy, budget)
        )
        accept[node] = set(accepted_order)
        joint_logp = joint_logp + logp
        joint_entropy = joint_entropy + entropy
    active = tuple(
        i for i, eid in enumerate(edge_ids)
        if i in accept.get(edges[eid][0], ()) and i in accept.get(edges[eid][1], ())
    )
    return DecentralizedAction(tuple(per_agent), active, joint_logp, joint_entropy)


def deterministic_decentralized_action(logits, edge_ids, *, edges, budgets) -> tuple:
    """The temperature -> 0 limit: each node accepts its top-``budget`` incident edges with logit
    >= 0 (ties broken by edge id, like the deployed decoder); an edge is active iff both endpoints
    accept it. Equivalent to ``local_mutual_assemble`` over the same primitives, returned as global
    edge indices."""
    incident: dict = defaultdict(list)
    for i, eid in enumerate(edge_ids):
        u, v = edges[eid]
        score = float(logits[i])
        incident[u].append((score, eid, i))
        incident[v].append((score, eid, i))
    accept: dict = {}
    for node, lst in incident.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        accept[node] = {idx for score, eid, idx in lst[: int(budgets.get(node, 0))] if score >= 0.0}
    return tuple(
        i for i, eid in enumerate(edge_ids)
        if i in accept.get(edges[eid][0], ()) and i in accept.get(edges[eid][1], ())
    )
