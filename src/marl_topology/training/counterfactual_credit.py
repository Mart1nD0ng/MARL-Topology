"""Phase 8b (Technical-Spec S9.4-9.5): COMA per-agent counterfactual credit on the action-conditioned
Q critic.

Upgrades the shared single-step scene advantage ``A_s = R_s - V(s)`` (R7 Graph-MAPPO) to a PER-AGENT
counterfactual advantage (Spec S9.4):

    A_i^E = Q_E(s, S) - E_{S~_i ~ pi_i}[ Q_E(s, S~_i, S_{-i}) ].

The baseline ``b_i = (1/K_cf) sum_k Q(s, S~_i^(k), S_{-i})`` is a Monte-Carlo estimate over ``K_cf``
INDEPENDENT BCSP draws ``S~_i ~ pi_i``. Each counterfactual:

  - is UNORDERED (a BCSP subset) and INDEPENDENT of the actual ``S_i`` given ``(o_i, S_{-i})`` -- the
    COMA unbiasedness condition (Spec S9.4 last line): :func:`sample_subset` reads only the node's
    ``(theta_i, b_i)``, never the realized ``S_i``;
  - fixes the other agents' subsets ``S_{-i}`` and RE-PASSES THE SAME mutual decoder (an edge is active
    iff BOTH endpoints accept it -- byte-identical to ``sample_decentralized_bcsp_action``'s decode, so
    train == deploy);
  - re-evaluates the LEARNED ``Q`` -- a critic forward, which is FREE: no evaluator/simulator call, so
    the evaluator budget stays 1 call/scene (== EMA, the Spec S9.8 fair-budget basis).

All of it is ``no_grad``: ``A_i`` is a detached rollout target for the per-agent PPO ratio; the Q critic
is trained by its OWN ``(Q(s,S) - R)^2`` regression (Spec S9.7), never by the counterfactuals. Lives
under ``training/`` (training-only; the deployed actor never sees ``Q``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from marl_topology.training.budget_conditioned_subset import sample_subset
from marl_topology.training.graph_mappo import critic_q_value


def acceptance_map(per_agent_actions) -> dict:
    """``node -> {global edge indices the node accepts}`` (mirrors the BCSP sampler's accept map:
    ``accept[node] = {incident[j] for j in accepted_local}``)."""
    return {pa.node_id: {pa.incident_edge_indices[j] for j in pa.accepted_local_indices}
            for pa in per_agent_actions}


def mutual_active_indices(accept: dict, edge_ids, edges) -> tuple:
    """Re-pass the mutual decoder: global edge index ``i`` active iff BOTH endpoints accept ``i``.
    IDENTICAL to :func:`sample_decentralized_bcsp_action`'s decode (train == deploy)."""
    return tuple(i for i, eid in enumerate(edge_ids)
                 if i in accept.get(edges[eid][0], ()) and i in accept.get(edges[eid][1], ()))


def _active_onehot(active_indices, num_edges: int, *, device, dtype) -> Tensor:
    oh = torch.zeros(num_edges, device=device, dtype=dtype)
    if active_indices:
        oh[list(active_indices)] = 1.0
    return oh


def within_scene_credit_variance(advantages: dict) -> float:
    """Population variance of the per-agent advantages within ONE scene -- the 'credit resolution' a
    method provides. The R7 shared scene advantage (every agent gets the same ``r - V``) has variance
    0 BY CONSTRUCTION; the COMA per-agent credit has variance > 0 whenever agents differ in marginal
    contribution. This is the structural reason a shared advantage cannot represent per-agent credit
    (Spec S13 counterfactual rank correlation motivation)."""
    vals = list(advantages.values())
    if len(vals) <= 1:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


@dataclass(frozen=True, slots=True)
class CounterfactualCredit:
    """Per-agent COMA result. ``advantages``/``baselines`` keyed by ``node_id``; ``q_actual`` is the
    action-conditioned value of the realized joint action; ``actual_active`` the realized active set."""
    advantages: dict
    baselines: dict
    q_actual: float
    actual_active: tuple


def per_agent_counterfactual_credit(
    q_of,
    *,
    per_agent_actions,
    edge_ids,
    edges,
    logits: Tensor,
    temperature: float,
    k_cf: int,
    generator: torch.Generator | None = None,
) -> CounterfactualCredit:
    """Generic per-agent COMA credit given a Q ORACLE ``q_of(active_indices) -> float`` (Spec S9.4).

    For each agent i: draw ``k_cf`` BCSP subsets ``S~_i ~ pi_i`` (from ``(theta_i = logits[inc_i]/T,
    b_i)`` ONLY -> independent of the realized ``S_i``), fix ``S_{-i}``, re-pass the mutual decoder, and
    score the resulting active set with ``q_of``. ``A_i = q_of(S) - mean_k q_of(S~_i^k, S_{-i})``.

    The Q oracle is the only injected dependency, so the SAME verified decode/sample logic serves both
    (a) the LEARNED critic on the budget-neutral training path (``q_of`` = a critic forward, see
    :func:`counterfactual_advantages`) and (b) the TRUE evaluator for the offline credit-fidelity
    DIAGNOSTIC (``q_of`` = the environment reward of the decoded topology -- those evaluator calls are
    diagnostic-only and never enter training, which stays at 1 evaluator call/scene).
    """
    accept = acceptance_map(per_agent_actions)
    actual_active = mutual_active_indices(accept, edge_ids, edges)
    q_actual = q_of(actual_active)
    advantages, baselines = {}, {}
    for pa in per_agent_actions:
        inc = pa.incident_edge_indices
        if not inc:                          # an isolated node has no action -> no counterfactual credit
            advantages[pa.node_id] = 0.0
            baselines[pa.node_id] = q_actual
            continue
        theta = (torch.stack([logits[i] for i in inc]) / temperature).detach()
        qs = []
        for _ in range(int(k_cf)):
            # S~_i ~ pi_i, drawn from (theta_i, b_i) ONLY -> independent of the actual S_i
            cf_local = sample_subset(theta, pa.budget, generator=generator)
            accept_cf = dict(accept)
            accept_cf[pa.node_id] = {inc[j] for j in cf_local}         # fix S_{-i}, swap in S~_i
            qs.append(q_of(mutual_active_indices(accept_cf, edge_ids, edges)))
        b_i = sum(qs) / len(qs)
        baselines[pa.node_id] = b_i
        advantages[pa.node_id] = q_actual - b_i
    return CounterfactualCredit(advantages, baselines, q_actual, actual_active)


def counterfactual_advantages(
    critic,
    *,
    node_features: Tensor,
    edge_features: Tensor,
    edge_index: Tensor,
    edge_ids,
    edges,
    per_agent_actions,
    logits: Tensor,
    temperature: float,
    k_cf: int,
    node_mean: Tensor,
    node_std: Tensor,
    edge_mean: Tensor,
    edge_std: Tensor,
    generator: torch.Generator | None = None,
) -> CounterfactualCredit:
    """Per-agent ``A_i^E = Q(s,S) - (1/K_cf) sum_k Q(s, S~_i^(k), S_{-i})`` with the LEARNED Q critic.

    ``logits`` are the actor's per-edge logits over the global edge list (the same logits the rollout
    sampled from); agent ``i``'s policy is ``BCSP(theta_i = logits[incident_i] / T, b_i)``. Requires
    ``critic.critic_sees_action`` True. Returns a :class:`CounterfactualCredit` (all values detached).
    A thin wrapper over :func:`per_agent_counterfactual_credit` with ``q_of`` = a no_grad critic forward.
    """
    num_edges = int(edge_features.shape[0])

    def q_of(active_indices) -> float:
        oh = _active_onehot(active_indices, num_edges, device=edge_features.device, dtype=edge_features.dtype)
        return float(critic_q_value(critic, node_features, edge_features, edge_index, oh,
                                    node_mean=node_mean, node_std=node_std,
                                    edge_mean=edge_mean, edge_std=edge_std))

    with torch.no_grad():
        return per_agent_counterfactual_credit(
            q_of, per_agent_actions=per_agent_actions, edge_ids=edge_ids, edges=edges,
            logits=logits, temperature=temperature, k_cf=k_cf, generator=generator)
