"""Phase 9 (Technical-Spec S10): SCQ closed-form counterfactual supervision for the centralized Q critic.

TRAINING-ONLY (S10.1: SCQ is NOT a decoder; deployment never calls it). For a few selected LOCAL
counterfactuals it queries the REAL evaluator for the EXACT single-step difference (S10.2):

    DeltaR_i^env = R(S) - R(S~_i, S_-i)

and trains the Q critic so its PREDICTED difference matches that exact value:

    L_SCQ = mean_i [ (Q(s, S) - Q(s, S~_i, S_-i)) - DeltaR_i^env ]^2 .

This directly calibrates Q on the per-agent marginal -- the Q-fidelity bottleneck the Phase-8 re-review
exposed (the learned-Q COMA credit tracked the true marginal only at rho~0.17). Properties (Spec S10.5):
the counterfactuals are UNORDERED subset edits re-passed through the SAME mutual decoder; SCQ is critic
supervision + witness discovery, NOT a policy baseline -- it enters ONLY the critic loss, never the
actor gradient. SCQ is NOT budget-neutral: each unique counterfactual costs one extra evaluator call
(reported as ``counterfactual_calls``); the actual-action reward is reused from the rollout, and
proposals that the mutual decoder maps to an already-seen topology share the evaluator cache (dedup).

Lives under ``training/`` (training-only). ``q_of`` returns a grad-on critic value (the critic is being
trained); ``reward_of`` returns the float environment reward (the supervision target, no grad).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from marl_topology.training.budget_conditioned_subset import inclusion_marginals, sample_subset
from marl_topology.training.counterfactual_credit import acceptance_map, mutual_active_indices


@dataclass(frozen=True, slots=True)
class SCQTargets:
    """The EVALUATOR side of SCQ, computed ONCE per scene per update (so the extra evaluator calls are
    paid once, not per critic epoch). ``actual_active`` + a list of ``(active_cf, delta_R)`` where
    ``delta_R = R(S) - R(S~_i, S_-i)`` is the exact env difference. The counters are the S10.5 audit
    fields; ``counterfactual_calls`` = EXTRA evaluator calls spent (the budget)."""
    actual_active: tuple
    targets: tuple              # tuple[(tuple active_cf, float delta_R), ...]
    counterfactual_calls: int
    unique_subset_count: int
    duplicate_topology_count: int


@dataclass(frozen=True, slots=True)
class SCQResult:
    """SCQ supervision outcome (the all-in-one path). ``loss`` is grad-on (trains the critic);
    ``mean_abs_residual`` = critic_difference_error |Q_diff - DeltaR|."""
    loss: Tensor
    counterfactual_calls: int
    unique_subset_count: int
    duplicate_topology_count: int
    mean_abs_residual: float


@dataclass(frozen=True, slots=True)
class SCQScoreWeights:
    """Weights for the Spec S10.4 counterfactual-candidate score
    ``s_e^cf = alpha|chi_e| + beta*boundary(z_e) + gamma*mutualConflict(e) + eta*bridgeScore(e)
    - zeta*cost(e)``. ``zeta`` defaults to 0 (cost is inert unless a per-edge cost is supplied)."""
    alpha: float = 1.0   # inclusion-marginal sensitivity |chi_e|
    beta: float = 1.0    # closeness to the logit gate boundary
    gamma: float = 1.0   # mutual-acceptance conflict between endpoints
    eta: float = 1.0     # structural bridge importance
    zeta: float = 0.0    # per-edge cost penalty (energy/budget)


def _bridge_score(e: int, active, edges, edge_ids) -> float:
    """1.0 iff global edge ``e`` is a BRIDGE of the active topology (removing it increases the number
    of connected components over the fixed active node set), else 0.0. Only active edges can bridge."""
    if e not in set(active):
        return 0.0
    nodes = set()
    for i in active:
        u, v = edges[edge_ids[i]]
        nodes.add(u)
        nodes.add(v)

    def ncomp(active_subset) -> int:
        adj = defaultdict(set)
        for i in active_subset:
            u, v = edges[edge_ids[i]]
            adj[u].add(v)
            adj[v].add(u)
        seen, comp = set(), 0
        for n in nodes:
            if n in seen:
                continue
            comp += 1
            stack = [n]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(adj[x] - seen)
        return comp

    return 1.0 if ncomp([i for i in active if i != e]) > ncomp(list(active)) else 0.0


def sensitivity_score(
    e: int,
    mu_local: float,
    *,
    logits: Tensor,
    accept: dict,
    edges,
    edge_ids,
    active,
    weights: SCQScoreWeights,
    edge_cost=None,
) -> float:
    """Spec S10.4 candidate score for global edge ``e`` (``mu_local`` = its BCSP inclusion marginal in
    the proposing node). chi = mu(1-mu) (inclusion uncertainty), boundary = 1/(1+|z_e|), mutualConflict
    = endpoints disagree on accepting e, bridgeScore = e bridges the active topology, cost = supplied
    per-edge cost (or 0)."""
    chi = float(mu_local) * (1.0 - float(mu_local))
    boundary = 1.0 / (1.0 + abs(float(logits[e])))
    u, v = edges[edge_ids[e]]
    conflict = 1.0 if ((e in accept.get(u, ())) != (e in accept.get(v, ()))) else 0.0
    bridge = _bridge_score(e, active, edges, edge_ids)
    cost = float(edge_cost[e]) if edge_cost is not None else 0.0
    w = weights
    return w.alpha * chi + w.beta * boundary + w.gamma * conflict + w.eta * bridge - w.zeta * cost


def select_topM_counterfactuals(
    per_agent_actions,
    *,
    logits: Tensor,
    temperature: float,
    edges,
    edge_ids,
    M: int,
    accept: dict,
    actual_active,
    weights: SCQScoreWeights | None = None,
    edge_cost=None,
) -> list:
    """Spec S10.4: score per-agent add/remove/swap edits of each incident edge by ``s_e^cf`` and return
    the top-``M`` ``(node_id, new_accepted_global_set)`` proposals (deduped). Edits are UNORDERED subset
    operations (re-decoded by the caller). These candidates DEPEND on the realized action S_i -- valid
    because SCQ is critic supervision, NOT a policy baseline (Spec S10.4/10.5)."""
    w = weights or SCQScoreWeights()
    scored: list = []
    for pa in per_agent_actions:
        inc = pa.incident_edge_indices
        if not inc:
            continue
        theta = (torch.stack([logits[i] for i in inc]) / temperature).detach()
        mu = inclusion_marginals(theta, pa.budget)        # P(e in S_i) per local incident edge
        a_i = set(accept.get(pa.node_id, set()))
        for li, e in enumerate(inc):
            if e in a_i:                                  # remove
                new = a_i - {e}
            elif len(a_i) < pa.budget:                    # add (budget allows)
                new = a_i | {e}
            elif a_i:                                     # swap: add e, drop the lowest-logit accepted
                drop = min(a_i, key=lambda g: float(logits[g]))
                new = (a_i - {drop}) | {e}
            else:
                new = {e}
            s = sensitivity_score(e, float(mu[li]), logits=logits, accept=accept, edges=edges,
                                  edge_ids=edge_ids, active=actual_active, weights=w, edge_cost=edge_cost)
            scored.append((s, pa.node_id, frozenset(new)))
    scored.sort(key=lambda t: -t[0])
    out, seen = [], set()
    for _s, nid, new in scored:
        key = (nid, new)
        if key in seen:
            continue
        seen.add(key)
        out.append((nid, set(new)))
        if len(out) >= int(M):
            break
    return out


def scq_counterfactual_targets(
    reward_of,
    *,
    per_agent_actions,
    edge_ids,
    edges,
    logits: Tensor,
    temperature: float,
    scq_m: int,
    r_actual: float | None = None,
    generator: torch.Generator | None = None,
    selection: str = "simple",
    score_weights: SCQScoreWeights | None = None,
    edge_cost=None,
) -> SCQTargets:
    """Build SCQ counterfactual targets: select up to ``scq_m`` candidates, fix ``S_-i``, re-pass the
    mutual decoder, and query the REAL evaluator ONCE per UNIQUE non-trivial topology for the exact
    ``delta_R = R(S) - R(S~_i, S_-i)`` (Spec S10.2). ``selection``: ``"simple"`` (9a -- first scq_m
    agents, one BCSP sample each) or ``"topM"`` (9b -- Spec S10.4 sensitivity-guided add/remove/swap).
    The actual-action reward is reused via ``r_actual``; no-ops and duplicate topologies are skipped
    (S10.5). This is the ONLY place SCQ spends evaluator budget."""
    accept = acceptance_map(per_agent_actions)
    actual_active = mutual_active_indices(accept, edge_ids, edges)
    if r_actual is None:
        r_actual = reward_of(actual_active)

    if selection == "topM":
        candidates = select_topM_counterfactuals(
            per_agent_actions, logits=logits, temperature=temperature, edges=edges, edge_ids=edge_ids,
            M=scq_m, accept=accept, actual_active=actual_active, weights=score_weights, edge_cost=edge_cost)
    elif selection == "simple":
        candidates = []
        for pa in [p for p in per_agent_actions if p.incident_edge_indices][: int(scq_m)]:
            inc = pa.incident_edge_indices
            theta = (torch.stack([logits[i] for i in inc]) / temperature).detach()
            cf_local = sample_subset(theta, pa.budget, generator=generator)
            candidates.append((pa.node_id, {inc[j] for j in cf_local}))
    else:
        raise ValueError(f"unknown selection {selection!r} (want 'simple' or 'topM')")

    targets: list = []
    seen: set = set()
    calls = 0
    dup = 0
    for node_id, new_accepted in candidates:
        accept_cf = dict(accept)
        accept_cf[node_id] = set(new_accepted)                       # fix S_-i, swap in S~_i (unordered)
        active_cf = mutual_active_indices(accept_cf, edge_ids, edges)
        if active_cf == actual_active:
            continue
        if active_cf in seen:
            dup += 1
            continue
        seen.add(active_cf)
        delta_R = float(r_actual) - float(reward_of(active_cf))      # EXTRA evaluator call (budget)
        calls += 1
        targets.append((active_cf, delta_R))
    return SCQTargets(actual_active, tuple(targets), calls, len(seen), dup)


def scq_loss_from_targets(q_of, scq: SCQTargets) -> tuple[Tensor, float]:
    """L_SCQ = mean_i [(Q(s,S) - Q(s,S~_i,S_-i)) - delta_R_i]^2 from cached targets. ``q_of`` is grad-on
    (the critic, recomputed each critic epoch); the evaluator targets are fixed. Returns (loss,
    mean_abs_residual). No actor is involved -- SCQ enters ONLY the critic loss (Spec S10.4/10.5)."""
    q_actual = q_of(scq.actual_active)
    if not scq.targets:
        return q_actual.sum() * 0.0, 0.0                            # differentiable zero
    res = torch.stack([(q_actual - q_of(active_cf)) - delta_R for active_cf, delta_R in scq.targets])
    return (res ** 2).mean(), float(res.abs().mean().detach())


def scq_consistency_loss(
    q_of,
    reward_of,
    *,
    per_agent_actions,
    edge_ids,
    edges,
    logits: Tensor,
    temperature: float,
    scq_m: int,
    r_actual: float | None = None,
    generator: torch.Generator | None = None,
) -> SCQResult:
    """All-in-one SCQ loss (targets + loss in one call). Convenience for tests / single-pass use; the
    trunk computes :func:`scq_counterfactual_targets` ONCE per update and re-applies
    :func:`scq_loss_from_targets` each critic epoch so the evaluator budget is paid once."""
    scq = scq_counterfactual_targets(reward_of, per_agent_actions=per_agent_actions, edge_ids=edge_ids,
                                     edges=edges, logits=logits, temperature=temperature, scq_m=scq_m,
                                     r_actual=r_actual, generator=generator)
    loss, mar = scq_loss_from_targets(q_of, scq)
    return SCQResult(loss, scq.counterfactual_calls, scq.unique_subset_count,
                     scq.duplicate_topology_count, mar)
