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

from dataclasses import dataclass

import torch
from torch import Tensor

from marl_topology.training.budget_conditioned_subset import sample_subset
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
) -> SCQTargets:
    """Sample up to ``scq_m`` UNORDERED counterfactual subsets ``S~_i ~ pi_i`` (from ``(theta_i, b_i)``
    only), fix ``S_-i``, re-pass the mutual decoder, and query the REAL evaluator ONCE per UNIQUE
    non-trivial topology for the exact ``delta_R = R(S) - R(S~_i, S_-i)`` (Spec S10.2). The actual-action
    reward is reused via ``r_actual`` (no extra call). No-ops (decode back to S) and duplicates are
    skipped (S10.5: shared evaluator cache). This is the ONLY place SCQ spends evaluator budget."""
    accept = acceptance_map(per_agent_actions)
    actual_active = mutual_active_indices(accept, edge_ids, edges)
    if r_actual is None:
        r_actual = reward_of(actual_active)
    agents = [pa for pa in per_agent_actions if pa.incident_edge_indices][: int(scq_m)]
    targets: list = []
    seen: set = set()
    calls = 0
    dup = 0
    for pa in agents:                                    # 9a: first scq_m; 9b: sensitivity-guided top-M
        inc = pa.incident_edge_indices
        theta = (torch.stack([logits[i] for i in inc]) / temperature).detach()
        cf_local = sample_subset(theta, pa.budget, generator=generator)
        accept_cf = dict(accept)
        accept_cf[pa.node_id] = {inc[j] for j in cf_local}          # fix S_-i, swap in S~_i
        active_cf = mutual_active_indices(accept_cf, edge_ids, edges)
        if active_cf == actual_active:
            continue
        if active_cf in seen:
            dup += 1
            continue
        seen.add(active_cf)
        delta_R = float(r_actual) - float(reward_of(active_cf))     # EXTRA evaluator call (budget)
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
