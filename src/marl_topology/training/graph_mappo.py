"""Graph-MAPPO update primitives for the single-step CTDE trunk (Engineering-Plan Phase 7).

Pure functions: the single-step (T=1) advantage, the PPO-clip actor surrogate on the Phase-6
joint log-prob, and the two headline critic diagnostics (explained variance, Schulman approximate
KL). No GAE / gamma / lambda / bootstrap -- the setting is a contextual bandit, so the advantage
is the immediate reward minus the value baseline (this IS GAE at T=1; introducing a temporal
return here would be D6-inert dead code until the dynamic env exists, Phase 5/11).

These live under ``training/`` (torch-exempt) and are TRAINING-ONLY -- the centralized critic that
produces ``value`` is never reachable from the deployed actor (D1).
"""

from __future__ import annotations

import torch
from torch import Tensor


def critic_scene_value(
    critic,
    node_features: Tensor,
    edge_features: Tensor,
    edge_index: Tensor,
    *,
    node_mean: Tensor,
    node_std: Tensor,
    edge_mean: Tensor,
    edge_std: Tensor,
) -> Tensor:
    """``V(scene)`` from the standardized node/edge features (one real scene -> all-ones masks).

    Spec S8.4: the critic TRAIN forward must keep its gradient -- so this helper is NOT wrapped in
    ``no_grad``. The UPDATE caller invokes it directly (grad on, ``optimizer.step()`` then moves the
    critic); the ROLLOUT caller wraps it in ``with torch.no_grad()`` for the detached baseline value.
    No oracle/teacher label is fed -> the critic never leaks into the deployed actor (D1).
    Device-preserving (masks created on the features' device).
    """
    nf = ((node_features - node_mean) / node_std).unsqueeze(0)
    ef = ((edge_features - edge_mean) / edge_std).unsqueeze(0)
    node_mask = torch.ones(1, node_features.shape[0], device=node_features.device)
    edge_mask = torch.ones(1, edge_features.shape[0], device=edge_features.device)
    return critic(nf, ef, edge_index.unsqueeze(0), node_mask, edge_mask)[0]


def graph_mappo_advantage(reward: float, value: Tensor) -> Tensor:
    """Single-step advantage ``A = reward - V(scene).detach()``. The value is detached so the
    actor's PPO gradient never flows into the critic through the advantage (the critic is trained
    by its own ``(reward - V)^2`` loss)."""
    return reward - value.detach()


def ppo_clip_actor_loss(
    logp_new: Tensor,
    logp_old: Tensor,
    advantage: Tensor,
    clip_eps: float = 0.2,
) -> tuple[Tensor, dict]:
    """PPO clipped surrogate. ELEMENT-WISE in its inputs: ``ratio = exp(logp_new - logp_old)`` per
    element, loss ``-mean(min(ratio*A, clip(ratio)*A))``.

    Feed PER-AGENT-flattened arrays (one element per ``(scene, agent)`` pair, with the scene's
    advantage repeated across its agents) to get the Spec-S9.2 PER-AGENT ratio ``rho_{s,i}`` -- NOT a
    joint ratio (summing per-agent logps into a per-scene scalar first would give the forbidden joint
    ratio whose variance explodes with the agent count). Returns ``(loss, info)`` with the per-element
    ratio, clip fraction, and Schulman's approximate KL. At inner epoch 0 (``logp_new == logp_old``)
    ``ratio == 1`` so the loss reduces to ``-mean(A)`` and ``approx_kl == 0``.
    """
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        info = {
            "ratio": ratio.detach(),
            "clip_fraction": (torch.abs(ratio - 1.0) > clip_eps).double().mean(),
            "approx_kl": _approx_kl_from_ratio(ratio, logp_new - logp_old),
        }
    return loss, info


def approx_kl(logp_new: Tensor, logp_old: Tensor) -> Tensor:
    """Schulman's positive approximate KL estimator ``E[(ratio - 1) - log(ratio)]`` >= 0, with
    ``log(ratio) = logp_new - logp_old``. Zero iff the two policies agree on every sample."""
    log_ratio = logp_new - logp_old
    return _approx_kl_from_ratio(torch.exp(log_ratio), log_ratio)


def _approx_kl_from_ratio(ratio: Tensor, log_ratio: Tensor) -> Tensor:
    return ((ratio - 1.0) - log_ratio).mean()


def explained_variance(rewards, values) -> float:
    """``1 - Var(reward - value)/Var(reward)`` over the minibatch (population variance).

    ``== 1`` when the critic is perfect, ``== 0`` when it only predicts the mean (no better than
    the EMA baseline), ``< 0`` when anti-correlated (a collapse alarm). Returns a ``0.0`` sentinel
    (not NaN, not a spurious huge magnitude) when ``Var(reward)`` is ~0: EV = 1 - Var(r-V)/Var(r) is
    UNDEFINED at zero reward variance, and a tiny-but-nonzero ``Var(r)`` (near-constant rewards, e.g.
    a low-temperature smoke) divided into a normal ``Var(r-V)`` yields a meaningless huge-negative EV.
    The threshold guards that degeneracy -- it is the metric being undefined, NOT a critic collapse.
    """
    r = [float(x) for x in rewards]
    v = [float(x) for x in values]
    n = len(r)
    if n == 0:
        return 0.0
    mean_r = sum(r) / n
    var_r = sum((x - mean_r) ** 2 for x in r) / n
    if var_r < 1e-8:  # reward variance ~ 0 -> EV is undefined (see docstring)
        return 0.0
    resid = [ri - vi for ri, vi in zip(r, v)]
    mean_e = sum(resid) / n
    var_e = sum((x - mean_e) ** 2 for x in resid) / n
    return 1.0 - var_e / var_r
