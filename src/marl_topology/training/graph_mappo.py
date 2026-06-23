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
    """PPO clipped surrogate on the per-scene joint log-prob.

    ``ratio = exp(logp_new - logp_old)``; the loss is ``-mean(min(ratio*A, clip(ratio)*A))``.
    Returns ``(loss, info)`` with the per-sample ratio, the clip fraction, and Schulman's
    approximate KL. At inner epoch 0 (``logp_new == logp_old``) ``ratio == 1`` so the loss reduces
    to ``-mean(A)`` and ``approx_kl == 0``.
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
    (not NaN) when ``Var(reward) == 0`` (the low-reward-variance warm-started case).
    """
    r = [float(x) for x in rewards]
    v = [float(x) for x in values]
    n = len(r)
    if n == 0:
        return 0.0
    mean_r = sum(r) / n
    var_r = sum((x - mean_r) ** 2 for x in r) / n
    if var_r == 0.0:
        return 0.0
    resid = [ri - vi for ri, vi in zip(r, v)]
    mean_e = sum(resid) / n
    var_e = sum((x - mean_e) ** 2 for x in resid) / n
    return 1.0 - var_e / var_r
