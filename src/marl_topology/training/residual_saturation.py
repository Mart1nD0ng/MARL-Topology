"""R1 (Belief-Guided Residual PPO) — standardization-across-all-frames, saturation metrics, raw-logit L2.

Q14 showed the residual logits saturate at the tanh rail; these helpers (a) standardize features over ALL
train frames (not just frame 0, which mis-scales stale features at later frames), (b) measure the saturation
(Contract v4 §9 Saturation/Degeneracy report), (c) penalize large raw logits, and (d) quantify the
Effect-on-Decision delta between recurrent and memoryless arms (Contract v4 §4).
"""

from __future__ import annotations

import torch
from torch import Tensor

from marl_topology.training.decentralized_distillation import feature_standardization


def feature_standardization_all_frames(scenes, *, return_count: bool = False):
    """Pool observations across ALL frames of ALL scenes (prev = [] each frame -> deterministic, and it
    covers the per-frame stale-CSI / motion variation that frame-0-only standardization misses), then reuse
    the canonical ``feature_standardization``. Returns ((nf_mean, ef_mean), (nf_std, ef_std))[, n_obs]."""
    obs = []
    for s in scenes:
        for t in range(s.n_frames):
            obs.append(s.observation(t, []))
    mean, std = feature_standardization(obs)
    if return_count:
        return mean, std, len(obs)
    return mean, std


def saturation_metrics(raw: Tensor, logits: Tensor, logit_scale: float) -> dict:
    """Saturation / degeneracy metrics for a discrete-action logit head (Contract v4 §9). ``frac_logit_near_
    rail`` is |logit| > 0.95*logit_scale (the head-specific rail); ``frac_abs_logit_gt_9_5`` is the legacy
    +-10-head metric for cross-stage comparison."""
    raw = raw.detach().reshape(-1)
    logits = logits.detach().reshape(-1)
    if raw.numel() == 0:
        return {"raw_abs_mean": 0.0, "raw_abs_p95": 0.0, "frac_abs_raw_gt_30": 0.0,
                "frac_abs_logit_gt_9_5": 0.0, "frac_logit_near_rail": 0.0, "n": 0}
    ra = raw.abs()
    la = logits.abs()
    return {
        "raw_abs_mean": float(ra.mean()),
        "raw_abs_p95": float(ra.quantile(0.95)),
        "frac_abs_raw_gt_30": float((ra > 30.0).double().mean()),
        "frac_abs_logit_gt_9_5": float((la > 9.5).double().mean()),
        "frac_logit_near_rail": float((la > 0.95 * float(logit_scale)).double().mean()),
        "n": int(raw.numel()),
    }


def raw_logit_l2_penalty(raw: Tensor) -> Tensor:
    """Mean squared raw logit -- penalizing it keeps the tanh head off the (zero-gradient) saturation rail."""
    if raw.numel() == 0:
        return raw.new_zeros(())
    return (raw.reshape(-1) ** 2).mean()


def recurrent_vs_memoryless_delta(logits_rec: Tensor, logits_mem: Tensor,
                                  action_rec: Tensor | None = None,
                                  action_mem: Tensor | None = None) -> dict:
    """Effect-on-Decision deltas (Contract v4 §4): does carrying the hidden state change the LOGITS and the
    final binary action, not just the forward pass? ``action_*`` are boolean flip masks (logit>0 if omitted)."""
    lr = logits_rec.detach().reshape(-1)
    lm = logits_mem.detach().reshape(-1)
    logit_delta = float((lr - lm).abs().max()) if lr.numel() else 0.0
    if action_rec is None:
        action_rec = lr > 0
    if action_mem is None:
        action_mem = lm > 0
    ar = action_rec.detach().reshape(-1).bool()
    am = action_mem.detach().reshape(-1).bool()
    action_delta = float((ar != am).double().mean()) if ar.numel() else 0.0
    return {
        "recurrent_memoryless_logit_delta": logit_delta,
        "recurrent_memoryless_action_delta": action_delta,
        "behaviorally_equal_to_baseline": bool(action_delta == 0.0 and logit_delta < 1e-6),
    }
