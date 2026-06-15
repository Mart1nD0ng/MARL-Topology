"""Clipped actor and centralized value losses for Stage 24."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class ClippedPolicyValueLossInputs:
    new_logprobs: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    value_predictions: torch.Tensor
    returns: torch.Tensor
    entropies: torch.Tensor
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    def __post_init__(self) -> None:
        shapes = {
            self.new_logprobs.reshape(-1).shape,
            self.old_logprobs.reshape(-1).shape,
            self.advantages.reshape(-1).shape,
            self.value_predictions.reshape(-1).shape,
            self.returns.reshape(-1).shape,
            self.entropies.reshape(-1).shape,
        }
        if len(shapes) != 1:
            raise ValueError("loss input tensors must have matching flat shape")
        if self.clip_eps <= 0.0:
            raise ValueError("clip_eps must be positive")
        if self.value_coef < 0.0:
            raise ValueError("value_coef must be nonnegative")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be nonnegative")
        for name in (
            "new_logprobs",
            "old_logprobs",
            "advantages",
            "value_predictions",
            "returns",
            "entropies",
        ):
            if not torch.isfinite(getattr(self, name)).all().item():
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ClippedPolicyValueLossResult:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_mean: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max: torch.Tensor
    advantage_mean: torch.Tensor
    advantage_std: torch.Tensor
    value_prediction_mean: torch.Tensor
    return_mean: torch.Tensor
    explained_variance: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy_mean",
            "approx_kl",
            "clip_fraction",
            "ratio_mean",
            "ratio_max",
            "advantage_mean",
            "advantage_std",
            "value_prediction_mean",
            "return_mean",
            "explained_variance",
        ):
            value = getattr(self, name)
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")

    def to_payload(self, *, grad_norm: float | None = None) -> dict[str, float]:
        payload = {
            "total_loss": _scalar(self.total_loss),
            "policy_loss": _scalar(self.policy_loss),
            "value_loss": _scalar(self.value_loss),
            "entropy": _scalar(self.entropy_mean),
            "approx_kl": _scalar(self.approx_kl),
            "clip_fraction": _scalar(self.clip_fraction),
            "ratio_mean": _scalar(self.ratio_mean),
            "ratio_max": _scalar(self.ratio_max),
            "advantage_mean": _scalar(self.advantage_mean),
            "advantage_std": _scalar(self.advantage_std),
            "value_prediction_mean": _scalar(self.value_prediction_mean),
            "return_mean": _scalar(self.return_mean),
            "explained_variance": _scalar(self.explained_variance),
        }
        if grad_norm is not None:
            payload["grad_norm"] = float(grad_norm)
        return payload


def clipped_policy_value_loss(
    inputs: ClippedPolicyValueLossInputs,
) -> ClippedPolicyValueLossResult:
    new_logprobs = inputs.new_logprobs.reshape(-1)
    old_logprobs = inputs.old_logprobs.reshape(-1)
    advantages = inputs.advantages.reshape(-1)
    value_predictions = inputs.value_predictions.reshape(-1)
    returns = inputs.returns.reshape(-1)
    entropies = inputs.entropies.reshape(-1)

    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - inputs.clip_eps, 1.0 + inputs.clip_eps) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(value_predictions, returns)
    entropy_mean = entropies.mean()
    total_loss = policy_loss + inputs.value_coef * value_loss - inputs.entropy_coef * entropy_mean
    approx_kl = ((ratio - 1.0) - log_ratio).mean().clamp_min(0.0)
    clip_fraction = (
        (torch.abs(ratio - 1.0) > inputs.clip_eps).to(dtype=ratio.dtype).mean()
    )
    explained_variance = _explained_variance(returns, value_predictions)
    return ClippedPolicyValueLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio.mean(),
        ratio_max=ratio.max(),
        advantage_mean=advantages.mean(),
        advantage_std=advantages.std(unbiased=False),
        value_prediction_mean=value_predictions.mean(),
        return_mean=returns.mean(),
        explained_variance=explained_variance,
    )


def _explained_variance(returns: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    variance = returns.var(unbiased=False)
    if float(variance.detach().cpu().item()) <= 1e-12:
        return returns.new_tensor(0.0)
    return 1.0 - (returns - predictions).var(unbiased=False) / variance


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(()).item())
