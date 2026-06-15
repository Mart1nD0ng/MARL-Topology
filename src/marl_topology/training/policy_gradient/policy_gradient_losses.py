"""Small Stage 23 proposal-policy gradient loss helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PolicyGradientLossInputs:
    new_logprob: torch.Tensor
    old_logprob: torch.Tensor
    training_signal: torch.Tensor
    baseline: torch.Tensor
    clip_epsilon: float = 0.2
    normalize_advantage: bool = False


@dataclass(frozen=True, slots=True)
class PolicyGradientLossResult:
    loss: torch.Tensor
    advantage_mean: float
    approximate_kl: float
    clip_fraction: float
    ratio_mean: float

    def to_payload(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach().cpu().item()),
            "advantage_mean": self.advantage_mean,
            "approximate_kl": self.approximate_kl,
            "clip_fraction": self.clip_fraction,
            "ratio_mean": self.ratio_mean,
        }


def clipped_reinforce_loss(inputs: PolicyGradientLossInputs) -> PolicyGradientLossResult:
    """Return a clipped proposal-policy loss with a batch-mean baseline."""

    if inputs.clip_epsilon < 0.0:
        raise ValueError("clip_epsilon must be nonnegative")
    new_logprob = inputs.new_logprob.reshape(-1)
    old_logprob = inputs.old_logprob.reshape(-1).detach()
    signal_values = inputs.training_signal.reshape(-1).detach()
    baseline = inputs.baseline.reshape(-1).detach()
    if not (
        new_logprob.shape == old_logprob.shape == signal_values.shape == baseline.shape
    ):
        raise ValueError("loss tensors must have matching flat shapes")
    for name, tensor in {
        "new_logprob": new_logprob,
        "old_logprob": old_logprob,
        "training_signal": signal_values,
        "baseline": baseline,
    }.items():
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be finite")

    advantage = signal_values - baseline
    if inputs.normalize_advantage and advantage.numel() > 1:
        std = advantage.std(unbiased=False)
        if float(std.item()) > 1e-8:
            advantage = (advantage - advantage.mean()) / std

    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped = torch.clamp(
        ratio,
        1.0 - inputs.clip_epsilon,
        1.0 + inputs.clip_epsilon,
    ) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = float((old_logprob - new_logprob.detach()).mean().abs().item())
    clip_fraction = float(
        (torch.abs(ratio.detach() - 1.0) > inputs.clip_epsilon).float().mean().item()
    )
    return PolicyGradientLossResult(
        loss=loss,
        advantage_mean=float(advantage.mean().item()),
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        ratio_mean=float(ratio.detach().mean().item()),
    )
