"""Return and advantage helpers for the Stage 24 critic baseline path."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AdvantageConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True, slots=True)
class AdvantageResult:
    advantages: torch.Tensor
    returns: torch.Tensor
    raw_advantages: torch.Tensor
    value_baseline_used: bool
    normalized: bool

    def __post_init__(self) -> None:
        if self.advantages.shape != self.returns.shape:
            raise ValueError("advantages and returns must have matching shape")
        if self.advantages.shape != self.raw_advantages.shape:
            raise ValueError("raw_advantages shape must match advantages")
        for name in ("advantages", "returns", "raw_advantages"):
            if not torch.isfinite(getattr(self, name)).all().item():
                raise ValueError(f"{name} must be finite")

    def to_payload(self) -> dict[str, object]:
        return {
            "advantage_mean": float(self.advantages.mean().detach().cpu().item()),
            "advantage_std": float(self.advantages.std(unbiased=False).detach().cpu().item()),
            "return_mean": float(self.returns.mean().detach().cpu().item()),
            "value_baseline_used": self.value_baseline_used,
            "normalized": self.normalized,
        }


def compute_gae_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    *,
    num_scenarios: int,
    rollout_steps: int,
    config: AdvantageConfig | None = None,
) -> AdvantageResult:
    """Compute finite generalized advantages and returns."""

    cfg = config or AdvantageConfig()
    flat_count = num_scenarios * rollout_steps
    if rewards.reshape(-1).shape[0] != flat_count:
        raise ValueError("rewards do not match rollout dimensions")
    if values.reshape(-1).shape[0] != flat_count:
        raise ValueError("values do not match rollout dimensions")
    if masks.reshape(-1).shape[0] != flat_count:
        raise ValueError("masks do not match rollout dimensions")

    signal = rewards.reshape(num_scenarios, rollout_steps)
    baseline = values.reshape(num_scenarios, rollout_steps)
    active_masks = masks.reshape(num_scenarios, rollout_steps)
    raw_advantages = torch.zeros_like(signal)
    next_advantage = torch.zeros((num_scenarios,), dtype=signal.dtype, device=signal.device)
    for step_index in range(rollout_steps - 1, -1, -1):
        if step_index == rollout_steps - 1:
            next_value = torch.zeros((num_scenarios,), dtype=signal.dtype, device=signal.device)
        else:
            next_value = baseline[:, step_index + 1]
        step_mask = active_masks[:, step_index]
        delta = (
            signal[:, step_index]
            + cfg.gamma * next_value * step_mask
            - baseline[:, step_index]
        )
        next_advantage = delta + cfg.gamma * cfg.gae_lambda * step_mask * next_advantage
        raw_advantages[:, step_index] = next_advantage

    returns = raw_advantages + baseline
    advantages = raw_advantages
    normalized = False
    if cfg.normalize_advantages and advantages.numel() > 1:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        if torch.isfinite(std).item() and float(std.detach().cpu().item()) > cfg.epsilon:
            advantages = (advantages - mean) / (std + cfg.epsilon)
            normalized = True

    return AdvantageResult(
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        raw_advantages=raw_advantages.reshape(-1),
        value_baseline_used=True,
        normalized=normalized,
    )
