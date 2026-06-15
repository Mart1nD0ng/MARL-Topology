"""Supervised loss functions for Stage 10 dry-run and later warm starts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from marl_topology.models.centralized_mlp_critic import CentralizedCriticTensorOutput

from .supervised_batching import SupervisedLearningBatch


STAGE10_SUPERVISED_LOSS_COMPONENTS = (
    "actor_edge_bce",
    "actor_edge_ranking",
    "actor_edge_delta_diagnostic",
    "critic_value_mse",
    "critic_feasibility_bce",
    "critic_consensus_mse",
    "critic_latency_mse",
    "critic_energy_mse",
    "critic_edge_delta_add_mse",
    "critic_edge_delta_remove_mse",
    "critic_edge_delta_keep_mse",
)


class SupervisedLossViolation(ValueError):
    """Raised when supervised loss inputs are incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class SupervisedLossReport:
    components: dict[str, float]
    total_loss: float
    actor_logit_shape: tuple[int, ...]
    critic_batch_size: int
    critic_edge_count: int
    backward_called: bool = False
    checkpoint_written: bool = False
    artifact_written: bool = False
    update_rule_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "components": dict(self.components),
            "total_loss": self.total_loss,
            "actor_logit_shape": list(self.actor_logit_shape),
            "critic_batch_size": self.critic_batch_size,
            "critic_edge_count": self.critic_edge_count,
            "backward_called": self.backward_called,
            "update_rule_used": self.update_rule_used,
            "checkpoint_written": self.checkpoint_written,
            "artifact_written": self.artifact_written,
        }


def compute_supervised_loss_report(
    *,
    actor_logits: torch.Tensor,
    critic_output: CentralizedCriticTensorOutput,
    batch: SupervisedLearningBatch,
) -> SupervisedLossReport:
    components = {}
    components.update(actor_loss_components(actor_logits=actor_logits, batch=batch))
    components.update(critic_loss_components(critic_output=critic_output, batch=batch))
    total = sum(components.values())
    if not torch.isfinite(torch.tensor(total)).item():
        raise SupervisedLossViolation("total supervised loss is not finite")
    return SupervisedLossReport(
        components={name: float(value) for name, value in components.items()},
        total_loss=float(total),
        actor_logit_shape=tuple(actor_logits.shape),
        critic_batch_size=batch.critic_batch.batch_size,
        critic_edge_count=batch.critic_batch.edge_count,
    )


def actor_loss_components(
    *,
    actor_logits: torch.Tensor,
    batch: SupervisedLearningBatch,
) -> dict[str, float]:
    logits = actor_logits.reshape(-1)
    targets = batch.actor_edge_targets.reshape(-1)
    if logits.shape != targets.shape:
        raise SupervisedLossViolation("actor logits and targets must align")
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    ranking = _pairwise_ranking_loss(logits, targets)
    delta = F.mse_loss(torch.sigmoid(logits), batch.actor_edge_delta_targets.reshape(-1))
    return {
        "actor_edge_bce": _finite_loss_value("actor_edge_bce", bce),
        "actor_edge_ranking": _finite_loss_value("actor_edge_ranking", ranking),
        "actor_edge_delta_diagnostic": _finite_loss_value(
            "actor_edge_delta_diagnostic",
            delta,
        ),
    }


def critic_loss_components(
    *,
    critic_output: CentralizedCriticTensorOutput,
    batch: SupervisedLearningBatch,
) -> dict[str, float]:
    target = batch.critic_targets
    critic_output.assert_shapes(
        batch_size=batch.critic_batch.batch_size,
        edge_count=batch.critic_batch.edge_count,
    )
    components = {
        "critic_value_mse": F.mse_loss(critic_output.value, target.value),
        "critic_feasibility_bce": F.binary_cross_entropy(
            critic_output.feasibility.clamp(1e-6, 1.0 - 1e-6),
            target.feasibility,
        ),
        "critic_consensus_mse": F.mse_loss(
            critic_output.consensus_success_probability,
            target.consensus_success_probability,
        ),
        "critic_latency_mse": F.mse_loss(critic_output.latency, target.latency),
        "critic_energy_mse": F.mse_loss(critic_output.energy, target.energy),
        "critic_edge_delta_add_mse": _masked_mse(
            critic_output.edge_delta_add,
            target.edge_delta_add,
            target.edge_mask,
        ),
        "critic_edge_delta_remove_mse": _masked_mse(
            critic_output.edge_delta_remove,
            target.edge_delta_remove,
            target.edge_mask,
        ),
        "critic_edge_delta_keep_mse": _masked_mse(
            critic_output.edge_delta_keep,
            target.edge_delta_keep,
            target.edge_mask,
        ),
    }
    return {name: _finite_loss_value(name, value) for name, value in components.items()}


def _pairwise_ranking_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positives = logits[targets > 0.5]
    negatives = logits[targets <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    margins = 1.0 - (positives.reshape(-1, 1) - negatives.reshape(1, -1))
    return torch.relu(margins).mean()


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise SupervisedLossViolation("masked MSE tensor shapes must align")
    active = mask.to(dtype=torch.bool)
    if not active.any().item():
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return F.mse_loss(prediction[active], target[active])


def _finite_loss_value(name: str, value: torch.Tensor) -> float:
    if value.ndim != 0:
        raise SupervisedLossViolation(f"{name} must be a scalar")
    if not torch.isfinite(value).item():
        raise SupervisedLossViolation(f"{name} is not finite")
    return float(value.detach().cpu().item())
