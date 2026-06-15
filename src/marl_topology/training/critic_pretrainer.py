"""Centralized critic and edge-delta pretraining utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F

from marl_topology.models import CentralizedMLPCriticBaseline, CentralizedMLPCriticConfig

from .supervised_batching import SupervisedLearningBatch


STAGE12_CRITIC_PRETRAINING_STAGE_ID = "stage_12_centralized_critic_edge_delta_pretraining"


class CriticPretrainingViolation(ValueError):
    """Raised when critic pretraining gates fail structurally."""


@dataclass(frozen=True, slots=True)
class CriticPretrainingConfig:
    seed: int = 12
    epochs: int = 220
    learning_rate: float = 0.01
    hidden_dim: int = 96
    collapse_std_threshold: float = 1e-4
    minimum_rank_signal: float = -0.05
    minimum_balanced_sign_accuracy: float = 0.35
    model_id: str = "centralized_mlp_critic_stage12_pretrained"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise CriticPretrainingViolation("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise CriticPretrainingViolation("learning_rate must be positive")
        if self.hidden_dim <= 0:
            raise CriticPretrainingViolation("hidden_dim must be positive")


@dataclass(frozen=True, slots=True)
class CriticPretrainingResult:
    stage_id: str
    model_id: str
    initial_loss: float
    final_loss: float
    update_steps: int
    parameter_checksum_before: float
    parameter_checksum_after: float
    critic_parameter_update_performed: bool
    actor_training_performed: bool
    checkpoint_written: bool
    fidelity_report: dict[str, object]
    fidelity_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "model_id": self.model_id,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "update_steps": self.update_steps,
            "parameter_checksum_before": self.parameter_checksum_before,
            "parameter_checksum_after": self.parameter_checksum_after,
            "critic_parameter_update_performed": self.critic_parameter_update_performed,
            "actor_training_performed": self.actor_training_performed,
            "checkpoint_written": self.checkpoint_written,
            "fidelity_report": dict(self.fidelity_report),
            "fidelity_gate_passed": self.fidelity_gate_passed,
        }


def pretrain_centralized_critic(
    batch: SupervisedLearningBatch,
    *,
    config: CriticPretrainingConfig | None = None,
) -> tuple[CentralizedMLPCriticBaseline, CriticPretrainingResult]:
    cfg = config or CriticPretrainingConfig()
    torch.manual_seed(cfg.seed)
    model = CentralizedMLPCriticBaseline(
        CentralizedMLPCriticConfig(
            hidden_dim=cfg.hidden_dim,
            edge_output_dim=batch.critic_batch.edge_count,
        )
    )
    before = parameter_checksum(model)
    update_rule = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    initial_loss = float(_critic_total_loss(model, batch).item())
    for _ in range(cfg.epochs):
        update_rule.zero_grad()
        loss = _critic_total_loss(model, batch)
        loss.backward()
        update_rule.step()
    final_loss = float(_critic_total_loss(model, batch).item())
    after = parameter_checksum(model)
    with torch.no_grad():
        output = model.predict_tensor_batch(batch.critic_batch)
    fidelity = build_critic_fidelity_report(output, batch, config=cfg)
    gate_passed = bool(fidelity["fidelity_gate_passed"])
    return model, CriticPretrainingResult(
        stage_id=STAGE12_CRITIC_PRETRAINING_STAGE_ID,
        model_id=cfg.model_id,
        initial_loss=initial_loss,
        final_loss=final_loss,
        update_steps=cfg.epochs,
        parameter_checksum_before=before,
        parameter_checksum_after=after,
        critic_parameter_update_performed=before != after,
        actor_training_performed=False,
        checkpoint_written=False,
        fidelity_report=fidelity,
        fidelity_gate_passed=gate_passed,
    )


def build_critic_fidelity_report(
    output,
    batch: SupervisedLearningBatch,
    *,
    config: CriticPretrainingConfig | None = None,
) -> dict[str, object]:
    cfg = config or CriticPretrainingConfig()
    target = batch.critic_targets
    mask = target.edge_mask
    add_pred = output.edge_delta_add[mask].detach()
    add_target = target.edge_delta_add[mask].detach()
    remove_pred = output.edge_delta_remove[mask].detach()
    remove_target = target.edge_delta_remove[mask].detach()
    consensus_error = _mae(
        output.consensus_success_probability,
        target.consensus_success_probability,
    )
    latency_error = _mae(output.latency, target.latency)
    energy_error = _mae(output.energy, target.energy)
    feasibility_accuracy = _binary_accuracy(output.feasibility, target.feasibility)
    add_rank = _spearman(add_pred, add_target)
    remove_rank = _spearman(remove_pred, remove_target)
    add_sign = _balanced_sign_accuracy(add_pred, add_target)
    remove_sign = _balanced_sign_accuracy(remove_pred, remove_target)
    collapsed = (
        float(output.consensus_success_probability.std(unbiased=False).item())
        <= cfg.collapse_std_threshold
        or float(output.edge_delta_add[mask].std(unbiased=False).item())
        <= cfg.collapse_std_threshold
    )
    helpful_precision = _positive_precision(add_pred, add_target)
    harmful_recall = _negative_recall(remove_pred, remove_target)
    gate_passed = (
        not collapsed
        and add_rank >= cfg.minimum_rank_signal
        and remove_rank >= cfg.minimum_rank_signal
        and add_sign >= cfg.minimum_balanced_sign_accuracy
        and remove_sign >= cfg.minimum_balanced_sign_accuracy
    )
    return {
        "add_edge_delta_ranking_spearman": add_rank,
        "remove_edge_delta_ranking_spearman": remove_rank,
        "balanced_sign_accuracy_add": add_sign,
        "balanced_sign_accuracy_remove": remove_sign,
        "harmful_edge_recall": harmful_recall,
        "helpful_edge_precision": helpful_precision,
        "feasibility_classification_accuracy": feasibility_accuracy,
        "consensus_regression_mae": consensus_error,
        "latency_regression_mae": latency_error,
        "energy_regression_mae": energy_error,
        "calibration_error_by_head": {
            "feasibility": _mae(output.feasibility, target.feasibility),
            "consensus_success_probability": consensus_error,
        },
        "rare_safety_sample_diagnostics": {
            "feasible_positive_count": int((target.feasibility > 0.5).sum().item()),
            "infeasible_count": int((target.feasibility <= 0.5).sum().item()),
        },
        "constant_or_collapsed_predictions": collapsed,
        "fidelity_gate_passed": gate_passed,
    }


def parameter_checksum(module: torch.nn.Module) -> float:
    return float(
        sum(parameter.detach().double().sum().item() for parameter in module.parameters())
    )


def _critic_total_loss(
    model: CentralizedMLPCriticBaseline,
    batch: SupervisedLearningBatch,
) -> torch.Tensor:
    output = model.predict_tensor_batch(batch.critic_batch)
    target = batch.critic_targets
    return (
        F.mse_loss(output.value, target.value)
        + F.binary_cross_entropy(
            output.feasibility.clamp(1e-6, 1.0 - 1e-6),
            target.feasibility,
        )
        + F.mse_loss(
            output.consensus_success_probability,
            target.consensus_success_probability,
        )
        + F.mse_loss(output.latency, target.latency)
        + F.mse_loss(output.energy, target.energy)
        + _masked_mse(output.edge_delta_add, target.edge_delta_add, target.edge_mask)
        + _masked_mse(output.edge_delta_remove, target.edge_delta_remove, target.edge_mask)
        + _masked_mse(output.edge_delta_keep, target.edge_delta_keep, target.edge_mask)
    )


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.to(dtype=torch.bool)
    if not active.any().item():
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return F.mse_loss(prediction[active], target[active])


def _mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(prediction.detach() - target.detach())).item())


def _binary_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction.detach() >= 0.5
    expected = target.detach() >= 0.5
    return float((predicted == expected).float().mean().item())


def _balanced_sign_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    positive = target > 0.0
    negative = target < 0.0
    pos_acc = (
        float((prediction[positive] > 0.0).float().mean().item())
        if positive.any().item()
        else 0.0
    )
    neg_acc = (
        float((prediction[negative] < 0.0).float().mean().item())
        if negative.any().item()
        else 0.0
    )
    if positive.any().item() and negative.any().item():
        return 0.5 * (pos_acc + neg_acc)
    return max(pos_acc, neg_acc)


def _positive_precision(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted_positive = prediction > 0.0
    if not predicted_positive.any().item():
        return 0.0
    return float((target[predicted_positive] > 0.0).float().mean().item())


def _negative_recall(prediction: torch.Tensor, target: torch.Tensor) -> float:
    actual_negative = target < 0.0
    if not actual_negative.any().item():
        return 0.0
    return float((prediction[actual_negative] < 0.0).float().mean().item())


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.numel() < 2 or target.std(unbiased=False).item() == 0.0:
        return 0.0
    left = _rank(prediction.detach().cpu())
    right = _rank(target.detach().cpu())
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = sqrt(float((left_centered**2).sum().item() * (right_centered**2).sum().item()))
    if denom == 0.0:
        return 0.0
    return float((left_centered * right_centered).sum().item() / denom)


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), dtype=torch.float32)
    return ranks
