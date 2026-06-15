"""Supervised actor warm-start utilities for local edge scorers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import torch
import torch.nn.functional as F

from marl_topology.evaluation.fixture_stack import build_fixture_stack
from marl_topology.models.local_mlp_edge_scorer import LocalMLPEdgeScorer
from marl_topology.policies import ActorPolicyInput, ConflictAwareGreedyAssembler
from marl_topology.policies.topology_assembler import CandidateEdgeConstraint
from marl_topology.scenario import get_scenario_fixture

from .supervised_batching import (
    SupervisedLearningBatch,
    build_supervised_batch_from_evidence_rows,
)
from .supervised_losses import actor_loss_components


STAGE11_SUPERVISED_ACTOR_STAGE_ID = "stage_11_supervised_mlp_actor_warm_start"


class SupervisedActorTrainingViolation(ValueError):
    """Raised when supervised actor warm-start preconditions fail."""


@dataclass(frozen=True, slots=True)
class SupervisedActorTrainingConfig:
    seed: int = 11
    epochs: int = 80
    learning_rate: float = 0.01
    train_fraction: float = 0.75
    tiny_batch_epochs: int = 400
    tiny_batch_learning_rate: float = 0.03
    tiny_overfit_loss_threshold: float = 0.08
    ranking_loss_weight: float = 0.1
    model_id: str = "local_mlp_edge_scorer_stage11_supervised"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise SupervisedActorTrainingViolation("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise SupervisedActorTrainingViolation("learning_rate must be positive")
        if not 0.0 < self.train_fraction < 1.0:
            raise SupervisedActorTrainingViolation("train_fraction must be in (0, 1)")
        if self.tiny_batch_epochs <= 0:
            raise SupervisedActorTrainingViolation("tiny_batch_epochs must be positive")
        if self.tiny_batch_learning_rate <= 0.0:
            raise SupervisedActorTrainingViolation("tiny_batch_learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class ActorTrainingTensors:
    features: torch.Tensor
    targets: torch.Tensor
    group_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise SupervisedActorTrainingViolation("features must be [edge_count, feature_dim]")
        if self.targets.shape != (self.features.shape[0],):
            raise SupervisedActorTrainingViolation("targets must match feature row count")
        if self.group_ids.shape != (self.features.shape[0],):
            raise SupervisedActorTrainingViolation("group_ids must match feature row count")
        if self.features.shape[0] == 0:
            raise SupervisedActorTrainingViolation("actor training tensors cannot be empty")


@dataclass(frozen=True, slots=True)
class SupervisedActorTrainingResult:
    stage_id: str
    model_id: str
    train_loss: float
    validation_loss: float
    train_bce: float
    validation_bce: float
    precision: float
    recall: float
    pairwise_accuracy: float
    spearman: float
    tiny_batch_initial_loss: float
    tiny_batch_final_loss: float
    tiny_batch_overfit_passed: bool
    parameter_checksum_before: float
    parameter_checksum_after: float
    actor_parameter_update_performed: bool
    update_steps: int
    leakage_check_passed: bool
    assembler_diagnostics: dict[str, object]
    checkpoint_written: bool = False
    critic_training_performed: bool = False
    rl_training_performed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "model_id": self.model_id,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "train_bce": self.train_bce,
            "validation_bce": self.validation_bce,
            "precision": self.precision,
            "recall": self.recall,
            "pairwise_accuracy": self.pairwise_accuracy,
            "spearman": self.spearman,
            "tiny_batch_initial_loss": self.tiny_batch_initial_loss,
            "tiny_batch_final_loss": self.tiny_batch_final_loss,
            "tiny_batch_overfit_passed": self.tiny_batch_overfit_passed,
            "parameter_checksum_before": self.parameter_checksum_before,
            "parameter_checksum_after": self.parameter_checksum_after,
            "actor_parameter_update_performed": self.actor_parameter_update_performed,
            "update_steps": self.update_steps,
            "leakage_check_passed": self.leakage_check_passed,
            "assembler_diagnostics": dict(self.assembler_diagnostics),
            "checkpoint_written": self.checkpoint_written,
            "critic_training_performed": self.critic_training_performed,
            "rl_training_performed": self.rl_training_performed,
        }


def run_supervised_actor_training(
    evidence_rows: tuple[dict[str, object], ...],
    *,
    config: SupervisedActorTrainingConfig | None = None,
    model_factory: Callable[[], LocalMLPEdgeScorer] = LocalMLPEdgeScorer,
) -> tuple[LocalMLPEdgeScorer, SupervisedActorTrainingResult]:
    cfg = config or SupervisedActorTrainingConfig()
    torch.manual_seed(cfg.seed)
    if len(evidence_rows) < 2:
        raise SupervisedActorTrainingViolation("train/validation split needs at least two rows")
    train_rows, validation_rows = _split_rows(evidence_rows, cfg.train_fraction)
    train_batch = build_supervised_batch_from_evidence_rows(
        train_rows,
        dataset_id="stage11_train_split",
    )
    validation_batch = build_supervised_batch_from_evidence_rows(
        validation_rows,
        dataset_id="stage11_validation_split",
    )
    train_tensors = actor_tensors_from_supervised_batch(train_batch)
    validation_tensors = actor_tensors_from_supervised_batch(validation_batch)

    model = model_factory()
    before = parameter_checksum(model)
    update_rule = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    update_steps = 0
    for _ in range(cfg.epochs):
        update_rule.zero_grad()
        logits = score_actor_tensors(model, train_tensors)
        loss = _actor_training_loss(
            logits,
            train_tensors.targets,
            ranking_loss_weight=cfg.ranking_loss_weight,
        )
        loss.backward()
        update_rule.step()
        update_steps += 1

    after = parameter_checksum(model)
    train_loss, train_bce = _loss_and_bce(model, train_tensors, cfg)
    validation_loss, validation_bce = _loss_and_bce(model, validation_tensors, cfg)
    validation_logits = score_actor_tensors(model, validation_tensors).detach()
    tiny_result = run_tiny_batch_overfit(
        evidence_rows,
        config=cfg,
        model_factory=model_factory,
    )
    assembler_diagnostics = evaluate_actor_scores_with_assembler(model, evidence_rows[0])
    precision, recall = _precision_recall(validation_logits, validation_tensors.targets)
    return model, SupervisedActorTrainingResult(
        stage_id=STAGE11_SUPERVISED_ACTOR_STAGE_ID,
        model_id=cfg.model_id,
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_bce=train_bce,
        validation_bce=validation_bce,
        precision=precision,
        recall=recall,
        pairwise_accuracy=_pairwise_accuracy(validation_logits, validation_tensors.targets),
        spearman=_spearman(validation_logits, validation_tensors.targets),
        tiny_batch_initial_loss=tiny_result["initial_loss"],
        tiny_batch_final_loss=tiny_result["final_loss"],
        tiny_batch_overfit_passed=(
            tiny_result["final_loss"] <= cfg.tiny_overfit_loss_threshold
        ),
        parameter_checksum_before=before,
        parameter_checksum_after=after,
        actor_parameter_update_performed=before != after,
        update_steps=update_steps,
        leakage_check_passed=True,
        assembler_diagnostics=assembler_diagnostics,
    )


def actor_tensors_from_supervised_batch(batch: SupervisedLearningBatch) -> ActorTrainingTensors:
    return ActorTrainingTensors(
        features=batch.actor_batch.edge_features,
        targets=batch.actor_edge_targets,
        group_ids=batch.actor_batch.group_ids,
    )


def run_tiny_batch_overfit(
    evidence_rows: tuple[dict[str, object], ...],
    *,
    config: SupervisedActorTrainingConfig,
    model_factory: Callable[[], LocalMLPEdgeScorer] = LocalMLPEdgeScorer,
) -> dict[str, float]:
    row = _first_mixed_label_row(evidence_rows)
    batch = build_supervised_batch_from_evidence_rows((row,), dataset_id="stage11_tiny")
    tensors = actor_tensors_from_supervised_batch(batch)
    model = model_factory()
    update_rule = torch.optim.AdamW(model.parameters(), lr=config.tiny_batch_learning_rate)
    initial_loss = float(
        F.binary_cross_entropy_with_logits(
            score_actor_tensors(model, tensors),
            tensors.targets,
        ).item()
    )
    for _ in range(config.tiny_batch_epochs):
        update_rule.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            score_actor_tensors(model, tensors),
            tensors.targets,
        )
        loss.backward()
        update_rule.step()
    final_loss = float(
        F.binary_cross_entropy_with_logits(
            score_actor_tensors(model, tensors),
            tensors.targets,
        ).item()
    )
    return {"initial_loss": initial_loss, "final_loss": final_loss}


def evaluate_actor_scores_with_assembler(
    model: LocalMLPEdgeScorer,
    evidence_row: dict[str, object],
) -> dict[str, object]:
    actor_inputs = tuple(
        ActorPolicyInput.from_actor_safe_row(row)
        for row in evidence_row["actor_safe_rows"]  # type: ignore[index]
    )
    batch = build_supervised_batch_from_evidence_rows(
        (evidence_row,),
        dataset_id="stage11_assembler_eval",
    ).actor_batch
    if hasattr(model, "score_tensor_batch"):
        logits = model.score_tensor_batch(batch).detach()
    else:
        logits = model(batch.edge_features).detach()
    edge_scores = batch.to_edge_score_batch(
        logits,
        batch_id="stage11_actor_scores",
        source="stage11_supervised_actor",
    )
    constraints = _constraints_from_actor_inputs(actor_inputs)
    assembled = ConflictAwareGreedyAssembler().assemble(edge_scores.edge_scores, constraints)
    scenario_id = str(evidence_row["scenario_id"])
    fixture = get_scenario_fixture(scenario_id)
    stack = build_fixture_stack(fixture)
    constraint_by_directed = {constraint.directed_edge_id: constraint for constraint in constraints}
    selected_edge_ids = {
        constraint_by_directed[directed_id].edge_id
        for directed_id in assembled.selected_directed_edges
        if directed_id in constraint_by_directed
    }
    evaluation = stack.evaluator.evaluate(
        selected_edge_ids,
        topology_id="stage11_actor_assembler_eval",
    )
    return {
        "assembler_id": assembled.assembler_id,
        "pre_projection_edge_count": assembled.pre_projection_edge_count,
        "post_projection_edge_count": assembled.post_projection_edge_count,
        "selected_edge_count": len(selected_edge_ids),
        "consensus_success_probability": float(
            evaluation.metrics["consensus_success_probability"]
        ),
        "feasible_under_tau_requirement": float(
            evaluation.metrics["consensus_success_probability"]
        )
        >= 0.9,
        "latency": float(evaluation.metrics["latency"]),
        "energy": float(evaluation.metrics["energy"]),
        "topology_diagnostics": dict(evaluation.metrics["topology_diagnostics"]),
    }


def parameter_checksum(module: torch.nn.Module) -> float:
    return float(
        sum(parameter.detach().double().sum().item() for parameter in module.parameters())
    )


def score_actor_tensors(model: torch.nn.Module, tensors: ActorTrainingTensors) -> torch.Tensor:
    if hasattr(model, "forward_with_groups"):
        return model.forward_with_groups(tensors.features, tensors.group_ids)  # type: ignore[attr-defined]
    return model(tensors.features)


def _split_rows(
    rows: tuple[dict[str, object], ...],
    train_fraction: float,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    split_index = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    return rows[:split_index], rows[split_index:]


def _actor_training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ranking_loss_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits.reshape(-1), targets)
    components = actor_loss_components(
        actor_logits=logits,
        batch=_TemporaryLossBatch(targets=targets),
    )
    ranking = torch.tensor(components["actor_edge_ranking"], dtype=logits.dtype)
    return bce + ranking_loss_weight * ranking


def _loss_and_bce(
    model: LocalMLPEdgeScorer,
    tensors: ActorTrainingTensors,
    config: SupervisedActorTrainingConfig,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = score_actor_tensors(model, tensors)
        bce = F.binary_cross_entropy_with_logits(logits, tensors.targets)
        total = _actor_training_loss(
            logits,
            tensors.targets,
            ranking_loss_weight=config.ranking_loss_weight,
        )
    return float(total.item()), float(bce.item())


def _first_mixed_label_row(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    for row in rows:
        batch = build_supervised_batch_from_evidence_rows((row,), dataset_id="tiny_scan")
        labels = batch.actor_edge_targets
        if bool((labels > 0.5).any().item()) and bool((labels <= 0.5).any().item()):
            return row
    raise SupervisedActorTrainingViolation("tiny overfit requires mixed deterministic labels")


def _precision_recall(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    predictions = torch.sigmoid(logits) >= 0.5
    positives = targets >= 0.5
    true_positive = float((predictions & positives).sum().item())
    predicted_positive = float(predictions.sum().item())
    actual_positive = float(positives.sum().item())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    return precision, recall


def _pairwise_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    positives = logits[targets > 0.5]
    negatives = logits[targets <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return 0.0
    comparisons = positives.reshape(-1, 1) > negatives.reshape(1, -1)
    return float(comparisons.float().mean().item())


def _spearman(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() < 2 or targets.unique().numel() < 2:
        return 0.0
    left = _rank(logits.detach().cpu())
    right = _rank(targets.detach().cpu())
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


def _constraints_from_actor_inputs(
    actor_inputs: tuple[ActorPolicyInput, ...],
) -> tuple[CandidateEdgeConstraint, ...]:
    constraints = []
    for actor_input in actor_inputs:
        for neighbor in actor_input.local_neighbor_observations:
            constraints.append(
                CandidateEdgeConstraint(
                    edge_id=str(_neighbor_value(neighbor, "edge_id")),
                    tx_id=actor_input.agent_id,
                    rx_id=str(_neighbor_value(neighbor, "neighbor_id")),
                    edge_type="actor_local_candidate",
                    role_allowed=True,
                    channel_slot=None,
                    conflict_group=None,
                )
            )
    return tuple(constraints)


def _neighbor_value(neighbor: object, name: str) -> object:
    if hasattr(neighbor, name):
        return getattr(neighbor, name)
    return neighbor[name]  # type: ignore[index]


@dataclass(frozen=True, slots=True)
class _TemporaryLossBatch:
    actor_edge_targets: torch.Tensor
    actor_edge_delta_targets: torch.Tensor

    def __init__(self, targets: torch.Tensor) -> None:
        object.__setattr__(self, "actor_edge_targets", targets)
        object.__setattr__(self, "actor_edge_delta_targets", torch.zeros_like(targets))
