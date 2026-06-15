"""Stage 21 supervised MLP/GNN rerun on assembler-aware targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_DATASET_ID,
    build_stage21_objective_stack_evidence_dataset,
    build_stage21_objective_stack_evidence_report,
)
from marl_topology.models import (
    LocalGNNEdgeScorer,
    LocalGNNEdgeScorerConfig,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
)
from marl_topology.training.stage19_supervised_actor_stack import (
    STAGE19_FEATURE_FIELDS,
    Stage19ActorBatch,
    build_stage19_actor_batch,
    build_stage19_pair_indices,
    _edge_logits,
    _split_indices,
    _train_edge_model,
)

STAGE21_SUPERVISED_RERUN_ID = "stage21_supervised_mlp_gnn_on_assembler_aware_targets"


class Stage21SupervisedRerunViolation(ValueError):
    """Raised when Stage 21 supervised rerun crosses its boundary."""


@dataclass(frozen=True, slots=True)
class Stage21SupervisedRerunConfig:
    seed: int = 21
    epochs: int = 100
    learning_rate: float = 0.01
    train_fraction: float = 0.75
    ranking_loss_weight: float = 0.25

    @property
    def stage19_compatible_config(self):
        from marl_topology.training.stage19_supervised_actor_stack import (
            Stage19SupervisedActorStackConfig,
        )

        return Stage19SupervisedActorStackConfig(
            seed=self.seed,
            epochs=self.epochs,
            temporal_epochs=1,
            learning_rate=self.learning_rate,
            train_fraction=self.train_fraction,
            ranking_loss_weight=self.ranking_loss_weight,
        )

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise Stage21SupervisedRerunViolation("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise Stage21SupervisedRerunViolation("learning_rate must be positive")
        if not 0.0 < self.train_fraction < 1.0:
            raise Stage21SupervisedRerunViolation("train_fraction must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class Stage21TrainedActorRerun:
    dataset: object
    batch: Stage19ActorBatch
    ranking_pairs: tuple[tuple[int, int, float, float], ...]
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    mlp_model: LocalMLPEdgeScorer
    gnn_model: LocalGNNEdgeScorer
    report: Mapping[str, object]


def train_stage21_supervised_actor_models(
    *,
    config: Stage21SupervisedRerunConfig | None = None,
) -> Stage21TrainedActorRerun:
    """Train MLP and GNN actor scorers on Stage 21 evidence in memory."""

    cfg = config or Stage21SupervisedRerunConfig()
    torch.manual_seed(cfg.seed)
    evidence_build = build_stage21_objective_stack_evidence_report()
    dataset = evidence_build.dataset
    if not bool(evidence_build.report["stage21_evidence_ready_for_supervised_rerun"]):
        raise Stage21SupervisedRerunViolation("Stage 21 evidence readiness gate failed")

    batch = build_stage19_actor_batch(dataset)  # structural Stage18-compatible rows
    ranking_pairs = build_stage19_pair_indices(dataset, batch)
    if not ranking_pairs:
        raise Stage21SupervisedRerunViolation("Stage 21 requires pairwise ranking targets")
    train_indices, validation_indices = _split_indices(len(batch.samples), cfg.train_fraction)
    stage19_cfg = cfg.stage19_compatible_config

    mlp = LocalMLPEdgeScorer(
        LocalMLPEdgeScorerConfig(
            model_id="stage21_local_mlp_on_assembler_aware_targets",
            input_dim=len(STAGE19_FEATURE_FIELDS),
        )
    )
    mlp_result = _train_edge_model(
        mlp,
        batch=batch,
        ranking_pairs=ranking_pairs,
        train_indices=train_indices,
        validation_indices=validation_indices,
        config=stage19_cfg,
        model_label="MLP",
    )
    mlp_logits = _edge_logits(mlp, batch).detach()
    mlp_result = {
        **mlp_result,
        "utility_prediction_calibration": _utility_prediction_calibration(
            mlp_logits,
            batch,
            validation_indices,
        ),
        "accepted_vs_rejected_edge_discrimination": _accepted_rejected_discrimination(
            mlp_logits,
            batch,
            validation_indices,
        ),
    }

    gnn = LocalGNNEdgeScorer(
        LocalGNNEdgeScorerConfig(
            model_id="stage21_local_gnn_on_assembler_aware_targets",
            input_dim=len(STAGE19_FEATURE_FIELDS),
        )
    )
    gnn_result = _train_edge_model(
        gnn,
        batch=batch,
        ranking_pairs=ranking_pairs,
        train_indices=train_indices,
        validation_indices=validation_indices,
        config=stage19_cfg,
        model_label="GNN",
    )
    gnn_logits = _edge_logits(gnn, batch).detach()
    gnn_result = {
        **gnn_result,
        "utility_prediction_calibration": _utility_prediction_calibration(
            gnn_logits,
            batch,
            validation_indices,
        ),
        "accepted_vs_rejected_edge_discrimination": _accepted_rejected_discrimination(
            gnn_logits,
            batch,
            validation_indices,
        ),
    }

    comparison = _comparison(mlp_result, gnn_result)
    report = {
        "stage": STAGE21_SUPERVISED_RERUN_ID,
        "source_dataset_id": STAGE21_DATASET_ID,
        "feature_schema_id": batch.feature_schema_id,
        "feature_count": len(STAGE19_FEATURE_FIELDS),
        "sample_count": len(batch.samples),
        "ranking_pair_count": len(ranking_pairs),
        "target_distribution_summary": evidence_build.report["target_distribution"],
        "mlp": mlp_result,
        "gnn": gnn_result,
        "comparison": comparison,
        "models_rerun": ["MLP", "GNN"],
        "gru_lstm_rerun": False,
        "actor_input_uses_stage21_actor_safe_view": True,
        "no_actor_leakage": bool(
            evidence_build.report["actor_safe_view_has_no_forbidden_fields"]
            and evidence_build.report["actor_target_view_has_no_global_delta_fields"]
        ),
        "critic_global_targets_actor_input": False,
        "reward_weight_tuning_performed": False,
        "final_tau_selected": False,
        "policy_gradient_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "coma_allowed": False,
        "transformer_allowed": False,
        "v5_modified": False,
    }
    return Stage21TrainedActorRerun(
        dataset=dataset,
        batch=batch,
        ranking_pairs=ranking_pairs,
        train_indices=train_indices,
        validation_indices=validation_indices,
        mlp_model=mlp,
        gnn_model=gnn,
        report=report,
    )


def run_stage21_supervised_actor_rerun(
    *,
    config: Stage21SupervisedRerunConfig | None = None,
) -> dict[str, object]:
    return dict(train_stage21_supervised_actor_models(config=config).report)


def _utility_prediction_calibration(
    logits: torch.Tensor,
    batch: Stage19ActorBatch,
    indices: torch.Tensor,
) -> dict[str, object]:
    probabilities = torch.sigmoid(logits[indices])
    targets = batch.targets[indices]
    errors = torch.abs(probabilities - targets)
    return {
        "mean_absolute_error": float(errors.mean().item()) if errors.numel() else 0.0,
        "max_absolute_error": float(errors.max().item()) if errors.numel() else 0.0,
        "prediction_mean": float(probabilities.mean().item()) if probabilities.numel() else 0.0,
        "target_mean": float(targets.mean().item()) if targets.numel() else 0.0,
    }


def _accepted_rejected_discrimination(
    logits: torch.Tensor,
    batch: Stage19ActorBatch,
    indices: torch.Tensor,
) -> dict[str, object]:
    active = set(int(index) for index in indices.tolist())
    high_indices = [
        index
        for index in active
        if float(batch.targets[index].item()) >= 0.7
    ]
    low_indices = [
        index
        for index in active
        if float(batch.targets[index].item()) < 0.35
    ]
    probabilities = torch.sigmoid(logits)
    high_mean = _mean_tensor(probabilities[high_indices]) if high_indices else 0.0
    low_mean = _mean_tensor(probabilities[low_indices]) if low_indices else 0.0
    return {
        "high_priority_validation_count": len(high_indices),
        "low_priority_validation_count": len(low_indices),
        "high_priority_mean_prediction": high_mean,
        "low_priority_mean_prediction": low_mean,
        "accepted_rejected_score_gap": high_mean - low_mean,
    }


def _comparison(
    mlp: Mapping[str, object],
    gnn: Mapping[str, object],
) -> dict[str, object]:
    best = min((mlp, gnn), key=lambda item: float(item["validation_loss"]))
    return {
        "validation_loss_delta_gnn_minus_mlp": float(gnn["validation_loss"])
        - float(mlp["validation_loss"]),
        "pairwise_accuracy_delta_gnn_minus_mlp": float(gnn["pairwise_accuracy"])
        - float(mlp["pairwise_accuracy"]),
        "recommended_starting_actor": str(best["model_label"]),
        "recommended_starting_actor_reason": (
            "Lowest validation loss on Stage 21 assembler-aware soft/ranking targets."
        ),
    }


def _mean_tensor(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item())
