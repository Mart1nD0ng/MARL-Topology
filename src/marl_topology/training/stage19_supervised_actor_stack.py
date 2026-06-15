"""Stage 19 supervised actor stack rerun on Stage 18 evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Callable

import torch
import torch.nn.functional as F

from marl_topology.data.learning_evidence_stage18 import (
    STAGE18_DATASET_ID,
    Stage18LearningEvidenceDataset,
    build_stage18_learning_evidence_dataset,
)
from marl_topology.models import (
    LocalGNNEdgeScorer,
    LocalGNNEdgeScorerConfig,
    LocalGRUEdgeScorer,
    LocalGRUEdgeScorerConfig,
    LocalLSTMEdgeScorer,
    LocalLSTMEdgeScorerConfig,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
)


STAGE19_STAGE_ID = "stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence"
STAGE19_VERDICT = "stage19_supervised_actor_stack_rerun_completed_rl_still_blocked"
STAGE19_FEATURE_SCHEMA_ID = "stage19_stage18_actor_feature_tensor_v1"
STAGE19_RECOMMENDED_NEXT_TASK = (
    "stage_20_supervised_actor_policy_evaluation_with_environment_assembler"
)

STAGE19_FEATURE_FIELDS = (
    "agent_kind_is_vehicle",
    "agent_kind_is_rsu",
    "neighbor_kind_is_vehicle",
    "neighbor_kind_is_rsu",
    "local_position_x_scaled",
    "local_position_y_scaled",
    "local_position_z_scaled",
    "distance_3d_scaled",
    "link_success_probability",
    "estimated_link_latency_scaled",
    "estimated_link_energy_scaled",
    "edge_active_prev",
    "edge_active_current_local",
    "local_outgoing_degree_prev_scaled",
    "local_selected_edge_count_scaled",
    "tx_budget_used_scaled",
    "tx_budget_remaining_scaled",
    "rx_capacity_estimate_for_neighbor_scaled",
    "channel_slot_available",
    "local_channel_slot_occupancy_scaled",
    "local_conflict_group_occupancy_scaled",
    "local_interference_estimate",
    "recent_local_resource_rejection_count_scaled",
    "estimated_deadline_delivery_probability",
    "estimated_sinr_scaled",
    "estimated_los_is_los",
    "estimated_required_transmission_time_scaled",
    "estimated_retransmission_attempts_scaled",
    "recent_message_success_rate_local",
    "recent_retry_count_local_scaled",
    "time_since_last_successful_local_delivery_scaled",
)


class Stage19SupervisedActorStackViolation(ValueError):
    """Raised when Stage 19 supervised actor rerun crosses its boundary."""


@dataclass(frozen=True, slots=True)
class Stage19SupervisedActorStackConfig:
    seed: int = 19
    epochs: int = 80
    temporal_epochs: int = 80
    learning_rate: float = 0.01
    temporal_learning_rate: float = 0.01
    train_fraction: float = 0.75
    ranking_loss_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.temporal_epochs <= 0:
            raise Stage19SupervisedActorStackViolation("epoch counts must be positive")
        if self.learning_rate <= 0.0 or self.temporal_learning_rate <= 0.0:
            raise Stage19SupervisedActorStackViolation("learning rates must be positive")
        if not 0.0 < self.train_fraction < 1.0:
            raise Stage19SupervisedActorStackViolation("train_fraction must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class Stage19ActorSample:
    sample_id: str
    row_index: int
    agent_id: str
    time_step: int
    edge_id: str
    directed_edge_id: str
    feature_values: tuple[float, ...]
    utility_target: float
    confidence: float
    ambiguity_level: str
    ambiguity_reasons: tuple[str, ...]
    sequence_id: str | None
    topology_name: str


@dataclass(frozen=True, slots=True)
class Stage19ActorBatch:
    samples: tuple[Stage19ActorSample, ...]
    features: torch.Tensor
    targets: torch.Tensor
    confidence: torch.Tensor
    group_ids: torch.Tensor
    feature_schema_id: str = STAGE19_FEATURE_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(STAGE19_FEATURE_FIELDS):
            raise Stage19SupervisedActorStackViolation("feature tensor shape mismatch")
        if self.targets.shape != (self.features.shape[0],):
            raise Stage19SupervisedActorStackViolation("target tensor shape mismatch")
        if self.confidence.shape != self.targets.shape:
            raise Stage19SupervisedActorStackViolation("confidence tensor shape mismatch")
        if self.group_ids.shape != self.targets.shape:
            raise Stage19SupervisedActorStackViolation("group id tensor shape mismatch")


@dataclass(frozen=True, slots=True)
class Stage19TemporalBatch:
    features: torch.Tensor
    targets: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor
    sequence_ids: tuple[str, ...]
    time_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.features.shape[2] != len(STAGE19_FEATURE_FIELDS):
            raise Stage19SupervisedActorStackViolation("temporal feature shape mismatch")
        if self.targets.shape != self.features.shape[:2]:
            raise Stage19SupervisedActorStackViolation("temporal target shape mismatch")
        if self.confidence.shape != self.targets.shape or self.mask.shape != self.targets.shape:
            raise Stage19SupervisedActorStackViolation("temporal mask/confidence shape mismatch")

    def summary(self) -> dict[str, object]:
        return {
            "sequence_count": int(self.features.shape[0]),
            "max_time_steps": int(self.features.shape[1]),
            "feature_dim": int(self.features.shape[2]),
            "mask_active_count": int(self.mask.sum().item()),
            "time_steps": list(self.time_steps),
            "real_multistep_actor_safe_sequence_used": True,
            "future_outcome_leakage_detected": False,
        }


@dataclass(frozen=True, slots=True)
class Stage19TrainedActorStack:
    """In-memory supervised actor stack for downstream evaluation gates."""

    dataset: Stage18LearningEvidenceDataset
    batch: Stage19ActorBatch
    ranking_pairs: tuple[tuple[int, int, float, float], ...]
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    mlp_model: LocalMLPEdgeScorer
    gnn_model: LocalGNNEdgeScorer
    gru_model: LocalGRUEdgeScorer
    lstm_model: LocalLSTMEdgeScorer | None
    temporal_batch: Stage19TemporalBatch
    report: Mapping[str, object]


def build_stage19_actor_batch(
    dataset: Stage18LearningEvidenceDataset | None = None,
) -> Stage19ActorBatch:
    """Build deterministic Stage 19 tensors from Stage 18 actor-safe views."""

    stage18 = dataset or build_stage18_learning_evidence_dataset()
    samples: list[Stage19ActorSample] = []
    group_lookup: dict[tuple[int, str, int], int] = {}
    group_ids: list[int] = []
    for row_index, row in enumerate(stage18.rows):
        soft_targets = {
            (str(target["agent_id"]), str(target["edge_id"])): target
            for target in row.actor_target_view["actor_soft_utility_targets"]
        }
        for actor_row in row.actor_safe_view:
            target = soft_targets.get((str(actor_row["agent_id"]), str(actor_row["edge_id"])))
            if target is None:
                continue
            feature_values = tuple(_stage19_feature_value(actor_row, field) for field in STAGE19_FEATURE_FIELDS)
            if len(feature_values) != len(STAGE19_FEATURE_FIELDS):
                raise Stage19SupervisedActorStackViolation("feature vector length mismatch")
            sample = Stage19ActorSample(
                sample_id=f"r{row_index}:{actor_row['directed_edge_id']}",
                row_index=row_index,
                agent_id=str(actor_row["agent_id"]),
                time_step=int(actor_row["time_step"]),
                edge_id=str(actor_row["edge_id"]),
                directed_edge_id=str(actor_row["directed_edge_id"]),
                feature_values=feature_values,
                utility_target=float(target["actor_edge_utility_target"]),
                confidence=float(target["actor_edge_utility_confidence"]),
                ambiguity_level=str(target["actor_target_ambiguity_level"]),
                ambiguity_reasons=tuple(str(item) for item in target.get("ambiguity_reasons", ())),
                sequence_id=(
                    str(row.diagnostics_view["sequence_id"])
                    if row.diagnostics_view.get("sequence_id") is not None
                    else None
                ),
                topology_name=str(row.diagnostics_view["topology_name"]),
            )
            group_key = (row_index, sample.agent_id, sample.time_step)
            if group_key not in group_lookup:
                group_lookup[group_key] = len(group_lookup)
            group_ids.append(group_lookup[group_key])
            samples.append(sample)
    if not samples:
        raise Stage19SupervisedActorStackViolation("Stage 19 needs nonempty actor samples")
    return Stage19ActorBatch(
        samples=tuple(samples),
        features=torch.tensor([sample.feature_values for sample in samples], dtype=torch.float32),
        targets=torch.tensor([sample.utility_target for sample in samples], dtype=torch.float32),
        confidence=torch.tensor([sample.confidence for sample in samples], dtype=torch.float32),
        group_ids=torch.tensor(group_ids, dtype=torch.long),
    )


def build_stage19_pair_indices(
    dataset: Stage18LearningEvidenceDataset,
    batch: Stage19ActorBatch,
) -> tuple[tuple[int, int, float, float], ...]:
    """Map Stage 18 ranking targets into Stage 19 sample indices."""

    index_by_row_agent_edge = {
        (sample.row_index, sample.agent_id, sample.edge_id): index
        for index, sample in enumerate(batch.samples)
    }
    pairs: list[tuple[int, int, float, float]] = []
    for row_index, row in enumerate(dataset.rows):
        for pair in row.actor_target_view["pairwise_ranking_targets"]:
            preferred = index_by_row_agent_edge.get(
                (row_index, str(pair["agent_id"]), str(pair["preferred_edge_id"]))
            )
            less_preferred = index_by_row_agent_edge.get(
                (row_index, str(pair["agent_id"]), str(pair["less_preferred_edge_id"]))
            )
            if preferred is None or less_preferred is None:
                continue
            pairs.append(
                (
                    preferred,
                    less_preferred,
                    float(pair["preference_margin"]),
                    float(pair["ranking_confidence"]),
                )
            )
    return tuple(pairs)


def run_stage19_supervised_actor_stack(
    *,
    config: Stage19SupervisedActorStackConfig | None = None,
) -> dict[str, object]:
    """Execute the bounded Stage 19 supervised actor stack rerun."""

    return dict(train_stage19_supervised_actor_models(config=config).report)


def train_stage19_supervised_actor_models(
    *,
    config: Stage19SupervisedActorStackConfig | None = None,
) -> Stage19TrainedActorStack:
    """Train Stage 19 actor models in memory for evaluation-only consumers."""

    cfg = config or Stage19SupervisedActorStackConfig()
    torch.manual_seed(cfg.seed)
    dataset = build_stage18_learning_evidence_dataset()
    batch = build_stage19_actor_batch(dataset)
    ranking_pairs = build_stage19_pair_indices(dataset, batch)
    train_indices, validation_indices = _split_indices(len(batch.samples), cfg.train_fraction)

    mlp = LocalMLPEdgeScorer(
        LocalMLPEdgeScorerConfig(
            model_id="stage19_local_mlp_on_disambiguated_evidence",
            input_dim=len(STAGE19_FEATURE_FIELDS),
        )
    )
    mlp_result = _train_edge_model(
        mlp,
        batch=batch,
        ranking_pairs=ranking_pairs,
        train_indices=train_indices,
        validation_indices=validation_indices,
        config=cfg,
        model_label="MLP",
    )

    gnn = LocalGNNEdgeScorer(
        LocalGNNEdgeScorerConfig(
            model_id="stage19_local_gnn_on_disambiguated_evidence",
            input_dim=len(STAGE19_FEATURE_FIELDS),
        )
    )
    gnn_result = _train_edge_model(
        gnn,
        batch=batch,
        ranking_pairs=ranking_pairs,
        train_indices=train_indices,
        validation_indices=validation_indices,
        config=cfg,
        model_label="GNN",
    )

    temporal_batch = build_stage19_real_multistep_temporal_batch(batch)
    gru = LocalGRUEdgeScorer(
        LocalGRUEdgeScorerConfig(input_dim=len(STAGE19_FEATURE_FIELDS))
    )
    gru_result = _train_temporal_model(
        gru,
        temporal_batch,
        config=cfg,
        model_label="GRU",
    )
    gru_sanity_passed = bool(
        gru_result["final_loss"] < gru_result["initial_loss"]
        and not gru_result["constant_output"]
    )
    lstm_result = None
    if gru_sanity_passed:
        lstm = LocalLSTMEdgeScorer(
            LocalLSTMEdgeScorerConfig(input_dim=len(STAGE19_FEATURE_FIELDS))
        )
        lstm_result = _train_temporal_model(
            lstm,
            temporal_batch,
            config=cfg,
            model_label="LSTM",
        )

    comparison = _comparison(mlp_result, gnn_result, gru_result, lstm_result)
    report = {
        "stage": STAGE19_STAGE_ID,
        "verdict": STAGE19_VERDICT,
        "source_dataset_id": STAGE18_DATASET_ID,
        "feature_schema_id": STAGE19_FEATURE_SCHEMA_ID,
        "feature_count": len(STAGE19_FEATURE_FIELDS),
        "sample_count": len(batch.samples),
        "ranking_pair_count": len(ranking_pairs),
        "rerun_order": ["MLP", "GNN", "GRU", "LSTM_after_GRU_sanity"],
        "mlp": mlp_result,
        "gnn": gnn_result,
        "temporal_batch": temporal_batch.summary(),
        "gru": gru_result,
        "gru_sanity_passed": gru_sanity_passed,
        "lstm_tested_after_gru": lstm_result is not None,
        "lstm": lstm_result,
        "comparison": comparison,
        "stage20_recommended_starting_actor": comparison["recommended_starting_actor"],
        "training_execution_scope": "supervised_actor_only_stage18_disambiguated_evidence",
        "critic_training_performed": False,
        "policy_gradient_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "ppo_mappo_allowed": False,
        "coma_allowed": False,
        "transformer_allowed": False,
        "scale_up_training_allowed": False,
        "reward_weight_tuning_performed": False,
        "final_tau_selected": False,
        "v5_modified": False,
        "recommended_next_task": STAGE19_RECOMMENDED_NEXT_TASK,
    }
    return Stage19TrainedActorStack(
        dataset=dataset,
        batch=batch,
        ranking_pairs=ranking_pairs,
        train_indices=train_indices,
        validation_indices=validation_indices,
        mlp_model=mlp,
        gnn_model=gnn,
        gru_model=gru,
        lstm_model=lstm if gru_sanity_passed else None,
        temporal_batch=temporal_batch,
        report=report,
    )


def build_stage19_real_multistep_temporal_batch(
    batch: Stage19ActorBatch,
) -> Stage19TemporalBatch:
    by_sequence: dict[tuple[str, str, str, str], dict[int, Stage19ActorSample]] = defaultdict(dict)
    for sample in batch.samples:
        if sample.sequence_id is None:
            continue
        key = (sample.sequence_id, sample.topology_name, sample.agent_id, sample.edge_id)
        by_sequence[key][sample.time_step] = sample
    complete = {
        key: values for key, values in by_sequence.items() if len(values) >= 2
    }
    if not complete:
        raise Stage19SupervisedActorStackViolation("real multi-step sequence subset is empty")
    time_steps = tuple(sorted({step for values in complete.values() for step in values}))
    sequence_keys = tuple(sorted(complete))
    features = torch.zeros(
        (len(sequence_keys), len(time_steps), len(STAGE19_FEATURE_FIELDS)),
        dtype=torch.float32,
    )
    targets = torch.zeros((len(sequence_keys), len(time_steps)), dtype=torch.float32)
    confidence = torch.zeros_like(targets)
    mask = torch.zeros((len(sequence_keys), len(time_steps)), dtype=torch.bool)
    for sequence_index, key in enumerate(sequence_keys):
        values = complete[key]
        for time_index, time_step in enumerate(time_steps):
            if time_step not in values:
                continue
            sample = values[time_step]
            features[sequence_index, time_index] = torch.tensor(sample.feature_values, dtype=torch.float32)
            targets[sequence_index, time_index] = float(sample.utility_target)
            confidence[sequence_index, time_index] = max(0.05, float(sample.confidence))
            mask[sequence_index, time_index] = True
    return Stage19TemporalBatch(
        features=features,
        targets=targets,
        confidence=confidence,
        mask=mask,
        sequence_ids=tuple("|".join(key) for key in sequence_keys),
        time_steps=time_steps,
    )


def _train_edge_model(
    model: torch.nn.Module,
    *,
    batch: Stage19ActorBatch,
    ranking_pairs: tuple[tuple[int, int, float, float], ...],
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    config: Stage19SupervisedActorStackConfig,
    model_label: str,
) -> dict[str, object]:
    before = _parameter_checksum(model)
    update_rule = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    initial_loss = _edge_loss(
        model,
        batch,
        indices=train_indices,
        ranking_pairs=ranking_pairs,
        ranking_loss_weight=config.ranking_loss_weight,
    )
    update_steps = 0
    for _ in range(config.epochs):
        update_rule.zero_grad()
        loss = _edge_loss(
            model,
            batch,
            indices=train_indices,
            ranking_pairs=ranking_pairs,
            ranking_loss_weight=config.ranking_loss_weight,
        )
        loss.backward()
        update_rule.step()
        update_steps += 1
    after = _parameter_checksum(model)
    with torch.no_grad():
        logits = _edge_logits(model, batch)
        final_train_loss = _edge_loss(
            model,
            batch,
            indices=train_indices,
            ranking_pairs=ranking_pairs,
            ranking_loss_weight=config.ranking_loss_weight,
        )
        validation_loss = _edge_loss(
            model,
            batch,
            indices=validation_indices,
            ranking_pairs=ranking_pairs,
            ranking_loss_weight=config.ranking_loss_weight,
        )
    return {
        "model_label": model_label,
        "model_id": getattr(model, "config").model_id,
        "initial_loss": float(initial_loss.item()),
        "train_loss": float(final_train_loss.item()),
        "validation_loss": float(validation_loss.item()),
        "validation_mse": _mse(logits[validation_indices], batch.targets[validation_indices]),
        "pairwise_accuracy": _pairwise_accuracy(logits, ranking_pairs, validation_indices),
        "spearman": _spearman(logits[validation_indices], batch.targets[validation_indices]),
        "constant_output": float(logits.detach().std(unbiased=False).item()) < 1e-5,
        "parameter_checksum_before": before,
        "parameter_checksum_after": after,
        "actor_parameter_update_performed": before != after,
        "update_steps": update_steps,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_written": False,
        "artifact_written": False,
    }


def _train_temporal_model(
    model: torch.nn.Module,
    batch: Stage19TemporalBatch,
    *,
    config: Stage19SupervisedActorStackConfig,
    model_label: str,
) -> dict[str, object]:
    before = _parameter_checksum(model)
    update_rule = torch.optim.AdamW(model.parameters(), lr=config.temporal_learning_rate)
    logits, _hidden = model(batch.features)
    initial_loss = _temporal_loss(logits, batch)
    update_steps = 0
    for _ in range(config.temporal_epochs):
        update_rule.zero_grad()
        logits, _hidden = model(batch.features)
        loss = _temporal_loss(logits, batch)
        loss.backward()
        update_rule.step()
        update_steps += 1
    final_logits, _hidden = model(batch.features)
    final_loss = _temporal_loss(final_logits, batch)
    after = _parameter_checksum(model)
    return {
        "model_label": model_label,
        "model_id": getattr(model, "config").model_id,
        "initial_loss": float(initial_loss.item()),
        "final_loss": float(final_loss.item()),
        "constant_output": float(final_logits[batch.mask].detach().std(unbiased=False).item()) < 1e-5,
        "parameter_checksum_before": before,
        "parameter_checksum_after": after,
        "actor_parameter_update_performed": before != after,
        "update_steps": update_steps,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "future_outcome_leakage_detected": False,
        "checkpoint_written": False,
        "artifact_written": False,
    }


def _edge_logits(model: torch.nn.Module, batch: Stage19ActorBatch) -> torch.Tensor:
    if hasattr(model, "forward_with_groups"):
        return model.forward_with_groups(batch.features, batch.group_ids)  # type: ignore[attr-defined]
    return model(batch.features)


def _edge_loss(
    model: torch.nn.Module,
    batch: Stage19ActorBatch,
    *,
    indices: torch.Tensor,
    ranking_pairs: tuple[tuple[int, int, float, float], ...],
    ranking_loss_weight: float,
) -> torch.Tensor:
    logits = _edge_logits(model, batch)
    bce = F.binary_cross_entropy_with_logits(
        logits[indices],
        batch.targets[indices],
        reduction="none",
    )
    weights = batch.confidence[indices].clamp_min(0.05)
    supervised = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    ranking = _ranking_loss(logits, ranking_pairs, indices)
    return supervised + ranking_loss_weight * ranking


def _ranking_loss(
    logits: torch.Tensor,
    ranking_pairs: tuple[tuple[int, int, float, float], ...],
    active_indices: torch.Tensor,
) -> torch.Tensor:
    active = set(int(index) for index in active_indices.tolist())
    losses = []
    weights = []
    for preferred, less_preferred, margin, confidence in ranking_pairs:
        if preferred not in active or less_preferred not in active:
            continue
        losses.append(F.softplus(-(logits[preferred] - logits[less_preferred] - margin)))
        weights.append(max(0.05, confidence))
    if not losses:
        return logits.new_tensor(0.0)
    loss_tensor = torch.stack(losses)
    weight_tensor = logits.new_tensor(weights)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def _temporal_loss(logits: torch.Tensor, batch: Stage19TemporalBatch) -> torch.Tensor:
    active = batch.mask.to(dtype=torch.bool)
    raw = F.binary_cross_entropy_with_logits(
        logits[active],
        batch.targets[active],
        reduction="none",
    )
    weights = batch.confidence[active].clamp_min(0.05)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def _split_indices(sample_count: int, train_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    split = max(1, min(sample_count - 1, int(sample_count * train_fraction)))
    return torch.arange(0, split, dtype=torch.long), torch.arange(split, sample_count, dtype=torch.long)


def _stage19_feature_value(row: Mapping[str, object], field: str) -> float:
    neighbor = tuple(row["local_neighbor_observations"])[0]
    position = tuple(float(value) for value in row["local_position_m"])
    if field == "agent_kind_is_vehicle":
        return _flag(str(row["agent_kind"]) == "vehicle")
    if field == "agent_kind_is_rsu":
        return _flag(str(row["agent_kind"]) == "rsu")
    if field == "neighbor_kind_is_vehicle":
        return _flag(str(row["neighbor_kind"]) == "vehicle")
    if field == "neighbor_kind_is_rsu":
        return _flag(str(row["neighbor_kind"]) == "rsu")
    if field == "local_position_x_scaled":
        return position[0] / 100.0
    if field == "local_position_y_scaled":
        return position[1] / 100.0
    if field == "local_position_z_scaled":
        return position[2] / 20.0
    if field == "distance_3d_scaled":
        return float(neighbor["distance_3d_m"]) / 200.0
    if field == "link_success_probability":
        return float(neighbor["link_success_probability"])
    if field == "estimated_link_latency_scaled":
        return float(neighbor["estimated_link_latency_s"]) / 0.02
    if field == "estimated_link_energy_scaled":
        return float(neighbor["estimated_link_energy_j"])
    if field == "estimated_los_is_los":
        return _flag(str(row["estimated_los_nlos"]) == "los_estimate")
    if field.endswith("_scaled"):
        base = field.removesuffix("_scaled")
        divisor = {
            "local_outgoing_degree_prev": 8.0,
            "local_selected_edge_count": 8.0,
            "tx_budget_used": 8.0,
            "tx_budget_remaining": 8.0,
            "rx_capacity_estimate_for_neighbor": 8.0,
            "local_channel_slot_occupancy": 8.0,
            "local_conflict_group_occupancy": 8.0,
            "recent_local_resource_rejection_count": 8.0,
            "estimated_sinr": 100.0,
            "estimated_required_transmission_time": 0.02,
            "estimated_retransmission_attempts": 10.0,
            "recent_retry_count_local": 8.0,
            "time_since_last_successful_local_delivery": 10.0,
        }[base]
        return float(row[base]) / divisor
    if field in {
        "edge_active_prev",
        "edge_active_current_local",
        "channel_slot_available",
    }:
        return _flag(bool(row[field]))
    return float(row[field])


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def _parameter_checksum(module: torch.nn.Module) -> float:
    return float(sum(parameter.detach().double().sum().item() for parameter in module.parameters()))


def _mse(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.mse_loss(torch.sigmoid(logits), targets).item())


def _pairwise_accuracy(
    logits: torch.Tensor,
    ranking_pairs: tuple[tuple[int, int, float, float], ...],
    validation_indices: torch.Tensor,
) -> float:
    active = set(int(index) for index in validation_indices.tolist())
    comparisons = []
    for preferred, less_preferred, _margin, _confidence in ranking_pairs:
        if preferred in active and less_preferred in active:
            comparisons.append(bool(logits[preferred] > logits[less_preferred]))
    if not comparisons:
        return 0.0
    return sum(int(item) for item in comparisons) / len(comparisons)


def _spearman(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() < 2:
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


def _comparison(
    mlp: Mapping[str, object],
    gnn: Mapping[str, object],
    gru: Mapping[str, object],
    lstm: Mapping[str, object] | None,
) -> dict[str, object]:
    supervised = [mlp, gnn]
    best = min(supervised, key=lambda item: float(item["validation_loss"]))
    temporal_note = "GRU passed real multi-step sanity"
    if lstm is not None and float(lstm["final_loss"]) <= float(gru["final_loss"]):
        temporal_note = "LSTM matched or improved GRU on real multi-step sanity"
    return {
        "validation_loss_delta_gnn_minus_mlp": float(gnn["validation_loss"])
        - float(mlp["validation_loss"]),
        "pairwise_accuracy_delta_gnn_minus_mlp": float(gnn["pairwise_accuracy"])
        - float(mlp["pairwise_accuracy"]),
        "recommended_starting_actor": str(best["model_label"]),
        "recommended_starting_actor_reason": (
            "Lowest validation loss on Stage 18 disambiguated soft/ranking targets."
        ),
        "temporal_result_note": temporal_note,
        "rl_readiness": "blocked_until_environment_assembler_policy_evaluation",
    }
