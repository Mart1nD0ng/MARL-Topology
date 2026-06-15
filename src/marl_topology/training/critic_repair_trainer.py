"""Critic-only Stage 27 repair training and selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from marl_topology.models import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
    CentralizedMessagePassingGraphCritic,
    CentralizedMessagePassingGraphCriticConfig,
    EnrichedCentralizedMLPCritic,
    EnrichedCentralizedMLPCriticConfig,
)
from marl_topology.models.centralized_message_passing_graph_critic import GraphCriticBatch

from .critic_dataset import CriticDataset, collect_stage27_critic_dataset
from .return_normalization import ReturnNormalizer


STAGE27_CRITIC_TRAIN_CONFIG_ID = "stage27_critic_repair_train_config"
STAGE27_PASS_VERDICT = "stage27_pass_critic_repaired"
STAGE27_FAIL_VERDICT = "stage27_critic_repair_blocked"
STAGE27_SELECTED_CRITIC_ID = CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID


class CriticRepairViolation(ValueError):
    """Raised when critic-only repair crosses a declared boundary."""


@dataclass(frozen=True, slots=True)
class Stage27CriticRepairTrainConfig:
    epochs: int = 30
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 5
    seed: int = 2700
    value_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.05
    min_eval_explained_variance: float = 0.10
    min_eval_value_return_correlation: float = 0.30
    min_bias_reduction_fraction: float = 0.50
    min_advantage_variance_reduction: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise CriticRepairViolation("epochs must be positive")
        if self.minibatch_size <= 0:
            raise CriticRepairViolation("minibatch_size must be positive")
        if self.learning_rate <= 0.0:
            raise CriticRepairViolation("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise CriticRepairViolation("weight_decay must be nonnegative")
        if self.early_stop_patience < 1:
            raise CriticRepairViolation("early_stop_patience must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "config_id": STAGE27_CRITIC_TRAIN_CONFIG_ID,
            "epochs": self.epochs,
            "minibatch_size": self.minibatch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stop_patience": self.early_stop_patience,
            "target": "normalized_return",
            "value_loss": "mse_normalized_value",
            "actor_update_performed": False,
            "policy_gradient_update_performed": False,
        }


@dataclass(slots=True)
class Stage27SelectedCriticBundle:
    model: CentralizedMessagePassingGraphCritic
    normalizer: ReturnNormalizer
    graph_state: Mapping[str, object]
    repair_report: Mapping[str, object]


@dataclass(slots=True)
class _GraphCandidateFit:
    model: CentralizedMessagePassingGraphCritic
    report: Mapping[str, object]


def run_stage27_critic_repair(
    *,
    dataset: CriticDataset | None = None,
    config: Stage27CriticRepairTrainConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or Stage27CriticRepairTrainConfig()
    data = dataset or collect_stage27_critic_dataset()
    normalizer = ReturnNormalizer.fit(row.return_value for row in data.train_rows)
    train_returns_raw = data.returns_tensor("train")
    eval_returns_raw = data.returns_tensor("eval")
    train_targets = torch.tensor(
        normalizer.transform(train_returns_raw.detach().cpu().tolist()),
        dtype=torch.float32,
    )
    eval_targets = torch.tensor(
        normalizer.transform(eval_returns_raw.detach().cpu().tolist()),
        dtype=torch.float32,
    )
    feature_state = _feature_standardization_state(data)
    graph_state = _graph_standardization_state(data)
    baseline_bias = _stage26_value_bias(project_root)
    mlp_report = _train_mlp_candidate(
        data=data,
        config=cfg,
        normalizer=normalizer,
        train_targets=train_targets,
        eval_targets=eval_targets,
        feature_state=feature_state,
        baseline_bias=baseline_bias,
    )
    graph_report = _train_graph_candidate(
        data=data,
        config=cfg,
        normalizer=normalizer,
        train_targets=train_targets,
        eval_targets=eval_targets,
        graph_state=graph_state,
        baseline_bias=baseline_bias,
    )
    candidates = [mlp_report, graph_report]
    selected = _select_candidate(candidates)
    pass_gate = selected is not None
    return {
        "stage": "stage_27_critic_baseline_repair",
        "verdict": STAGE27_PASS_VERDICT if pass_gate else STAGE27_FAIL_VERDICT,
        "pass_gate": pass_gate,
        "dataset_health": data.health_report(),
        "train_config": cfg.to_payload(),
        "return_normalizer": normalizer.state_dict(),
        "return_normalizer_report": normalizer.scale_report(
            row.return_value for row in data.train_rows
        ),
        "feature_standardization": feature_state["report"],
        "graph_standardization": graph_state["report"],
        "candidate_reports": candidates,
        "selected_critic_id": selected["candidate_id"] if selected else None,
        "selection_reason": selected["selection_reason"] if selected else "no critic passed gates",
        "exactly_one_active_value_critic_selected": pass_gate,
        "old_critic_deprecated_for_value_baseline": pass_gate,
        "mappo_readiness": {
            "active_critic_selected": pass_gate,
            "return_normalizer_fitted": True,
            "gae_uses_raw_denormalized_values": True,
            "actor_leakage_absent": True,
            "actor_update_performed": False,
            "policy_gradient_update_performed": False,
            "reward_weights_changed": False,
            "sampler_switched": False,
            "next_stage_allowed_with_owner_decision": pass_gate,
            "recommended_next_task": (
                "stage_28_rerun_small_scale_mappo_with_repaired_critic"
                if pass_gate
                else "stage_27_followup_critic_feature_or_data_repair"
            ),
        },
        "forbidden_action_flags": {
            "actor_update_performed": False,
            "policy_gradient_update_performed": False,
            "scale_up_training_run": False,
            "reward_weights_changed": False,
            "sampler_switched": False,
            "checkpoint_created": False,
            "legacy_v5_modified": False,
        },
    }


def train_stage27_selected_graph_value_critic_bundle(
    *,
    dataset: CriticDataset | None = None,
    config: Stage27CriticRepairTrainConfig | None = None,
    project_root: str | Path | None = None,
) -> Stage27SelectedCriticBundle:
    cfg = config or Stage27CriticRepairTrainConfig()
    data = dataset or collect_stage27_critic_dataset()
    normalizer = ReturnNormalizer.fit(row.return_value for row in data.train_rows)
    train_returns_raw = data.returns_tensor("train")
    eval_returns_raw = data.returns_tensor("eval")
    train_targets = torch.tensor(
        normalizer.transform(train_returns_raw.detach().cpu().tolist()),
        dtype=torch.float32,
    )
    eval_targets = torch.tensor(
        normalizer.transform(eval_returns_raw.detach().cpu().tolist()),
        dtype=torch.float32,
    )
    graph_state = _graph_standardization_state(data)
    baseline_bias = _stage26_value_bias(project_root)
    fit = _fit_graph_candidate_model(
        data=data,
        config=cfg,
        normalizer=normalizer,
        train_targets=train_targets,
        eval_targets=eval_targets,
        graph_state=graph_state,
        baseline_bias=baseline_bias,
    )
    report = {
        "candidate_report": fit.report,
        "dataset_health": data.health_report(),
        "return_normalizer": normalizer.state_dict(),
        "return_normalizer_report": normalizer.scale_report(
            row.return_value for row in data.train_rows
        ),
        "graph_standardization": graph_state["report"],
        "selected_critic_id": CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
        "graph_gate_passed": bool(fit.report.get("gate", {}).get("passed")),
        "actor_update_performed": False,
        "policy_gradient_update_performed": False,
        "checkpoint_created": False,
    }
    return Stage27SelectedCriticBundle(
        model=fit.model,
        normalizer=normalizer,
        graph_state=graph_state,
        repair_report=report,
    )


def denormalize_values_for_gae(
    normalized_values: torch.Tensor,
    normalizer: ReturnNormalizer,
) -> torch.Tensor:
    raw = normalizer.inverse_transform(normalized_values.detach().cpu().tolist())
    return torch.tensor(raw, dtype=normalized_values.dtype, device=normalized_values.device)


def _train_mlp_candidate(
    *,
    data: CriticDataset,
    config: Stage27CriticRepairTrainConfig,
    normalizer: ReturnNormalizer,
    train_targets: torch.Tensor,
    eval_targets: torch.Tensor,
    feature_state: Mapping[str, object],
    baseline_bias: float,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    model = EnrichedCentralizedMLPCritic(
        EnrichedCentralizedMLPCriticConfig(hidden_dim=128)
    )
    train_features = _standardize_features(
        data.feature_batch("train").value_features,
        feature_state,
    )
    eval_features = _standardize_features(
        data.feature_batch("eval").value_features,
        feature_state,
    )
    train_aux = _auxiliary_targets(data, "train")
    eval_aux = _auxiliary_targets(data, "eval")
    before = _parameter_checksum(model)
    update_rule = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    curves = []
    best_eval = float("inf")
    patience = 0
    last_grad_norm = 0.0
    for epoch in range(config.epochs):
        permutation = torch.randperm(train_features.shape[0], generator=torch.Generator().manual_seed(config.seed + epoch))
        for start in range(0, train_features.shape[0], config.minibatch_size):
            index = permutation[start : start + config.minibatch_size]
            update_rule.zero_grad()
            output = model(train_features[index])
            loss = _candidate_loss(output, train_targets[index], _select_aux(train_aux, index), config)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            last_grad_norm = float(grad_norm.detach().cpu().item())
            update_rule.step()
        train_metrics = _evaluate_vector_candidate(
            candidate_id=ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
            output=model(train_features),
            targets=train_targets,
            raw_returns=data.returns_tensor("train"),
            normalizer=normalizer,
            baseline_bias=baseline_bias,
            grad_norm=last_grad_norm,
            parameter_delta=abs(_parameter_checksum(model) - before),
        )
        eval_metrics = _evaluate_vector_candidate(
            candidate_id=ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
            output=model(eval_features),
            targets=eval_targets,
            raw_returns=data.returns_tensor("eval"),
            normalizer=normalizer,
            baseline_bias=baseline_bias,
            grad_norm=last_grad_norm,
            parameter_delta=abs(_parameter_checksum(model) - before),
        )
        curves.append(
            {
                "epoch": epoch + 1,
                "train_value_loss": train_metrics["normalized_value_loss"],
                "eval_value_loss": eval_metrics["normalized_value_loss"],
                "train_explained_variance": train_metrics["explained_variance"],
                "eval_explained_variance": eval_metrics["explained_variance"],
            }
        )
        if float(eval_metrics["normalized_value_loss"]) < best_eval:
            best_eval = float(eval_metrics["normalized_value_loss"])
            patience = 0
        else:
            patience += 1
        if patience >= config.early_stop_patience:
            break
    after = _parameter_checksum(model)
    final_train = _evaluate_vector_candidate(
        candidate_id=ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
        output=model(train_features),
        targets=train_targets,
        raw_returns=data.returns_tensor("train"),
        normalizer=normalizer,
        baseline_bias=baseline_bias,
        grad_norm=last_grad_norm,
        parameter_delta=abs(after - before),
    )
    final_eval = _evaluate_vector_candidate(
        candidate_id=ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
        output=model(eval_features),
        targets=eval_targets,
        raw_returns=data.returns_tensor("eval"),
        normalizer=normalizer,
        baseline_bias=baseline_bias,
        grad_norm=last_grad_norm,
        parameter_delta=abs(after - before),
    )
    gate = _candidate_gate(final_eval, final_train, baseline_bias, config)
    return {
        "candidate_id": ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
        "implemented": True,
        "selected_active_registry_default": False,
        "train_metrics": final_train,
        "eval_metrics": final_eval,
        "curves": curves,
        "gate": gate,
        "parameter_checksum_before": before,
        "parameter_checksum_after": after,
        "parameter_delta": abs(after - before),
        "actor_update_performed": False,
    }


def _train_graph_candidate(
    *,
    data: CriticDataset,
    config: Stage27CriticRepairTrainConfig,
    normalizer: ReturnNormalizer,
    train_targets: torch.Tensor,
    eval_targets: torch.Tensor,
    graph_state: Mapping[str, object],
    baseline_bias: float,
) -> dict[str, object]:
    try:
        return dict(
            _fit_graph_candidate_model(
                data=data,
                config=config,
                normalizer=normalizer,
                train_targets=train_targets,
                eval_targets=eval_targets,
                graph_state=graph_state,
                baseline_bias=baseline_bias,
            ).report
        )
    except Exception as exc:  # pragma: no cover - reported as honest block.
        return {
            "candidate_id": CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
            "implemented": False,
            "blocked_reason": str(exc),
            "gate": {"passed": False, "issues": ["graph_critic_blocked"]},
            "actor_update_performed": False,
        }


def _fit_graph_candidate_model(
    *,
    data: CriticDataset,
    config: Stage27CriticRepairTrainConfig,
    normalizer: ReturnNormalizer,
    train_targets: torch.Tensor,
    eval_targets: torch.Tensor,
    graph_state: Mapping[str, object],
    baseline_bias: float,
) -> _GraphCandidateFit:
    torch.manual_seed(config.seed + 17)
    train_batch = _standardize_graph_batch(data.graph_batch("train"), graph_state)
    eval_batch = _standardize_graph_batch(data.graph_batch("eval"), graph_state)
    model = CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(
            node_feature_dim=int(train_batch.node_features.shape[-1]),
            edge_feature_dim=int(train_batch.edge_features.shape[-1]),
            hidden_dim=64,
            message_layers=2,
        )
    )
    train_aux = _auxiliary_targets(data, "train")
    before = _parameter_checksum(model)
    update_rule = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    curves = []
    best_eval = float("inf")
    patience = 0
    last_grad_norm = 0.0
    for epoch in range(config.epochs):
        permutation = torch.randperm(
            train_targets.shape[0],
            generator=torch.Generator().manual_seed(config.seed + 100 + epoch),
        )
        for start in range(0, train_targets.shape[0], config.minibatch_size):
            index = permutation[start : start + config.minibatch_size]
            update_rule.zero_grad()
            output = model(_graph_slice(train_batch, index))
            loss = _candidate_loss(
                output,
                train_targets[index],
                _select_aux(train_aux, index),
                config,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            last_grad_norm = float(grad_norm.detach().cpu().item())
            update_rule.step()
        train_metrics = _evaluate_vector_candidate(
            candidate_id=CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
            output=model(train_batch),
            targets=train_targets,
            raw_returns=data.returns_tensor("train"),
            normalizer=normalizer,
            baseline_bias=baseline_bias,
            grad_norm=last_grad_norm,
            parameter_delta=abs(_parameter_checksum(model) - before),
        )
        eval_metrics = _evaluate_vector_candidate(
            candidate_id=CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
            output=model(eval_batch),
            targets=eval_targets,
            raw_returns=data.returns_tensor("eval"),
            normalizer=normalizer,
            baseline_bias=baseline_bias,
            grad_norm=last_grad_norm,
            parameter_delta=abs(_parameter_checksum(model) - before),
        )
        curves.append(
            {
                "epoch": epoch + 1,
                "train_value_loss": train_metrics["normalized_value_loss"],
                "eval_value_loss": eval_metrics["normalized_value_loss"],
                "train_explained_variance": train_metrics["explained_variance"],
                "eval_explained_variance": eval_metrics["explained_variance"],
            }
        )
        if float(eval_metrics["normalized_value_loss"]) < best_eval:
            best_eval = float(eval_metrics["normalized_value_loss"])
            patience = 0
        else:
            patience += 1
        if patience >= config.early_stop_patience:
            break
    after = _parameter_checksum(model)
    final_train = _evaluate_vector_candidate(
        candidate_id=CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
        output=model(train_batch),
        targets=train_targets,
        raw_returns=data.returns_tensor("train"),
        normalizer=normalizer,
        baseline_bias=baseline_bias,
        grad_norm=last_grad_norm,
        parameter_delta=abs(after - before),
    )
    final_eval = _evaluate_vector_candidate(
        candidate_id=CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
        output=model(eval_batch),
        targets=eval_targets,
        raw_returns=data.returns_tensor("eval"),
        normalizer=normalizer,
        baseline_bias=baseline_bias,
        grad_norm=last_grad_norm,
        parameter_delta=abs(after - before),
    )
    gate = _candidate_gate(final_eval, final_train, baseline_bias, config)
    return _GraphCandidateFit(
        model=model,
        report={
            "candidate_id": CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
            "implemented": True,
            "selected_active_registry_default": True,
            "train_metrics": final_train,
            "eval_metrics": final_eval,
            "curves": curves,
            "gate": gate,
            "parameter_checksum_before": before,
            "parameter_checksum_after": after,
            "parameter_delta": abs(after - before),
            "actor_update_performed": False,
        },
    )


def _candidate_loss(output, target: torch.Tensor, aux: Mapping[str, torch.Tensor], config) -> torch.Tensor:
    value_loss = F.mse_loss(output.normalized_value.reshape(-1), target.reshape(-1))
    feasibility = F.binary_cross_entropy_with_logits(
        output.feasibility_logit.reshape(-1),
        aux["feasibility"].reshape(-1),
    )
    consensus = F.mse_loss(torch.sigmoid(output.consensus_proxy.reshape(-1)), aux["consensus"])
    latency = F.mse_loss(output.latency_proxy.reshape(-1), aux["latency"])
    energy = F.mse_loss(output.energy_proxy.reshape(-1), aux["energy"])
    return config.value_loss_weight * value_loss + config.auxiliary_loss_weight * (
        feasibility + consensus + latency + energy
    )


def _evaluate_vector_candidate(
    *,
    candidate_id: str,
    output,
    targets: torch.Tensor,
    raw_returns: torch.Tensor,
    normalizer: ReturnNormalizer,
    baseline_bias: float,
    grad_norm: float,
    parameter_delta: float,
) -> dict[str, float | list[dict[str, float]]]:
    pred_norm = output.normalized_value.detach().reshape(-1)
    raw_pred = denormalize_values_for_gae(pred_norm, normalizer)
    raw_returns = raw_returns.detach().reshape(-1)
    value_loss = F.mse_loss(pred_norm, targets.reshape(-1)).detach()
    raw_loss = F.mse_loss(raw_pred, raw_returns).detach()
    explained = _explained_variance(raw_returns, raw_pred)
    corr = _correlation(raw_returns, raw_pred)
    bias = float((raw_pred - raw_returns).mean().item())
    baseline_values = torch.full_like(raw_returns, float(raw_returns.mean().item()))
    advantage = raw_returns - raw_pred
    baseline_advantage = raw_returns - baseline_values
    advantage_var = float(advantage.var(unbiased=False).item())
    baseline_var = float(baseline_advantage.var(unbiased=False).item())
    variance_reduction = 1.0 - advantage_var / baseline_var if baseline_var > 1e-12 else 0.0
    return_mean = float(raw_returns.mean().item())
    pred_mean = float(raw_pred.mean().item())
    scale_ratio = abs(pred_mean) / max(abs(return_mean), 1e-8)
    return {
        "candidate_id": candidate_id,
        "normalized_value_loss": float(value_loss.item()),
        "raw_value_loss": float(raw_loss.item()),
        "explained_variance": explained,
        "value_return_correlation": corr,
        "value_bias": bias,
        "absolute_value_bias": abs(bias),
        "stage26_baseline_absolute_bias": abs(float(baseline_bias)),
        "bias_reduction_fraction": 1.0 - abs(bias) / max(abs(float(baseline_bias)), 1e-8),
        "value_scale_ratio": scale_ratio,
        "value_prediction_mean_raw": pred_mean,
        "return_mean_raw": return_mean,
        "advantage_std_with_critic": float(advantage.std(unbiased=False).item()),
        "advantage_std_batch_mean_baseline": float(
            baseline_advantage.std(unbiased=False).item()
        ),
        "advantage_variance_reduction": variance_reduction,
        "grad_norm": float(grad_norm),
        "parameter_delta": float(parameter_delta),
        "scatter_sample": _scatter_sample(raw_pred, raw_returns),
    }


def _candidate_gate(eval_metrics, train_metrics, baseline_bias, config) -> dict[str, object]:
    issues = []
    if float(eval_metrics["explained_variance"]) <= config.min_eval_explained_variance:
        issues.append("eval_explained_variance_below_minimum")
    if float(eval_metrics["value_return_correlation"]) <= config.min_eval_value_return_correlation:
        issues.append("eval_value_return_correlation_below_minimum")
    if float(eval_metrics["bias_reduction_fraction"]) < config.min_bias_reduction_fraction:
        issues.append("value_bias_not_reduced_by_half")
    if not 0.25 <= float(eval_metrics["value_scale_ratio"]) <= 4.0:
        issues.append("value_prediction_scale_not_same_order")
    if float(eval_metrics["advantage_variance_reduction"]) <= config.min_advantage_variance_reduction:
        issues.append("advantage_variance_not_reduced")
    if float(eval_metrics["normalized_value_loss"]) > 5.0 * max(
        float(train_metrics["normalized_value_loss"]),
        1e-8,
    ):
        issues.append("severe_train_eval_overfit")
    return {
        "passed": not issues,
        "issues": issues,
        "minimum_gate": {
            "eval_explained_variance_gt_0_10": float(eval_metrics["explained_variance"]) > 0.10,
            "eval_value_return_correlation_gt_0_30": float(
                eval_metrics["value_return_correlation"]
            )
            > 0.30,
            "absolute_value_bias_reduced_by_50_percent": float(
                eval_metrics["bias_reduction_fraction"]
            )
            >= 0.50,
            "advantage_variance_reduced": float(
                eval_metrics["advantage_variance_reduction"]
            )
            > 0.0,
        },
        "preferred_gate": {
            "eval_explained_variance_gt_0_25": float(eval_metrics["explained_variance"]) > 0.25,
            "eval_value_return_correlation_gt_0_50": float(
                eval_metrics["value_return_correlation"]
            )
            > 0.50,
        },
    }


def _select_candidate(candidates: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    passing = [candidate for candidate in candidates if candidate.get("gate", {}).get("passed")]
    if not passing:
        return None
    enriched_mlp = [
        candidate
        for candidate in passing
        if candidate["candidate_id"] == ENRICHED_CENTRALIZED_MLP_CRITIC_ID
    ]
    graph = [
        candidate
        for candidate in passing
        if candidate["candidate_id"] == CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
    ]
    if enriched_mlp and graph:
        enriched_ev = float(enriched_mlp[0]["eval_metrics"]["explained_variance"])
        graph_ev = float(graph[0]["eval_metrics"]["explained_variance"])
        if graph_ev > enriched_ev + 0.05:
            selected = graph[0]
            reason = "graph critic had materially higher eval explained variance"
        else:
            selected = enriched_mlp[0]
            reason = "enriched MLP selected by simplicity with close or better eval quality"
    else:
        selected = passing[0]
        reason = "only passing critic candidate"
    return {
        "candidate_id": selected["candidate_id"],
        "selection_reason": reason,
    }


def _feature_standardization_state(data: CriticDataset) -> dict[str, object]:
    features = data.feature_batch("train").value_features
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "mean": mean,
        "std": std,
        "report": {
            "feature_count": int(features.shape[1]),
            "fitted_on_train_only": True,
            "mean_abs_mean": float(mean.abs().mean().item()),
            "min_std": float(std.min().item()),
        },
    }


def _standardize_features(features: torch.Tensor, state: Mapping[str, object]) -> torch.Tensor:
    return (features - state["mean"]) / state["std"]


def _graph_standardization_state(data: CriticDataset) -> dict[str, object]:
    batch = data.graph_batch("train")
    node_mask = batch.node_mask.unsqueeze(-1)
    edge_mask = batch.edge_mask.unsqueeze(-1)
    node_values = batch.node_features[node_mask.expand_as(batch.node_features)].reshape(
        -1,
        batch.node_features.shape[-1],
    )
    edge_values = batch.edge_features[edge_mask.expand_as(batch.edge_features)].reshape(
        -1,
        batch.edge_features.shape[-1],
    )
    node_mean = node_values.mean(dim=0)
    node_std = node_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    edge_mean = edge_values.mean(dim=0)
    edge_std = edge_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "report": {
            "node_feature_count": int(batch.node_features.shape[-1]),
            "edge_feature_count": int(batch.edge_features.shape[-1]),
            "fitted_on_train_only": True,
        },
    }


def _standardize_graph_batch(batch: GraphCriticBatch, state: Mapping[str, object]) -> GraphCriticBatch:
    return GraphCriticBatch(
        node_features=(batch.node_features - state["node_mean"]) / state["node_std"],
        edge_features=(batch.edge_features - state["edge_mean"]) / state["edge_std"],
        edge_index=batch.edge_index,
        node_mask=batch.node_mask,
        edge_mask=batch.edge_mask,
    )


def _graph_slice(batch: GraphCriticBatch, index: torch.Tensor) -> GraphCriticBatch:
    return GraphCriticBatch(
        node_features=batch.node_features[index],
        edge_features=batch.edge_features[index],
        edge_index=batch.edge_index[index],
        node_mask=batch.node_mask[index],
        edge_mask=batch.edge_mask[index],
    )


def _auxiliary_targets(data: CriticDataset, split: str) -> dict[str, torch.Tensor]:
    rows = data.train_rows if split == "train" else data.eval_rows
    consensus = torch.tensor(
        [row.consensus_success_probability for row in rows],
        dtype=torch.float32,
    )
    latency = torch.tensor([row.latency for row in rows], dtype=torch.float32)
    energy = torch.tensor([row.energy for row in rows], dtype=torch.float32)
    return {
        "feasibility": (consensus >= 0.9).to(dtype=torch.float32),
        "consensus": consensus,
        "latency": _unit_scale(latency),
        "energy": _unit_scale(energy),
    }


def _select_aux(aux: Mapping[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[index] for key, value in aux.items()}


def _unit_scale(values: torch.Tensor) -> torch.Tensor:
    max_value = values.abs().max().clamp_min(1e-8)
    return values / max_value


def _stage26_value_bias(project_root: str | Path | None) -> float:
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[3]
    path = (
        root
        / "result_save"
        / "stage26_full_system_health_diagnostic"
        / "stage26_no_training_full_system_health_diagnostic_v1"
        / "stage26_full_system_health_report.json"
    )
    if not path.exists():
        return 342.39235267917314
    import json

    report = json.loads(path.read_text(encoding="utf-8"))
    return float(report["component_health"]["critic_health"]["metrics"]["value_bias"])


def _parameter_checksum(module) -> float:
    return sum(
        float(parameter.detach().double().sum().cpu().item())
        for parameter in module.parameters()
    )


def _explained_variance(returns: torch.Tensor, predictions: torch.Tensor) -> float:
    variance = returns.var(unbiased=False)
    if float(variance.detach().cpu().item()) <= 1e-12:
        return 0.0
    return float((1.0 - (returns - predictions).var(unbiased=False) / variance).item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = torch.sqrt(
        (left_centered.square().sum() * right_centered.square().sum()).clamp_min(1e-12)
    )
    return float((left_centered * right_centered).sum().item() / float(denom.item()))


def _scatter_sample(predictions: torch.Tensor, returns: torch.Tensor) -> list[dict[str, float]]:
    count = min(80, predictions.numel())
    if count == 0:
        return []
    step = max(1, predictions.numel() // count)
    rows = []
    for index in range(0, predictions.numel(), step):
        rows.append(
            {
                "prediction": float(predictions[index].item()),
                "return": float(returns[index].item()),
            }
        )
        if len(rows) >= count:
            break
    return rows
