"""Frozen-policy critic-only dataset collection for Stage 27."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite

import torch

from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_TAU_REQUIREMENT_MIN,
    Stage21EvaluationContext,
)
from marl_topology.models import LocalGNNEdgeScorer
from marl_topology.models.centralized_message_passing_graph_critic import GraphCriticBatch
from marl_topology.objectives import SurrogateSignalInput, evaluate_reward_surrogate
from marl_topology.training.policy_gradient import pilot_runner as stage23_pg
from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    get_active_policy_gradient_sampler,
)

from .critic_features import (
    CriticDiagnosticFeatureRecord,
    CriticFeatureBatch,
    CriticValueFeatureRecord,
    PreviousStepSummary,
    build_diagnostic_feature_record,
    build_feature_batch,
    build_value_feature_record,
    value_feature_health,
)
from .mappo.trainer import Stage24LoopConfig, _parameter_checksum, _row_contexts, _stage23_config


STAGE27_CRITIC_DATASET_CONFIG_ID = "stage27_critic_repair_dataset_config"
STAGE27_FROZEN_ACTOR_POLICY_ID = "stage27_frozen_local_gnn_policy_no_actor_update"


class CriticDatasetViolation(ValueError):
    """Raised when critic dataset collection crosses a boundary."""


@dataclass(frozen=True, slots=True)
class Stage27CriticDatasetConfig:
    train_scenarios: int = 32
    eval_scenarios: int = 16
    seeds: tuple[int, ...] = (2701, 2702, 2703, 2704)
    rollout_steps: int = 16
    gamma: float = 0.99
    top_k: int = 3
    endpoint_budget: int = 1
    tau_requirement_min: float = STAGE21_TAU_REQUIREMENT_MIN

    def __post_init__(self) -> None:
        if self.train_scenarios <= 0 or self.eval_scenarios <= 0:
            raise CriticDatasetViolation("scenario counts must be positive")
        if self.rollout_steps <= 0:
            raise CriticDatasetViolation("rollout_steps must be positive")
        if len(self.seeds) < 2:
            raise CriticDatasetViolation("at least two seeds are required")
        if self.train_transitions < 512:
            raise CriticDatasetViolation("Stage 27 requires at least 512 train transitions")
        if self.eval_transitions < 128:
            raise CriticDatasetViolation("Stage 27 requires at least 128 eval transitions")
        if self.tau_requirement_min != STAGE21_TAU_REQUIREMENT_MIN:
            raise CriticDatasetViolation("tau requirement remains fixed")

    @property
    def train_transitions(self) -> int:
        return self.train_scenarios * self.rollout_steps * len(self.seeds)

    @property
    def eval_transitions(self) -> int:
        return self.eval_scenarios * self.rollout_steps * len(self.seeds)

    def to_payload(self) -> dict[str, object]:
        return {
            "config_id": STAGE27_CRITIC_DATASET_CONFIG_ID,
            "train_scenarios": self.train_scenarios,
            "eval_scenarios": self.eval_scenarios,
            "seeds": list(self.seeds),
            "rollout_steps": self.rollout_steps,
            "train_transitions": self.train_transitions,
            "eval_transitions": self.eval_transitions,
            "gamma": self.gamma,
            "top_k": self.top_k,
            "endpoint_budget": self.endpoint_budget,
            "actor_update_performed": False,
            "policy_gradient_update_performed": False,
        }


@dataclass(frozen=True, slots=True)
class CriticDatasetRow:
    split: str
    scenario_id: str
    time_step: int
    seed: int
    source_index: int
    scenario_slot: int
    value_features: CriticValueFeatureRecord
    diagnostic_features: CriticDiagnosticFeatureRecord
    graph_node_features: tuple[tuple[float, ...], ...]
    graph_edge_features: tuple[tuple[float, ...], ...]
    graph_edge_index: tuple[tuple[int, int], ...]
    reward : float
    return_value: float
    done: bool
    mask: float
    selected_edges: tuple[str, ...]
    sampler_id: str
    proposal_logprob: float
    proposal_entropy: float
    consensus_success_probability: float
    latency: float
    energy: float
    reward_components: Mapping[str, float]
    topology_diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        numeric = (
            self.reward,
            self.return_value,
            self.mask,
            self.proposal_logprob,
            self.proposal_entropy,
            self.consensus_success_probability,
            self.latency,
            self.energy,
        )
        if any(not isfinite(value) for value in numeric):
            raise CriticDatasetViolation("critic dataset row has non-finite numeric values")
        if self.split not in {"train", "eval"}:
            raise CriticDatasetViolation("split must be train or eval")

    def to_payload(self) -> dict[str, object]:
        return {
            "split": self.split,
            "scenario_id": self.scenario_id,
            "time_step": self.time_step,
            "seed": self.seed,
            "source_index": self.source_index,
            "scenario_slot": self.scenario_slot,
            "value_critic_features": self.value_features.as_mapping(),
            "diagnostic_features": self.diagnostic_features.as_mapping(),
            "reward": self.reward,
            "return": self.return_value,
            "done": self.done,
            "mask": self.mask,
            "sampler_id": self.sampler_id,
            "selected_topology_diagnostics": dict(self.topology_diagnostics),
            "reward_components": dict(self.reward_components),
            "outcome_metrics": {
                "consensus_success_probability": self.consensus_success_probability,
                "latency": self.latency,
                "energy": self.energy,
            },
        }


@dataclass(frozen=True, slots=True)
class CriticDataset:
    rows: tuple[CriticDatasetRow, ...]
    config: Stage27CriticDatasetConfig
    actor_checksum_before: float
    actor_checksum_after: float
    sampler_id: str = ACTIVE_POLICY_GRADIENT_SAMPLER_ID
    actor_update_performed: bool = False
    policy_gradient_update_performed: bool = False

    def __post_init__(self) -> None:
        if not self.rows:
            raise CriticDatasetViolation("critic dataset cannot be empty")
        if self.actor_checksum_before != self.actor_checksum_after:
            raise CriticDatasetViolation("actor checksum changed during critic data collection")
        if self.actor_update_performed or self.policy_gradient_update_performed:
            raise CriticDatasetViolation("Stage 27 dataset collection must be critic-only")

    @property
    def train_rows(self) -> tuple[CriticDatasetRow, ...]:
        return tuple(row for row in self.rows if row.split == "train")

    @property
    def eval_rows(self) -> tuple[CriticDatasetRow, ...]:
        return tuple(row for row in self.rows if row.split == "eval")

    def feature_batch(self, split: str) -> CriticFeatureBatch:
        rows = self._split_rows(split)
        return build_feature_batch(
            (row.value_features for row in rows),
            (row.diagnostic_features for row in rows),
        )

    def returns_tensor(self, split: str) -> torch.Tensor:
        return torch.tensor([row.return_value for row in self._split_rows(split)], dtype=torch.float32)

    def training_signal_tensor(self, split: str) -> torch.Tensor:
        return torch.tensor([row.reward for row in self._split_rows(split)], dtype=torch.float32)

    def graph_batch(self, split: str) -> GraphCriticBatch:
        return graph_batch_from_rows(self._split_rows(split))

    def _split_rows(self, split: str) -> tuple[CriticDatasetRow, ...]:
        if split == "train":
            return self.train_rows
        if split == "eval":
            return self.eval_rows
        raise CriticDatasetViolation("split must be train or eval")

    def health_report(self) -> dict[str, object]:
        train = self.train_rows
        eval_rows = self.eval_rows
        rewards = [row.reward for row in self.rows]
        returns = [row.return_value for row in self.rows]
        source_keys = [f"{row.split}:{row.scenario_id}:slot{row.scenario_slot}" for row in self.rows]
        duplicate_rate = 1.0 - len(set(source_keys)) / len(source_keys)
        feasible = [
            row.consensus_success_probability >= self.config.tau_requirement_min
            for row in self.rows
        ]
        return {
            "config": self.config.to_payload(),
            "train_transition_count": len(train),
            "eval_transition_count": len(eval_rows),
            "return_variance": _variance(returns),
            "reward_variance": _variance(rewards),
            "minimum_train_transition_gate": len(train) >= 512,
            "minimum_eval_transition_gate": len(eval_rows) >= 128,
            "actor_update_performed": False,
            "policy_gradient_update_performed": False,
            "feature_health": value_feature_health(row.value_features for row in self.rows),
            "duplicate_context_rate": duplicate_rate,
            "feasible_ratio": sum(float(item) for item in feasible) / len(feasible),
            "reward_distribution": _summary(rewards),
            "return_distribution": _summary(returns),
            "sampler_id": self.sampler_id,
            "actor_policy_id": STAGE27_FROZEN_ACTOR_POLICY_ID,
        }


def collect_stage27_critic_dataset(
    config: Stage27CriticDatasetConfig | None = None,
) -> CriticDataset:
    cfg = config or Stage27CriticDatasetConfig()
    torch.manual_seed(cfg.seeds[0])
    contexts = _row_contexts()
    train_sources, eval_sources = _split_sources(contexts)
    actor = LocalGNNEdgeScorer()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor_before = _parameter_checksum(actor)
    sampler = get_active_policy_gradient_sampler()
    reward_config = stage23_pg.build_stage23_reward_surrogate_config()
    stage_config = _stage_config(cfg)
    policy_config = _stage23_config(stage_config)
    rows: list[CriticDatasetRow] = []
    with torch.no_grad():
        for split, sources, scenario_count in (
            ("train", train_sources, cfg.train_scenarios),
            ("eval", eval_sources, cfg.eval_scenarios),
        ):
            slots = _expanded_slots(sources, scenario_count)
            for seed in cfg.seeds:
                rows.extend(
                    _collect_split_rows(
                        split=split,
                        slots=slots,
                        actor=actor,
                        sampler=sampler,
                        reward_config=reward_config,
                        policy_config=policy_config,
                        config=cfg,
                        seed=seed,
                    )
                )
    actor_after = _parameter_checksum(actor)
    return CriticDataset(
        rows=tuple(rows),
        config=cfg,
        actor_checksum_before=actor_before,
        actor_checksum_after=actor_after,
        sampler_id=sampler.sampler_id,
    )


def graph_batch_from_rows(rows: Sequence[CriticDatasetRow]) -> GraphCriticBatch:
    if not rows:
        raise CriticDatasetViolation("graph batch requires rows")
    max_nodes = max(len(row.graph_node_features) for row in rows)
    max_edges = max(len(row.graph_edge_features) for row in rows)
    node_dim = len(rows[0].graph_node_features[0])
    edge_dim = len(rows[0].graph_edge_features[0])
    node_tensor = torch.zeros((len(rows), max_nodes, node_dim), dtype=torch.float32)
    edge_tensor = torch.zeros((len(rows), max_edges, edge_dim), dtype=torch.float32)
    edge_index = torch.zeros((len(rows), max_edges, 2), dtype=torch.long)
    node_mask = torch.zeros((len(rows), max_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((len(rows), max_edges), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        node_count = len(row.graph_node_features)
        edge_count = len(row.graph_edge_features)
        node_tensor[row_index, :node_count] = torch.tensor(row.graph_node_features, dtype=torch.float32)
        edge_tensor[row_index, :edge_count] = torch.tensor(row.graph_edge_features, dtype=torch.float32)
        edge_index[row_index, :edge_count] = torch.tensor(row.graph_edge_index, dtype=torch.long)
        node_mask[row_index, :node_count] = True
        edge_mask[row_index, :edge_count] = True
    return GraphCriticBatch(
        node_features=node_tensor,
        edge_features=edge_tensor,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )


def _collect_split_rows(
    *,
    split: str,
    slots: Sequence[tuple[object, Stage21EvaluationContext, int]],
    actor: LocalGNNEdgeScorer,
    sampler,
    reward_config,
    policy_config,
    config: Stage27CriticDatasetConfig,
    seed: int,
) -> list[CriticDatasetRow]:
    rows: list[CriticDatasetRow] = []
    for scenario_slot, (row, context, source_index) in enumerate(slots):
        previous = PreviousStepSummary()
        trajectory_indices = []
        for step_index in range(config.rollout_steps):
            sample_seed = seed + scenario_slot * 10_000 + step_index * 101
            value_features = build_value_feature_record(
                row=row,
                context=context,
                previous=previous,
                step_index=step_index,
                seed=sample_seed,
            )
            view = stage23_pg._actor_physical_logit_view(actor, row, policy_config)
            rng = torch.Generator().manual_seed(sample_seed)
            sample = sampler.sample(view.physical_logits, view.mask, view.config, rng)
            assembly = stage23_pg._assemble_physical_proposal(row.actor_safe_view, view, sample)
            evaluation = context.evaluator.evaluate(assembly.selected_physical_edges)
            metrics = evaluation.metrics
            signal = evaluate_reward_surrogate(
                SurrogateSignalInput(
                    consensus_success_probability=float(
                        metrics["consensus_success_probability"]
                    ),
                    latency=float(metrics["latency"]),
                    energy=float(metrics["energy"]),
                    topology_diagnostics=metrics["topology_diagnostics"],
                ),
                reward_config,
            )
            projection = stage23_pg._projection_diagnostics(view, sample, assembly)
            diagnostic_features = build_diagnostic_feature_record(
                row=row,
                context=context,
                selected_edges=assembly.selected_physical_edges,
                consensus_success_probability=float(metrics["consensus_success_probability"]),
                latency=float(metrics["latency"]),
                energy=float(metrics["energy"]),
                reward_surrogate=float(signal.training_signal_value),
                projection_diagnostics=projection,
            )
            graph_parts = _graph_parts(
                row=row,
                context=context,
                previous_selected_edges=previous.selected_edges,
                step_index=step_index,
            )
            dataset_row = CriticDatasetRow(
                split=split,
                scenario_id=str(context.fixture.fixture_id),
                time_step=step_index,
                seed=sample_seed,
                source_index=source_index,
                scenario_slot=scenario_slot,
                value_features=value_features,
                diagnostic_features=diagnostic_features,
                graph_node_features=graph_parts["node_features"],
                graph_edge_features=graph_parts["edge_features"],
                graph_edge_index=graph_parts["edge_index"],
                reward=float(signal.training_signal_value),
                return_value=0.0,
                done=step_index == config.rollout_steps - 1,
                mask=0.0 if step_index == config.rollout_steps - 1 else 1.0,
                selected_edges=tuple(assembly.selected_physical_edges),
                sampler_id=sample.sampler_id,
                proposal_logprob=float(sample.logprob.detach().cpu().item()),
                proposal_entropy=float(sample.entropy.detach().cpu().item()),
                consensus_success_probability=float(metrics["consensus_success_probability"]),
                latency=float(metrics["latency"]),
                energy=float(metrics["energy"]),
                reward_components={
                    "reliability_violation": float(signal.reliability_violation),
                    "reliability_penalty": float(signal.reliability_penalty),
                    "latency_penalty": float(signal.latency_penalty),
                    "energy_penalty": float(signal.energy_penalty),
                    "normalized_latency": float(signal.normalized_latency),
                    "normalized_energy": float(signal.normalized_energy),
                },
                topology_diagnostics=dict(metrics["topology_diagnostics"]),
            )
            trajectory_indices.append(len(rows))
            rows.append(dataset_row)
            previous = PreviousStepSummary(
                selected_edges=tuple(assembly.selected_physical_edges),
                projection_rejection_rate=float(projection["top_proposal_rejection_rate"]),
                top_rejection_rate=float(projection["top_proposal_rejection_rate"]),
                rejection_by_reason=dict(projection["rejection_by_reason"]),
                reward_summary=float(signal.training_signal_value),
                consensus_success_probability=float(metrics["consensus_success_probability"]),
                latency=float(metrics["latency"]),
                energy=float(metrics["energy"]),
                violation_indicator=(
                    1.0
                    if float(metrics["consensus_success_probability"])
                    < config.tau_requirement_min
                    else 0.0
                ),
            )
        running = 0.0
        for row_index in reversed(trajectory_indices):
            running = rows[row_index].reward + config.gamma * running * rows[row_index].mask
            rows[row_index] = replace(rows[row_index], return_value=running)
    return rows


def _split_sources(contexts: Sequence[tuple[object, Stage21EvaluationContext]]):
    if len(contexts) < 2:
        raise CriticDatasetViolation("at least two source contexts are required")
    split_index = max(1, int(round(len(contexts) * 0.7)))
    if split_index >= len(contexts):
        split_index = len(contexts) - 1
    train = tuple((row, context, index) for index, (row, context) in enumerate(contexts[:split_index]))
    eval_rows = tuple(
        (row, context, index)
        for index, (row, context) in enumerate(contexts[split_index:], start=split_index)
    )
    return train, eval_rows


def _expanded_slots(
    sources: Sequence[tuple[object, Stage21EvaluationContext, int]],
    count: int,
) -> tuple[tuple[object, Stage21EvaluationContext, int], ...]:
    return tuple(sources[index % len(sources)] for index in range(count))


def _stage_config(config: Stage27CriticDatasetConfig):
    return Stage24LoopConfig(
        mode="micro",
        seed=config.seeds[0],
        num_scenarios=16,
        rollout_steps=config.rollout_steps,
        top_k=config.top_k,
        endpoint_budget=config.endpoint_budget,
    )


def _graph_parts(
    *,
    row: object,
    context: Stage21EvaluationContext,
    previous_selected_edges: Sequence[str],
    step_index: int,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    node_ids = tuple(context.graph.node_ids)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    actor_rows = tuple(row.actor_safe_view)
    by_agent: dict[str, list[Mapping[str, object]]] = {}
    for actor_row in actor_rows:
        by_agent.setdefault(str(actor_row["agent_id"]), []).append(actor_row)
    previous_degree = Counter()
    for edge_id in previous_selected_edges:
        left, right = _split_edge(edge_id)
        previous_degree[left] += 1
        previous_degree[right] += 1
    node_features = []
    for node_id in node_ids:
        records = by_agent.get(node_id, [])
        role = _node_role(node_id)
        node_features.append(
            (
                1.0 if role == "vehicle" else 0.0,
                1.0 if role == "rsu" else 0.0,
                1.0 if role == "base_station" else 0.0,
                float(previous_degree[node_id]),
                _avg(records, "tx_budget_used"),
                _avg(records, "tx_budget_remaining"),
                _avg(records, "rx_capacity_estimate_for_neighbor"),
                float(step_index),
            )
        )
    edge_features = []
    edge_index = []
    previous_set = set(previous_selected_edges)
    for edge in context.graph.edges:
        record = context.link_records[edge.edge_id]
        left_role = _node_role(edge.node_u)
        right_role = _node_role(edge.node_v)
        edge_features.append(
            (
                float(record.link_success_probability),
                float(record.link_success_probability),
                float(record.latency_s),
                float(record.energy_j),
                1.0 if edge.edge_id in previous_set else 0.0,
                float(edge.distance_3d_m),
                _role_code(left_role),
                _role_code(right_role),
            )
        )
        edge_index.append((node_index[edge.node_u], node_index[edge.node_v]))
    return {
        "node_features": tuple(node_features),
        "edge_features": tuple(edge_features),
        "edge_index": tuple(edge_index),
    }


def _avg(records: Sequence[Mapping[str, object]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def _split_edge(edge_id: str) -> tuple[str, str]:
    parts = str(edge_id).split("--")
    if len(parts) != 2:
        return str(edge_id), ""
    return parts[0], parts[1]


def _node_role(node_id: str) -> str:
    lowered = str(node_id).lower()
    if lowered.startswith("veh"):
        return "vehicle"
    if lowered.startswith("rsu"):
        return "rsu"
    if lowered.startswith("bs") or "base" in lowered:
        return "base_station"
    return "unknown"


def _role_code(role: str) -> float:
    return {"vehicle": 0.25, "rsu": 0.5, "base_station": 0.75}.get(role, 0.0)


def _summary(values: Iterable[float]) -> dict[str, float]:
    vals = [float(value) for value in values]
    if not vals:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(vals) / len(vals)
    return {
        "count": len(vals),
        "mean": mean,
        "std": _variance(vals) ** 0.5,
        "min": min(vals),
        "max": max(vals),
    }


def _variance(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((value - mean) ** 2 for value in vals) / len(vals)
