"""Stage 20 supervised actor policy evaluation through topology assembler."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from marl_topology.data.learning_evidence_stage16 import (
    STAGE16_TAU_REQUIREMENT_MIN,
    _stage16_topology_variants,
    iter_stage16_evidence_scenario_fixtures,
)
from marl_topology.evaluation.fixture_stack import EvaluationStack, build_fixture_stack
from marl_topology.policies import PolicyBaselines
from marl_topology.policies.edge_scores import EdgeScoreRecord
from marl_topology.policies.topology_assembler import (
    AssemblerConfig,
    CandidateEdgeConstraint,
    ConflictAwareGreedyAssembler,
    DEPLOYMENT_ASSEMBLER_FORBIDDEN_FIELDS,
)
from marl_topology.training.stage19_supervised_actor_stack import (
    STAGE19_FEATURE_FIELDS,
    Stage19SupervisedActorStackConfig,
    Stage19TrainedActorStack,
    train_stage19_supervised_actor_models,
)
import marl_topology.training.torch_backend as _torch_backend

torch = _torch_backend.torch


STAGE20_STAGE_ID = "stage_20_supervised_actor_policy_evaluation_with_environment_assembler"
STAGE20_VERDICT = "stage20_supervised_actor_policy_evaluated_rl_still_blocked"
STAGE20_RECOMMENDED_NEXT_IF_PASS = (
    "stage_21_owner_decision_on_bounded_policy_gradient_readiness_review"
)
STAGE20_RECOMMENDED_NEXT_IF_BLOCKED = (
    "stage_21_assembler_aware_supervised_target_refinement"
)


class Stage20PolicyEvaluationViolation(ValueError):
    """Raised when Stage 20 evaluation crosses a declared boundary."""


@dataclass(frozen=True, slots=True)
class Stage20PolicyEvaluationConfig:
    seed: int = 20
    stage19_epochs: int = 80
    stage19_temporal_epochs: int = 80
    tau_requirement_min: float = STAGE16_TAU_REQUIREMENT_MIN
    projection_rejection_rate_warn: float = 0.75

    def __post_init__(self) -> None:
        if self.stage19_epochs <= 0 or self.stage19_temporal_epochs <= 0:
            raise Stage20PolicyEvaluationViolation("Stage 19 epoch counts must be positive")
        if not 0.0 <= self.tau_requirement_min <= 1.0:
            raise Stage20PolicyEvaluationViolation("tau_requirement_min must be in [0, 1]")
        if not 0.0 <= self.projection_rejection_rate_warn <= 1.0:
            raise Stage20PolicyEvaluationViolation("projection rejection threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Stage20EvaluationRow:
    model_label: str
    row_index: int
    scenario_id: str
    source_fixture: str
    topology_name: str
    time_step: int
    sequence_id: str | None
    selected_edge_ids: tuple[str, ...]
    selected_directed_edges: tuple[str, ...]
    pre_projection_edge_count: int
    post_projection_edge_count: int
    rejected_edge_count: int
    rejection_reason_counts: Mapping[str, int]
    metrics: Mapping[str, object]
    baseline_metrics: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_label": self.model_label,
            "row_index": self.row_index,
            "scenario_id": self.scenario_id,
            "source_fixture": self.source_fixture,
            "topology_name": self.topology_name,
            "time_step": self.time_step,
            "sequence_id": self.sequence_id,
            "selected_edge_ids": list(self.selected_edge_ids),
            "selected_directed_edges": list(self.selected_directed_edges),
            "pre_projection_edge_count": self.pre_projection_edge_count,
            "post_projection_edge_count": self.post_projection_edge_count,
            "rejected_edge_count": self.rejected_edge_count,
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "metrics": _jsonable_mapping(self.metrics),
            "baseline_metrics": {
                name: _jsonable_mapping(metrics)
                for name, metrics in self.baseline_metrics.items()
            },
        }


def run_stage20_supervised_actor_policy_evaluation(
    *,
    config: Stage20PolicyEvaluationConfig | None = None,
) -> dict[str, object]:
    """Evaluate Stage 19 supervised actor scores through the Stage 8 assembler."""

    cfg = config or Stage20PolicyEvaluationConfig()
    trained = train_stage19_supervised_actor_models(
        config=Stage19SupervisedActorStackConfig(
            seed=cfg.seed,
            epochs=cfg.stage19_epochs,
            temporal_epochs=cfg.stage19_temporal_epochs,
        )
    )
    stacks = _fixture_stack_lookup()
    row_scores = _score_records_by_model_and_row(trained)
    evaluation_rows: dict[str, tuple[Stage20EvaluationRow, ...]] = {}
    for model_label, rows in row_scores.items():
        evaluated: list[Stage20EvaluationRow] = []
        for row_index in sorted(rows):
            if not rows[row_index]:
                continue
            evaluated.append(
                _evaluate_row(
                    trained=trained,
                    row_index=row_index,
                    model_label=model_label,
                    scores=rows[row_index],
                    stacks=stacks,
                    tau_requirement_min=cfg.tau_requirement_min,
                )
            )
        evaluation_rows[model_label] = tuple(evaluated)

    summaries = {
        model_label: _summarize_evaluation_rows(
            rows,
            tau_requirement_min=cfg.tau_requirement_min,
        )
        for model_label, rows in evaluation_rows.items()
    }
    baseline_summary = _baseline_summary(evaluation_rows, tau_requirement_min=cfg.tau_requirement_min)
    best_all_row_actor = _select_best_all_row_actor(summaries)
    next_gate_passed = _stage21_policy_gradient_readiness_gate(
        summaries=summaries,
        baseline_summary=baseline_summary,
        best_actor=best_all_row_actor,
        projection_rejection_rate_warn=cfg.projection_rejection_rate_warn,
    )
    return {
        "stage": STAGE20_STAGE_ID,
        "verdict": STAGE20_VERDICT,
        "source_stage19_verdict": trained.report["verdict"],
        "source_dataset_id": trained.report["source_dataset_id"],
        "feature_schema_id": trained.report["feature_schema_id"],
        "feature_count": len(STAGE19_FEATURE_FIELDS),
        "tau_requirement_min": cfg.tau_requirement_min,
        "model_evaluation_order": ["MLP", "GNN", "GRU_real_multistep_subset", "LSTM_after_GRU_sanity"],
        "assembler_id": "stage20_conflict_aware_greedy_environment_projection",
        "assembler_type": "ConflictAwareGreedyAssembler",
        "environment_side_topology_assembler_used": True,
        "actor_outputs_final_topology": False,
        "actor_outputs_edge_scores_only": True,
        "assembler_uses_oracle_reward_objective_consensus": False,
        "oracle_diagnostic_used_for_actor_input": False,
        "baseline_full_graph_is_oracle": False,
        "evaluation_rows": {
            model_label: [row.to_dict() for row in rows]
            for model_label, rows in evaluation_rows.items()
        },
        "model_summaries": summaries,
        "baseline_summary": baseline_summary,
        "best_all_row_actor": best_all_row_actor,
        "stage21_policy_gradient_readiness_gate_passed": next_gate_passed,
        "recommended_next_task": STAGE20_RECOMMENDED_NEXT_IF_PASS
        if next_gate_passed
        else STAGE20_RECOMMENDED_NEXT_IF_BLOCKED,
        "policy_gradient_allowed": False,
        "ppo_mappo_allowed": False,
        "coma_allowed": False,
        "transformer_allowed": False,
        "scale_up_training_allowed": False,
        "critic_training_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "reward_weight_tuning_performed": False,
        "final_tau_selected": False,
        "v5_modified": False,
    }


def _score_records_by_model_and_row(
    trained: Stage19TrainedActorStack,
) -> dict[str, dict[int, list[EdgeScoreRecord]]]:
    result: dict[str, dict[int, list[EdgeScoreRecord]]] = {
        "MLP": defaultdict(list),
        "GNN": defaultdict(list),
    }
    with torch.no_grad():
        mlp_logits = trained.mlp_model(trained.batch.features)
        gnn_logits = trained.gnn_model.forward_with_groups(
            trained.batch.features,
            trained.batch.group_ids,
        )
    for model_label, logits in (("MLP", mlp_logits), ("GNN", gnn_logits)):
        for index, sample in enumerate(trained.batch.samples):
            result[model_label][sample.row_index].append(
                _edge_score_from_sample(sample, float(logits[index].item()), model_label)
            )

    temporal_scores = _temporal_score_lookup(trained)
    for model_label, keyed_scores in temporal_scores.items():
        result[model_label] = defaultdict(list)
        sample_lookup = {
            (
                sample.sequence_id,
                sample.topology_name,
                sample.agent_id,
                sample.edge_id,
                sample.time_step,
            ): sample
            for sample in trained.batch.samples
            if sample.sequence_id is not None
        }
        for key, logit in keyed_scores.items():
            sample = sample_lookup.get(key)
            if sample is None:
                continue
            result[model_label][sample.row_index].append(
                _edge_score_from_sample(sample, logit, model_label)
            )
    return result


def _temporal_score_lookup(
    trained: Stage19TrainedActorStack,
) -> dict[str, dict[tuple[str, str, str, str, int], float]]:
    lookup: dict[str, dict[tuple[str, str, str, str, int], float]] = {}
    with torch.no_grad():
        gru_logits, _hidden = trained.gru_model(trained.temporal_batch.features)
        model_logits = {"GRU": gru_logits}
        if trained.lstm_model is not None:
            lstm_logits, _hidden = trained.lstm_model(trained.temporal_batch.features)
            model_logits["LSTM"] = lstm_logits
    for model_label, logits in model_logits.items():
        keyed: dict[tuple[str, str, str, str, int], float] = {}
        for sequence_index, sequence_id in enumerate(trained.temporal_batch.sequence_ids):
            source_sequence_id, topology_name, agent_id, edge_id = sequence_id.split("|")
            for time_index, time_step in enumerate(trained.temporal_batch.time_steps):
                if not bool(trained.temporal_batch.mask[sequence_index, time_index].item()):
                    continue
                keyed[(source_sequence_id, topology_name, agent_id, edge_id, time_step)] = float(
                    logits[sequence_index, time_index].item()
                )
        lookup[model_label] = keyed
    return lookup


def _edge_score_from_sample(sample, logit: float, model_label: str) -> EdgeScoreRecord:
    neighbor_id = sample.directed_edge_id.split("->", 1)[1]
    return EdgeScoreRecord(
        agent_id=sample.agent_id,
        neighbor_id=neighbor_id,
        edge_id=sample.edge_id,
        directed_edge_id=sample.directed_edge_id,
        score=logit,
        probability=float(torch.sigmoid(torch.tensor(logit)).item()),
        score_source=f"stage20_{model_label.lower()}_supervised_actor_score",
        time_step=sample.time_step,
    )


def _evaluate_row(
    *,
    trained: Stage19TrainedActorStack,
    row_index: int,
    model_label: str,
    scores: Iterable[EdgeScoreRecord],
    stacks: Mapping[tuple[str, int], EvaluationStack],
    tau_requirement_min: float,
) -> Stage20EvaluationRow:
    row = trained.dataset.rows[row_index]
    diagnostics = row.diagnostics_view
    source_fixture = str(diagnostics["source_fixture"])
    time_step = int(diagnostics["time_step"])
    stack = stacks[(source_fixture, time_step)]
    constraints = _candidate_constraints(row.actor_safe_view)
    assembler = ConflictAwareGreedyAssembler(
        AssemblerConfig(
            assembler_id="stage20_conflict_aware_greedy_environment_projection",
            mode="conflict_aware_greedy",
            tx_capacity=_tx_capacity(row.actor_safe_view),
            rx_capacity=_rx_capacity(row.actor_safe_view),
            deterministic=True,
        )
    )
    metadata = {"agent_roles": _agent_roles(row.actor_safe_view)}
    _assert_metadata_actor_safe_for_assembler(metadata)
    assembled = assembler.assemble(scores, constraints, metadata=metadata)
    selected_edge_ids = _selected_edge_ids_from_assembly(scores, assembled.selected_directed_edges)
    evaluation = stack.evaluator.evaluate(
        set(selected_edge_ids),
        topology_id=f"stage20:{model_label}:{source_fixture}:{row_index}",
    )
    baseline_metrics = _baseline_metrics_for_row(
        stack,
        topology_name=str(diagnostics["topology_name"]),
        tau_requirement_min=tau_requirement_min,
    )
    return Stage20EvaluationRow(
        model_label=model_label,
        row_index=row_index,
        scenario_id=str(diagnostics["scenario_id"]),
        source_fixture=source_fixture,
        topology_name=str(diagnostics["topology_name"]),
        time_step=time_step,
        sequence_id=(
            str(diagnostics["sequence_id"])
            if diagnostics.get("sequence_id") is not None
            else None
        ),
        selected_edge_ids=tuple(sorted(selected_edge_ids)),
        selected_directed_edges=assembled.selected_directed_edges,
        pre_projection_edge_count=int(assembled.pre_projection_edge_count),
        post_projection_edge_count=int(assembled.post_projection_edge_count),
        rejected_edge_count=len(assembled.rejected_edges),
        rejection_reason_counts=dict(assembled.diagnostics["rejection_reason_counts"]),
        metrics=evaluation.metrics,
        baseline_metrics=baseline_metrics,
    )


def _fixture_stack_lookup() -> dict[tuple[str, int], EvaluationStack]:
    return {
        (fixture.fixture_id, time_step): build_fixture_stack(fixture)
        for fixture, time_step, _sequence_id in iter_stage16_evidence_scenario_fixtures()
    }


def _candidate_constraints(
    actor_safe_view: Iterable[Mapping[str, object]],
) -> tuple[CandidateEdgeConstraint, ...]:
    constraints = []
    for row in actor_safe_view:
        constraints.append(
            CandidateEdgeConstraint(
                edge_id=str(row["edge_id"]),
                tx_id=str(row["agent_id"]),
                rx_id=str(row["neighbor_id"]),
                edge_type="stage20_actor_local_candidate",
                role_allowed=True,
                channel_slot=None,
                conflict_group=f"physical_edge:{row['edge_id']}",
                tx_capacity_cost=1.0,
                rx_capacity_cost=1.0,
                valid_candidate=bool(row["channel_slot_available"]),
            )
        )
    return tuple(constraints)


def _tx_capacity(actor_safe_view: Iterable[Mapping[str, object]]) -> dict[str, float]:
    capacity: dict[str, float] = {}
    for row in actor_safe_view:
        agent_id = str(row["agent_id"])
        capacity[agent_id] = max(
            capacity.get(agent_id, 0.0),
            float(row["tx_budget_remaining"]),
        )
    return capacity


def _rx_capacity(actor_safe_view: Iterable[Mapping[str, object]]) -> dict[str, float]:
    capacity: dict[str, float] = {}
    for row in actor_safe_view:
        neighbor_id = str(row["neighbor_id"])
        capacity[neighbor_id] = max(
            capacity.get(neighbor_id, 0.0),
            float(row["rx_capacity_estimate_for_neighbor"]),
        )
    return capacity


def _agent_roles(actor_safe_view: Iterable[Mapping[str, object]]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in actor_safe_view:
        roles[str(row["agent_id"])] = str(row["agent_kind"])
        roles[str(row["neighbor_id"])] = str(row["neighbor_kind"])
    return roles


def _assert_metadata_actor_safe_for_assembler(metadata: Mapping[str, object]) -> None:
    forbidden = sorted(set(metadata) & DEPLOYMENT_ASSEMBLER_FORBIDDEN_FIELDS)
    if forbidden:
        raise Stage20PolicyEvaluationViolation(
            f"forbidden Stage 20 assembler metadata: {forbidden}"
        )


def _selected_edge_ids_from_assembly(
    scores: Iterable[EdgeScoreRecord],
    selected_directed_edges: Iterable[str],
) -> tuple[str, ...]:
    by_directed = {score.directed_edge_id: score.edge_id for score in scores}
    return tuple(
        sorted({by_directed[directed_edge_id] for directed_edge_id in selected_directed_edges})
    )


def _baseline_metrics_for_row(
    stack: EvaluationStack,
    *,
    topology_name: str,
    tau_requirement_min: float,
) -> dict[str, Mapping[str, object]]:
    full = stack.evaluator.evaluate(
        set(stack.graph.edge_ids),
        topology_id="stage20_baseline:full_graph",
    )
    greedy_edges = PolicyBaselines.greedy_reliability(
        stack.graph,
        stack.evaluator.link_records,
    ).edge_ids
    greedy = stack.evaluator.evaluate(
        set(greedy_edges),
        topology_id="stage20_baseline:greedy_reliability",
    )
    source_edges = _source_topology_edges(stack, topology_name)
    source = stack.evaluator.evaluate(
        set(source_edges),
        topology_id=f"stage20_baseline:source:{topology_name}",
    )
    oracle = stack.oracle.solve(random_seed=20).evaluation
    result = {
        "full_graph": _metric_summary(full.metrics, tau_requirement_min=tau_requirement_min),
        "greedy_reliability": _metric_summary(greedy.metrics, tau_requirement_min=tau_requirement_min),
        "source_topology": _metric_summary(source.metrics, tau_requirement_min=tau_requirement_min),
    }
    if oracle is not None:
        result["oracle_diagnostic"] = _metric_summary(
            oracle.metrics,
            tau_requirement_min=tau_requirement_min,
        )
    return result


def _source_topology_edges(stack: EvaluationStack, topology_name: str) -> tuple[str, ...]:
    if topology_name == "oracle_candidate":
        oracle = stack.oracle.solve(random_seed=16).evaluation
        return tuple(oracle.selected_edge_ids) if oracle is not None else ()
    variants = _stage16_topology_variants(stack.evaluator)
    return tuple(variants.get(topology_name, ()))


def _summarize_evaluation_rows(
    rows: tuple[Stage20EvaluationRow, ...],
    *,
    tau_requirement_min: float,
) -> dict[str, object]:
    if not rows:
        return {
            "row_count": 0,
            "tau_feasible_count": 0,
            "tau_violation_rate": 1.0,
            "mean_consensus_success_probability": 0.0,
            "mean_latency": 0.0,
            "mean_energy": 0.0,
            "mean_selected_edge_count": 0.0,
            "projection_rejection_rate": 0.0,
            "nonempty_topology_rate": 0.0,
            "rejection_reason_counts": {},
        }
    probabilities = [
        float(row.metrics["consensus_success_probability"])
        for row in rows
    ]
    latencies = [float(row.metrics["latency"]) for row in rows]
    energies = [float(row.metrics["energy"]) for row in rows]
    selected_counts = [len(row.selected_edge_ids) for row in rows]
    total_pre = sum(row.pre_projection_edge_count for row in rows)
    total_rejected = sum(row.rejected_edge_count for row in rows)
    reason_counts = Counter(
        reason
        for row in rows
        for reason, count in row.rejection_reason_counts.items()
        for _ in range(int(count))
    )
    feasible_count = sum(probability >= tau_requirement_min for probability in probabilities)
    return {
        "row_count": len(rows),
        "tau_feasible_count": feasible_count,
        "tau_feasible_rate": feasible_count / len(rows),
        "tau_violation_rate": 1.0 - feasible_count / len(rows),
        "mean_consensus_success_probability": _mean(probabilities),
        "min_consensus_success_probability": min(probabilities),
        "max_consensus_success_probability": max(probabilities),
        "mean_latency": _mean(latencies),
        "mean_energy": _mean(energies),
        "mean_selected_edge_count": _mean(selected_counts),
        "projection_rejection_rate": total_rejected / max(1, total_pre),
        "nonempty_topology_rate": sum(count > 0 for count in selected_counts) / len(rows),
        "rejection_reason_counts": dict(reason_counts),
    }


def _baseline_summary(
    evaluation_rows: Mapping[str, tuple[Stage20EvaluationRow, ...]],
    *,
    tau_requirement_min: float,
) -> dict[str, object]:
    rows = next(iter(evaluation_rows.values()), ())
    result: dict[str, object] = {}
    for baseline_name in ("full_graph", "greedy_reliability", "source_topology", "oracle_diagnostic"):
        metrics = [
            row.baseline_metrics[baseline_name]
            for row in rows
            if baseline_name in row.baseline_metrics
        ]
        probabilities = [
            float(metric["consensus_success_probability"])
            for metric in metrics
        ]
        if not metrics:
            continue
        feasible_count = sum(probability >= tau_requirement_min for probability in probabilities)
        result[baseline_name] = {
            "row_count": len(metrics),
            "tau_feasible_count": feasible_count,
            "tau_feasible_rate": feasible_count / len(metrics),
            "mean_consensus_success_probability": _mean(probabilities),
            "mean_latency": _mean(float(metric["latency"]) for metric in metrics),
            "mean_energy": _mean(float(metric["energy"]) for metric in metrics),
            "is_oracle": baseline_name == "oracle_diagnostic",
        }
    return result


def _select_best_all_row_actor(summaries: Mapping[str, Mapping[str, object]]) -> str:
    candidates = {
        label: summary
        for label, summary in summaries.items()
        if label in {"MLP", "GNN"} and int(summary["row_count"]) > 0
    }
    if not candidates:
        return "none"
    return max(
        candidates,
        key=lambda label: (
            float(candidates[label]["tau_feasible_rate"]),
            float(candidates[label]["mean_consensus_success_probability"]),
            -float(candidates[label]["mean_latency"]),
            -float(candidates[label]["mean_energy"]),
        ),
    )


def _stage21_policy_gradient_readiness_gate(
    *,
    summaries: Mapping[str, Mapping[str, object]],
    baseline_summary: Mapping[str, object],
    best_actor: str,
    projection_rejection_rate_warn: float,
) -> bool:
    if best_actor not in summaries:
        return False
    best = summaries[best_actor]
    greedy = baseline_summary.get("greedy_reliability", {})
    if not isinstance(greedy, Mapping):
        return False
    return (
        float(best["tau_feasible_rate"]) >= float(greedy.get("tau_feasible_rate", 1.0))
        and float(best["projection_rejection_rate"]) <= projection_rejection_rate_warn
        and float(best["nonempty_topology_rate"]) > 0.0
    )


def _metric_summary(
    metrics: Mapping[str, object],
    *,
    tau_requirement_min: float,
) -> dict[str, object]:
    probability = float(metrics["consensus_success_probability"])
    return {
        "consensus_success": int(metrics["consensus_success"]),
        "consensus_success_probability": probability,
        "tau_feasible": probability >= tau_requirement_min,
        "latency": float(metrics["latency"]),
        "energy": float(metrics["energy"]),
        "topology_diagnostics": _jsonable_mapping(
            metrics.get("topology_diagnostics", {})
            if isinstance(metrics.get("topology_diagnostics", {}), Mapping)
            else {}
        ),
    }


def _jsonable_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            result[str(key)] = _jsonable_mapping(value)
        elif isinstance(value, tuple | list):
            result[str(key)] = list(value)
        else:
            result[str(key)] = value
    return result


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if isfinite(float(value))]
    if not items:
        return 0.0
    return sum(items) / len(items)
