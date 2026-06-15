"""Selected Stage 22 physical-link evidence over the final objective stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from marl_topology.data.actor_feature_rebuild import (
    actor_safe_view_has_no_forbidden_fields,
    build_stage18_actor_safe_feature_rows,
)
from marl_topology.data.learning_evidence import EdgeDeltaTarget
from marl_topology.data.stage21_assembler_aware_targets import (
    agent_roles_from_actor_safe_view,
    build_stage21_conflict_aware_assembler,
    candidate_constraints_from_actor_safe_view,
    edge_score_records_from_actor_safe_view,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_EVALUATOR_ID,
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PHYSICS_REGIME_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    STAGE21_TAU_REQUIREMENT_MIN,
    Stage21EvaluationContext,
    Stage21ObjectiveEvaluation,
    build_stage21_objective_stack_contexts,
)
from marl_topology.policies import UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID

from .stage22_objective_teacher import (
    STAGE22_TAU_REQUIREMENT_MIN,
    aggregate_stage22_target_distribution,
    actor_target_view_has_no_forbidden_global_fields,
    build_stage22_actor_target_view,
    build_stage22_critic_target_view,
    critic_targets_are_critic_only,
    select_objective_aware_teacher,
    TeacherCandidate,
)


STAGE22_STAGE_ID = "stage_22_action_semantics_ab_full_gnn_repair"
STAGE22_DATASET_ID = "stage22_action_semantics_ab_objective_teacher_evidence_v1"
STAGE22_PHYSICAL_EVALUATOR_ID = STAGE21_EVALUATOR_ID
STAGE22_SELECTED_SEMANTICS = (UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,)


class Stage22EvidenceViolation(ValueError):
    """Raised when Stage 22 evidence crosses action/evaluator boundaries."""


@dataclass(frozen=True, slots=True)
class Stage22ObjectiveEvaluation:
    scenario_id: str
    topology_id: str
    action_semantics_id: str
    selected_directed_edges: tuple[str, ...]
    selected_physical_edges: tuple[str, ...]
    metrics: Mapping[str, object]
    per_primary_reliability: Mapping[str, float]
    evaluator_id: str
    physics_regime_id: str
    protocol_model_id: str
    objective_contract_id: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Stage22EvidenceRow:
    evidence_id: str
    action_semantics_id: str
    scenario_id: str
    time_step: int
    topology_id: str
    topology_name: str
    evaluator_id: str
    physics_regime_id: str
    protocol_model_id: str
    objective_contract_id: str
    selected_directed_edges: tuple[str, ...]
    selected_physical_edges: tuple[str, ...]
    consensus_success_probability: float
    per_primary_reliability: Mapping[str, float]
    latency: float
    energy: float
    feasible_under_tau_requirement: bool
    actor_safe_view: tuple[Mapping[str, object], ...]
    actor_target_view: Mapping[str, object]
    critic_target_view: Mapping[str, object]
    diagnostics_view: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.action_semantics_id != UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID:
            raise Stage22EvidenceViolation("Stage 23 preflight permits only selected physical semantics")
        if self.physics_regime_id != STAGE21_PHYSICS_REGIME_ID:
            raise Stage22EvidenceViolation("Stage 22 evidence must use finite-blocklength physics")
        if self.protocol_model_id != STAGE21_PROTOCOL_MODEL_ID:
            raise Stage22EvidenceViolation("Stage 22 evidence must use Stage 4 expected-initiator PBFT")
        if self.objective_contract_id != STAGE21_OBJECTIVE_CONTRACT_ID:
            raise Stage22EvidenceViolation("Stage 22 evidence must use the Stage 5 objective contract")
        if not actor_safe_view_has_no_forbidden_fields(self.actor_safe_view):
            raise Stage22EvidenceViolation("actor-safe view contains forbidden fields")
        if not actor_target_view_has_no_forbidden_global_fields(self.actor_target_view):
            raise Stage22EvidenceViolation("actor target view contains forbidden global fields")
        if not critic_targets_are_critic_only(self.critic_target_view):
            raise Stage22EvidenceViolation("critic target view must remain critic-only")


@dataclass(frozen=True, slots=True)
class Stage22ActionSemanticsDataset:
    dataset_id: str
    action_semantics_id: str
    rows: tuple[Stage22EvidenceRow, ...]
    readiness: Mapping[str, object]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def actor_edge_sample_count(self) -> int:
        return sum(len(row.actor_safe_view) for row in self.rows)


@dataclass(frozen=True, slots=True)
class Stage22EvidenceBuild:
    datasets: Mapping[str, Stage22ActionSemanticsDataset]
    report: Mapping[str, object]


@lru_cache(maxsize=1)
def build_stage22_action_semantics_evidence() -> Stage22EvidenceBuild:
    datasets = {
        semantics_id: _build_dataset_for_semantics(semantics_id)
        for semantics_id in STAGE22_SELECTED_SEMANTICS
    }
    report = {
        "stage": STAGE22_STAGE_ID,
        "dataset_id": STAGE22_DATASET_ID,
        "active_action_semantics": list(STAGE22_SELECTED_SEMANTICS),
        "stage23_preflight_removed_ab_loser_from_executable_code": True,
        "main_evidence_uses_stage3_stage4_stack": True,
        "silent_fallback_to_simple_link_or_min_link": False,
        "tau_requirement_min": STAGE21_TAU_REQUIREMENT_MIN,
        "datasets": {
            semantics_id: _dataset_summary(dataset)
            for semantics_id, dataset in datasets.items()
        },
        "target_distribution": {
            semantics_id: aggregate_stage22_target_distribution(
                row.actor_target_view for row in dataset.rows
            )
            for semantics_id, dataset in datasets.items()
        },
        "no_actor_leakage": all(
            actor_safe_view_has_no_forbidden_fields(row.actor_safe_view)
            and actor_target_view_has_no_forbidden_global_fields(row.actor_target_view)
            for dataset in datasets.values()
            for row in dataset.rows
        ),
        "critic_targets_are_critic_only": all(
            critic_targets_are_critic_only(row.critic_target_view)
            for dataset in datasets.values()
            for row in dataset.rows
        ),
        "policy_gradient_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "v5_modified": False,
    }
    return Stage22EvidenceBuild(datasets=datasets, report=report)


def _build_dataset_for_semantics(action_semantics_id: str) -> Stage22ActionSemanticsDataset:
    rows: list[Stage22EvidenceRow] = []
    previous_selected_by_key: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for context in build_stage21_objective_stack_contexts():
        source_evaluation = context.evaluator.evaluate(
            context.topology_variants["greedy_reliability_raw"],
            topology_id=f"stage22:source:{context.fixture.fixture_id}:t{context.time_step}",
        )
        source_row = _source_learning_row(context, source_evaluation, "stage22_source_greedy")
        previous_key = (str(context.sequence_id or ""), action_semantics_id, context.time_step - 1)
        actor_safe_view = build_stage18_actor_safe_feature_rows(
            source_row,
            previous_selected_edges=previous_selected_by_key.get(previous_key, ()),
        )
        teacher_result, rejected_reasons = _teacher_for_context(
            context,
            action_semantics_id=action_semantics_id,
            actor_safe_view=actor_safe_view,
        )
        evaluation = _evaluate_teacher_selected(context, action_semantics_id, teacher_result)
        edge_deltas = _edge_delta_targets(context, evaluation, action_semantics_id)
        actor_target_view = build_stage22_actor_target_view(
            action_semantics_id=action_semantics_id,
            actor_safe_view=actor_safe_view,
            teacher_result=teacher_result,
            rejected_reasons_by_edge=rejected_reasons,
        )
        critic_target_view = build_stage22_critic_target_view(
            action_semantics_id=action_semantics_id,
            teacher_result=teacher_result,
            edge_delta_rows=(target.to_dict() for target in edge_deltas),
        )
        rows.append(
            Stage22EvidenceRow(
                evidence_id=f"stage22:{action_semantics_id}:{context.fixture.fixture_id}:t{context.time_step}",
                action_semantics_id=action_semantics_id,
                scenario_id=evaluation.scenario_id,
                time_step=context.time_step,
                topology_id=evaluation.topology_id,
                topology_name="objective_aware_teacher_projected",
                evaluator_id=evaluation.evaluator_id,
                physics_regime_id=evaluation.physics_regime_id,
                protocol_model_id=evaluation.protocol_model_id,
                objective_contract_id=evaluation.objective_contract_id,
                selected_directed_edges=evaluation.selected_directed_edges,
                selected_physical_edges=evaluation.selected_physical_edges,
                consensus_success_probability=float(evaluation.metrics["consensus_success_probability"]),
                per_primary_reliability=evaluation.per_primary_reliability,
                latency=float(evaluation.metrics["latency"]),
                energy=float(evaluation.metrics["energy"]),
                feasible_under_tau_requirement=(
                    float(evaluation.metrics["consensus_success_probability"])
                    >= STAGE21_TAU_REQUIREMENT_MIN
                ),
                actor_safe_view=actor_safe_view,
                actor_target_view=actor_target_view,
                critic_target_view=critic_target_view,
                diagnostics_view={
                    "source_fixture": context.fixture.fixture_id,
                    "sequence_id": context.sequence_id,
                    "time_step": context.time_step,
                    "topology_name": "objective_aware_teacher_projected",
                    "action_semantics_id": action_semantics_id,
                    "teacher_summary": teacher_result.to_summary(),
                    "evaluator_id": evaluation.evaluator_id,
                    "fallback_used": False,
                },
            )
        )
        previous_selected_by_key[
            (str(context.sequence_id or ""), action_semantics_id, context.time_step)
        ] = evaluation.selected_physical_edges
    return Stage22ActionSemanticsDataset(
        dataset_id=f"{STAGE22_DATASET_ID}:{action_semantics_id}",
        action_semantics_id=action_semantics_id,
        rows=tuple(rows),
        readiness=_readiness(rows),
    )


def _teacher_for_context(
    context: Stage21EvaluationContext,
    *,
    action_semantics_id: str,
    actor_safe_view: tuple[Mapping[str, object], ...],
) -> tuple[object, dict[str, tuple[str, ...]]]:
    if action_semantics_id != UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID:
        raise Stage22EvidenceViolation("only selected physical-link semantics are executable")
    candidates = _physical_teacher_candidates(context, actor_safe_view)

    def evaluate(candidate: TeacherCandidate) -> Mapping[str, object]:
        result = context.evaluator.evaluate(candidate.selected_physical_edges)
        return {
            "consensus_success_probability": result.metrics["consensus_success_probability"],
            "latency": result.metrics["latency"],
            "energy": result.metrics["energy"],
        }

    teacher = select_objective_aware_teacher(
        action_semantics_id=action_semantics_id,
        candidates=candidates,
        evaluate_candidate=evaluate,
        search_method="small_graph_exhaustive_physical_objective_search",
    )
    return teacher, _projected_rejection_reasons(actor_safe_view, teacher.best_evaluation.candidate)


def _evaluate_teacher_selected(
    context: Stage21EvaluationContext,
    action_semantics_id: str,
    teacher_result,
) -> Stage22ObjectiveEvaluation:
    candidate = teacher_result.best_evaluation.candidate
    if action_semantics_id != UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID:
        raise Stage22EvidenceViolation("only selected physical-link semantics are executable")
    evaluation = context.evaluator.evaluate(
        candidate.selected_physical_edges,
        topology_id=f"stage22:{action_semantics_id}:{context.fixture.fixture_id}:teacher:t{context.time_step}",
    )
    return _physical_stage22_evaluation(
        action_semantics_id=action_semantics_id,
        selected_physical_edges=candidate.selected_physical_edges,
        evaluation=evaluation,
    )


def _physical_stage22_evaluation(
    *,
    action_semantics_id: str,
    selected_physical_edges: tuple[str, ...],
    evaluation: Stage21ObjectiveEvaluation,
) -> Stage22ObjectiveEvaluation:
    return Stage22ObjectiveEvaluation(
        scenario_id=evaluation.scenario_id,
        topology_id=evaluation.topology_id,
        action_semantics_id=action_semantics_id,
        selected_directed_edges=(),
        selected_physical_edges=selected_physical_edges,
        metrics=evaluation.metrics,
        per_primary_reliability=evaluation.per_primary_reliability,
        evaluator_id=STAGE22_PHYSICAL_EVALUATOR_ID,
        physics_regime_id=evaluation.physics_regime_id,
        protocol_model_id=evaluation.protocol_model_id,
        objective_contract_id=evaluation.objective_contract_id,
        diagnostics={
            **dict(evaluation.diagnostics),
            "action_semantics_id": action_semantics_id,
            "selected_physical_edge_count": len(selected_physical_edges),
            "fallback_used": False,
        },
    )


def _physical_teacher_candidates(
    context: Stage21EvaluationContext,
    actor_safe_view: tuple[Mapping[str, object], ...],
) -> tuple[TeacherCandidate, ...]:
    edge_ids = tuple(context.graph.edge_ids)
    raw_candidates = [
        ("empty", ()),
        ("full_graph_projected", edge_ids),
        ("greedy_reliability_projected", tuple(context.topology_variants["greedy_reliability_raw"])),
    ]
    for count in range(1, len(edge_ids) + 1):
        for combo in combinations(edge_ids, count):
            raw_candidates.append((f"physical_combo_{count}_{len(raw_candidates)}", tuple(sorted(combo))))
    candidates: list[TeacherCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for candidate_id, raw_edges in raw_candidates:
        projected = _project_physical_candidate(actor_safe_view, raw_edges)
        if projected in seen and candidate_id not in {"full_graph_projected", "greedy_reliability_projected"}:
            continue
        seen.add(projected)
        candidates.append(
            TeacherCandidate(
                candidate_id=candidate_id,
                selected_directed_edges=(),
                selected_physical_edges=projected,
                source="assembler_projected_objective_search",
            )
        )
    return tuple(candidates)


def _project_physical_candidate(
    actor_safe_view: tuple[Mapping[str, object], ...],
    physical_edges: tuple[str, ...],
) -> tuple[str, ...]:
    from marl_topology.policies import (
        PHYSICAL_LINK_ASSEMBLER_ID,
        AssemblerConfig,
        PhysicalLinkConflictAwareAssembler,
        aggregate_endpoint_scores_to_physical_links,
    )
    from marl_topology.data.stage21_assembler_aware_targets import (
        rx_capacity_from_actor_safe_view,
        tx_capacity_from_actor_safe_view,
    )

    scores = edge_score_records_from_actor_safe_view(
        actor_safe_view,
        score_source="stage22_projected_physical_teacher_candidate",
        proposed_edge_ids=set(physical_edges),
        include_non_proposed=False,
    )
    physical_scores = aggregate_endpoint_scores_to_physical_links(scores)
    assembly = PhysicalLinkConflictAwareAssembler(
        AssemblerConfig(
            assembler_id=PHYSICAL_LINK_ASSEMBLER_ID,
            mode="physical_link_conflict_aware_greedy",
            tx_capacity=tx_capacity_from_actor_safe_view(actor_safe_view),
            rx_capacity=rx_capacity_from_actor_safe_view(actor_safe_view),
            deterministic=True,
        )
    ).assemble(
        physical_scores,
        candidate_constraints_from_actor_safe_view(actor_safe_view),
        metadata={},
    )
    return tuple(assembly.selected_physical_edges)


def _projected_rejection_reasons(
    actor_safe_view: tuple[Mapping[str, object], ...],
    teacher_candidate: TeacherCandidate,
) -> dict[str, tuple[str, ...]]:
    proposed_physical = set(teacher_candidate.selected_physical_edges)
    scores = edge_score_records_from_actor_safe_view(
        actor_safe_view,
        score_source="stage22_teacher_projection_diagnostic",
        proposed_edge_ids=proposed_physical,
        include_non_proposed=False,
    )
    assembler = build_stage21_conflict_aware_assembler(actor_safe_view)
    assembled = assembler.assemble(
        scores,
        candidate_constraints_from_actor_safe_view(actor_safe_view),
        metadata={"agent_roles": agent_roles_from_actor_safe_view(actor_safe_view)},
    )
    return {
        edge_id: tuple(reason.value for reason in reasons)
        for edge_id, reasons in assembled.rejection_reasons.items()
    }


def _source_learning_row(
    context: Stage21EvaluationContext,
    evaluation: Stage21ObjectiveEvaluation,
    topology_name: str,
):
    from marl_topology.data.stage21_objective_stack_evidence import _source_learning_evidence_row
    from marl_topology.policies import build_local_observations

    observations = build_local_observations(
        scene=context.fixture.scene,
        graph=context.graph,
        link_records=context.evaluator.link_records,
        time_step=context.time_step,
    )
    return _source_learning_evidence_row(
        evaluation=evaluation,
        topology_name=topology_name,
        observations=observations,
        context=context,
        learning_targets=(),
    )


def _edge_delta_targets(
    context: Stage21EvaluationContext,
    evaluation: Stage22ObjectiveEvaluation,
    action_semantics_id: str,
) -> tuple[EdgeDeltaTarget, ...]:
    selected = set(evaluation.selected_physical_edges)
    baseline_probability = float(evaluation.metrics["consensus_success_probability"])
    baseline_latency = float(evaluation.metrics["latency"])
    baseline_energy = float(evaluation.metrics["energy"])
    baseline_feasible = int(baseline_probability >= STAGE21_TAU_REQUIREMENT_MIN)
    targets: list[EdgeDeltaTarget] = []
    for edge_id in context.graph.edge_ids:
        changed = set(selected)
        if edge_id in changed:
            changed.remove(edge_id)
            action_type = "remove_edge"
        else:
            changed.add(edge_id)
            action_type = "add_edge"
        changed_eval = context.evaluator.evaluate(
            changed,
            topology_id=f"stage22:{action_semantics_id}:{evaluation.topology_id}:{action_type}:{edge_id}",
        )
        changed_probability = float(changed_eval.metrics["consensus_success_probability"])
        changed_latency = float(changed_eval.metrics["latency"])
        changed_energy = float(changed_eval.metrics["energy"])
        changed_feasible = int(changed_probability >= STAGE21_TAU_REQUIREMENT_MIN)
        targets.append(
            EdgeDeltaTarget(
                topology_id=evaluation.topology_id,
                edge_id=edge_id,
                action_type=action_type,
                delta_consensus_success_probability=changed_probability - baseline_probability,
                delta_latency=changed_latency - baseline_latency,
                delta_energy=changed_energy - baseline_energy,
                delta_feasibility=changed_feasible - baseline_feasible,
                delta_reward_surrogate_diagnostic={
                    "stage22_objective_diagnostic_only": 0.0,
                },
            )
        )
    return tuple(targets)


def _readiness(rows: list[Stage22EvidenceRow]) -> dict[str, object]:
    return {
        "row_count": len(rows),
        "tau_feasible_rows": sum(row.feasible_under_tau_requirement for row in rows),
        "all_rows_declare_action_semantics_id": all(bool(row.action_semantics_id) for row in rows),
        "all_rows_declare_evaluator_id": all(bool(row.evaluator_id) for row in rows),
        "uses_final_objective_stack": all(
            row.physics_regime_id == STAGE21_PHYSICS_REGIME_ID
            and row.protocol_model_id == STAGE21_PROTOCOL_MODEL_ID
            and row.objective_contract_id == STAGE21_OBJECTIVE_CONTRACT_ID
            for row in rows
        ),
        "fallback_used": False,
        "ready_for_selected_semantics_training": bool(rows),
    }


def _dataset_summary(dataset: Stage22ActionSemanticsDataset) -> dict[str, object]:
    feasible = sum(row.feasible_under_tau_requirement for row in dataset.rows)
    return {
        "dataset_id": dataset.dataset_id,
        "action_semantics_id": dataset.action_semantics_id,
        "row_count": dataset.row_count,
        "actor_edge_sample_count": dataset.actor_edge_sample_count,
        "tau_feasible_rate": feasible / dataset.row_count if dataset.row_count else 0.0,
        "readiness": dict(dataset.readiness),
    }
