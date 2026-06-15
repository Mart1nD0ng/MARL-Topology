"""Stage 16 learning evidence quality improvement.

This module builds an in-memory evidence fixture set and quality report. It
does not train models, create checkpoints, write result artifacts, or introduce
new policy architectures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from marl_topology.evaluation.fixture_stack import build_fixture_stack
from marl_topology.geometry3d import Point3D
from marl_topology.policies import PolicyBaselines, build_local_observations
from marl_topology.scenario import ScenarioFixture
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D
from marl_topology.topology.evaluator import TopologyEvaluation, TopologyEvaluator

from .learning_evidence import (
    EdgeDeltaTarget,
    LearningEvidenceDataset,
    LearningEvidenceRow,
    build_learning_evidence_row,
)
from .learning_evidence_quality import evaluate_learning_evidence_quality


STAGE16_STAGE_ID = "stage_16_learning_evidence_quality_improvement"
STAGE16_DATASET_ID = "stage16_learning_evidence_quality_dataset_v1"
STAGE16_VERDICT = "stage16_evidence_quality_improved_scale_up_training_blocked"
STAGE16_RECOMMENDED_NEXT_TASK = (
    "stage_16_continue_evidence_expansion_before_rerun_stage11_to_stage15"
)
STAGE16_SEQUENCE_ID = "stage16_real_mobility_actor_safe_sequence_v1"
STAGE16_TAU_REQUIREMENT_MIN = 0.9
NEAR_THRESHOLD_BAND = 0.05


@dataclass(frozen=True, slots=True)
class Stage16LearningEvidenceBuild:
    """Stage 16 in-memory dataset and data-quality report."""

    dataset: LearningEvidenceDataset
    report: Mapping[str, object]


def build_stage16_learning_evidence_dataset() -> LearningEvidenceDataset:
    """Build Stage 16 evidence rows without training or artifact writes."""

    rows: list[LearningEvidenceRow] = []
    all_targets: list[EdgeDeltaTarget] = []
    for fixture, time_step, sequence_id in iter_stage16_evidence_scenario_fixtures():
        stack = build_fixture_stack(fixture)
        observations = build_local_observations(
            scene=stack.scene,
            graph=stack.graph,
            link_records=stack.evaluator.link_records,
            time_step=time_step,
        )
        variants = _stage16_topology_variants(stack.evaluator)
        oracle_result = stack.oracle.solve(random_seed=16)
        if oracle_result.evaluation is not None:
            variants["oracle_candidate"] = oracle_result.evaluation.selected_edge_ids

        latency_reference_s, energy_reference_j = _normalization_references(
            stack.evaluator,
            variants.values(),
        )
        oracle_evaluation = oracle_result.evaluation
        for topology_name, selected_edges in variants.items():
            selected = tuple(sorted(set(selected_edges)))
            evaluation = stack.evaluator.evaluate(
                selected,
                topology_id=f"stage16:{fixture.fixture_id}:{topology_name}:t{time_step}",
            )
            edge_targets = _build_stage16_edge_delta_targets(
                stack.evaluator,
                selected,
                tau_requirement_min=STAGE16_TAU_REQUIREMENT_MIN,
                topology_id=evaluation.topology_id,
                latency_reference_s=latency_reference_s,
                energy_reference_j=energy_reference_j,
            )
            all_targets.extend(edge_targets)
            rows.append(
                build_learning_evidence_row(
                    evaluation,
                    topology_name=topology_name,
                    observations=observations,
                    graph_node_ids=stack.graph.node_ids,
                    candidate_edge_ids=stack.graph.edge_ids,
                    tau_requirement_min=STAGE16_TAU_REQUIREMENT_MIN,
                    learning_targets=(target.to_dict() for target in edge_targets),
                    diagnostics=_row_diagnostics(
                        fixture=fixture,
                        evaluation=evaluation,
                        evaluator=stack.evaluator,
                        topology_name=topology_name,
                        oracle_evaluation=oracle_evaluation,
                        time_step=time_step,
                        sequence_id=sequence_id,
                    ),
                )
            )

    return LearningEvidenceDataset(
        dataset_id=STAGE16_DATASET_ID,
        rows=tuple(rows),
        edge_delta_targets=tuple(all_targets),
    )


def build_stage16_learning_evidence_report() -> Stage16LearningEvidenceBuild:
    """Build the Stage 16 quality report without executing training."""

    dataset = build_stage16_learning_evidence_dataset()
    base_quality = evaluate_learning_evidence_quality(dataset).to_dict()
    coverage = _coverage_summary(dataset)
    edge_delta = _edge_delta_rebuild_summary(dataset)
    contradiction = _identical_local_observation_contradictions(dataset)
    data_quality_answers = _data_quality_answers(
        coverage=coverage,
        contradiction=contradiction,
    )
    report = {
        "stage": STAGE16_STAGE_ID,
        "verdict": STAGE16_VERDICT,
        "dataset_id": dataset.dataset_id,
        "tau_requirement_min": STAGE16_TAU_REQUIREMENT_MIN,
        "row_count": dataset.row_count,
        "target_count": dataset.target_count,
        "evidence_coverage": coverage,
        "edge_delta_target_rebuild": edge_delta,
        "actor_observation_label_contradictions": contradiction,
        "base_quality_report": base_quality,
        "data_quality_answers": data_quality_answers,
        "scale_up_training_allowed": False,
        "training_execution_allowed": False,
        "ppo_mappo_allowed": False,
        "coma_allowed": False,
        "transformer_allowed": False,
        "checkpoint_creation_allowed": False,
        "artifact_written": False,
        "model_expansion_allowed": False,
        "v5_migration_performed": False,
        "recommended_next_task": STAGE16_RECOMMENDED_NEXT_TASK,
    }
    return Stage16LearningEvidenceBuild(dataset=dataset, report=report)


def iter_stage16_evidence_scenario_fixtures() -> tuple[
    tuple[ScenarioFixture, int, str | None],
    ...
]:
    """Return deterministic Stage 16 fixtures, including real time-step scenes."""

    static: tuple[tuple[ScenarioFixture, int, str | None], ...] = tuple(
        (fixture, 0, None)
        for fixture in (
            _stage16_tau_feasible_fixture(),
            _stage16_near_threshold_fixture(),
            _stage16_hard_infeasible_fixture(),
            _stage16_sparse_full_resource_fixture(),
            _stage16_center_primary_fixture(),
            _stage16_weak_primary_fixture(),
            _stage16_interference_penalty_proxy_fixture(),
        )
    )
    sequence = tuple(
        (fixture, time_step, STAGE16_SEQUENCE_ID)
        for time_step, fixture in enumerate(_stage16_real_multistep_fixtures())
    )
    return (*static, *sequence)


def _stage16_topology_variants(
    evaluator: TopologyEvaluator,
) -> dict[str, tuple[str, ...]]:
    sparse_quorum = _leader_quorum_edges(evaluator)
    single_best = sparse_quorum[:1]
    return {
        "empty": (),
        "single_best_edge": single_best,
        "sparse_quorum": sparse_quorum,
        "greedy": PolicyBaselines.greedy_reliability(
            evaluator.graph,
            evaluator.link_records,
        ).edge_ids,
        "full_graph": PolicyBaselines.full(evaluator.graph).edge_ids,
    }


def _leader_quorum_edges(evaluator: TopologyEvaluator) -> tuple[str, ...]:
    leader = evaluator.graph.node_ids[0]
    incident = [
        edge.edge_id
        for edge in evaluator.graph.edges
        if edge.node_u == leader or edge.node_v == leader
    ]
    ranked = sorted(
        incident,
        key=lambda edge_id: (
            -evaluator.link_records[edge_id].link_success_probability,
            evaluator.link_records[edge_id].energy_j,
            edge_id,
        ),
    )
    target_edges = max(0, evaluator.consensus_config.quorum_size - 1)
    if len(ranked) >= target_edges:
        return tuple(sorted(ranked[:target_edges]))
    return tuple(sorted(PolicyBaselines.greedy_reliability(evaluator.graph, evaluator.link_records).edge_ids))


def _build_stage16_edge_delta_targets(
    evaluator: TopologyEvaluator,
    selected_edge_ids: Iterable[str],
    *,
    tau_requirement_min: float,
    topology_id: str,
    latency_reference_s: float,
    energy_reference_j: float,
) -> tuple[EdgeDeltaTarget, ...]:
    selected = set(selected_edge_ids)
    baseline = evaluator.evaluate(selected, topology_id=f"{topology_id}:baseline")
    targets: list[EdgeDeltaTarget] = []
    for edge_id in evaluator.graph.edge_ids:
        targets.append(
            _target_for_change(
                evaluator,
                baseline=baseline,
                changed=selected,
                topology_id=topology_id,
                edge_id=edge_id,
                action_type="keep_edge",
                tau_requirement_min=tau_requirement_min,
                latency_reference_s=latency_reference_s,
                energy_reference_j=energy_reference_j,
            )
        )
        changed = set(selected)
        if edge_id in selected:
            changed.remove(edge_id)
            action_type = "remove_edge"
        else:
            changed.add(edge_id)
            action_type = "add_edge"
        targets.append(
            _target_for_change(
                evaluator,
                baseline=baseline,
                changed=changed,
                topology_id=topology_id,
                edge_id=edge_id,
                action_type=action_type,
                tau_requirement_min=tau_requirement_min,
                latency_reference_s=latency_reference_s,
                energy_reference_j=energy_reference_j,
            )
        )
    return tuple(targets)


def _target_for_change(
    evaluator: TopologyEvaluator,
    *,
    baseline: TopologyEvaluation,
    changed: set[str],
    topology_id: str,
    edge_id: str,
    action_type: str,
    tau_requirement_min: float,
    latency_reference_s: float,
    energy_reference_j: float,
) -> EdgeDeltaTarget:
    changed_evaluation = evaluator.evaluate(
        changed,
        topology_id=f"{topology_id}:{action_type}:{edge_id}",
    )
    baseline_probability = float(baseline.metrics["consensus_success_probability"])
    changed_probability = float(changed_evaluation.metrics["consensus_success_probability"])
    baseline_latency = float(baseline.metrics["latency"])
    changed_latency = float(changed_evaluation.metrics["latency"])
    baseline_energy = float(baseline.metrics["energy"])
    changed_energy = float(changed_evaluation.metrics["energy"])
    baseline_components = _surrogate_components(
        baseline,
        tau_requirement_min=tau_requirement_min,
        latency_reference_s=latency_reference_s,
        energy_reference_j=energy_reference_j,
    )
    changed_components = _surrogate_components(
        changed_evaluation,
        tau_requirement_min=tau_requirement_min,
        latency_reference_s=latency_reference_s,
        energy_reference_j=energy_reference_j,
    )
    return EdgeDeltaTarget(
        topology_id=topology_id,
        edge_id=edge_id,
        action_type=action_type,
        delta_consensus_success_probability=changed_probability - baseline_probability,
        delta_latency=changed_latency - baseline_latency,
        delta_energy=changed_energy - baseline_energy,
        delta_feasibility=int(changed_probability >= tau_requirement_min)
        - int(baseline_probability >= tau_requirement_min),
        delta_reward_surrogate_diagnostic={
            key: changed_components[key] - baseline_components[key]
            for key in baseline_components
        },
    )


def _surrogate_components(
    evaluation: TopologyEvaluation,
    *,
    tau_requirement_min: float,
    latency_reference_s: float,
    energy_reference_j: float,
) -> dict[str, float]:
    probability = float(evaluation.metrics["consensus_success_probability"])
    reliability_gap_penalty = max(0.0, tau_requirement_min - probability) ** 2
    return {
        "delta_reliability_gap_penalty": reliability_gap_penalty,
        "delta_normalized_latency": float(evaluation.metrics["latency"]) / latency_reference_s,
        "delta_normalized_energy": float(evaluation.metrics["energy"]) / energy_reference_j,
        "delta_component_sum_diagnostic": reliability_gap_penalty
        + float(evaluation.metrics["latency"]) / latency_reference_s
        + float(evaluation.metrics["energy"]) / energy_reference_j,
    }


def _normalization_references(
    evaluator: TopologyEvaluator,
    topology_sets: Iterable[Iterable[str]],
) -> tuple[float, float]:
    evaluations = [evaluator.evaluate(tuple(edges)) for edges in topology_sets]
    return (
        max(1e-12, max(float(row.metrics["latency"]) for row in evaluations)),
        max(1e-12, max(float(row.metrics["energy"]) for row in evaluations)),
    )


def _row_diagnostics(
    *,
    fixture: ScenarioFixture,
    evaluation: TopologyEvaluation,
    evaluator: TopologyEvaluator,
    topology_name: str,
    oracle_evaluation: TopologyEvaluation | None,
    time_step: int,
    sequence_id: str | None,
) -> dict[str, object]:
    probability = float(evaluation.metrics["consensus_success_probability"])
    selected = tuple(evaluation.selected_edge_ids)
    tags = set(_fixture_tags(fixture.fixture_id))
    if probability >= STAGE16_TAU_REQUIREMENT_MIN:
        tags.add("tau_requirement_min_0_9_feasible_example")
    if abs(probability - STAGE16_TAU_REQUIREMENT_MIN) <= NEAR_THRESHOLD_BAND:
        tags.add("near_threshold_example")
    if topology_name == "sparse_quorum" and probability >= STAGE16_TAU_REQUIREMENT_MIN:
        tags.add("sparse_feasible_topology")
    if topology_name == "full_graph" and "resource_dominated_family" in tags:
        tags.add("full_graph_resource_dominated_topology")
    if sequence_id is not None:
        tags.add("real_multistep_actor_safe_sequence")
    if topology_name in {"empty", "single_best_edge"} and "hard_infeasible_family" in tags:
        tags.add("infeasible_hard_example")
    return {
        "scenario_family": fixture.description,
        "fixture_id": fixture.fixture_id,
        "topology_family": _topology_family(topology_name, selected, evaluator.graph.edge_ids),
        "topology_label_role": _topology_label_role(topology_name),
        "coverage_tags": tuple(sorted(tags)),
        "is_full_graph_baseline": topology_name == "full_graph"
        or set(selected) == set(evaluator.graph.edge_ids),
        "is_oracle_candidate": topology_name == "oracle_candidate",
        "is_deployment_actor_input": False,
        "is_real_multistep_actor_safe_sequence": sequence_id is not None,
        "sequence_id": sequence_id,
        "time_step": time_step,
        "primary_position_role": _primary_position_role(fixture.fixture_id),
        "oracle_gap": _row_oracle_gap(evaluation, oracle_evaluation),
        "tau_requirement_min": STAGE16_TAU_REQUIREMENT_MIN,
    }


def _topology_family(
    topology_name: str,
    selected_edges: tuple[str, ...],
    full_edge_ids: tuple[str, ...],
) -> str:
    if topology_name == "empty" or not selected_edges:
        return "weak_or_disconnected_baseline"
    if topology_name == "full_graph" or set(selected_edges) == set(full_edge_ids):
        return "dense_full_graph_baseline"
    if topology_name == "oracle_candidate":
        return "oracle_candidate_diagnostic"
    if topology_name in {"sparse_quorum", "greedy"}:
        return "sparse_candidate"
    return "partial_candidate"


def _topology_label_role(topology_name: str) -> str:
    if topology_name == "oracle_candidate":
        return "oracle_training_diagnostic_only"
    if topology_name in {"sparse_quorum", "greedy"}:
        return "heuristic_training_diagnostic_only"
    return "baseline_evaluation"


def _row_oracle_gap(
    evaluation: TopologyEvaluation,
    oracle_evaluation: TopologyEvaluation | None,
) -> dict[str, float | None]:
    if oracle_evaluation is None:
        return {
            "delta_consensus_success_probability_to_oracle": None,
            "delta_latency_to_oracle": None,
            "delta_energy_to_oracle": None,
        }
    return {
        "delta_consensus_success_probability_to_oracle": float(
            oracle_evaluation.metrics["consensus_success_probability"]
        )
        - float(evaluation.metrics["consensus_success_probability"]),
        "delta_latency_to_oracle": float(evaluation.metrics["latency"])
        - float(oracle_evaluation.metrics["latency"]),
        "delta_energy_to_oracle": float(evaluation.metrics["energy"])
        - float(oracle_evaluation.metrics["energy"]),
    }


def _fixture_tags(fixture_id: str) -> tuple[str, ...]:
    tags: list[str] = []
    if "tau_feasible" in fixture_id:
        tags.append("tau_feasible_family")
    if "near_threshold" in fixture_id:
        tags.append("near_threshold_family")
    if "hard_infeasible" in fixture_id:
        tags.extend(("hard_infeasible_family", "infeasible_hard_example"))
    if "sparse_full_resource" in fixture_id:
        tags.extend(("resource_dominated_family", "sparse_full_contrast_family"))
    if "interference_penalty_proxy" in fixture_id:
        tags.extend(("interference_penalty_example", "resource_dominated_family"))
    if "center_primary" in fixture_id:
        tags.append("center_primary_example")
    if "weak_primary" in fixture_id:
        tags.append("weak_primary_example")
    return tuple(tags)


def _primary_position_role(fixture_id: str) -> str | None:
    if "center_primary" in fixture_id:
        return "center_primary"
    if "weak_primary" in fixture_id:
        return "weak_primary"
    return None


def _coverage_summary(dataset: LearningEvidenceDataset) -> dict[str, object]:
    tag_counts: Counter[str] = Counter()
    sequence_steps: dict[str, set[int]] = defaultdict(set)
    for row in dataset.rows:
        for tag in row.diagnostics.get("coverage_tags", ()):
            tag_counts[str(tag)] += 1
        sequence_id = row.diagnostics.get("sequence_id")
        if sequence_id is not None:
            sequence_steps[str(sequence_id)].add(int(row.diagnostics.get("time_step", 0)))

    sparse_full = _sparse_full_tradeoff(dataset)
    weak_center = _weak_center_primary_contrast(dataset)
    return {
        "coverage_tag_counts": dict(sorted(tag_counts.items())),
        "tau_ge_0_9_feasible_row_count": sum(
            1
            for row in dataset.rows
            if row.consensus_success_probability >= STAGE16_TAU_REQUIREMENT_MIN
        ),
        "near_threshold_row_count": tag_counts["near_threshold_example"],
        "infeasible_hard_row_count": tag_counts["infeasible_hard_example"],
        "sparse_feasible_topology_count": tag_counts["sparse_feasible_topology"],
        "full_graph_resource_dominated_topology_count": tag_counts[
            "full_graph_resource_dominated_topology"
        ],
        "interference_penalty_example_count": tag_counts["interference_penalty_example"],
        "weak_primary_row_count": tag_counts["weak_primary_example"],
        "center_primary_row_count": tag_counts["center_primary_example"],
        "real_multistep_sequence_count": len(sequence_steps),
        "real_multistep_time_steps": {
            sequence_id: sorted(steps) for sequence_id, steps in sequence_steps.items()
        },
        "sparse_full_resource_tradeoff": sparse_full,
        "weak_center_primary_contrast": weak_center,
    }


def _sparse_full_tradeoff(dataset: LearningEvidenceDataset) -> dict[str, object]:
    rows_by_scenario: dict[str, dict[str, LearningEvidenceRow]] = defaultdict(dict)
    for row in dataset.rows:
        rows_by_scenario[row.scenario_id][row.topology_name] = row
    tradeoffs: list[dict[str, object]] = []
    for scenario_id, rows in rows_by_scenario.items():
        sparse = rows.get("sparse_quorum")
        full = rows.get("full_graph")
        if sparse is None or full is None:
            continue
        if not sparse.feasible_under_tau_requirement:
            continue
        if sparse.latency <= full.latency and sparse.energy <= full.energy:
            tradeoffs.append(
                {
                    "scenario_id": scenario_id,
                    "sparse_probability": sparse.consensus_success_probability,
                    "full_probability": full.consensus_success_probability,
                    "sparse_latency": sparse.latency,
                    "full_latency": full.latency,
                    "sparse_energy": sparse.energy,
                    "full_energy": full.energy,
                    "full_edge_count": len(full.selected_edges),
                    "sparse_edge_count": len(sparse.selected_edges),
                }
            )
    return {
        "tradeoff_count": len(tradeoffs),
        "tradeoffs": tradeoffs,
        "has_sparse_feasible_resource_advantage": bool(tradeoffs),
    }


def _weak_center_primary_contrast(dataset: LearningEvidenceDataset) -> dict[str, object]:
    sparse_rows = [
        row
        for row in dataset.rows
        if row.topology_name == "sparse_quorum"
        and row.diagnostics.get("primary_position_role") in {"weak_primary", "center_primary"}
    ]
    by_role = {str(row.diagnostics["primary_position_role"]): row for row in sparse_rows}
    weak = by_role.get("weak_primary")
    center = by_role.get("center_primary")
    if weak is None or center is None:
        return {"has_contrast": False, "reason": "missing weak or center primary row"}
    return {
        "has_contrast": True,
        "weak_primary_probability": weak.consensus_success_probability,
        "center_primary_probability": center.consensus_success_probability,
        "probability_delta_center_minus_weak": center.consensus_success_probability
        - weak.consensus_success_probability,
        "weak_primary_feasible": weak.feasible_under_tau_requirement,
        "center_primary_feasible": center.feasible_under_tau_requirement,
    }


def _edge_delta_rebuild_summary(dataset: LearningEvidenceDataset) -> dict[str, object]:
    action_counts = Counter(target.action_type for target in dataset.edge_delta_targets)
    helpful_targets = [
        target
        for target in dataset.edge_delta_targets
        if _target_helpfulness(target) == "helpful"
    ]
    harmful_targets = [
        target
        for target in dataset.edge_delta_targets
        if _target_helpfulness(target) == "harmful"
    ]
    probability_deltas = [
        target.delta_consensus_success_probability for target in dataset.edge_delta_targets
    ]
    latency_deltas = [target.delta_latency for target in dataset.edge_delta_targets]
    energy_deltas = [target.delta_energy for target in dataset.edge_delta_targets]
    feasibility_deltas = [target.delta_feasibility for target in dataset.edge_delta_targets]
    return {
        "action_type_counts": dict(sorted(action_counts.items())),
        "positive_probability_delta_count": sum(1 for value in probability_deltas if value > 0.0),
        "negative_probability_delta_count": sum(1 for value in probability_deltas if value < 0.0),
        "zero_probability_delta_count": sum(1 for value in probability_deltas if value == 0.0),
        "helpful_edge_count": len(helpful_targets),
        "harmful_edge_count": len(harmful_targets),
        "keep_edge_count": action_counts.get("keep_edge", 0),
        "feasibility_changing_delta_count": sum(1 for value in feasibility_deltas if value != 0),
        "positive_feasibility_delta_count": sum(1 for value in feasibility_deltas if value > 0),
        "negative_feasibility_delta_count": sum(1 for value in feasibility_deltas if value < 0),
        "delta_consensus_success_probability_nonzero_count": sum(
            1 for value in probability_deltas if value != 0.0
        ),
        "delta_latency_nonzero_count": sum(1 for value in latency_deltas if value != 0.0),
        "delta_energy_nonzero_count": sum(1 for value in energy_deltas if value != 0.0),
        "delta_surrogate_diagnostic_nonzero_count": sum(
            1
            for target in dataset.edge_delta_targets
            if any(value != 0.0 for value in target.delta_reward_surrogate_diagnostic.values())
        ),
        "positive_negative_balance": {
            "helpful_to_harmful_ratio": (
                len(helpful_targets) / len(harmful_targets) if harmful_targets else None
            ),
            "has_helpful_and_harmful_edges": bool(helpful_targets and harmful_targets),
        },
        "rare_safety_samples": _rare_safety_samples(dataset),
        "oracle_gap": _oracle_gap(dataset),
    }


def _target_helpfulness(target: EdgeDeltaTarget) -> str:
    if target.action_type == "keep_edge":
        return "neutral"
    component_sum_delta = float(
        target.delta_reward_surrogate_diagnostic.get("delta_component_sum_diagnostic", 0.0)
    )
    if target.delta_feasibility > 0 or target.delta_consensus_success_probability > 0.0:
        return "helpful"
    if target.delta_feasibility < 0 or target.delta_consensus_success_probability < 0.0:
        return "harmful"
    if component_sum_delta < 0.0:
        return "helpful"
    if component_sum_delta > 0.0:
        return "harmful"
    return "neutral"


def _rare_safety_samples(dataset: LearningEvidenceDataset) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for row in dataset.rows:
        reason = _rare_reason(row)
        if reason is None:
            continue
        rows.append(
            {
                "scenario_id": row.scenario_id,
                "topology_name": row.topology_name,
                "consensus_success_probability": row.consensus_success_probability,
                "latency": row.latency,
                "energy": row.energy,
                "reason": reason,
            }
        )
    return {"rare_sample_count": len(rows), "rows": rows}


def _rare_reason(row: LearningEvidenceRow) -> str | None:
    tags = set(str(tag) for tag in row.diagnostics.get("coverage_tags", ()))
    if row.topology_name == "empty":
        return "empty_disconnected_topology"
    if "infeasible_hard_example" in tags and row.energy > 0.0:
        return "scheduled_but_requirement_infeasible_topology"
    if (
        row.diagnostics.get("is_full_graph_baseline")
        and not row.feasible_under_tau_requirement
        and row.energy > 0.0
    ):
        return "full_graph_not_feasible_under_tau_requirement"
    if "near_threshold_example" in tags:
        return "near_tau_boundary_sample"
    return None


def _oracle_gap(dataset: LearningEvidenceDataset) -> dict[str, object]:
    rows = []
    for row in dataset.rows:
        gap = dict(row.diagnostics.get("oracle_gap", {}))
        probability_gap = gap.get("delta_consensus_success_probability_to_oracle")
        if probability_gap is None:
            continue
        rows.append({"scenario_id": row.scenario_id, "topology_name": row.topology_name, **gap})
    return {
        "oracle_comparable_row_count": len(rows),
        "max_probability_gap": max(
            (float(row["delta_consensus_success_probability_to_oracle"]) for row in rows),
            default=None,
        ),
        "rows": rows,
    }


def _identical_local_observation_contradictions(
    dataset: LearningEvidenceDataset,
) -> dict[str, object]:
    labels_by_fingerprint: dict[tuple[object, ...], set[str]] = defaultdict(set)
    examples: list[dict[str, object]] = []
    for row in dataset.rows:
        edge_target_labels = _edge_target_labels(row.learning_targets)
        for actor_row in row.actor_safe_rows:
            for neighbor in actor_row["local_neighbor_observations"]:
                edge_id = str(neighbor.edge_id)
                if edge_id not in edge_target_labels:
                    continue
                fingerprint = (
                    actor_row["agent_id"],
                    edge_id,
                    round(float(neighbor.distance_3d_m), 6),
                    round(float(neighbor.link_success_probability), 6),
                    round(float(neighbor.estimated_link_latency_s), 12),
                    round(float(neighbor.estimated_link_energy_j), 6),
                )
                labels_by_fingerprint[fingerprint].add(edge_target_labels[edge_id])

    for fingerprint, labels in labels_by_fingerprint.items():
        if len(labels) <= 1:
            continue
        examples.append(
            {
                "agent_id": fingerprint[0],
                "edge_id": fingerprint[1],
                "labels": sorted(labels),
            }
        )
        if len(examples) >= 5:
            break
    contradictory_count = sum(1 for labels in labels_by_fingerprint.values() if len(labels) > 1)
    return {
        "contradictory_fingerprint_count": contradictory_count,
        "fingerprints_checked": len(labels_by_fingerprint),
        "has_identical_local_observations_with_contradictory_labels": contradictory_count > 0,
        "example_count": len(examples),
        "examples": examples,
    }


def _edge_target_labels(
    learning_targets: Iterable[Mapping[str, object]],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for target in learning_targets:
        action_type = str(target.get("action_type", ""))
        edge_id = str(target.get("edge_id", ""))
        if action_type == "keep_edge" or not edge_id:
            continue
        probability_delta = float(target.get("delta_consensus_success_probability", 0.0))
        feasibility_delta = int(target.get("delta_feasibility", 0))
        if feasibility_delta > 0 or probability_delta > 0.0:
            labels[edge_id] = "helpful"
        elif feasibility_delta < 0 or probability_delta < 0.0:
            labels[edge_id] = "harmful"
        else:
            labels[edge_id] = "neutral"
    return labels


def _data_quality_answers(
    *,
    coverage: Mapping[str, object],
    contradiction: Mapping[str, object],
) -> dict[str, object]:
    tau_feasible_count = int(coverage["tau_ge_0_9_feasible_row_count"])
    sparse_feasible_count = int(coverage["sparse_feasible_topology_count"])
    real_multistep_present = int(coverage["real_multistep_sequence_count"]) > 0
    contradictions_present = bool(
        contradiction["has_identical_local_observations_with_contradictory_labels"]
    )
    return {
        "current_data_sufficient_to_continue_supervised_training": False,
        "current_data_sufficient_for_smoke_or_dry_run_only": True,
        "tau_ge_0_9_feasible_positive_examples_present": tau_feasible_count > 0,
        "sufficient_tau_ge_0_9_feasible_positive_examples": tau_feasible_count >= 20,
        "sparse_feasible_examples_present": sparse_feasible_count > 0,
        "sufficient_sparse_feasible_examples": sparse_feasible_count >= 8,
        "real_multistep_evidence_present": real_multistep_present,
        "actor_observable_features_sufficient_to_disambiguate_edge_labels": (
            not contradictions_present
        ),
        "identical_local_observations_with_contradictory_labels": contradictions_present,
        "recommended_next_step": "continue_evidence_expansion_before_rerun_stage11_to_stage15",
        "rerun_stage11_to_stage15_now_recommended": False,
        "scale_up_training_still_blocked": True,
    }


def _stage16_tau_feasible_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_tau_feasible_examples",
        "Stage 16 close geometry with tau >= 0.9 feasible sparse examples.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(5.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 5.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(5.0, 5.0, 1.5)),
        ),
        max_candidate_distance_m=20.0,
        link_reference_distance_m=500.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage16_near_threshold_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_near_threshold_examples",
        "Stage 16 near tau boundary geometry for threshold-sensitive evidence.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(11.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 11.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(11.0, 11.0, 1.5)),
        ),
        max_candidate_distance_m=25.0,
        link_reference_distance_m=112.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage16_hard_infeasible_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_hard_infeasible_examples",
        "Stage 16 hard infeasible split where quorum cannot be reached.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(22.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(180.0, 0.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(220.0, 0.0, 1.5)),
        ),
        max_candidate_distance_m=35.0,
        link_reference_distance_m=120.0,
        expected_oracle_status="infeasible",
        expected_full_graph_success=False,
    )


def _stage16_sparse_full_resource_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_sparse_full_resource_dominated",
        "Stage 16 sparse feasible topology with dense full graph resource overhead.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(8.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 8.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(8.0, 8.0, 1.5)),
            Node3D("veh_3", NodeKind.VEHICLE, Point3D(4.0, 4.0, 1.5)),
        ),
        max_candidate_distance_m=20.0,
        quorum_size=3,
        link_reference_distance_m=500.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage16_center_primary_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_center_primary_contrast",
        "Stage 16 centered primary with short local links.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(9.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(-9.0, 0.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(0.0, 9.0, 1.5)),
            Node3D("veh_3", NodeKind.VEHICLE, Point3D(0.0, -9.0, 1.5)),
        ),
        max_candidate_distance_m=25.0,
        link_reference_distance_m=160.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage16_weak_primary_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_weak_primary_contrast",
        "Stage 16 weak edge primary with follower cluster away from the leader.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(68.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(74.0, 4.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(80.0, -4.0, 1.5)),
            Node3D("veh_3", NodeKind.VEHICLE, Point3D(86.0, 0.0, 1.5)),
        ),
        max_candidate_distance_m=90.0,
        link_reference_distance_m=160.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage16_interference_penalty_proxy_fixture() -> ScenarioFixture:
    return _fixture(
        "stage16_interference_penalty_proxy",
        "Stage 16 dense topology resource/interference penalty proxy.",
        (
            Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
            Node3D("veh_0", NodeKind.VEHICLE, Point3D(6.0, 0.0, 1.5)),
            Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 6.0, 1.5)),
            Node3D("veh_2", NodeKind.VEHICLE, Point3D(6.0, 6.0, 1.5)),
            Node3D("veh_3", NodeKind.VEHICLE, Point3D(3.0, 9.0, 1.5)),
        ),
        max_candidate_distance_m=18.0,
        quorum_size=3,
        link_reference_distance_m=450.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage16_real_multistep_fixtures() -> tuple[ScenarioFixture, ...]:
    positions_by_step = (
        (
            Point3D(8.0, 0.0, 1.5),
            Point3D(0.0, 8.0, 1.5),
            Point3D(8.0, 8.0, 1.5),
        ),
        (
            Point3D(10.0, 0.0, 1.5),
            Point3D(0.0, 9.0, 1.5),
            Point3D(10.0, 8.0, 1.5),
        ),
        (
            Point3D(13.0, 0.0, 1.5),
            Point3D(0.0, 10.0, 1.5),
            Point3D(13.0, 8.0, 1.5),
        ),
    )
    fixtures = []
    for time_step, positions in enumerate(positions_by_step):
        fixtures.append(
            _fixture(
                f"stage16_real_multistep_actor_safe_t{time_step}",
                "Stage 16 real multi-step actor-safe sequence from deterministic mobility.",
                (
                    Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                    Node3D("veh_0", NodeKind.VEHICLE, positions[0]),
                    Node3D("veh_1", NodeKind.VEHICLE, positions[1]),
                    Node3D("veh_2", NodeKind.VEHICLE, positions[2]),
                ),
                max_candidate_distance_m=25.0,
                link_reference_distance_m=180.0,
                expected_oracle_status="feasible",
                expected_full_graph_success=True,
            )
        )
    return tuple(fixtures)


def _fixture(
    fixture_id: str,
    description: str,
    nodes: tuple[Node3D, ...],
    *,
    max_candidate_distance_m: float,
    link_reference_distance_m: float,
    expected_oracle_status: str,
    expected_full_graph_success: bool,
    quorum_size: int = 3,
) -> ScenarioFixture:
    return ScenarioFixture(
        fixture_id=fixture_id,
        description=description,
        scene=Scene3D(
            scenario_id=fixture_id,
            nodes=nodes,
            physics_regime="stage2_deterministic_distance",
        ),
        max_candidate_distance_m=max_candidate_distance_m,
        quorum_size=quorum_size,
        success_probability_threshold=0.45,
        deadline_s=0.01,
        link_reference_distance_m=link_reference_distance_m,
        expected_oracle_status=expected_oracle_status,
        expected_full_graph_success=expected_full_graph_success,
    )
