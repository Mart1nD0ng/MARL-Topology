"""Stage 7 completion dataset build and quality exit gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from marl_topology.evaluation.fixture_stack import build_fixture_stack
from marl_topology.geometry3d import Point3D
from marl_topology.policies import PolicyBaselines, build_local_observations
from marl_topology.scenario import ScenarioFixture
from marl_topology.scenario.scene import Node3D, NodeKind, Scene3D
from marl_topology.topology.evaluator import TopologyEvaluation, TopologyEvaluator
from marl_topology.training import build_valid_stage5_10_dry_run_manifest

from .learning_evidence import (
    STAGE7_0_ARTIFACT_SCOPE,
    EdgeDeltaTarget,
    EvidenceArtifactWriteResult,
    LearningEvidenceDataset,
    LearningEvidenceRow,
    build_learning_evidence_row,
    write_learning_evidence_artifact,
)
from .learning_evidence_quality import evaluate_learning_evidence_quality


STAGE7_COMPLETION_STAGE_ID = (
    "stage_7_completion_learning_evidence_dataset_build_quality_exit_gate"
)
STAGE7_COMPLETION_VERDICT = "stage7_completion_exit_gate_passed_training_blocked"
STAGE7_COMPLETION_DATASET_ID = "stage7_completion_learning_evidence_dataset_v1"
STAGE7_COMPLETION_RUN_ID = "stage7_completion_learning_evidence_dataset_v1"
STAGE7_COMPLETION_RECOMMENDED_NEXT_TASK = (
    "stage_8_0_actor_policy_interface_contract_with_owner_approval"
)
TAU_REQUIREMENT_MIN = 0.9
TOPOLOGY_VARIANT_NAMES = (
    "empty",
    "random",
    "sparse_heuristic",
    "full_graph",
    "greedy",
    "oracle_candidate",
)


@dataclass(frozen=True, slots=True)
class Stage7CompletionBuild:
    """Stage 7 completion dataset and quality report."""

    dataset: LearningEvidenceDataset
    report: Mapping[str, object]
    manifest: Mapping[str, object]


def build_stage7_completion_dataset() -> LearningEvidenceDataset:
    """Build scenario/topology evidence rows for the Stage 7 exit gate."""

    rows: list[LearningEvidenceRow] = []
    all_targets: list[EdgeDeltaTarget] = []
    for fixture in iter_stage7_completion_scenario_fixtures():
        stack = build_fixture_stack(fixture)
        observations = build_local_observations(
            scene=stack.scene,
            graph=stack.graph,
            link_records=stack.evaluator.link_records,
            time_step=0,
        )
        variants = _topology_variants(stack.evaluator, random_seed=7)
        oracle_result = stack.oracle.solve(random_seed=7)
        if oracle_result.evaluation is not None:
            variants["oracle_candidate"] = oracle_result.evaluation.selected_edge_ids

        oracle_evaluation = oracle_result.evaluation
        latency_reference_s, energy_reference_j = _normalization_references(
            stack.evaluator,
            variants.values(),
        )
        for topology_name in TOPOLOGY_VARIANT_NAMES:
            if topology_name not in variants:
                continue
            selected = tuple(sorted(set(variants[topology_name])))
            evaluation = stack.evaluator.evaluate(
                selected,
                topology_id=f"stage7:{fixture.fixture_id}:{topology_name}",
            )
            edge_targets = build_stage7_completion_edge_delta_targets(
                stack.evaluator,
                selected,
                tau_requirement_min=TAU_REQUIREMENT_MIN,
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
                    tau_requirement_min=TAU_REQUIREMENT_MIN,
                    learning_targets=(target.to_dict() for target in edge_targets),
                    diagnostics=_row_diagnostics(
                        fixture=fixture,
                        evaluator=stack.evaluator,
                        evaluation=evaluation,
                        topology_name=topology_name,
                        oracle_evaluation=oracle_evaluation,
                    ),
                )
            )

    return LearningEvidenceDataset(
        dataset_id=STAGE7_COMPLETION_DATASET_ID,
        rows=tuple(rows),
        edge_delta_targets=tuple(all_targets),
    )


def build_stage7_completion_report() -> Stage7CompletionBuild:
    """Build the Stage 7 completion report without writing artifacts."""

    dataset = build_stage7_completion_dataset()
    quality = evaluate_learning_evidence_quality(dataset)
    completion = _completion_quality(dataset, quality.to_dict())
    manifest = build_stage7_completion_manifest()
    return Stage7CompletionBuild(
        dataset=dataset,
        report=completion,
        manifest=manifest,
    )


def build_stage7_completion_manifest() -> dict[str, object]:
    """Build the manifest used for owner-approved evidence export."""

    return build_valid_stage5_10_dry_run_manifest(
        overrides={
            "run_id": STAGE7_COMPLETION_RUN_ID,
            "stage_id": STAGE7_COMPLETION_STAGE_ID,
            "owner_approval_id": "owner_approved_stage7_completion_evidence_export",
            "config_id": "stage7_completion_learning_evidence_config_v1",
            "scenario_set_id": "stage7_completion_scenario_set_v1",
            "split_id": "stage7_completion_all_rows_no_train_split",
            "seed": 7,
            "seed_group_id": "stage7_completion_deterministic_seed_group",
            "contract_ids": [
                "metric_governance",
                "objective_contract_stage5_0",
                "reward_surrogate_contract_stage5_2",
                "dec_pomdp_contract",
                "stage7_learning_evidence_dataset_contract",
                "stage7_completion_exit_gate",
            ],
            "artifact_scope": STAGE7_0_ARTIFACT_SCOPE,
        }
    )


def write_stage7_completion_artifact(
    *,
    project_root: str | Path,
    owner_approved_evidence_export: bool,
) -> dict[str, object]:
    """Write evidence-only completion artifacts after manifest validation."""

    build = build_stage7_completion_report()
    write_result = write_learning_evidence_artifact(
        build.dataset,
        manifest=build.manifest,
        project_root=project_root,
        owner_approved_evidence_export=owner_approved_evidence_export,
    )
    artifact_dir = Path(write_result.artifact_dir).resolve(strict=False)
    quality_path = artifact_dir / "quality_report.json"
    quality_path.write_bytes(
        json.dumps(build.report, indent=2, sort_keys=True).encode("utf-8")
    )
    return {
        **write_result.to_dict(),
        "quality_report_path": str(quality_path),
        "artifact_file_names": sorted(path.name for path in artifact_dir.iterdir()),
        "manifest_validated": True,
        "artifact_scope": STAGE7_0_ARTIFACT_SCOPE,
    }


def iter_stage7_completion_scenario_fixtures() -> tuple[ScenarioFixture, ...]:
    """Return deterministic scenario fixtures used for Stage 7 completion."""

    from marl_topology.scenario import iter_scenario_fixtures

    return (
        *iter_scenario_fixtures(),
        _stage7_close_reference_fixture(),
        _stage7_sparse_vs_full_tradeoff_fixture(),
    )


def build_stage7_completion_edge_delta_targets(
    evaluator: TopologyEvaluator,
    selected_edge_ids: Iterable[str],
    *,
    tau_requirement_min: float,
    topology_id: str,
    latency_reference_s: float,
    energy_reference_j: float,
) -> tuple[EdgeDeltaTarget, ...]:
    """Build add/remove/keep/swap edge-delta learning targets."""

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
        if edge_id in selected:
            changed = set(selected)
            changed.remove(edge_id)
            action_type = "remove_edge"
        else:
            changed = set(selected)
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

    selected_sorted = tuple(sorted(selected))
    unselected_sorted = tuple(edge for edge in evaluator.graph.edge_ids if edge not in selected)
    for remove_edge_id in selected_sorted:
        for add_edge_id in unselected_sorted:
            changed = set(selected)
            changed.remove(remove_edge_id)
            changed.add(add_edge_id)
            targets.append(
                _target_for_change(
                    evaluator,
                    baseline=baseline,
                    changed=changed,
                    topology_id=topology_id,
                    edge_id=f"{remove_edge_id}->{add_edge_id}",
                    action_type="swap_edge",
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


def _completion_quality(
    dataset: LearningEvidenceDataset,
    base_quality: Mapping[str, object],
) -> dict[str, object]:
    rows = [row.to_dict() for row in dataset.rows]
    class_balance = _class_balance(dataset)
    sparse_tradeoff = _sparse_vs_full_tradeoff(dataset)
    edge_distribution = _edge_delta_distribution(dataset)
    rare_safety = _rare_safety_samples(dataset)
    oracle_gap = _oracle_gap(dataset)
    scenario_diversity = _scenario_diversity(dataset)
    predictability = _actor_observable_predictability_risk(dataset)
    exit_criteria = {
        "scenario_topology_evidence_rows": dataset.row_count > 0,
        "oracle_or_heuristic_topology_label": _has_oracle_or_heuristic(dataset),
        "edge_delta_learning_target": dataset.target_count > 0
        and "swap_edge" in edge_distribution["action_type_counts"],
        "views_separated": _views_separated(dataset),
        "learnable_signal_present": _learnable_signal_present(dataset),
        "oracle_label_not_actor_input": _oracle_not_actor_input(dataset),
        "surrogate_not_actor_observation": _surrogate_not_actor_observation(dataset),
        "manifest_validated_artifact_write_supported": True,
        "no_training": True,
        "no_model_implementation": True,
    }
    stage7_exit_ready = all(exit_criteria.values()) and int(base_quality["blocking_issue_count"]) == 0
    return {
        "stage": STAGE7_COMPLETION_STAGE_ID,
        "verdict": STAGE7_COMPLETION_VERDICT if stage7_exit_ready else "stage7_completion_blocked",
        "dataset_id": dataset.dataset_id,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "row_count": dataset.row_count,
        "target_count": dataset.target_count,
        "required_topology_variants": list(TOPOLOGY_VARIANT_NAMES),
        "scenario_topology_evidence_rows": rows,
        "class_balance": class_balance,
        "sparse_vs_full_tradeoff": sparse_tradeoff,
        "edge_delta_distribution": edge_distribution,
        "rare_safety_samples": rare_safety,
        "oracle_gap": oracle_gap,
        "scenario_diversity": scenario_diversity,
        "actor_observable_predictability_risk": predictability,
        "base_quality_report": base_quality,
        "exit_criteria": exit_criteria,
        "stage7_exit_ready": stage7_exit_ready,
        "stage8_policy_interface_ready": stage7_exit_ready,
        "training_execution_ready": False,
        "checkpoint_creation_allowed": False,
        "model_implementation_allowed": False,
        "artifact_scope": STAGE7_0_ARTIFACT_SCOPE,
        "recommended_next_task": STAGE7_COMPLETION_RECOMMENDED_NEXT_TASK,
    }


def _topology_variants(
    evaluator: TopologyEvaluator,
    *,
    random_seed: int,
) -> dict[str, tuple[str, ...]]:
    graph = evaluator.graph
    baselines = {
        "empty": PolicyBaselines.empty(graph).edge_ids,
        "random": PolicyBaselines.random(graph, seed=random_seed).edge_ids,
        "sparse_heuristic": _sparse_heuristic_edges(evaluator),
        "full_graph": PolicyBaselines.full(graph).edge_ids,
        "greedy": PolicyBaselines.greedy_reliability(graph, evaluator.link_records).edge_ids,
    }
    return {name: tuple(sorted(edges)) for name, edges in baselines.items()}


def _sparse_heuristic_edges(evaluator: TopologyEvaluator) -> tuple[str, ...]:
    selected: set[str] = set()
    remaining = set(evaluator.graph.edge_ids)
    best_evaluation = evaluator.evaluate(selected)
    target_edges = max(0, evaluator.consensus_config.quorum_size - 1)
    while remaining and len(selected) < target_edges:
        candidates: list[tuple[tuple[float, float, float, float, str], str, TopologyEvaluation]] = []
        for edge_id in remaining:
            evaluation = evaluator.evaluate(selected | {edge_id})
            candidates.append(
                (
                    (
                        float(evaluation.consensus.reachable_node_count),
                        float(evaluation.metrics["consensus_success_probability"]),
                        -float(evaluation.metrics["latency"]),
                        -float(evaluation.metrics["energy"]),
                        edge_id,
                    ),
                    edge_id,
                    evaluation,
                )
            )
        _, chosen_edge, chosen_evaluation = max(candidates, key=lambda item: item[0])
        selected.add(chosen_edge)
        remaining.remove(chosen_edge)
        best_evaluation = chosen_evaluation
        if best_evaluation.consensus.consensus_success:
            break
    return tuple(sorted(selected))


def _normalization_references(
    evaluator: TopologyEvaluator,
    topology_sets: Iterable[Iterable[str]],
) -> tuple[float, float]:
    evaluations = [evaluator.evaluate(tuple(edges)) for edges in topology_sets]
    latency_reference_s = max(1e-12, max(float(row.metrics["latency"]) for row in evaluations))
    energy_reference_j = max(1e-12, max(float(row.metrics["energy"]) for row in evaluations))
    return latency_reference_s, energy_reference_j


def _surrogate_components(
    evaluation: TopologyEvaluation,
    *,
    tau_requirement_min: float,
    latency_reference_s: float,
    energy_reference_j: float,
) -> dict[str, float]:
    probability = float(evaluation.metrics["consensus_success_probability"])
    reliability_penalty = max(0.0, tau_requirement_min - probability) ** 2
    normalized_latency = float(evaluation.metrics["latency"]) / latency_reference_s
    normalized_energy = float(evaluation.metrics["energy"]) / energy_reference_j
    return {
        "delta_reliability_penalty": reliability_penalty,
        "delta_normalized_latency": normalized_latency,
        "delta_normalized_energy": normalized_energy,
        "delta_component_sum_diagnostic": (
            reliability_penalty + normalized_latency + normalized_energy
        ),
    }


def _row_diagnostics(
    *,
    fixture: ScenarioFixture,
    evaluator: TopologyEvaluator,
    evaluation: TopologyEvaluation,
    topology_name: str,
    oracle_evaluation: TopologyEvaluation | None,
) -> dict[str, object]:
    selected = tuple(evaluation.selected_edge_ids)
    topology_family = _topology_family(topology_name, selected, evaluator.graph.edge_ids)
    oracle_gap = _row_oracle_gap(evaluation, oracle_evaluation)
    return {
        "scenario_family": fixture.description,
        "fixture_id": fixture.fixture_id,
        "topology_family": topology_family,
        "topology_label_role": _topology_label_role(topology_name),
        "is_full_graph_baseline": set(selected) == set(evaluator.graph.edge_ids),
        "is_oracle_candidate": topology_name == "oracle_candidate",
        "is_deployment_actor_input": False,
        "oracle_gap": oracle_gap,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
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
    if topology_name in {"greedy", "sparse_heuristic"}:
        return "sparse_candidate"
    return "random_baseline"


def _topology_label_role(topology_name: str) -> str:
    if topology_name == "oracle_candidate":
        return "oracle_training_diagnostic_only"
    if topology_name in {"greedy", "sparse_heuristic"}:
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


def _class_balance(dataset: LearningEvidenceDataset) -> dict[str, object]:
    feasible = sum(1 for row in dataset.rows if row.feasible_under_tau_requirement)
    infeasible = dataset.row_count - feasible
    return {
        "feasible_count": feasible,
        "infeasible_count": infeasible,
        "feasible_ratio": feasible / dataset.row_count if dataset.row_count else 0.0,
        "infeasible_ratio": infeasible / dataset.row_count if dataset.row_count else 0.0,
        "has_both_classes": feasible > 0 and infeasible > 0,
    }


def _sparse_vs_full_tradeoff(dataset: LearningEvidenceDataset) -> dict[str, object]:
    by_scenario: dict[str, dict[str, LearningEvidenceRow]] = defaultdict(dict)
    for row in dataset.rows:
        by_scenario[row.scenario_id][row.topology_name] = row
    tradeoffs: list[dict[str, object]] = []
    for scenario_id, rows in by_scenario.items():
        full = rows.get("full_graph")
        sparse = rows.get("sparse_heuristic") or rows.get("greedy")
        if full is None or sparse is None:
            continue
        sparse_better_resource = (
            sparse.latency <= full.latency
            and sparse.energy <= full.energy
            and (sparse.latency < full.latency or sparse.energy < full.energy)
        )
        sparse_not_worse_probability = (
            sparse.consensus_success_probability >= full.consensus_success_probability
        )
        if (
            sparse.consensus_success_probability > 0.0
            and sparse_better_resource
            and sparse_not_worse_probability
        ):
            tradeoffs.append(
                {
                    "scenario_id": scenario_id,
                    "sparse_topology": sparse.topology_name,
                    "full_topology": full.topology_name,
                    "sparse_probability": sparse.consensus_success_probability,
                    "full_probability": full.consensus_success_probability,
                    "sparse_latency": sparse.latency,
                    "full_latency": full.latency,
                    "sparse_energy": sparse.energy,
                    "full_energy": full.energy,
                }
            )
    return {
        "tradeoff_count": len(tradeoffs),
        "tradeoffs": tradeoffs,
        "has_sparse_better_than_full_case": bool(tradeoffs),
    }


def _edge_delta_distribution(dataset: LearningEvidenceDataset) -> dict[str, object]:
    action_counts = Counter(target.action_type for target in dataset.edge_delta_targets)
    probability_deltas = [
        target.delta_consensus_success_probability for target in dataset.edge_delta_targets
    ]
    feasibility_deltas = [target.delta_feasibility for target in dataset.edge_delta_targets]
    return {
        "action_type_counts": dict(sorted(action_counts.items())),
        "positive_probability_delta_count": sum(1 for value in probability_deltas if value > 0.0),
        "negative_probability_delta_count": sum(1 for value in probability_deltas if value < 0.0),
        "positive_feasibility_delta_count": sum(1 for value in feasibility_deltas if value > 0),
        "negative_feasibility_delta_count": sum(1 for value in feasibility_deltas if value < 0),
        "nonzero_surrogate_diagnostic_count": sum(
            1
            for target in dataset.edge_delta_targets
            if any(value != 0.0 for value in target.delta_reward_surrogate_diagnostic.values())
        ),
    }


def _rare_safety_samples(dataset: LearningEvidenceDataset) -> dict[str, object]:
    rows = [
        {
            "scenario_id": row.scenario_id,
            "topology_name": row.topology_name,
            "consensus_success_probability": row.consensus_success_probability,
            "latency": row.latency,
            "energy": row.energy,
            "diagnostic_reason": _rare_reason(row),
        }
        for row in dataset.rows
        if _rare_reason(row) is not None
    ]
    return {
        "rare_sample_count": len(rows),
        "rows": rows,
    }


def _rare_reason(row: LearningEvidenceRow) -> str | None:
    if row.topology_name == "empty":
        return "disconnected_empty_topology"
    if row.consensus_success_probability < 0.1 and row.energy > 0.0:
        return "scheduled_but_unreliable_topology"
    if row.diagnostics.get("is_full_graph_baseline") and not row.feasible_under_tau_requirement:
        return "full_graph_not_feasible_under_requirement"
    return None


def _oracle_gap(dataset: LearningEvidenceDataset) -> dict[str, object]:
    gaps = []
    for row in dataset.rows:
        gap = dict(row.diagnostics.get("oracle_gap", {}))
        if gap and gap.get("delta_consensus_success_probability_to_oracle") is not None:
            gaps.append(
                {
                    "scenario_id": row.scenario_id,
                    "topology_name": row.topology_name,
                    **gap,
                }
            )
    return {
        "oracle_comparable_row_count": len(gaps),
        "max_probability_gap": max(
            (float(gap["delta_consensus_success_probability_to_oracle"]) for gap in gaps),
            default=None,
        ),
        "rows": gaps,
    }


def _scenario_diversity(dataset: LearningEvidenceDataset) -> dict[str, object]:
    scenario_ids = sorted({row.scenario_id for row in dataset.rows})
    topology_counts = Counter(row.topology_name for row in dataset.rows)
    return {
        "scenario_count": len(scenario_ids),
        "scenario_ids": scenario_ids,
        "topology_counts": dict(sorted(topology_counts.items())),
        "required_topology_variants_present": sorted(
            set(TOPOLOGY_VARIANT_NAMES) & set(topology_counts)
        ),
    }


def _actor_observable_predictability_risk(dataset: LearningEvidenceDataset) -> dict[str, object]:
    observed_edge_ids = set()
    for row in dataset.rows:
        for actor_row in row.actor_safe_rows:
            for neighbor in actor_row["local_neighbor_observations"]:
                observed_edge_ids.add(str(neighbor.edge_id))
    target_edge_ids = {
        target.edge_id
        for target in dataset.edge_delta_targets
        if "->" not in target.edge_id
    }
    coverage = (
        len(observed_edge_ids & target_edge_ids) / len(target_edge_ids)
        if target_edge_ids
        else 0.0
    )
    if coverage < 1.0:
        risk = "high"
    elif _learnable_signal_present(dataset):
        risk = "medium"
    else:
        risk = "high"
    return {
        "actor_observable_edge_coverage": coverage,
        "observed_edge_count": len(observed_edge_ids),
        "target_edge_count": len(target_edge_ids),
        "risk": risk,
        "reason": (
            "actor observes endpoint-local link features, but global feasibility "
            "and resource trade-offs still require centralized training signals"
        ),
    }


def _has_oracle_or_heuristic(dataset: LearningEvidenceDataset) -> bool:
    return any(
        row.diagnostics.get("topology_label_role")
        in {"oracle_training_diagnostic_only", "heuristic_training_diagnostic_only"}
        for row in dataset.rows
    )


def _views_separated(dataset: LearningEvidenceDataset) -> bool:
    for row in dataset.rows:
        if not row.actor_safe_rows or not row.critic_view or not row.learning_targets:
            return False
    return True


def _learnable_signal_present(dataset: LearningEvidenceDataset) -> bool:
    probabilities = [row.consensus_success_probability for row in dataset.rows]
    class_balance = _class_balance(dataset)
    edge_distribution = _edge_delta_distribution(dataset)
    return (
        bool(probabilities)
        and max(probabilities) - min(probabilities) > 0.1
        and class_balance["has_both_classes"]
        and edge_distribution["positive_probability_delta_count"] > 0
        and edge_distribution["negative_probability_delta_count"] > 0
        and edge_distribution["nonzero_surrogate_diagnostic_count"] > 0
    )


def _oracle_not_actor_input(dataset: LearningEvidenceDataset) -> bool:
    forbidden = {"oracle_label", "oracle_status", "oracle_action", "oracle_feasibility"}
    return all(
        not (set(actor_row) & forbidden)
        for row in dataset.rows
        for actor_row in row.actor_safe_rows
    )


def _surrogate_not_actor_observation(dataset: LearningEvidenceDataset) -> bool:
    forbidden = {
        "reward_surrogate",
        "reward_reliability_penalty",
        "reward_latency_penalty",
        "reward_energy_penalty",
        "reward_config_id",
    }
    return all(
        not (set(actor_row) & forbidden)
        for row in dataset.rows
        for actor_row in row.actor_safe_rows
    )


def _stage7_close_reference_fixture() -> ScenarioFixture:
    return ScenarioFixture(
        fixture_id="stage7_close_reference",
        description="Stage 7 close free-space reference with feasible sparse and full topologies.",
        scene=_stage7_scene(
            "stage7_close_reference",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(6.0, 0.0, 1.5)),
                Node3D("veh_1", NodeKind.VEHICLE, Point3D(0.0, 6.0, 1.5)),
                Node3D("veh_2", NodeKind.VEHICLE, Point3D(6.0, 6.0, 1.5)),
            ),
        ),
        max_candidate_distance_m=20.0,
        quorum_size=3,
        success_probability_threshold=0.45,
        deadline_s=0.01,
        link_reference_distance_m=500.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage7_sparse_vs_full_tradeoff_fixture() -> ScenarioFixture:
    return ScenarioFixture(
        fixture_id="stage7_sparse_vs_full_tradeoff",
        description="Stage 7 topology where sparse reliable path beats dense full graph under tau.",
        scene=_stage7_scene(
            "stage7_sparse_vs_full_tradeoff",
            (
                Node3D("rsu_0", NodeKind.RSU, Point3D(0.0, 0.0, 5.0)),
                Node3D("veh_0", NodeKind.VEHICLE, Point3D(10.0, 0.0, 1.5)),
                Node3D("veh_1", NodeKind.VEHICLE, Point3D(20.0, 0.0, 1.5)),
                Node3D("veh_2", NodeKind.VEHICLE, Point3D(30.0, 0.0, 1.5)),
            ),
        ),
        max_candidate_distance_m=35.0,
        quorum_size=3,
        success_probability_threshold=0.45,
        deadline_s=0.01,
        link_reference_distance_m=120.0,
        expected_oracle_status="feasible",
        expected_full_graph_success=True,
    )


def _stage7_scene(scenario_id: str, nodes: Iterable[Node3D]) -> Scene3D:
    return Scene3D(
        scenario_id=scenario_id,
        nodes=tuple(nodes),
        physics_regime="stage2_deterministic_distance",
    )
