"""Learning evidence dataset boundaries for Stage 7.

The builders in this module create in-memory evidence rows from existing
topology evaluations. They keep deployment actor inputs separate from critic
views, learning targets, and diagnostics. A small guarded writer is included
for evidence-only artifacts after manifest validation and owner approval.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from types import MappingProxyType
from pathlib import Path

from marl_topology.env import ActorObservation
from marl_topology.topology.evaluator import TopologyEvaluation, TopologyEvaluator
from marl_topology.training import validate_run_manifest_dry_run

from .actor_batch import ActorSafeBatch, build_actor_safe_batch_from_observations


STAGE7_0_LEARNING_EVIDENCE_STAGE_ID = (
    "stage_7_0_learning_evidence_dataset_generation_with_owner_approval"
)
STAGE7_0_VERDICT = "learning_evidence_dataset_ready_model_training_blocked"
STAGE7_0_ARTIFACT_SCOPE = "evidence_dataset_only"
STAGE7_0_RECOMMENDED_NEXT_TASK = (
    "stage_7_1_learning_evidence_data_quality_report_with_owner_approval"
)
TARGET_ROLE_LEARNING_ONLY = "learning_target_only"
REQUIRED_LEARNING_EVIDENCE_VIEWS = (
    "actor_safe_view",
    "critic_centralized_view",
    "learning_target_view",
    "diagnostics_view",
)
ACTOR_VIEW_FORBIDDEN_FIELDS = frozenset(
    {
        "candidate_edge_ids",
        "centralized_state",
        "consensus_success",
        "consensus_success_probability",
        "critic_features",
        "critic_view",
        "energy",
        "feasible_under_tau_requirement",
        "future_channel_state",
        "future_consensus_outcome",
        "global_graph",
        "global_topology",
        "latency",
        "learning_targets",
        "objective_metrics",
        "oracle_action",
        "oracle_feasibility",
        "oracle_label",
        "oracle_status",
        "reward_config_id",
        "reward_energy_penalty",
        "reward_latency_penalty",
        "reward_reliability_penalty",
        "reward_surrogate",
        "selected_edges",
        "selected_edge_ids",
        "topology_diagnostics",
    }
)


class LearningEvidenceViolation(ValueError):
    """Raised when Stage 7 evidence boundaries are violated."""


@dataclass(frozen=True, slots=True)
class EdgeDeltaTarget:
    """Training-only edge counterfactual target."""

    topology_id: str
    edge_id: str
    action_type: str
    delta_consensus_success_probability: float
    delta_latency: float
    delta_energy: float
    delta_feasibility: int
    delta_reward_surrogate_diagnostic: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    target_role: str = TARGET_ROLE_LEARNING_ONLY

    def __post_init__(self) -> None:
        if self.action_type not in {"add_edge", "remove_edge", "keep_edge", "swap_edge"}:
            raise LearningEvidenceViolation(f"unknown edge action type: {self.action_type}")
        if self.target_role != TARGET_ROLE_LEARNING_ONLY:
            raise LearningEvidenceViolation("edge-delta targets must remain learning-target only")

    def to_dict(self) -> dict[str, object]:
        return {
            "topology_id": self.topology_id,
            "edge_id": self.edge_id,
            "action_type": self.action_type,
            "delta_consensus_success_probability": self.delta_consensus_success_probability,
            "delta_latency": self.delta_latency,
            "delta_energy": self.delta_energy,
            "delta_feasibility": self.delta_feasibility,
            "delta_reward_surrogate_diagnostic": dict(self.delta_reward_surrogate_diagnostic),
            "target_role": self.target_role,
        }


@dataclass(frozen=True, slots=True)
class LearningEvidenceRow:
    """One topology evidence row with separated data views."""

    scenario_id: str
    topology_id: str
    topology_name: str
    selected_edges: tuple[str, ...]
    consensus_success_probability: float
    latency: float
    energy: float
    topology_diagnostics: Mapping[str, object]
    feasible_under_tau_requirement: bool
    actor_safe_rows: tuple[Mapping[str, object], ...]
    critic_view: Mapping[str, object]
    learning_targets: tuple[Mapping[str, object], ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise LearningEvidenceViolation("consensus success probability must be in [0, 1]")
        if self.latency < 0:
            raise LearningEvidenceViolation("latency must be nonnegative")
        if self.energy < 0:
            raise LearningEvidenceViolation("energy must be nonnegative")
        validate_actor_safe_view(self.actor_safe_rows)
        for target in self.learning_targets:
            if target.get("target_role") != TARGET_ROLE_LEARNING_ONLY:
                raise LearningEvidenceViolation("learning target row has wrong target_role")

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "topology_id": self.topology_id,
            "topology_name": self.topology_name,
            "selected_edges": list(self.selected_edges),
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.latency,
            "energy": self.energy,
            "topology_diagnostics": _jsonable(self.topology_diagnostics),
            "feasible_under_tau_requirement": self.feasible_under_tau_requirement,
            "actor_safe_rows": _jsonable(self.actor_safe_rows),
            "critic_view": _jsonable(self.critic_view),
            "learning_targets": _jsonable(self.learning_targets),
            "diagnostics": _jsonable(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class LearningEvidenceDataset:
    """In-memory Stage 7 evidence dataset."""

    dataset_id: str
    rows: tuple[LearningEvidenceRow, ...]
    edge_delta_targets: tuple[EdgeDeltaTarget, ...]
    views: tuple[str, ...] = REQUIRED_LEARNING_EVIDENCE_VIEWS

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise LearningEvidenceViolation("dataset_id must be declared")
        missing = sorted(set(REQUIRED_LEARNING_EVIDENCE_VIEWS) - set(self.views))
        if missing:
            raise LearningEvidenceViolation(f"missing evidence views: {missing}")

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def target_count(self) -> int:
        return len(self.edge_delta_targets)

    def summary(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "target_count": self.target_count,
            "views": list(self.views),
            "actor_safe_view_separated": True,
            "learning_target_view_separated": True,
            "checkpoint_creation_allowed": False,
            "training_execution_allowed": False,
            "model_implementation_allowed": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.summary(),
            "rows": [row.to_dict() for row in self.rows],
            "edge_delta_targets": [target.to_dict() for target in self.edge_delta_targets],
        }


@dataclass(frozen=True, slots=True)
class EvidenceArtifactWriteResult:
    """Result of an evidence-only guarded artifact write."""

    artifact_scope: str
    artifact_dir: str
    manifest_path: str
    evidence_path: str
    writes_performed: bool
    checkpoint_creation_allowed: bool = False
    training_execution_allowed: bool = False
    model_output_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_actor_safe_view(rows: Iterable[Mapping[str, object]]) -> None:
    """Reject objective, oracle, future, and global fields in actor-safe rows."""

    for index, row in enumerate(rows):
        forbidden = sorted(set(row) & ACTOR_VIEW_FORBIDDEN_FIELDS)
        if forbidden:
            raise LearningEvidenceViolation(
                f"actor_safe_view row {index} contains forbidden fields: {forbidden}"
            )


def build_learning_evidence_row(
    evaluation: TopologyEvaluation,
    *,
    topology_name: str,
    observations: Iterable[ActorObservation],
    graph_node_ids: tuple[str, ...],
    candidate_edge_ids: tuple[str, ...],
    tau_requirement_min: float,
    learning_targets: Iterable[Mapping[str, object]] = (),
    diagnostics: Mapping[str, object] | None = None,
) -> LearningEvidenceRow:
    """Build one Stage 7 evidence row from an existing topology evaluation."""

    if not 0.0 <= tau_requirement_min <= 1.0:
        raise LearningEvidenceViolation("tau_requirement_min must be in [0, 1]")

    actor_batch = build_actor_safe_batch_from_observations(
        observations,
        batch_id=f"actor_safe_view:{evaluation.topology_id}",
    )
    consensus_probability = float(evaluation.metrics["consensus_success_probability"])
    latency = float(evaluation.metrics["latency"])
    energy = float(evaluation.metrics["energy"])
    metric_names = tuple(sorted(evaluation.metrics.keys()))
    return LearningEvidenceRow(
        scenario_id=evaluation.scenario_id,
        topology_id=evaluation.topology_id,
        topology_name=topology_name,
        selected_edges=tuple(evaluation.selected_edge_ids),
        consensus_success_probability=consensus_probability,
        latency=latency,
        energy=energy,
        topology_diagnostics=evaluation.metrics["topology_diagnostics"],
        feasible_under_tau_requirement=consensus_probability >= tau_requirement_min,
        actor_safe_rows=actor_batch.rows,
        critic_view={
            "scenario_id": evaluation.scenario_id,
            "node_ids": tuple(graph_node_ids),
            "candidate_edge_ids": tuple(candidate_edge_ids),
            "selected_edge_ids": tuple(evaluation.selected_edge_ids),
            "metric_names": metric_names,
            "view_role": "critic_centralized_training_only",
        },
        learning_targets=tuple(dict(target) for target in learning_targets),
        diagnostics={
            "tau_requirement_min": tau_requirement_min,
            "source_topology_id": evaluation.topology_id,
            "evidence_stage": STAGE7_0_LEARNING_EVIDENCE_STAGE_ID,
            **dict(diagnostics or {}),
        },
    )


def build_edge_delta_targets(
    evaluator: TopologyEvaluator,
    selected_edge_ids: Iterable[str],
    *,
    tau_requirement_min: float,
    topology_id: str = "edge_delta_source",
) -> tuple[EdgeDeltaTarget, ...]:
    """Build add/remove/keep edge-delta targets for one selected topology."""

    selected = set(selected_edge_ids)
    baseline = evaluator.evaluate(selected, topology_id=f"{topology_id}:baseline")
    baseline_probability = float(baseline.metrics["consensus_success_probability"])
    baseline_latency = float(baseline.metrics["latency"])
    baseline_energy = float(baseline.metrics["energy"])
    baseline_feasible = int(baseline_probability >= tau_requirement_min)

    targets: list[EdgeDeltaTarget] = []
    for edge_id in evaluator.graph.edge_ids:
        targets.append(
            EdgeDeltaTarget(
                topology_id=topology_id,
                edge_id=edge_id,
                action_type="keep_edge",
                delta_consensus_success_probability=0.0,
                delta_latency=0.0,
                delta_energy=0.0,
                delta_feasibility=0,
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
        changed_evaluation = evaluator.evaluate(
            changed,
            topology_id=f"{topology_id}:{action_type}:{edge_id}",
        )
        changed_probability = float(changed_evaluation.metrics["consensus_success_probability"])
        changed_latency = float(changed_evaluation.metrics["latency"])
        changed_energy = float(changed_evaluation.metrics["energy"])
        changed_feasible = int(changed_probability >= tau_requirement_min)
        targets.append(
            EdgeDeltaTarget(
                topology_id=topology_id,
                edge_id=edge_id,
                action_type=action_type,
                delta_consensus_success_probability=changed_probability
                - baseline_probability,
                delta_latency=changed_latency - baseline_latency,
                delta_energy=changed_energy - baseline_energy,
                delta_feasibility=changed_feasible - baseline_feasible,
            )
        )
    return tuple(targets)


def build_learning_evidence_dataset(
    evaluator: TopologyEvaluator,
    *,
    observations: Iterable[ActorObservation],
    topology_variants: Mapping[str, Iterable[str]],
    tau_requirement_min: float,
    dataset_id: str = "stage7_0_demo_learning_evidence",
) -> LearningEvidenceDataset:
    """Build a minimal in-memory evidence dataset from named topology variants."""

    observation_tuple = tuple(observations)
    rows: list[LearningEvidenceRow] = []
    all_targets: list[EdgeDeltaTarget] = []
    for topology_name, selected_edges in topology_variants.items():
        selected = tuple(sorted(set(selected_edges)))
        evaluation = evaluator.evaluate(selected, topology_id=f"evidence:{topology_name}")
        edge_targets = build_edge_delta_targets(
            evaluator,
            selected,
            tau_requirement_min=tau_requirement_min,
            topology_id=evaluation.topology_id,
        )
        all_targets.extend(edge_targets)
        row = build_learning_evidence_row(
            evaluation,
            topology_name=topology_name,
            observations=observation_tuple,
            graph_node_ids=evaluator.graph.node_ids,
            candidate_edge_ids=evaluator.graph.edge_ids,
            tau_requirement_min=tau_requirement_min,
            learning_targets=(target.to_dict() for target in edge_targets),
            diagnostics={
                "topology_family": _topology_family(
                    topology_name=topology_name,
                    selected_edges=selected,
                    full_edge_ids=evaluator.graph.edge_ids,
                ),
                "is_full_graph_baseline": set(selected) == set(evaluator.graph.edge_ids),
                "is_oracle_candidate": topology_name.startswith("oracle_candidate"),
                "is_deployment_actor_input": False,
            },
        )
        rows.append(row)
    return LearningEvidenceDataset(
        dataset_id=dataset_id,
        rows=tuple(rows),
        edge_delta_targets=tuple(all_targets),
    )


def write_learning_evidence_artifact(
    dataset: LearningEvidenceDataset,
    *,
    manifest: Mapping[str, object],
    project_root: str | Path,
    owner_approved_evidence_export: bool,
) -> EvidenceArtifactWriteResult:
    """Write an evidence-only artifact after manifest and approval checks."""

    if not owner_approved_evidence_export:
        raise LearningEvidenceViolation("owner approval is required for evidence export")
    if manifest.get("artifact_scope") != STAGE7_0_ARTIFACT_SCOPE:
        raise LearningEvidenceViolation("artifact_scope must be evidence_dataset_only")

    validation = validate_run_manifest_dry_run(manifest, project_root=project_root)
    if not validation.is_valid:
        codes = ", ".join(validation.error_codes())
        raise LearningEvidenceViolation(f"run manifest validation failed: {codes}")

    root = Path(project_root).resolve(strict=False)
    allowed_root = (root / "result_save" / STAGE7_0_ARTIFACT_SCOPE).resolve(strict=False)
    run_id = str(manifest.get("run_id", dataset.dataset_id))
    artifact_dir = (allowed_root / run_id).resolve(strict=False)
    _ensure_within(artifact_dir, allowed_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir / "manifest.json"
    evidence_path = artifact_dir / "learning_evidence.json"
    manifest_path.write_bytes(
        json.dumps(_jsonable(dict(manifest)), indent=2, sort_keys=True).encode("utf-8")
    )
    evidence_path.write_bytes(
        json.dumps(dataset.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    )
    return EvidenceArtifactWriteResult(
        artifact_scope=STAGE7_0_ARTIFACT_SCOPE,
        artifact_dir=str(artifact_dir),
        manifest_path=str(manifest_path),
        evidence_path=str(evidence_path),
        writes_performed=True,
    )


def build_stage7_0_learning_evidence_report(
    dataset: LearningEvidenceDataset,
    *,
    artifact_writer_available: bool,
) -> dict[str, object]:
    """Return a report for Stage 7.0 without writing project artifacts."""

    nonempty_actor_rows = all(row.actor_safe_rows for row in dataset.rows)
    separated_targets = all(
        target.target_role == TARGET_ROLE_LEARNING_ONLY for target in dataset.edge_delta_targets
    )
    return {
        "stage": STAGE7_0_LEARNING_EVIDENCE_STAGE_ID,
        "verdict": STAGE7_0_VERDICT,
        "dataset_summary": dataset.summary(),
        "evidence_generated": dataset.row_count > 0 and dataset.target_count > 0,
        "actor_safe_rows_present": nonempty_actor_rows,
        "actor_safe_view_separated": nonempty_actor_rows,
        "learning_target_view_separated": separated_targets,
        "artifact_scope": STAGE7_0_ARTIFACT_SCOPE,
        "artifact_writer_available": artifact_writer_available,
        "artifact_written": False,
        "checkpoint_creation_allowed": False,
        "training_execution_allowed": False,
        "model_implementation_allowed": False,
        "v5_code_migrated": False,
        "recommended_next_task": STAGE7_0_RECOMMENDED_NEXT_TASK,
    }


def _topology_family(
    *,
    topology_name: str,
    selected_edges: tuple[str, ...],
    full_edge_ids: tuple[str, ...],
) -> str:
    if not selected_edges:
        return "weak_or_disconnected_baseline"
    if set(selected_edges) == set(full_edge_ids):
        return "dense_full_graph_baseline"
    if topology_name.startswith("oracle_candidate"):
        return "oracle_candidate_diagnostic"
    return "sparse_candidate"


def _ensure_within(candidate: Path, root: Path) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LearningEvidenceViolation("artifact path escaped evidence_dataset_only") from exc


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
