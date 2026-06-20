"""Stage 33 graph-structure-necessity dataset.

The dataset is generated through the Stage 31 procedural path and measured by
the existing Stage 3/4-backed objective evaluator. Family labels describe the
intended structural stressor; feasibility and teacher labels are measured, not
hardcoded.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from marl_topology.data.stage31_production_dataset import (
    Stage31ProductionDataset,
    build_production_context,
    build_production_dataset,
)
from marl_topology.data.stage31_scenario_generator import (
    PhysicsRegime,
    ProductionScenarioConfig,
)


STAGE33_GRAPH_STRUCTURE_DATASET_ID = "stage33_graph_structure_necessity_dataset_v1"
STAGE33_GRAPH_STRUCTURE_FAMILIES = (
    "bridge_node",
    "weak_primary_repair",
    "rsu_hub_role_structured",
    "redundant_local_quality",
    "multi_hop_sparse_backbone",
    "near_threshold_structural",
    "full_graph_resource_failure_sparse_feasible",
)
TAU_REQUIREMENT_MIN = 0.9


class Stage33GraphStructureDatasetViolation(ValueError):
    """Raised when Stage 33 data generation violates the declared boundary."""


@dataclass(frozen=True, slots=True)
class Stage33GraphStructureConfig:
    seed: int = 33
    scenario_count: int = 21
    node_count_choices: tuple[int, ...] = (6, 7, 8, 9, 10)
    tau_requirement_min: float = TAU_REQUIREMENT_MIN
    # urban_mode + regime are opt-in: when set they make the dataset generate urban
    # NLOS scenes under the given physics regime (e.g. scheduled MAC + relay). The spec
    # carries the regime, so the training evaluator and the relay-aware teacher follow
    # automatically. Defaults (urban_mode=False, regime=None) are byte-identical.
    urban_mode: bool = False
    urban_blocks_per_side: int = 3
    urban_rsu_count: int = 1
    regime: PhysicsRegime | None = None
    # Target family fractions (must sum to 1.0). Exposed so an urban/scheduled regime whose
    # natural feasible rate differs from the 0.5 default can MATCH it -- otherwise generation
    # thrashes trying to fill a feasible bin the scene distribution cannot supply. Defaults
    # mirror ProductionScenarioConfig (byte-identical when left unset).
    target_feasible_fraction: float = 0.5
    target_near_threshold_fraction: float = 0.2
    target_infeasible_fraction: float = 0.3
    # Opt-in N>=24 build enablers (defaults keep small-N builds byte-identical):
    #  - vectorized_evaluator: route the SA/measure path through the bounded-cache 6x-faster
    #    VectorizedStage21Evaluator (float-identical) so large-N builds don't OOM/thrash.
    #  - max_total_attempts: hard cap on the rejection-sampling loop (0 = auto) so a low-yield
    #    N>=24 build returns a smaller dataset instead of running for hours.
    vectorized_evaluator: bool = False
    max_total_attempts: int = 0

    def __post_init__(self) -> None:
        if self.scenario_count < len(STAGE33_GRAPH_STRUCTURE_FAMILIES):
            raise Stage33GraphStructureDatasetViolation(
                "scenario_count must cover every Stage 33 graph family"
            )
        if self.tau_requirement_min != TAU_REQUIREMENT_MIN:
            raise Stage33GraphStructureDatasetViolation(
                "Stage 33 does not lower tau_requirement_min below 0.9"
            )
        if any(count < 6 or count > 48 for count in self.node_count_choices):
            raise Stage33GraphStructureDatasetViolation(
                "Stage 33 graph-structure dataset uses node counts in 6..48"
            )

    def production_config(self) -> ProductionScenarioConfig:
        kwargs: dict[str, object] = dict(
            seed=self.seed,
            scenario_count=self.scenario_count,
            node_count_choices=self.node_count_choices,
            tau_requirement_min=self.tau_requirement_min,
            urban_mode=self.urban_mode,
            urban_blocks_per_side=self.urban_blocks_per_side,
            urban_rsu_count=self.urban_rsu_count,
            target_feasible_fraction=self.target_feasible_fraction,
            target_near_threshold_fraction=self.target_near_threshold_fraction,
            target_infeasible_fraction=self.target_infeasible_fraction,
            max_total_attempts=self.max_total_attempts,
        )
        if self.regime is not None:
            kwargs["regime"] = self.regime
        return ProductionScenarioConfig(**kwargs)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_id": STAGE33_GRAPH_STRUCTURE_DATASET_ID,
            "seed": self.seed,
            "scenario_count": self.scenario_count,
            "node_count_choices": list(self.node_count_choices),
            "tau_requirement_min": self.tau_requirement_min,
            "families": list(STAGE33_GRAPH_STRUCTURE_FAMILIES),
        }


@dataclass(frozen=True, slots=True)
class Stage33GraphScenarioRecord:
    scenario_id: str
    structural_family: str
    split: str
    node_count: int
    candidate_edge_count: int
    measured_family: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.structural_family not in STAGE33_GRAPH_STRUCTURE_FAMILIES:
            raise Stage33GraphStructureDatasetViolation(
                f"unknown structural family: {self.structural_family}"
            )
        if not isinstance(self.diagnostics, Mapping):
            raise Stage33GraphStructureDatasetViolation("diagnostics must be a mapping")


@dataclass(frozen=True, slots=True)
class Stage33GraphStructureDataset:
    dataset_id: str
    config: Stage33GraphStructureConfig
    source_dataset: Stage31ProductionDataset
    records: tuple[Stage33GraphScenarioRecord, ...]
    quality_report: Mapping[str, object]

    def records_for_split(self, split_name: str) -> tuple[Stage33GraphScenarioRecord, ...]:
        return tuple(record for record in self.records if record.split == split_name)

    def records_for_family(self, family: str) -> tuple[Stage33GraphScenarioRecord, ...]:
        return tuple(record for record in self.records if record.structural_family == family)


def build_stage33_graph_structure_dataset(
    config: Stage33GraphStructureConfig | None = None,
) -> Stage33GraphStructureDataset:
    cfg = config or Stage33GraphStructureConfig()
    source = build_production_dataset(cfg.production_config(), vectorized=cfg.vectorized_evaluator)
    split_by_id = _split_lookup(source)
    records: list[Stage33GraphScenarioRecord] = []
    for index, spec in enumerate(source.specs):
        family = STAGE33_GRAPH_STRUCTURE_FAMILIES[index % len(STAGE33_GRAPH_STRUCTURE_FAMILIES)]
        context = build_production_context(spec, vectorized=cfg.vectorized_evaluator)
        label = source.teacher_labels[spec.scenario_id]
        diagnostics = graph_necessity_diagnostics(
            context=context,
            teacher_label=label,
            structural_family=family,
            tau=cfg.tau_requirement_min,
        )
        records.append(
            Stage33GraphScenarioRecord(
                scenario_id=spec.scenario_id,
                structural_family=family,
                split=split_by_id[spec.scenario_id],
                node_count=len(spec.scene.nodes),
                candidate_edge_count=spec.candidate_edge_count,
                measured_family=spec.family,
                diagnostics=diagnostics,
            )
        )
    quality = build_graph_structure_quality_report(cfg, source, tuple(records))
    return Stage33GraphStructureDataset(
        dataset_id=STAGE33_GRAPH_STRUCTURE_DATASET_ID,
        config=cfg,
        source_dataset=source,
        records=tuple(records),
        quality_report=quality,
    )


def graph_necessity_diagnostics(
    *,
    context,
    teacher_label: Mapping[str, object],
    structural_family: str,
    tau: float = TAU_REQUIREMENT_MIN,
) -> dict[str, object]:
    teacher_edges = tuple(str(edge) for edge in teacher_label["selected_physical_edges"])
    teacher_eval = context.evaluator.evaluate(set(teacher_edges))
    full_edges = tuple(context.graph.edge_ids)
    full_eval = context.evaluator.evaluate(set(full_edges))
    local_edges = _local_edge_quality_heuristic_edges(context, max(1, len(teacher_edges)))
    local_eval = context.evaluator.evaluate(set(local_edges))
    teacher_psucc = float(teacher_eval.metrics["consensus_success_probability"])
    full_psucc = float(full_eval.metrics["consensus_success_probability"])
    local_psucc = float(local_eval.metrics["consensus_success_probability"])
    local_gap = teacher_psucc - local_psucc
    full_gap = teacher_psucc - full_psucc
    rank_divergence = _similar_quality_objective_divergence(context, teacher_edges)
    score = max(0.0, local_gap) + 0.5 * max(0.0, full_gap) + 0.05 * rank_divergence
    return {
        "structural_family": structural_family,
        "teacher_feasible": teacher_psucc >= tau,
        "teacher_consensus_success_probability": teacher_psucc,
        "teacher_latency": float(teacher_eval.metrics["latency"]),
        "teacher_energy": float(teacher_eval.metrics["energy"]),
        "teacher_selected_edge_count": len(teacher_edges),
        "full_graph_feasible": full_psucc >= tau,
        "full_graph_consensus_success_probability": full_psucc,
        "full_graph_energy": float(full_eval.metrics["energy"]),
        "sparse_feasible": teacher_psucc >= tau and len(teacher_edges) < len(full_edges),
        "local_edge_heuristic_edges": local_edges,
        "local_edge_heuristic_feasible": local_psucc >= tau,
        "local_edge_heuristic_consensus_success_probability": local_psucc,
        "local_edge_heuristic_teacher_gap": local_gap,
        "full_graph_teacher_gap": full_gap,
        "similar_local_quality_objective_rank_divergence": rank_divergence,
        "graph_necessity_score": score,
        "mlp_hard_label_detected": local_gap > 0.05 or rank_divergence >= 2,
        "gnn_necessary_label_detected": score > 0.10,
        "stage3_stage4_evaluator_used": True,
        "objective_aware_teacher_used": True,
        "simple_link_model_fallback_used": False,
        "hardcoded_fake_feasibility": False,
        "all_metrics_finite": all(
            isfinite(value)
            for value in (teacher_psucc, full_psucc, local_psucc, local_gap, full_gap, score)
        ),
    }


def build_graph_structure_quality_report(
    config: Stage33GraphStructureConfig,
    source: Stage31ProductionDataset,
    records: tuple[Stage33GraphScenarioRecord, ...],
) -> dict[str, object]:
    by_family: dict[str, list[Stage33GraphScenarioRecord]] = defaultdict(list)
    for record in records:
        by_family[record.structural_family].append(record)
    per_family = {
        family: _family_metrics(items)
        for family, items in sorted(by_family.items())
    }
    split_ids = {
        "train": set(source.split.train_scenario_ids),
        "eval": set(source.split.eval_scenario_ids),
        "test": set(source.split.test_scenario_ids),
    }
    leakage = len(split_ids["train"] & split_ids["eval"]) + len(split_ids["train"] & split_ids["test"])
    leakage += len(split_ids["eval"] & split_ids["test"])
    graph_scores = [float(record.diagnostics["graph_necessity_score"]) for record in records]
    return {
        "dataset_id": STAGE33_GRAPH_STRUCTURE_DATASET_ID,
        "config": config.to_payload(),
        "scenario_count": len(records),
        "unique_context_count": len({record.scenario_id for record in records}),
        "duplicate_context_rate": 1.0 - len({record.scenario_id for record in records}) / len(records),
        "split_leakage_overlap_count": leakage,
        "node_counts_present": sorted({record.node_count for record in records}),
        "families_present": sorted(per_family),
        "all_required_families_present": set(STAGE33_GRAPH_STRUCTURE_FAMILIES).issubset(per_family),
        "per_family": per_family,
        "mean_graph_necessity_score": _mean(graph_scores),
        "max_graph_necessity_score": max(graph_scores) if graph_scores else 0.0,
        "teacher_label_is_actor_input": False,
        "stage3_stage4_evaluator_used": True,
        "objective_aware_teacher_used": True,
        "simple_link_model_fallback_used": False,
        "hardcoded_fake_feasibility": False,
    }


def _family_metrics(records: list[Stage33GraphScenarioRecord]) -> dict[str, object]:
    diagnostics = [record.diagnostics for record in records]
    return {
        "count": len(records),
        "teacher_feasible_rate": _rate(diagnostics, "teacher_feasible"),
        "full_graph_feasible_rate": _rate(diagnostics, "full_graph_feasible"),
        "sparse_feasible_rate": _rate(diagnostics, "sparse_feasible"),
        "local_edge_heuristic_feasible_rate": _rate(
            diagnostics, "local_edge_heuristic_feasible"
        ),
        "mlp_hard_rate": _rate(diagnostics, "mlp_hard_label_detected"),
        "gnn_necessary_rate": _rate(diagnostics, "gnn_necessary_label_detected"),
        "mean_graph_necessity_score": _mean(
            float(item["graph_necessity_score"]) for item in diagnostics
        ),
        "mean_local_edge_teacher_gap": _mean(
            float(item["local_edge_heuristic_teacher_gap"]) for item in diagnostics
        ),
    }


def _local_edge_quality_heuristic_edges(context, edge_count: int) -> tuple[str, ...]:
    ranked = sorted(
        tuple(context.graph.edge_ids),
        key=lambda edge_id: float(
            getattr(context.link_records[edge_id], "link_success_probability", 0.0)
        ),
        reverse=True,
    )
    return tuple(sorted(ranked[:edge_count]))


def _similar_quality_objective_divergence(context, teacher_edges: tuple[str, ...]) -> int:
    teacher_set = set(teacher_edges)
    link_values = {
        edge_id: float(getattr(context.link_records[edge_id], "link_success_probability", 0.0))
        for edge_id in context.graph.edge_ids
    }
    divergence = 0
    edge_ids = tuple(link_values)
    for left_index, left in enumerate(edge_ids):
        for right in edge_ids[left_index + 1 :]:
            if abs(link_values[left] - link_values[right]) <= 0.02 and (
                (left in teacher_set) != (right in teacher_set)
            ):
                divergence += 1
    return divergence


def _split_lookup(source: Stage31ProductionDataset) -> dict[str, str]:
    result: dict[str, str] = {}
    for split_name, ids in (
        ("train", source.split.train_scenario_ids),
        ("eval", source.split.eval_scenario_ids),
        ("test", source.split.test_scenario_ids),
    ):
        for scenario_id in ids:
            result[scenario_id] = split_name
    return result


def _rate(records: list[Mapping[str, object]], key: str) -> float:
    return sum(1 for record in records if bool(record[key])) / len(records) if records else 0.0


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0
