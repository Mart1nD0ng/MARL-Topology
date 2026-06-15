"""Stage 31 Phase E: production-scale dataset with leakage-checked splits.

This module assembles the Stage 31 procedural scenarios into a production
dataset: per-scenario evaluation contexts, a scalable heuristic teacher label
(the best feasible topology, found by candidate enumeration so it scales past the
exhaustive-oracle edge cap), a deterministic train/eval/test split keyed on the
unique scenario context (no cross-split leakage), and a quality report and
manifest.

The teacher label is an evaluation/supervision target only; it never enters an
actor observation (same boundary as the exhaustive oracle).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from marl_topology.topology import CandidateGraph

from .stage21_objective_stack_evidence import (
    Stage21EvaluationContext,
    Stage21ObjectiveStackEvaluator,
)
from .stage31_scenario_generator import (
    ProductionScenarioConfig,
    ProductionScenarioSpec,
    best_feasible_topology,
    build_stack_config,
    enumerate_candidate_topologies,
    generate_production_scenarios,
    relay_aware_search_kwargs,
    scenario_fixture_from_spec,
    summarize_feasibility_distribution,
)

TAU_REQUIREMENT_MIN = 0.9
STAGE31_DATASET_ID = "stage31_production_scenario_dataset_v1"
STAGE31_PHYSICS_REGIME_ID = "urlcc_finite_blocklength_v1"
STAGE31_PROTOCOL_MODEL_ID = "stage4_expected_initiator_pbft_over_stage3_network_v1"


def build_scenario_evaluator(
    spec: ProductionScenarioSpec,
) -> tuple[CandidateGraph, Stage21ObjectiveStackEvaluator]:
    graph = CandidateGraph.from_scene(spec.scene, max_distance_m=None)
    evaluator = Stage21ObjectiveStackEvaluator(
        scene=spec.scene, graph=graph, config=build_stack_config(spec.regime)
    )
    return graph, evaluator


def build_teacher_label(
    spec: ProductionScenarioSpec, *, tau: float = TAU_REQUIREMENT_MIN
) -> dict[str, object]:
    """Scalable heuristic teacher: best feasible topology among candidates.

    Uses candidate enumeration + evaluation rather than exhaustive search, so it
    labels graphs beyond the 10-edge exhaustive-oracle cap. Teacher-only target.
    """

    from marl_topology.budgets import node_budgets_for_scene

    graph, evaluator = build_scenario_evaluator(spec)
    candidates = enumerate_candidate_topologies(graph, evaluator.link_records, spec.quorum_size)
    # Moderate SA budget for the BC teacher target (heavier than family-binning, lighter
    # than the 200/6 default so dataset build stays tractable under scheduled MAC + relay).
    search_kwargs = relay_aware_search_kwargs(
        spec.regime, graph, spec.scenario_id, sa_iters=80, restarts=2, polish=False
    )
    best = best_feasible_topology(
        evaluator, candidates, tau, node_budgets=node_budgets_for_scene(spec.scene),
        **search_kwargs,
    )
    return {
        "scenario_id": spec.scenario_id,
        "teacher_source": (
            "stage31_candidate_enumeration_best_feasible_relay_search_v3"
            if search_kwargs
            else "stage31_candidate_enumeration_best_feasible_budget_v2"
        ),
        "feasible_exists": bool(best["feasible_exists"]),
        "selected_physical_edges": tuple(best["edges"]),
        "selected_edge_count": int(best["edge_count"]),
        "consensus_success_probability": float(best["psucc"]),
        "latency": float(best["latency"]),
        "energy": float(best["energy"]),
        "candidate_edge_count": len(graph.edge_ids),
        "is_actor_input": False,
    }


def build_production_context(
    spec: ProductionScenarioSpec,
    *,
    time_step: int = 1,
    sequence_id: str | None = None,
) -> Stage21EvaluationContext:
    """Build a Stage 21 evaluation context for a procedural scenario."""

    graph, evaluator = build_scenario_evaluator(spec)
    variants = enumerate_candidate_topologies(graph, evaluator.link_records, spec.quorum_size)
    return Stage21EvaluationContext(
        fixture=scenario_fixture_from_spec(spec),
        time_step=time_step,
        sequence_id=sequence_id or spec.scenario_id,
        graph=graph,
        evaluator=evaluator,
        link_records=evaluator.link_records,
        topology_variants=variants,
    )


@dataclass(frozen=True, slots=True)
class LeakageCheckedSplit:
    train_scenario_ids: tuple[str, ...]
    eval_scenario_ids: tuple[str, ...]
    test_scenario_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        sets = [
            set(self.train_scenario_ids),
            set(self.eval_scenario_ids),
            set(self.test_scenario_ids),
        ]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                overlap = sets[left] & sets[right]
                if overlap:
                    raise ValueError(f"split leakage: overlapping scenarios {sorted(overlap)}")

    @property
    def overlap_count(self) -> int:
        return 0  # guaranteed by __post_init__


def split_scenarios(
    specs: Sequence[ProductionScenarioSpec],
    *,
    train_fraction: float = 0.7,
    eval_fraction: float = 0.15,
    seed: int = 31,
) -> LeakageCheckedSplit:
    """Deterministic context-keyed split with no cross-split scenario overlap."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    if train_fraction + eval_fraction >= 1.0:
        raise ValueError("train + eval fractions must leave room for a test split")

    ids = [spec.scenario_id for spec in specs]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(train_fraction * n)))
    n_eval = max(1, int(round(eval_fraction * n)))
    if n_train + n_eval >= n:
        n_eval = max(1, n - n_train - 1)
    train = tuple(sorted(ids[:n_train]))
    eval_ids = tuple(sorted(ids[n_train : n_train + n_eval]))
    test = tuple(sorted(ids[n_train + n_eval :]))
    return LeakageCheckedSplit(
        train_scenario_ids=train,
        eval_scenario_ids=eval_ids,
        test_scenario_ids=test,
    )


@dataclass(frozen=True, slots=True)
class Stage31ProductionDataset:
    dataset_id: str
    specs: tuple[ProductionScenarioSpec, ...]
    split: LeakageCheckedSplit
    teacher_labels: Mapping[str, Mapping[str, object]]
    quality_report: Mapping[str, object]

    def specs_for_split(self, split_name: str) -> tuple[ProductionScenarioSpec, ...]:
        ids = {
            "train": set(self.split.train_scenario_ids),
            "eval": set(self.split.eval_scenario_ids),
            "test": set(self.split.test_scenario_ids),
        }[split_name]
        return tuple(spec for spec in self.specs if spec.scenario_id in ids)


def build_production_dataset(
    config: ProductionScenarioConfig | None = None,
    *,
    train_fraction: float = 0.7,
    eval_fraction: float = 0.15,
) -> Stage31ProductionDataset:
    specs = generate_production_scenarios(config or ProductionScenarioConfig())
    split = split_scenarios(
        specs,
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        seed=(config.seed if config else 31),
    )
    teacher_labels = {spec.scenario_id: build_teacher_label(spec) for spec in specs}
    quality = build_dataset_quality_report(specs, split, teacher_labels)
    return Stage31ProductionDataset(
        dataset_id=STAGE31_DATASET_ID,
        specs=specs,
        split=split,
        teacher_labels=teacher_labels,
        quality_report=quality,
    )


def _split_feasibility(specs, ids):
    chosen = [s for s in specs if s.scenario_id in set(ids)]
    feasible = sum(int(s.feasible_exists) for s in chosen)
    return {
        "count": len(chosen),
        "feasible": feasible,
        "infeasible": len(chosen) - feasible,
    }


def build_dataset_quality_report(
    specs: Sequence[ProductionScenarioSpec],
    split: LeakageCheckedSplit,
    teacher_labels: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    feasibility = summarize_feasibility_distribution(specs)
    unique = len({spec.scenario_id for spec in specs})
    teacher_feasible = sum(
        int(bool(label["feasible_exists"])) for label in teacher_labels.values()
    )
    splits = {
        "train": _split_feasibility(specs, split.train_scenario_ids),
        "eval": _split_feasibility(specs, split.eval_scenario_ids),
        "test": _split_feasibility(specs, split.test_scenario_ids),
    }
    each_split_has_both = all(
        block["feasible"] > 0 and block["infeasible"] > 0 for block in splits.values()
    )
    return {
        "dataset_id": STAGE31_DATASET_ID,
        "scenario_count": len(specs),
        "unique_scenario_count": unique,
        "duplicate_context_rate": 1.0 - unique / len(specs) if specs else 0.0,
        "feasibility_distribution": feasibility,
        "teacher_feasible_count": teacher_feasible,
        "split_sizes": {name: block["count"] for name, block in splits.items()},
        "split_feasibility": splits,
        "split_leakage_overlap_count": split.overlap_count,
        "each_split_has_feasible_and_infeasible": each_split_has_both,
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "production_scale_ready": (
            len(specs) >= 100
            and split.overlap_count == 0
            and each_split_has_both
            and 0.2 <= feasibility["feasible_fraction"] <= 0.9
        ),
    }


def build_dataset_manifest(dataset: Stage31ProductionDataset) -> dict[str, object]:
    return {
        "dataset_id": dataset.dataset_id,
        "stage_id": "stage31_phase_e_scalable_teacher_labels_and_leakage_checked_split",
        "physics_regime_id": STAGE31_PHYSICS_REGIME_ID,
        "protocol_model_id": STAGE31_PROTOCOL_MODEL_ID,
        "objective_contract_id": "stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1",
        "tau_requirement_min": TAU_REQUIREMENT_MIN,
        "scenario_count": len(dataset.specs),
        "split_sizes": dataset.quality_report["split_sizes"],
        "teacher_label_source": "stage31_candidate_enumeration_best_feasible",
        "teacher_label_is_actor_input": False,
        "owner_approval_id": "owner_approved_stage31_production_readiness_unfreeze",
        "artifact_scope": "production_scenario_dataset_only",
    }
