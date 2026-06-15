"""Stage 31 Phase C: recalibrate the feasibility-first surrogate on real data.

This module builds objective records by evaluating candidate topologies for the
Stage 31 procedural scenarios through the existing finite-blocklength +
expected-initiator PBFT stack, recalibrates the latency/energy normalization
references on the *feasible* records (the project's ``feasible_positive_max_v1``
policy), and constructs the owner-approved feasibility-first surrogate config.

It does not run training, write checkpoints, or change tau (fixed at 0.9).
"""

from __future__ import annotations

from typing import Iterable, Sequence

from marl_topology.objectives.normalization import (
    NormalizationReferenceConfig,
    NormalizationReferenceRecord,
    select_normalization_references,
)
from marl_topology.objectives.surrogate_signal import (
    SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
    SurrogateSignalConfig,
)
from marl_topology.policies import PolicyBaselines
from marl_topology.topology import CandidateGraph

from .stage21_objective_stack_evidence import Stage21ObjectiveStackEvaluator
from .stage31_scenario_generator import (
    ProductionScenarioSpec,
    build_stack_config,
    enumerate_candidate_topologies,
)

TAU_REQUIREMENT_MIN = 0.9
STAGE31_SURROGATE_MODEL_ID = "stage31_feasibility_first_surrogate_v2"
STAGE31_SURROGATE_CONFIG_ID = "stage31_feasibility_first_recalibrated_v2"
STAGE31_NORMALIZATION_SOURCE_STAGE_ID = (
    "stage31_phase_b_procedural_scenario_generator_with_tau_feasibility_gradient"
)

# Recalibrated, owner-approved Stage 31 weights. The barrier (not the weights)
# guarantees feasibility-first ordering, so the weights only shape the gradients:
# equal latency/energy importance among feasible topologies, linear reliability
# progress while infeasible.
STAGE31_LATENCY_WEIGHT = 1.0
STAGE31_ENERGY_WEIGHT = 1.0
STAGE31_RELIABILITY_WEIGHT = 1.0
STAGE31_RELIABILITY_PENALTY_POWER = 1.0
STAGE31_CLIP_MAX = 4.0
STAGE31_FEASIBILITY_MARGIN = 1.0


def build_objective_records_from_specs(
    specs: Sequence[ProductionScenarioSpec],
) -> tuple[dict[str, object], ...]:
    """Evaluate candidate topologies per scenario; return objective records.

    Each record carries the registered objective quantities the surrogate and
    the reward-surface analyzer consume. These are evaluation records, not actor
    inputs (no oracle/global leakage).
    """

    records: list[dict[str, object]] = []
    for spec in specs:
        graph = CandidateGraph.from_scene(spec.scene, max_distance_m=None)
        evaluator = Stage21ObjectiveStackEvaluator(
            scene=spec.scene, graph=graph, config=build_stack_config(spec.regime)
        )
        candidates = enumerate_candidate_topologies(
            graph, evaluator.link_records, spec.quorum_size
        )
        for topology_name, edges in candidates.items():
            evaluation = evaluator.evaluate(set(edges))
            records.append(
                {
                    "row_id": f"{spec.scenario_id}:{topology_name}",
                    "policy_label": topology_name,
                    "scenario_id": spec.scenario_id,
                    "selected_edge_count": len(edges),
                    "candidate_edge_count": len(graph.edge_ids),
                    "consensus_success_probability": float(
                        evaluation.metrics["consensus_success_probability"]
                    ),
                    "latency": float(evaluation.metrics["latency"]),
                    "energy": float(evaluation.metrics["energy"]),
                }
            )
    return tuple(records)


def recalibrate_references(
    records: Iterable[dict[str, object]],
    *,
    tau_requirement_min: float = TAU_REQUIREMENT_MIN,
) -> NormalizationReferenceRecord:
    """Select latency/energy references from the feasible records."""

    config = NormalizationReferenceConfig(
        tau_requirement_min=tau_requirement_min,
        source_stage_id=STAGE31_NORMALIZATION_SOURCE_STAGE_ID,
        config_id="stage31_normalization_reference_config",
    )
    return select_normalization_references(records, config)


def build_stage31_surrogate_config(
    references: NormalizationReferenceRecord,
    *,
    tau: float = TAU_REQUIREMENT_MIN,
    latency_weight: float = STAGE31_LATENCY_WEIGHT,
    energy_weight: float = STAGE31_ENERGY_WEIGHT,
    reliability_weight: float = STAGE31_RELIABILITY_WEIGHT,
    reliability_penalty_power: float = STAGE31_RELIABILITY_PENALTY_POWER,
    clip_max: float = STAGE31_CLIP_MAX,
    feasibility_margin: float = STAGE31_FEASIBILITY_MARGIN,
) -> SurrogateSignalConfig:
    """Construct the owner-approved Stage 31 feasibility-first surrogate config."""

    if tau != TAU_REQUIREMENT_MIN:
        raise ValueError("Stage 31 keeps tau fixed at 0.9")
    return SurrogateSignalConfig(
        tau=tau,
        reliability_weight=reliability_weight,
        latency_weight=latency_weight,
        energy_weight=energy_weight,
        latency_reference_s=references.latency_reference_s,
        energy_reference_j=references.energy_reference_j,
        clip_min=0.0,
        clip_max=clip_max,
        reliability_penalty_power=reliability_penalty_power,
        structure=SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
        feasibility_margin=feasibility_margin,
        config_id=STAGE31_SURROGATE_CONFIG_ID,
        model_id=STAGE31_SURROGATE_MODEL_ID,
    )


def build_recalibrated_stage31_surrogate(
    specs: Sequence[ProductionScenarioSpec],
) -> tuple[SurrogateSignalConfig, NormalizationReferenceRecord, tuple[dict[str, object], ...]]:
    """End-to-end: records -> references -> feasibility-first config."""

    records = build_objective_records_from_specs(specs)
    references = recalibrate_references(records)
    config = build_stage31_surrogate_config(references)
    return config, references, records
