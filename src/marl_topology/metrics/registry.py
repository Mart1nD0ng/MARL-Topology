"""Registered metric names for Stage 2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    definition: str
    range_unit: str
    level: str
    used_for: str
    formula_source: str
    dependencies: tuple[str, ...]
    tests: tuple[str, ...]


_CORE_METRICS = {
    "consensus_success": MetricDefinition(
        name="consensus_success",
        definition="Binary success of one minimal consensus evaluation.",
        range_unit="0 or 1",
        level="topology evaluation",
        used_for="constraint, oracle, evaluation",
        formula_source="docs/METRIC_CONTRACT.md and docs/PROTOCOL_CONTRACT.md",
        dependencies=("selected_topology", "consensus_config", "link_records"),
        tests=("quorum boundary", "topology response", "metric name registration"),
    ),
    "consensus_success_probability": MetricDefinition(
        name="consensus_success_probability",
        definition="Stage 2 deterministic estimate of consensus success probability.",
        range_unit="[0, 1]",
        level="topology evaluation",
        used_for="constraint, oracle, evaluation",
        formula_source="docs/METRIC_CONTRACT.md and Stage 2 evaluator contract",
        dependencies=("selected_topology", "link_success_probability", "quorum_size"),
        tests=("probability range", "topology response"),
    ),
    "latency": MetricDefinition(
        name="latency",
        definition="Maximum selected-link latency in the minimal topology evaluation.",
        range_unit="seconds",
        level="topology evaluation",
        used_for="objective after reliability is feasible",
        formula_source="docs/METRIC_CONTRACT.md and simple link model",
        dependencies=("selected_link_records",),
        tests=("latency nonnegative",),
    ),
    "energy": MetricDefinition(
        name="energy",
        definition="Sum of selected-link energy in the minimal topology evaluation.",
        range_unit="joules",
        level="topology evaluation",
        used_for="objective after reliability is feasible",
        formula_source="docs/METRIC_CONTRACT.md and simple link model",
        dependencies=("selected_link_records",),
        tests=("energy nonnegative",),
    ),
    "topology_diagnostics": MetricDefinition(
        name="topology_diagnostics",
        definition="Diagnostic metadata for selected topology structure.",
        range_unit="mapping",
        level="topology evaluation",
        used_for="diagnostic",
        formula_source="docs/METRIC_CONTRACT.md",
        dependencies=("candidate_graph", "selected_topology"),
        tests=("full graph is baseline not oracle", "edge count diagnostic only"),
    ),
}

REGISTERED_METRICS = MappingProxyType(_CORE_METRICS)


def require_registered_metrics(metric_names: Iterable[str]) -> None:
    unknown = sorted(set(metric_names) - set(REGISTERED_METRICS))
    if unknown:
        raise ValueError(f"unregistered metric names: {unknown}")
