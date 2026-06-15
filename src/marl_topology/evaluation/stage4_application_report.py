"""Stage 4.7 PBFT application evaluation report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.protocol import PBFT_PROTOCOL_ACCOUNTING_MODEL_ID

from .stage4_baseline_oracle_review import (
    STAGE4_5_REVIEW_STAGE_ID,
    Stage45ReviewConfig,
    build_stage4_5_baseline_oracle_review,
)


STAGE4_7_REPORT_STAGE_ID = "stage_4_7_pbft_application_evaluation_report"


@dataclass(frozen=True, slots=True)
class Stage47EvaluationReportConfig:
    baseline_config: Stage45ReviewConfig = Stage45ReviewConfig()


@dataclass(frozen=True, slots=True)
class Stage47TopologySummary:
    name: str
    family: str
    selected_edge_count: int
    is_full_graph_baseline: bool
    is_oracle_candidate: bool
    is_deployment_actor_input: bool
    reliability_feasible: bool
    consensus_success_probability: float
    latency_s: float
    energy_j: float

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("name and family must be non-empty")
        if self.selected_edge_count < 0:
            raise ValueError("selected_edge_count must be nonnegative")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if self.latency_s < 0.0 or self.energy_j < 0.0:
            raise ValueError("latency_s and energy_j must be nonnegative")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "selected_edge_count": self.selected_edge_count,
            "is_full_graph_baseline": self.is_full_graph_baseline,
            "is_oracle_candidate": self.is_oracle_candidate,
            "is_deployment_actor_input": self.is_deployment_actor_input,
            "reliability_feasible": self.reliability_feasible,
            "consensus_success_probability": self.consensus_success_probability,
            "latency_s": self.latency_s,
            "energy_j": self.energy_j,
        }


def build_stage4_7_pbft_application_evaluation_report(
    config: Stage47EvaluationReportConfig | None = None,
) -> dict[str, object]:
    """Build a deterministic Stage 4.7 application evaluation report."""

    if config is None:
        config = Stage47EvaluationReportConfig()
    stage45_report = build_stage4_5_baseline_oracle_review(config.baseline_config)
    rows = tuple(_summary_from_stage45_row(row) for row in stage45_report["topology_rows"])
    metric_table = _metric_table(stage45_report["topology_rows"])
    require_registered_metrics(row["metric_name"] for row in metric_table)
    feasible_rows = tuple(row for row in rows if row.reliability_feasible)
    lowest_latency = _best_feasible(feasible_rows, key_name="latency")
    lowest_energy = _best_feasible(feasible_rows, key_name="energy")

    return {
        "stage": STAGE4_7_REPORT_STAGE_ID,
        "source_stage": stage45_report["stage"],
        "scenario": dict(stage45_report["scenario"]),
        "protocol": dict(stage45_report["protocol"]),
        "metric_governance": {
            "registered_metric_names": list(REGISTERED_METRICS.keys()),
            "metric_rows_are_registered": True,
            "new_metric_names_introduced": [],
            "wide_metrics_exported": False,
        },
        "topology_summaries": [row.to_payload() for row in rows],
        "metric_table": metric_table,
        "application_summary": {
            "reliability_threshold": stage45_report["protocol"]["reliability_threshold"],
            "evaluated_topology_count": len(rows),
            "feasible_topology_count": len(feasible_rows),
            "infeasible_topology_names": [
                row.name for row in rows if not row.reliability_feasible
            ],
            "feasible_topology_names": [row.name for row in feasible_rows],
            "lowest_latency_feasible_topology": (
                None if lowest_latency is None else lowest_latency.to_payload()
            ),
            "lowest_energy_feasible_topology": (
                None if lowest_energy is None else lowest_energy.to_payload()
            ),
            "oracle_candidate": dict(stage45_report["oracle_candidate"]),
        },
        "checks": _checks(stage45_report, rows, metric_table),
    }


def _summary_from_stage45_row(row: Mapping[str, object]) -> Stage47TopologySummary:
    metrics = row["metrics"]
    return Stage47TopologySummary(
        name=str(row["name"]),
        family=str(row["family"]),
        selected_edge_count=len(tuple(row["selected_edge_ids"])),
        is_full_graph_baseline=bool(row["is_full_graph_baseline"]),
        is_oracle_candidate=bool(row["is_oracle_candidate"]),
        is_deployment_actor_input=bool(row["is_deployment_actor_input"]),
        reliability_feasible=bool(row["reliability_feasible"]),
        consensus_success_probability=float(metrics["consensus_success_probability"]),
        latency_s=float(metrics["latency"]),
        energy_j=float(metrics["energy"]),
    )


def _metric_table(rows: object) -> tuple[Mapping[str, object], ...]:
    table: list[Mapping[str, object]] = []
    for row in rows:
        for metric_row in row["metric_rows"]:
            table.append(dict(metric_row))
    return tuple(table)


def _best_feasible(
    rows: tuple[Stage47TopologySummary, ...],
    *,
    key_name: str,
) -> Stage47TopologySummary | None:
    if not rows:
        return None
    if key_name == "latency":
        return min(rows, key=lambda row: (row.latency_s, row.energy_j, row.name))
    if key_name == "energy":
        return min(rows, key=lambda row: (row.energy_j, row.latency_s, row.name))
    raise ValueError(f"unknown feasible selection key: {key_name}")


def _checks(
    stage45_report: Mapping[str, object],
    rows: tuple[Stage47TopologySummary, ...],
    metric_table: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    metric_names = {str(row["metric_name"]) for row in metric_table}
    reliability_threshold = float(stage45_report["protocol"]["reliability_threshold"])
    return {
        "source_stage_is_stage4_5": stage45_report["stage"] == STAGE4_5_REVIEW_STAGE_ID,
        "metric_rows_are_registered": metric_names <= set(REGISTERED_METRICS),
        "no_new_metric_names_introduced": metric_names <= set(REGISTERED_METRICS),
        "reliability_constraint_separate_from_latency_energy": all(
            row.reliability_feasible
            == (row.consensus_success_probability >= reliability_threshold)
            for row in rows
        ),
        "full_graph_is_baseline_not_oracle": stage45_report["checks"][
            "full_graph_is_baseline_not_oracle"
        ],
        "oracle_candidate_is_not_actor_input": stage45_report["checks"][
            "oracle_candidate_is_not_actor_input"
        ],
        "protocol_accounting_present": all(
            row_payload["diagnostics"]["protocol_accounting"][
                "protocol_accounting_model_id"
            ]
            == PBFT_PROTOCOL_ACCOUNTING_MODEL_ID
            for row_payload in stage45_report["topology_rows"]
        ),
        "training_run": False,
        "v5_code_migrated": False,
        "reward_implemented": False,
    }
