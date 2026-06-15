#!/usr/bin/env python3
"""Generate Stage 30 closed-loop repair diagnostics and blocker review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation.stage30_repair_diagnostics import (  # noqa: E402
    build_stage30_closed_loop_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write-docs", action="store_true")
    args = parser.parse_args()
    report = build_stage30_closed_loop_report(project_root=ROOT)
    if not args.no_write_docs:
        write_stage30_docs(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] in {
        "stage30_ready_for_large_scale_training_owner_decision",
        "stage30_repair_loop_blocked_awaiting_owner_decision",
    } else 1


def write_stage30_docs(report: Mapping[str, object]) -> None:
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "STAGE30_CLOSED_LOOP_REPAIR_PLAN.md").write_text(
        _plan_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_CURRENT_HEALTH_SNAPSHOT.md").write_text(
        _health_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_REWARD_OBJECTIVE_REPAIR.md").write_text(
        _reward_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_PROJECTION_ALIGNMENT_REPAIR.md").write_text(
        _projection_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_RELIABILITY_MARGIN_REPAIR.md").write_text(
        _margin_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_DATA_EXPANSION_REPAIR.md").write_text(
        _data_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_ROOT_CAUSE_MATRIX.md").write_text(
        _root_cause_markdown(report),
        encoding="utf-8",
    )
    (docs / "STAGE30_REPAIR_LOOP_BLOCKER_REVIEW.md").write_text(
        _blocker_markdown(report),
        encoding="utf-8",
    )


def _plan_markdown(report: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Stage 30 Closed-Loop Repair Plan",
            "",
            "Stage 29 blocked scale-up because critic repair worked, but reward/objective alignment, projection friction, reliability margin, and data scale remained unresolved.",
            "",
            "## Control Model",
            "",
            "- Controlled object: pre-scale repair loop for the active policy-gradient stack.",
            "- Desired state: either large-scale readiness certificate or a blocker review with owner decision required.",
            "- State variables: component health, reliability margin, reward/objective rank alignment, projection rejection, data scale, critic health, and loop iteration count.",
            "- Sensors: Stage 26 diagnostics, Stage 27 critic report, Stage 28 repaired-critic pilot, Stage 29 decision packet, Stage 30 tests, and harness validation.",
            "- Actuators: diagnostic repair code, report generation, readiness gate, root-cause matrix, and owner-gated next-stage recommendation.",
            "- Disturbances: small duplicated data, stochastic policy-gradient outcomes, reward/projection coupling, and pressure to promote small-pilot evidence into scale-up.",
            "",
            "## Iteration Protocol",
            "",
            "Each iteration diagnoses, chooses the dominant blocker by priority, applies the smallest executable repair or repair candidate, validates with report evidence, updates the root-cause matrix, then either continues or stops.",
            "",
            "## Entropy Control",
            "",
            "Stage 30 uses internal iteration ids instead of new stage numbers. Durable behavior lives in `src/marl_topology/evaluation/stage30_repair_diagnostics.py`; generated docs are the human review surface.",
            "",
            "## Boundary",
            "",
            "- large-scale training remains blocked",
            "- final tau selection remains blocked",
            "- reward-weight sweeps remain blocked",
            "- sampler switching remains blocked",
            "- recurrent policy work remains blocked",
            "- owner decision is required for Stage 31",
        ]
    ) + "\n"


def _health_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Stage 30 Current Health Snapshot",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- iteration count: `{report['iteration_count']}`",
        f"- dominant blocker: `{report['dominant_blocker']['component']}`",  # type: ignore[index]
        f"- recommended next task: `{report['recommended_next_task']}`",
        "",
        "| Component | Status | Evidence | Confidence | Repair Class |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["component_health_snapshot"]:  # type: ignore[index]
        lines.append(
            f"| `{row['component']}` | `{row['status']}` | {row['evidence']} | `{row['confidence']}` | `{row['recommended_repair_class']}` |"
        )
    return "\n".join(lines) + "\n"


def _reward_markdown(report: Mapping[str, object]) -> str:
    repair = report["reward_objective_repair"]  # type: ignore[index]
    return "\n".join(
        [
            "# Stage 30 Reward Objective Repair",
            "",
            "Stage 30 implements an objective-order repair candidate, not an active training surrogate replacement. The candidate enforces feasibility-first ranking, then latency, then energy, with no reward-weight sweep.",
            "",
            f"- repair id: `{repair['repair_id']}`",
            f"- before inversion rate: `{repair['before_inversion_rate']}`",
            f"- after inversion rate: `{repair['after_inversion_rate']}`",
            f"- before inversion count: `{repair['before_inversion_count']}`",
            f"- after inversion count: `{repair['after_inversion_count']}`",
            f"- alignment improved: `{repair['alignment_improved']}`",
            f"- active training surrogate changed: `{repair['active_training_surrogate_changed']}`",
            f"- owner activation required: `{repair['owner_activation_required']}`",
            f"- weight sweep performed: `{repair['weight_sweep_performed']}`",
            "",
            "The repair improves the rank diagnostic but remains blocked for active training until the owner approves changing the Stage 5 surrogate structure.",
        ]
    ) + "\n"


def _projection_markdown(report: Mapping[str, object]) -> str:
    repair = report["projection_alignment_repair"]  # type: ignore[index]
    return "\n".join(
        [
            "# Stage 30 Projection Alignment Repair",
            "",
            f"- repair id: `{repair['repair_id']}`",
            f"- top proposal rejection delta: `{repair['top_proposal_rejection_delta']}`",
            f"- above-threshold rejection delta: `{repair['above_threshold_rejection_delta']}`",
            f"- selected edge count shift: `{repair['selected_edge_count_shift']}`",
            f"- projection friction decreased: `{repair['projection_friction_decreased']}`",
            f"- active actor loss changed: `{repair['active_actor_loss_changed']}`",
            "",
            f"Evidence: {repair['evidence']}",
            "",
            "Stage 30 records the projection-aware repair design but does not modify actor inputs or switch samplers.",
        ]
    ) + "\n"


def _margin_markdown(report: Mapping[str, object]) -> str:
    repair = report["reliability_margin_repair"]  # type: ignore[index]
    return "\n".join(
        [
            "# Stage 30 Reliability Margin Repair",
            "",
            f"- tau requirement min: `{repair['tau_requirement_min']}`",
            f"- tau changed: `{repair['tau_changed']}`",
            f"- mean margin: `{repair['mean_margin']}`",
            f"- baseline mean margin: `{repair['baseline_mean_margin']}`",
            f"- margin shift: `{repair['margin_shift']}`",
            f"- min margin: `{repair['min_margin']}`",
            f"- near-zero-margin ratio: `{repair['near_zero_margin_ratio']}`",
            f"- negative-margin ratio: `{repair['negative_margin_ratio']}`",
            f"- tau-feasible delta: `{repair['tau_feasible_rate_delta']}`",
            f"- violation delta: `{repair['violation_rate_delta']}`",
            f"- readiness margin passed: `{repair['readiness_margin_passed']}`",
            "",
            "The monitor is active in readiness assessment only. It does not lower tau or create a new final tau.",
        ]
    ) + "\n"


def _data_markdown(report: Mapping[str, object]) -> str:
    repair = report["data_expansion_repair"]  # type: ignore[index]
    return "\n".join(
        [
            "# Stage 30 Data Expansion Repair",
            "",
            f"- repair id: `{repair['repair_id']}`",
            f"- status from Stage 26: `{repair['status_from_stage26']}`",
            f"- score from Stage 26: `{repair['score_from_stage26']}`",
            f"- scenario data expanded: `{repair['scenario_data_expanded']}`",
            f"- sufficient for large scale: `{repair['sufficient_for_large_scale']}`",
            f"- owner scope required: `{repair['owner_scope_required']}`",
            f"- evidence: {repair['evidence_from_stage26']}",
            "",
            f"Recommended expansion: {repair['recommended_expansion']}",
        ]
    ) + "\n"


def _root_cause_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Stage 30 Root Cause Matrix",
        "",
        "| Component | Status | Evidence | Repair Applied | Effect | Remaining Risk | Next Action | Confidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["root_cause_matrix"]:  # type: ignore[index]
        lines.append(
            f"| `{row['component']}` | `{row['status']}` | {row['evidence']} | {row['repair_applied']} | {row['after_repair_effect']} | {row['remaining_risk']} | {row['next_action']} | `{row['confidence']}` |"
        )
    return "\n".join(lines) + "\n"


def _blocker_markdown(report: Mapping[str, object]) -> str:
    review = report["blocker_review"]  # type: ignore[index]
    readiness = report["large_scale_readiness"]  # type: ignore[index]
    return "\n".join(
        [
            "# Stage 30 Repair Loop Blocker Review",
            "",
            f"- max iterations reached: `{review['max_iterations_reached']}`",
            f"- large-scale readiness passed: `{review['large_scale_readiness_passed']}`",
            f"- most important remaining blocker: `{review['most_important_remaining_blocker']}`",
            f"- recommended owner decision: `{review['recommended_owner_decision']}`",
            f"- readiness issues: `{readiness['issues']}`",
            "",
            "Stage 30 stops here. Large-scale training is not authorized.",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
