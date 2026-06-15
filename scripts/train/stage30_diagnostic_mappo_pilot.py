#!/usr/bin/env python3
"""Produce the Stage 30 diagnostic MAPPO pilot assessment."""

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
    parser.add_argument("--no-write-doc", action="store_true")
    args = parser.parse_args()
    report = build_stage30_closed_loop_report(project_root=ROOT)
    pilot = build_pilot_report(report)
    if not args.no_write_doc:
        (ROOT / "docs" / "STAGE30_DIAGNOSTIC_MAPPO_PILOT.md").write_text(
            _pilot_markdown(pilot),
            encoding="utf-8",
        )
    print(json.dumps(pilot, indent=2, sort_keys=True))
    return 0 if pilot["script_passed"] else 1


def build_pilot_report(loop_report: Mapping[str, object]) -> dict[str, object]:
    pilot = dict(loop_report["diagnostic_pilot"])  # type: ignore[index]
    readiness = loop_report["large_scale_readiness"]  # type: ignore[index]
    return {
        "stage": "stage30_diagnostic_mappo_pilot",
        "script_passed": True,
        "new_training_run": pilot["new_training_run"],
        "source_pilot": pilot["source_pilot"],
        "completed_seed_count": pilot["completed_seed_count"],
        "all_seeds_completed": pilot["all_seeds_completed"],
        "critic_ev": pilot["critic_ev"],
        "critic_value_return_correlation": pilot["critic_value_return_correlation"],
        "latency_improved": pilot["latency_improved"],
        "energy_improved": pilot["energy_improved"],
        "surrogate_improved": pilot["surrogate_improved"],
        "projection_improved": pilot["projection_improved"],
        "reliability_large_scale_gate_passed": pilot["reliability_large_scale_gate_passed"],
        "objective_alignment_candidate_improved": pilot["objective_alignment_candidate_improved"],
        "diagnostic_pilot_passed": pilot["diagnostic_pilot_passed"],
        "large_scale_readiness_passed": readiness["passed"],
        "large_scale_training_allowed": readiness["large_scale_training_allowed"],
        "readiness_issues": readiness["issues"],
        "owner_decision_required": True,
    }


def _pilot_markdown(pilot: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Stage 30 Diagnostic MAPPO Pilot",
            "",
            "Stage 30 stops before running a new policy update because the repair loop reaches owner-gated reward/projection/data blockers. This document records the latest fixed small-scale pilot evidence used by the Stage 30 readiness gate.",
            "",
            f"- source pilot: `{pilot['source_pilot']}`",
            f"- new training run: `{pilot['new_training_run']}`",
            f"- completed seed count: `{pilot['completed_seed_count']}`",
            f"- all seeds completed: `{pilot['all_seeds_completed']}`",
            f"- critic EV: `{pilot['critic_ev']}`",
            f"- critic value-return correlation: `{pilot['critic_value_return_correlation']}`",
            f"- latency improved: `{pilot['latency_improved']}`",
            f"- energy improved: `{pilot['energy_improved']}`",
            f"- surrogate improved: `{pilot['surrogate_improved']}`",
            f"- projection improved: `{pilot['projection_improved']}`",
            f"- reliability large-scale gate passed: `{pilot['reliability_large_scale_gate_passed']}`",
            f"- objective-alignment candidate improved: `{pilot['objective_alignment_candidate_improved']}`",
            f"- diagnostic pilot passed: `{pilot['diagnostic_pilot_passed']}`",
            f"- large-scale readiness passed: `{pilot['large_scale_readiness_passed']}`",
            f"- large-scale training allowed: `{pilot['large_scale_training_allowed']}`",
            f"- readiness issues: `{pilot['readiness_issues']}`",
            "",
            "This script is intentionally a training-gate sensor, not a new training run.",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
