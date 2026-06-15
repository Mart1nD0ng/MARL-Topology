#!/usr/bin/env python3
"""Generate Stage 29 pre-scale decision review documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation.stage29_pre_scale_decision import (  # noqa: E402
    build_stage29_pre_scale_decision_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write-docs",
        action="store_true",
        help="Build and print the decision report without updating docs.",
    )
    args = parser.parse_args()
    report = build_stage29_pre_scale_decision_report(project_root=ROOT)
    if not args.no_write_docs:
        write_stage29_docs(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report.get("pass_gate")) else 1


def write_stage29_docs(report: Mapping[str, object]) -> None:
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "STAGE29_PRE_SCALE_DECISION_REVIEW.md").write_text(
        _main_markdown(report),
        encoding="utf-8",
    )
    (docs_dir / "STAGE29_SCALE_READINESS_SCORECARD.md").write_text(
        _scorecard_markdown(report),
        encoding="utf-8",
    )
    (docs_dir / "STAGE29_ROOT_CAUSE_AND_DECISION_PACKET.md").write_text(
        _decision_packet_markdown(report),
        encoding="utf-8",
    )


def _main_markdown(report: Mapping[str, object]) -> str:
    stage28 = report["stage28_evidence"]  # type: ignore[index]
    delta = stage28["eval_delta_mean"]  # type: ignore[index]
    critic = stage28["critic_health"]  # type: ignore[index]
    packet = report["decision_packet"]  # type: ignore[index]
    lines = [
        "# Stage 29 Pre-Scale Decision Review",
        "",
        "Stage 29 is a decision-review stage. It reads frozen Stage 26, Stage 27, and Stage 28 evidence only. It does not run training, tune reward weights, switch samplers, create checkpoints, select final tau, modify v5, or authorize scale-up.",
        "",
        "## Cybernetic Control Model",
        "",
        "- Controlled object: pre-scale owner decision gate after the repaired-critic small-scale rerun.",
        "- Desired state: decide whether the project is ready for a larger pilot, needs a repair stage, or should hold training.",
        "- State variables: seed completion, reliability deltas, latency/energy deltas, surrogate reward delta, projection rejection, critic health, Stage 26 prior blockers, and forbidden-action flags.",
        "- Sensors: Stage 26 full-system diagnostic, Stage 27 critic repair report, Stage 28 repaired-critic rerun report, PROJECT_STATE, tests, and harness validation.",
        "- Actuators used: report generation, harness task registration, tests, PROJECT_STATE synchronization, and next-stage recommendation only.",
        "- Disturbances: small/duplicated evidence data, reward/objective tension, projection friction, seed variance, and pressure to treat a small pilot as scale readiness.",
        "- Coupling map: critic repair improved the value baseline; remaining policy behavior is coupled to reward/objective alignment, assembler projection constraints, and limited scenario diversity.",
        "",
        "## Decision",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- pass gate: `{report['pass_gate']}`",
        f"- recommended option: `{packet['recommended_option']}`",
        f"- recommended next task: `{packet['recommended_next_task']}`",
        f"- larger pilot approved: `{packet['larger_pilot_approved']}`",
        f"- scale-up approved: `{packet['scale_up_approved']}`",
        f"- owner decision required: `{packet['owner_decision_required']}`",
        "",
        "## Stage 28 Evidence",
        "",
        f"- completed seeds: `{stage28['completed_seed_count']}` / `{stage28['seed_count']}`",
        f"- stop reasons: `{stage28['stop_reasons']}`",
        f"- critic explained variance: `{critic['mean_update_explained_variance']}`",
        f"- critic value-return correlation: `{critic['mean_value_return_correlation']}`",
        f"- normalized value loss: `{critic['mean_normalized_value_loss']}`",
        f"- tau-feasible delta vs supervised: `{delta['tau_feasible_rate_delta']}`",
        f"- violation delta vs supervised: `{delta['violation_rate_delta']}`",
        f"- latency delta vs supervised: `{delta['latency_delta']}`",
        f"- energy delta vs supervised: `{delta['energy_delta']}`",
        f"- surrogate reward delta vs supervised: `{delta['mean_reward_surrogate_delta']}`",
        f"- top proposal rejection delta vs supervised: `{delta['top_proposal_rejection_rate_delta']}`",
        "",
        "## Conclusion",
        "",
        "The repaired critic removed the Stage 25 value-baseline blocker, but the project is not ready for a larger pilot. The next work should repair reward/objective and projection alignment before more policy-gradient training. Scale-up, recurrent policy work, reward tuning, sampler switching, checkpoint creation, and final tau selection remain blocked pending owner decision.",
        "",
        "## Verification",
        "",
        "- `python scripts\\replay\\stage29_pre_scale_decision_review.py`",
        "- `python -m pytest -q`",
        "- `python harness\\scripts\\validate_tasks.py`",
    ]
    return "\n".join(lines) + "\n"


def _scorecard_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Stage 29 Scale Readiness Scorecard",
        "",
        "| Component | Status | Score | Blocks Larger Pilot | Blocks Scale-Up | Confidence |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in report["decision_scorecard"]:  # type: ignore[index]
        lines.append(
            f"| `{row['component']}` | `{row['status']}` | {row['score']} | `{row['blocks_larger_pilot']}` | `{row['blocks_scale_up']}` | `{row['confidence']}` |"
        )
    readiness = report["scale_readiness"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Readiness Verdict",
            "",
            f"- larger pilot ready: `{readiness['larger_pilot_ready']}`",
            f"- scale-up ready: `{readiness['scale_up_ready']}`",
            f"- critic ready: `{readiness['critic_ready']}`",
            f"- reward/objective ready: `{readiness['reward_objective_ready']}`",
            f"- projection ready: `{readiness['projection_ready']}`",
            f"- data ready for scale: `{readiness['data_ready_for_scale']}`",
            f"- reason: {readiness['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_packet_markdown(report: Mapping[str, object]) -> str:
    packet = report["decision_packet"]  # type: ignore[index]
    lines = [
        "# Stage 29 Root Cause and Decision Packet",
        "",
        "## Owner Decision",
        "",
        f"- recommended option: `{packet['recommended_option']}`",
        f"- recommended next task: `{packet['recommended_next_task']}`",
        f"- larger pilot approved: `{packet['larger_pilot_approved']}`",
        f"- scale-up approved: `{packet['scale_up_approved']}`",
        f"- owner decision required: `{packet['owner_decision_required']}`",
        f"- rationale: {packet['rationale']}",
        "",
        "## Root Cause Matrix",
        "",
        "| Component | Status | Evidence | Contribution | Confidence | Recommended Repair | Blocks Larger Pilot | Blocks Scale-Up | Blocks Recurrent Policy | Blocks Reward Tuning | Blocks Sampler Change |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["root_cause_matrix"]:  # type: ignore[index]
        lines.append(
            "| `{component}` | `{status}` | {evidence} | {contribution} | `{confidence}` | {repair} | `{larger}` | `{scale}` | `{recurrent}` | `{reward}` | `{sampler}` |".format(
                component=row["component"],
                status=row["status"],
                evidence=row["evidence"],
                contribution=row["likely_contribution"],
                confidence=row["confidence"],
                repair=row["recommended_repair"],
                larger=row["blocks_larger_pilot"],
                scale=row["blocks_scale_up"],
                recurrent=row["blocks_recurrent_policy"],
                reward=row["blocks_reward_tuning"],
                sampler=row["blocks_sampler_change"],
            )
        )
    lines.extend(
        [
            "",
            "## Options",
            "",
            *_option_lines(packet["owner_options"]),  # type: ignore[index]
            "",
            "## Boundary",
            "",
            "This packet recommends one next stage only. It does not self-authorize the next stage or any training run.",
        ]
    )
    return "\n".join(lines) + "\n"


def _option_lines(options: Mapping[str, str]) -> Sequence[str]:
    return [f"- `{key}`: {value}" for key, value in options.items()]


if __name__ == "__main__":
    raise SystemExit(main())
