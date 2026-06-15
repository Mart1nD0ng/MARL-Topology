#!/usr/bin/env python3
"""Generate Stage 26 full-system health reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.evaluation.stage26_health_diagnostics import (  # noqa: E402
    STAGE26_ARTIFACT_ROOT,
    build_stage26_full_system_health_report,
    component_rows,
    metrics_summary_rows,
    root_cause_rows,
)


COMPONENT_DOCS = {
    "harness_state_health": "STAGE26_HARNESS_STATE_HEALTH.md",
    "data_health": "STAGE26_DATA_HEALTH.md",
    "communication_health": "STAGE26_COMMUNICATION_HEALTH.md",
    "consensus_health": "STAGE26_CONSENSUS_HEALTH.md",
    "reward_objective_health": "STAGE26_REWARD_OBJECTIVE_HEALTH.md",
    "assembler_health": "STAGE26_ASSEMBLER_HEALTH.md",
    "sampler_health": "STAGE26_SAMPLER_HEALTH.md",
    "actor_health": "STAGE26_ACTOR_HEALTH.md",
    "critic_health": "STAGE26_CRITIC_HEALTH.md",
    "mappo_loop_health": "STAGE26_MAPPO_LOOP_HEALTH.md",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write-artifacts", action="store_true")
    args = parser.parse_args()
    report = build_stage26_full_system_health_report(project_root=ROOT)
    if not args.no_write_artifacts:
        write_stage26_outputs(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass_gate"] else 1


def write_stage26_outputs(report: Mapping[str, object]) -> None:
    artifact_dir = ROOT / STAGE26_ARTIFACT_ROOT
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "manifest.json", report["manifest"])
    _write_json(artifact_dir / "stage26_full_system_health_report.json", report)
    _write_csv(artifact_dir / "component_health_scorecard.csv", component_rows(report))
    _write_csv(artifact_dir / "root_cause_matrix.csv", root_cause_rows(report))
    _write_csv(artifact_dir / "health_metrics_summary.csv", metrics_summary_rows(report))
    _write_optional_scorecard_plot(artifact_dir, component_rows(report))

    docs_dir = ROOT / "docs"
    for component_id, filename in COMPONENT_DOCS.items():
        (docs_dir / filename).write_text(
            _component_markdown(report, component_id),
            encoding="utf-8",
        )
    (docs_dir / "STAGE26_FULL_SYSTEM_HEALTH_DIAGNOSTIC.md").write_text(
        _full_report_markdown(report),
        encoding="utf-8",
    )
    (docs_dir / "STAGE26_ROOT_CAUSE_MATRIX_AND_DECISION_PACKET.md").write_text(
        _decision_packet_markdown(report),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _write_optional_scorecard_plot(artifact_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [str(row["component"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    colors = [
        "#2f7d32" if row["status"] == "PASS" else "#b26a00" if row["status"] == "WARN" else "#b3261e"
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(scores)), scores, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Health score")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Stage 26 Component Health Scorecard")
    fig.tight_layout()
    fig.savefig(artifact_dir / "component_health_scorecard.png", dpi=150)
    plt.close(fig)


def _component_markdown(report: Mapping[str, object], component_id: str) -> str:
    component = report["component_health"][component_id]  # type: ignore[index]
    title = _title(component_id)
    lines = [
        f"# {title}",
        "",
        "## Cybernetic Diagnostic",
        "",
        f"- Controlled object: `{component_id}` in the active Stage 25 stack.",
        "- Desired state: enough finite, interpretable evidence to decide whether this component blocks a larger MAPPO pilot.",
        "- Sensors: frozen Stage 25 artifacts, Stage 21/22 evaluation contexts, source scans, manifest validation, and component metrics.",
        "- Actuators used in Stage 26: report generation only. No model updates, no sampler switch, no reward weight change, and no checkpoint creation.",
        "- Disturbances: small source context count, stochastic seed variance, projection effects, reward/objective tension, and critic value noise.",
        "",
        "## Status",
        "",
        f"- status: `{component['status']}`",
        f"- score: `{component['score']}`",
        f"- diagnosis labels: `{', '.join(component['diagnosis_labels'])}`",
        f"- blocks scale-up: `{component['blocks_scale_up']}`",
        f"- likely contribution: {component['likely_contribution_to_stage25_weak_improvement']}",
        f"- confidence: `{component['confidence']}`",
        "",
        "## Key Metrics",
        "",
    ]
    for key, value in component["metrics"].items():
        lines.append(f"- `{key}`: `{_short(value)}`")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            *[f"- {item}" for item in component["evidence"]],
            "",
            "## Residual Risk",
            "",
            "This diagnostic is evaluation-only and cannot prove that a future training run will improve. It identifies the next sensor or repair stage for owner decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def _full_report_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Stage 26 Full-System Health Diagnostic",
        "",
        "Stage 26 is a diagnostic stage, not a training stage. It reads frozen Stage 25 artifacts and Stage 21/22 evaluation contexts, then produces a component health scorecard and root-cause matrix.",
        "",
        "## Controlled System",
        "",
        "- Controlled object: active pre-scale policy-gradient stack, including project state, evidence, communication, consensus, reward surrogate, assembler, sampler, actor, critic, rollout loop, visualization, and scale-readiness gates.",
        "- Desired state: identify which components most plausibly explain Stage 25 weak improvement without changing the active stack.",
        "- Feedback loop: frozen evidence -> component metrics -> health scorecard -> root-cause matrix -> owner decision packet.",
        "- Forbidden actuators avoided: training updates, scale-up, reward tuning, sampler switching, COMA, Transformer, recurrent actor work, final tau selection, checkpoints, v5 writes, and unmanifested artifacts.",
        "",
        "## Summary",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- pass gate: `{report['pass_gate']}`",
        f"- recommended option: `{report['recommended_option']}`",
        f"- recommended next task: `{report['recommended_next_task']}`",
        "- scale-up approved: `False`",
        "",
        "## Component Scorecard",
        "",
        "| Component | Status | Score | Blocks Scale-Up | Labels |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in report["component_scorecard"]:  # type: ignore[index]
        lines.append(
            f"| `{row['component']}` | `{row['status']}` | {row['score']} | `{row['blocks_scale_up']}` | `{row['diagnosis_labels']}` |"
        )
    lines.extend(
        [
            "",
            "## Root Cause Summary",
            "",
            "The strongest Stage 25 limiter is critic health: value prediction explained variance was weak and value scale mismatch was large. Data size/duplication, reward/objective tension, projection friction, and one unstable seed are secondary blockers for scale-up.",
            "",
            "## Verification",
            "",
            "- `python scripts\\replay\\stage26_full_system_health_report.py`",
            "- `python -m pytest -q`",
            "- `python harness\\scripts\\validate_tasks.py`",
        ]
    )
    return "\n".join(lines) + "\n"


def _decision_packet_markdown(report: Mapping[str, object]) -> str:
    packet = report["decision_packet"]
    lines = [
        "# Stage 26 Root Cause Matrix and Decision Packet",
        "",
        "## Decision",
        "",
        f"- recommended option: `{packet['recommended_option']}`",
        f"- recommended next task: `{packet['recommended_next_task']}`",
        f"- scale-up approved: `{packet['scale_up_approved']}`",
        f"- owner decision required: `{packet['owner_decision_required']}`",
        f"- rationale: {packet['option_rationale']}",
        "",
        "## Root Cause Matrix",
        "",
        "| Component | Status | Evidence | Contribution | Confidence | Repair | Blocks Scale-Up | Blocks LSTM | Blocks Reward Tuning | Blocks Sampler Change |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["root_cause_matrix"]:  # type: ignore[index]
        lines.append(
            "| `{component}` | `{status}` | {evidence} | {contribution} | `{confidence}` | {repair} | `{scale}` | `{lstm}` | `{reward}` | `{sampler}` |".format(
                component=row["component"],
                status=row["status"],
                evidence=row["evidence"],
                contribution=row["likely_contribution_to_stage25_weak_improvement"],
                confidence=row["confidence"],
                repair=row["recommended_repair"],
                scale=row["blocks_scale_up"],
                lstm=row["blocks_lstm"],
                reward=row["blocks_reward_tuning"],
                sampler=row["blocks_sampler_change"],
            )
        )
    lines.extend(
        [
            "",
            "## Options",
            "",
        ]
    )
    for key, value in packet["options"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Owner Gate",
            "",
            "Stage 26 recommends the next task only. It does not authorize scale-up, LSTM, reward tuning, sampler switching, checkpoint creation, or any training run.",
        ]
    )
    return "\n".join(lines) + "\n"


def _title(component_id: str) -> str:
    return "Stage 26 " + component_id.replace("_", " ").title()


def _short(value: object) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 500 else text[:497] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
