#!/usr/bin/env python3
"""Generate Stage 25 training visualization tables and optional plots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.mappo.stage25_pilot import STAGE25_ARTIFACT_ROOT  # noqa: E402
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    validate_run_manifest_dry_run,
)


DEFAULT_REPORT_PATH = ROOT / STAGE25_ARTIFACT_ROOT / "training_report.json"


def build_visualization_report(
    *,
    training_report_path: Path = DEFAULT_REPORT_PATH,
    write_artifacts: bool = True,
) -> dict[str, object]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    artifact_dir = training_report_path.parent
    validation = validate_run_manifest_dry_run(report["manifest"], project_root=ROOT)
    if not validation.is_valid:
        raise RuntimeError(f"manifest validation failed: {validation.error_codes()}")

    training_rows = _training_curve_rows(report)
    objective_rows = _objective_curve_rows(report)
    ppo_rows = _ppo_rows(report)
    critic_rows = _critic_rows(report)
    projection_rows = _projection_rows(report)
    topology_rows = _topology_rows(report)
    reward_surface_rows = list(report["reward_surface_analysis"]["rows"])
    before_after_rows = _before_after_rows(report)
    matplotlib_available = False
    plot_files: list[str] = []

    if write_artifacts:
        _write_csv(artifact_dir / "training_curves.csv", training_rows)
        _write_csv(artifact_dir / "objective_curves.csv", objective_rows)
        _write_csv(artifact_dir / "ppo_diagnostics.csv", ppo_rows)
        _write_csv(artifact_dir / "critic_diagnostics.csv", critic_rows)
        _write_csv(artifact_dir / "projection_diagnostics.csv", projection_rows)
        _write_csv(artifact_dir / "topology_diagnostics.csv", topology_rows)
        _write_csv(artifact_dir / "reward_surface.csv", reward_surface_rows)
        _write_csv(artifact_dir / "before_after_comparison.csv", before_after_rows)
        matplotlib_available, plot_files = _write_optional_plots(
            artifact_dir=artifact_dir,
            training_rows=training_rows,
            objective_rows=objective_rows,
            ppo_rows=ppo_rows,
            projection_rows=projection_rows,
            topology_rows=topology_rows,
            reward_surface_rows=reward_surface_rows,
            before_after_rows=before_after_rows,
        )
        report.setdefault("pass_fail_gate", {})["visualization_report_generated"] = True
        report["visualization_report_generated"] = True
        report["visualization_artifact_files"] = [
            "visualization_report.json",
            "training_curves.csv",
            "objective_curves.csv",
            "ppo_diagnostics.csv",
            "critic_diagnostics.csv",
            "projection_diagnostics.csv",
            "topology_diagnostics.csv",
            "reward_surface.csv",
            "before_after_comparison.csv",
            *plot_files,
        ]
        training_report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    visualization_report = {
        "stage": "stage_25_training_visualization_report",
        "source_training_report": str(training_report_path),
        "artifact_dir": str(artifact_dir),
        "manifest_validation": validation.to_dict(),
        "matplotlib_available": matplotlib_available,
        "plot_files": plot_files,
        "csv_outputs": [
            "training_curves.csv",
            "objective_curves.csv",
            "ppo_diagnostics.csv",
            "critic_diagnostics.csv",
            "projection_diagnostics.csv",
            "topology_diagnostics.csv",
            "reward_surface.csv",
            "before_after_comparison.csv",
        ],
        "visualization_artifacts_written": write_artifacts,
        "required_views": {
            "training_curves": bool(training_rows),
            "objective_curves": bool(objective_rows),
            "ppo_diagnostics": bool(ppo_rows),
            "critic_diagnostics": bool(critic_rows),
            "projection_diagnostics": bool(projection_rows),
            "topology_diagnostics": bool(topology_rows),
            "reward_surface": bool(reward_surface_rows),
            "before_after_comparison": bool(before_after_rows),
        },
    }
    if write_artifacts:
        (artifact_dir / "visualization_report.json").write_text(
            json.dumps(visualization_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return visualization_report


def _training_curve_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for row in seed_report["update_metrics"]:
            rows.append(
                {
                    "seed": row["seed"],
                    "update_index": row["update_index"],
                    "phase": "train",
                    "reward": row["mean_reward_surrogate"],
                    "tau_feasible_rate": row["tau_feasible_rate"],
                    "violation_rate": row["violation_rate"],
                }
            )
        for row in seed_report["eval_metrics"]:
            rows.append(
                {
                    "seed": row["seed"],
                    "update_index": row["update_index"],
                    "phase": row["phase"],
                    "reward": row["mean_reward_surrogate"],
                    "tau_feasible_rate": row["tau_feasible_rate"],
                    "violation_rate": row["violation_rate"],
                }
            )
    return rows


def _objective_curve_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for row in seed_report["eval_metrics"]:
            rows.append(
                {
                    "seed": row["seed"],
                    "update_index": row["update_index"],
                    "phase": row["phase"],
                    "consensus_success_probability": row["mean_consensus_success_probability"],
                    "latency": row["mean_latency"],
                    "energy": row["mean_energy"],
                }
            )
    return rows


def _ppo_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    keys = (
        "seed",
        "update_index",
        "approx_kl",
        "entropy",
        "clip_fraction",
        "policy_loss",
        "value_loss",
        "grad_norm",
        "total_loss",
    )
    return [
        {key: row.get(key) for key in keys}
        for seed_report in report["seed_reports"]  # type: ignore[index]
        for row in seed_report["update_metrics"]
    ]


def _critic_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = [
        {
            "seed": row["seed"],
            "update_index": row["update_index"],
            "value_loss": row["value_loss"],
            "explained_variance": row["explained_variance"],
            "value_prediction_mean": row["value_prediction_mean"],
            "return_mean": row["return_mean"],
            "record_type": "update_summary",
        }
        for seed_report in report["seed_reports"]  # type: ignore[index]
        for row in seed_report["update_metrics"]
    ]
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for index, pair in enumerate(seed_report["critic_prediction_return_pairs"]):
            rows.append(
                {
                    "seed": seed_report["seed"],
                    "update_index": seed_report["completed_updates"],
                    "value_prediction": pair["value_prediction"],
                    "return": pair["return"],
                    "pair_index": index,
                    "record_type": "prediction_return_pair",
                }
            )
    return rows


def _projection_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for row in seed_report["eval_metrics"]:
            rows.append(
                {
                    "seed": row["seed"],
                    "update_index": row["update_index"],
                    "phase": row["phase"],
                    "top_proposal_rejection_rate": row["top_proposal_rejection_rate"],
                    "above_threshold_rejection_rate": row["above_threshold_rejection_rate"],
                    "rejection_by_reason": row["rejection_by_reason"],
                }
            )
    return rows


def _topology_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for row in seed_report["eval_metrics"]:
            rows.append(
                {
                    "seed": row["seed"],
                    "update_index": row["update_index"],
                    "phase": row["phase"],
                    "mean_selected_edge_count": row["mean_selected_edge_count"],
                    "selected_edge_count_distribution": row["selected_edge_count_distribution"],
                    "empty_graph_rate": row["empty_graph_rate"],
                    "full_graph_rate": row["full_graph_rate"],
                }
            )
    return rows


def _before_after_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    comparison = report["before_after_comparison"]
    rows = []
    for label, summary in (
        ("supervised_gnn", comparison["supervised_gnn_eval_mean"]),
        ("mappo_fine_tuned_gnn", comparison["mappo_fine_tuned_gnn_eval_mean"]),
    ):
        rows.append({"policy_label": label, **summary})
    for label, summary in comparison["baselines_from_first_seed_eval_split"].items():
        rows.append({"policy_label": label, **summary})
    return rows


def _write_optional_plots(
    *,
    artifact_dir: Path,
    training_rows: Sequence[Mapping[str, object]],
    objective_rows: Sequence[Mapping[str, object]],
    ppo_rows: Sequence[Mapping[str, object]],
    projection_rows: Sequence[Mapping[str, object]],
    topology_rows: Sequence[Mapping[str, object]],
    reward_surface_rows: Sequence[Mapping[str, object]],
    before_after_rows: Sequence[Mapping[str, object]],
) -> tuple[bool, list[str]]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False, []
    plot_files = []
    _line_plot(
        plt,
        artifact_dir / "training_curves.png",
        training_rows,
        y_fields=("reward", "tau_feasible_rate", "violation_rate"),
        title="Stage 25 Training Curves",
    )
    plot_files.append("training_curves.png")
    _line_plot(
        plt,
        artifact_dir / "objective_curves.png",
        objective_rows,
        y_fields=("consensus_success_probability", "latency", "energy"),
        title="Stage 25 Objective Curves",
    )
    plot_files.append("objective_curves.png")
    _line_plot(
        plt,
        artifact_dir / "ppo_diagnostics.png",
        ppo_rows,
        y_fields=("approx_kl", "entropy", "clip_fraction", "policy_loss", "value_loss"),
        title="Stage 25 PPO Diagnostics",
    )
    plot_files.append("ppo_diagnostics.png")
    _line_plot(
        plt,
        artifact_dir / "projection_diagnostics.png",
        projection_rows,
        y_fields=("top_proposal_rejection_rate", "above_threshold_rejection_rate"),
        title="Stage 25 Projection Diagnostics",
    )
    plot_files.append("projection_diagnostics.png")
    _line_plot(
        plt,
        artifact_dir / "topology_diagnostics.png",
        topology_rows,
        y_fields=("mean_selected_edge_count", "empty_graph_rate", "full_graph_rate"),
        title="Stage 25 Topology Diagnostics",
    )
    plot_files.append("topology_diagnostics.png")
    _reward_surface_plot(plt, artifact_dir / "reward_surface_scatter.png", reward_surface_rows)
    plot_files.append("reward_surface_scatter.png")
    _before_after_plot(plt, artifact_dir / "before_after_comparison.png", before_after_rows)
    plot_files.append("before_after_comparison.png")
    return True, plot_files


def _line_plot(plt, path: Path, rows: Sequence[Mapping[str, object]], *, y_fields: tuple[str, ...], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for field in y_fields:
        series = [
            (float(row.get("update_index", 0)), float(row[field]))
            for row in rows
            if row.get(field) is not None
        ]
        if not series:
            continue
        series.sort()
        ax.plot([item[0] for item in series], [item[1] for item in series], marker="o", label=field)
    ax.set_title(title)
    ax.set_xlabel("update")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _reward_surface_plot(plt, path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(
        [float(row["latency"]) for row in rows],
        [float(row["energy"]) for row in rows],
        c=[float(row["surrogate_reward"]) for row in rows],
        cmap="viridis",
        s=[40 if row.get("non_dominated") else 16 for row in rows],
    )
    ax.set_xlabel("latency")
    ax.set_ylabel("energy")
    ax.set_title("Reward Surface")
    fig.colorbar(scatter, ax=ax, label="surrogate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _before_after_plot(plt, path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    labels = [str(row["policy_label"]) for row in rows]
    rewards = [float(row.get("mean_reward_surrogate", 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, rewards)
    ax.set_title("Before/After Reward Comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-write-artifacts", action="store_true")
    args = parser.parse_args()
    report = build_visualization_report(
        training_report_path=args.training_report,
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
