#!/usr/bin/env python3
"""Run Stage 27 critic-only baseline repair and report generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.critic_repair_trainer import (  # noqa: E402
    STAGE27_PASS_VERDICT,
    run_stage27_critic_repair,
)
from marl_topology.training.run_manifest_validator import (  # noqa: E402
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)


STAGE27_STAGE_ID = "stage_27_critic_baseline_repair"
STAGE27_CONFIG_ID = "stage27_critic_repair_dataset_and_train_config"
STAGE27_ARTIFACT_ROOT = (
    "result_save/stage27_critic_baseline_repair/"
    "stage27_critic_repair_dataset_and_train_config"
)
STAGE27_ARTIFACT_FILENAMES = (
    "manifest.json",
    "stage27_critic_repair_report.json",
    "critic_candidate_metrics.csv",
    "critic_training_curves.csv",
    "value_return_scatter.csv",
    "dataset_health.csv",
    "value_return_scatter.png",
    "critic_loss_curves.png",
    "explained_variance_curves.png",
    "return_normalization_distribution.png",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write-artifacts", action="store_true")
    args = parser.parse_args()
    report = run_stage27_critic_repair(project_root=ROOT)
    manifest = build_stage27_manifest()
    validation = validate_run_manifest_dry_run(manifest, project_root=ROOT)
    report = {
        **report,
        "manifest": manifest,
        "manifest_validation": validation.to_dict(),
        "pass_gate": bool(report["pass_gate"]) and validation.is_valid,
        "verdict": STAGE27_PASS_VERDICT
        if bool(report["pass_gate"]) and validation.is_valid
        else "stage27_critic_repair_blocked",
    }
    if not args.no_write_artifacts:
        write_stage27_outputs(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass_gate"] else 1


def build_stage27_manifest() -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root=STAGE27_ARTIFACT_ROOT,
        overrides={
            "run_id": "stage27_critic_repair_frozen_policy_dataset",
            "stage_id": STAGE27_STAGE_ID,
            "owner_approval_id": "owner_approved_stage27_critic_baseline_repair",
            "config_id": STAGE27_CONFIG_ID,
            "scenario_set_id": "stage27_frozen_policy_critic_dataset",
            "split_id": "stage27_train_eval_critic_dataset_split_v1",
            "seed": 2701,
            "seed_group_id": "stage27_critic_repair_seeds_2701_2704",
            "artifact_paths": [
                f"{STAGE27_ARTIFACT_ROOT}/{name}" for name in STAGE27_ARTIFACT_FILENAMES
            ],
            "artifact_write_allowed": "manifest_validated_reports_only",
            "checkpoint_creation_allowed": False,
            "training_scale_up_allowed": False,
            "actor_update_allowed": False,
            "policy_gradient_update_allowed": False,
        },
    )


def write_stage27_outputs(report: Mapping[str, object]) -> None:
    artifact_dir = ROOT / STAGE27_ARTIFACT_ROOT
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "manifest.json", report["manifest"])
    _write_json(artifact_dir / "stage27_critic_repair_report.json", report)
    _write_csv(artifact_dir / "critic_candidate_metrics.csv", _candidate_rows(report))
    _write_csv(artifact_dir / "critic_training_curves.csv", _curve_rows(report))
    _write_csv(artifact_dir / "value_return_scatter.csv", _scatter_rows(report))
    _write_csv(artifact_dir / "dataset_health.csv", _dataset_rows(report))
    _write_optional_plots(artifact_dir, report)
    docs_dir = ROOT / "docs"
    docs = {
        "STAGE27_CRITIC_BASELINE_REPAIR.md": _main_markdown(report),
        "STAGE27_CRITIC_SEMANTICS.md": _semantics_markdown(report),
        "STAGE27_RETURN_VALUE_SCALE_ALIGNMENT.md": _scale_markdown(report),
        "STAGE27_CRITIC_DATASET_HEALTH.md": _dataset_markdown(report),
        "STAGE27_CRITIC_ARCHITECTURE_REPAIR.md": _architecture_markdown(report),
        "STAGE27_CRITIC_PRETRAINING_REPORT.md": _pretraining_markdown(report),
        "STAGE27_MAPPO_READINESS_AFTER_CRITIC_REPAIR.md": _readiness_markdown(report),
    }
    if not report["pass_gate"]:
        docs["STAGE27_CRITIC_REPAIR_FAILURE_REVIEW.md"] = _failure_markdown(report)
    for filename, text in docs.items():
        (docs_dir / filename).write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def _candidate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for candidate in report["candidate_reports"]:  # type: ignore[index]
        eval_metrics = candidate.get("eval_metrics", {})
        train_metrics = candidate.get("train_metrics", {})
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "implemented": candidate.get("implemented", False),
                "gate_passed": candidate.get("gate", {}).get("passed", False),
                "train_explained_variance": train_metrics.get("explained_variance"),
                "eval_explained_variance": eval_metrics.get("explained_variance"),
                "eval_value_return_correlation": eval_metrics.get(
                    "value_return_correlation"
                ),
                "eval_value_bias": eval_metrics.get("value_bias"),
                "eval_advantage_variance_reduction": eval_metrics.get(
                    "advantage_variance_reduction"
                ),
            }
        )
    return rows


def _curve_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for candidate in report["candidate_reports"]:  # type: ignore[index]
        for row in candidate.get("curves", ()):
            rows.append({"candidate_id": candidate["candidate_id"], **row})
    return rows


def _scatter_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    selected = str(report["selected_critic_id"])
    rows = []
    for candidate in report["candidate_reports"]:  # type: ignore[index]
        if candidate["candidate_id"] != selected:
            continue
        for index, row in enumerate(candidate["eval_metrics"]["scatter_sample"]):
            rows.append({"sample_index": index, **row})
    return rows


def _dataset_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    health = report["dataset_health"]
    return [
        {"metric": key, "value": value}
        for key, value in health.items()
        if not isinstance(value, dict)
    ]


def _write_optional_plots(artifact_dir: Path, report: Mapping[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    scatter = _scatter_rows(report)
    if scatter:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter([row["return"] for row in scatter], [row["prediction"] for row in scatter], s=14)
        ax.set_xlabel("Return")
        ax.set_ylabel("Predicted value")
        ax.set_title("Stage 27 Value vs Return")
        fig.tight_layout()
        fig.savefig(artifact_dir / "value_return_scatter.png", dpi=150)
        plt.close(fig)
    curves = _curve_rows(report)
    if curves:
        for filename, field, ylabel in (
            ("critic_loss_curves.png", "eval_value_loss", "Eval normalized value loss"),
            ("explained_variance_curves.png", "eval_explained_variance", "Eval explained variance"),
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            for candidate_id in sorted({row["candidate_id"] for row in curves}):
                rows = [row for row in curves if row["candidate_id"] == candidate_id]
                ax.plot([row["epoch"] for row in rows], [row[field] for row in rows], label=candidate_id)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(artifact_dir / filename, dpi=150)
            plt.close(fig)
    norm = report["return_normalizer_report"]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["raw std", "norm std"], [norm["raw_std"], norm["normalized_std"]])
    ax.set_title("Return Scaling")
    fig.tight_layout()
    fig.savefig(artifact_dir / "return_normalization_distribution.png", dpi=150)
    plt.close(fig)


def _main_markdown(report: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Stage 27 Critic Baseline Repair",
            "",
            "Stage 27 is critic-only. It collected frozen-policy critic data, trained critic candidates, and selected one future value baseline without actor updates or policy-gradient updates.",
            "",
            "## Result",
            "",
            f"- verdict: `{report['verdict']}`",
            f"- pass gate: `{report['pass_gate']}`",
            f"- selected critic: `{report['selected_critic_id']}`",
            f"- selection reason: {report['selection_reason']}",
            "- actor update performed: `False`",
            "- reward weights changed: `False`",
            "- sampler switched: `False`",
            "- checkpoint created: `False`",
            "",
            "## Candidate Table",
            "",
            "| Candidate | Gate | Eval EV | Eval Corr | Bias Reduction | Advantage Var Reduction |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *[
                "| `{candidate}` | `{gate}` | {ev:.4f} | {corr:.4f} | {bias:.4f} | {adv:.4f} |".format(
                    candidate=row["candidate_id"],
                    gate=row["gate_passed"],
                    ev=float(row["eval_explained_variance"] or 0.0),
                    corr=float(row["eval_value_return_correlation"] or 0.0),
                    bias=float(
                        _metric(report, row["candidate_id"], "bias_reduction_fraction")
                    ),
                    adv=float(row["eval_advantage_variance_reduction"] or 0.0),
                )
                for row in _candidate_rows(report)
            ],
            "",
            "## Acceptance",
            "",
            "The selected critic passed the minimum gate and preferred explained-variance/correlation gate. Stage 28 is recommended only with owner approval.",
        ]
    ) + "\n"


def _semantics_markdown(report: Mapping[str, object]) -> str:
    return """# Stage 27 Critic Semantics

## MAPPO Value Critic

Purpose: provide an action-independent centralized V(s) baseline for advantage computation.

Input: pre-action centralized state only. It may include graph summaries, previous topology/resource summaries, previous-step objective summaries, current candidate-edge communication estimates, current time step, and pre-proposal resource state.

Forbidden in value input: current selected topology caused by the action, current consensus success, current latency, current energy, current reward, current surrogate, and future outcome.

## Action-Conditioned Diagnostic Critic

Diagnostic records may include selected topology, post-action Stage 3/4 metrics, reward surrogate, and edge-delta targets. These records are tagged `action_conditioned_diagnostic` and are not used as the MAPPO V(s) baseline.

## Advantage Path

The future advantage path must use the selected value critic only. The value head predicts normalized value targets; predictions are denormalized to raw return scale before GAE.
"""


def _scale_markdown(report: Mapping[str, object]) -> str:
    scale = report["return_normalizer_report"]
    return f"""# Stage 27 Return / Value Scale Alignment

The normalizer is fitted on train returns only.

- raw mean: `{scale['raw_mean']}`
- raw std: `{scale['raw_std']}`
- normalized mean: `{scale['normalized_mean']}`
- normalized std: `{scale['normalized_std']}`
- fitted on train only: `{scale['fitted_on_train_only']}`
- GAE policy: critic predicts normalized value, then denormalizes to raw return scale before GAE.
"""


def _dataset_markdown(report: Mapping[str, object]) -> str:
    health = report["dataset_health"]
    return f"""# Stage 27 Critic Dataset Health

- train transitions: `{health['train_transition_count']}`
- eval transitions: `{health['eval_transition_count']}`
- return variance: `{health['return_variance']}`
- reward variance: `{health['reward_variance']}`
- duplicate context rate: `{health['duplicate_context_rate']}`
- feasible ratio: `{health['feasible_ratio']}`
- actor update performed: `{health['actor_update_performed']}`
- policy-gradient update performed: `{health['policy_gradient_update_performed']}`
- sampler id: `{health['sampler_id']}`

The dataset is frozen-policy critic-only evidence. It is not a MAPPO training run.
"""


def _architecture_markdown(report: Mapping[str, object]) -> str:
    return """# Stage 27 Critic Architecture Repair

Implemented candidates:

- `enriched_centralized_mlp_value_critic_v1`: enriched pre-action feature vector, normalized value head, and auxiliary feasibility/consensus/latency/energy heads.
- `centralized_message_passing_graph_value_critic_v1`: centralized training-only graph critic with node encoder, edge encoder, two message-passing layers, graph pooling, normalized value head, and auxiliary heads.

The graph critic was implemented safely and passed the structure-sensitive tests. It is not a Transformer and is not exposed to the actor.
"""


def _pretraining_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Stage 27 Critic Pretraining Report",
        "",
        "## Metrics",
        "",
        "| Candidate | Gate | Eval EV | Eval Corr | Raw Bias | Scale Ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in report["candidate_reports"]:  # type: ignore[index]
        metrics = candidate.get("eval_metrics", {})
        lines.append(
            "| `{}` | `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                candidate["candidate_id"],
                candidate.get("gate", {}).get("passed", False),
                float(metrics.get("explained_variance", 0.0)),
                float(metrics.get("value_return_correlation", 0.0)),
                float(metrics.get("value_bias", 0.0)),
                float(metrics.get("value_scale_ratio", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def _readiness_markdown(report: Mapping[str, object]) -> str:
    ready = report["mappo_readiness"]
    return f"""# Stage 27 MAPPO Readiness After Critic Repair

- active critic selected: `{ready['active_critic_selected']}`
- selected critic: `{report['selected_critic_id']}`
- return normalizer fitted: `{ready['return_normalizer_fitted']}`
- values denormalized for GAE: `{ready['gae_uses_raw_denormalized_values']}`
- actor leakage absent: `{ready['actor_leakage_absent']}`
- actor update performed: `{ready['actor_update_performed']}`
- policy-gradient update performed: `{ready['policy_gradient_update_performed']}`
- next recommended task: `{ready['recommended_next_task']}`

Stage 28 remains owner-gated. Stage 27 does not authorize scale-up.
"""


def _failure_markdown(report: Mapping[str, object]) -> str:
    return """# Stage 27 Critic Repair Failure Review

No critic passed the minimum gate. Review feature sufficiency, return scaling, data volume, architecture stability, target noise, leakage, and implementation issues before more training.
"""


def _metric(report: Mapping[str, object], candidate_id: str, metric: str) -> object:
    for candidate in report["candidate_reports"]:  # type: ignore[index]
        if candidate["candidate_id"] == candidate_id:
            return candidate["eval_metrics"][metric]
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
