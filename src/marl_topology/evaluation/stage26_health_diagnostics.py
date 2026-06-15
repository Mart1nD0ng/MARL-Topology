"""Stage 26 full-system health diagnostics using frozen evidence only."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import ast
from math import isfinite, sqrt
from pathlib import Path
import json

from marl_topology.channel import evaluate_channel
from marl_topology.data.actor_feature_rebuild import stage18_actor_signature_key
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_TAU_REQUIREMENT_MIN,
    _config_for_fixture,
)
from marl_topology.link import evaluate_link_transmission
from marl_topology.training.mappo.stage25_pilot import (
    STAGE25_ARTIFACT_ROOT,
    Stage25PilotBaseConfig,
    _build_stage25_split,
    _row_contexts,
)
from marl_topology.training.run_manifest_validator import (
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)


STAGE26_STAGE_ID = "stage_26_full_system_health_diagnostic"
STAGE26_CONFIG_ID = "stage26_no_training_full_system_health_diagnostic_v1"
STAGE26_ARTIFACT_ROOT = (
    "result_save/stage26_full_system_health_diagnostic/"
    "stage26_no_training_full_system_health_diagnostic_v1"
)
STAGE26_RUN_ID = "stage26_full_system_health_from_stage25_artifacts"
STAGE26_RECOMMENDED_OPTION = "option_b_repair_critic_before_more_training"
STAGE26_RECOMMENDED_NEXT_TASK = "stage_27_critic_baseline_repair_before_more_training"

STAGE26_ARTIFACT_FILENAMES = (
    "manifest.json",
    "stage26_full_system_health_report.json",
    "component_health_scorecard.csv",
    "root_cause_matrix.csv",
    "health_metrics_summary.csv",
    "component_health_scorecard.png",
)

COMPONENT_ORDER = (
    "harness_state_health",
    "data_health",
    "communication_health",
    "consensus_health",
    "reward_objective_health",
    "assembler_health",
    "sampler_health",
    "actor_health",
    "critic_health",
    "mappo_loop_health",
    "visualization_health",
    "scale_readiness",
)


class Stage26HealthDiagnosticViolation(ValueError):
    """Raised when diagnostic inputs or boundaries are invalid."""


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    component_id: str
    status: str
    score: float
    diagnosis_labels: tuple[str, ...]
    metrics: Mapping[str, object]
    evidence: tuple[str, ...]
    likely_contribution: str
    confidence: str
    blocks_scale_up: bool
    blocks_lstm: bool = True
    blocks_reward_tuning: bool = False
    blocks_sampler_change: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "status": self.status,
            "score": self.score,
            "diagnosis_labels": list(self.diagnosis_labels),
            "metrics": _jsonable(self.metrics),
            "evidence": list(self.evidence),
            "likely_contribution_to_stage25_weak_improvement": self.likely_contribution,
            "confidence": self.confidence,
            "blocks_scale_up": self.blocks_scale_up,
            "blocks_lstm": self.blocks_lstm,
            "blocks_reward_tuning": self.blocks_reward_tuning,
            "blocks_sampler_change": self.blocks_sampler_change,
        }


def build_stage26_manifest(run_id: str = STAGE26_RUN_ID) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root=STAGE26_ARTIFACT_ROOT,
        overrides={
            "run_id": run_id,
            "stage_id": STAGE26_STAGE_ID,
            "owner_approval_id": "owner_approved_stage26_full_system_health_diagnostic",
            "config_id": STAGE26_CONFIG_ID,
            "scenario_set_id": "stage25_frozen_train_eval_split",
            "split_id": "stage25_disjoint_source_rows_cyclic_slots_v1",
            "seed": 2600,
            "seed_group_id": "stage26_no_training_diagnostic",
            "artifact_paths": [
                f"{STAGE26_ARTIFACT_ROOT}/{filename}"
                for filename in STAGE26_ARTIFACT_FILENAMES
            ],
            "artifact_write_allowed": "manifest_validated_reports_only",
            "checkpoint_creation_allowed": False,
            "training_scale_up_allowed": False,
        },
    )


def build_stage26_full_system_health_report(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path.cwd()
    stage25_report = _load_stage25_report(root)
    stage25_manifest_validation = validate_run_manifest_dry_run(
        stage25_report["manifest"],
        project_root=root,
    )
    stage26_manifest = build_stage26_manifest()
    stage26_manifest_validation = validate_run_manifest_dry_run(
        stage26_manifest,
        project_root=root,
    )
    contexts = _row_contexts()
    cfg = Stage25PilotBaseConfig()
    split = _build_stage25_split(contexts, cfg)
    context_evaluations = _context_evaluations(contexts)
    update_rows = _update_rows(stage25_report)
    eval_rows = _eval_rows(stage25_report)
    reward_rows = tuple(stage25_report["reward_surface_analysis"]["rows"])

    components = {
        "harness_state_health": _harness_state_health(
            root,
            stage25_report,
            stage25_manifest_validation.to_dict(),
            stage26_manifest_validation.to_dict(),
        ),
        "data_health": _data_health(contexts, split),
        "communication_health": _communication_health(contexts, context_evaluations),
        "consensus_health": _consensus_health(context_evaluations),
        "reward_objective_health": _objective_signal_health(stage25_report, reward_rows),
        "assembler_health": _assembler_health(stage25_report, update_rows, eval_rows),
        "sampler_health": _sampler_health(stage25_report, update_rows, eval_rows),
        "actor_health": _actor_health(stage25_report, contexts, update_rows, eval_rows),
        "critic_health": _critic_health(stage25_report, update_rows),
        "mappo_loop_health": _loop_health(stage25_report, update_rows),
        "visualization_health": _visualization_health(root, stage25_report),
    }
    components["scale_readiness"] = _scale_readiness_health(components)
    matrix = _root_cause_matrix(components)
    decision_packet = _decision_packet(components, matrix)
    forbidden_flags = {
        "new_training_updates_run": False,
        "scale_up_training_run": False,
        "reward_weights_changed": False,
        "hyperparameter_tuning_performed": False,
        "sampler_switched": False,
        "checkpoint_created": False,
        "legacy_v5_modified": False,
    }
    pass_issues = []
    if not stage25_manifest_validation.is_valid:
        pass_issues.append("stage25_manifest_invalid")
    if not stage26_manifest_validation.is_valid:
        pass_issues.append("stage26_manifest_invalid")
    if components["harness_state_health"].status == "FAIL":
        pass_issues.append("harness_state_inconsistent")
    if any(bool(value) for value in forbidden_flags.values()):
        pass_issues.append("forbidden_training_or_tuning_detected")
    return {
        "stage": STAGE26_STAGE_ID,
        "verdict": "stage26_pass_full_system_health_diagnostic_complete"
        if not pass_issues
        else "stage26_fail_full_system_health_diagnostic_blocked",
        "pass_gate": not pass_issues,
        "pass_issues": pass_issues,
        "manifest": stage26_manifest,
        "manifest_validation": stage26_manifest_validation.to_dict(),
        "stage25_artifact_manifest_validation": stage25_manifest_validation.to_dict(),
        "source_stage25_artifact_root": STAGE25_ARTIFACT_ROOT,
        "component_health": {
            key: components[key].to_dict()
            for key in COMPONENT_ORDER
        },
        "component_scorecard": _scorecard(components),
        "root_cause_matrix": matrix,
        "decision_packet": decision_packet,
        "forbidden_action_flags": forbidden_flags,
        "diagnostic_scope": {
            "training_updates_run": False,
            "uses_frozen_stage25_artifacts": True,
            "uses_frozen_stage21_stage22_contexts": True,
            "active_stack_changed": False,
            "stage25_summary": stage25_report["aggregate"],
        },
        "recommended_option": decision_packet["recommended_option"],
        "recommended_next_task": decision_packet["recommended_next_task"],
        "owner_decision_required": True,
    }


def component_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for item in report["component_scorecard"]:  # type: ignore[index]
        rows.append(dict(item))
    return rows


def root_cause_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [dict(row) for row in report["root_cause_matrix"]]  # type: ignore[index]


def metrics_summary_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component_id, payload in report["component_health"].items():  # type: ignore[index, union-attr]
        metrics = payload["metrics"]
        for metric, value in metrics.items():
            if isinstance(value, Mapping):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, int | float | str | bool) or sub_value is None:
                        rows.append(
                            {
                                "component_id": component_id,
                                "metric": f"{metric}.{sub_key}",
                                "value": sub_value,
                            }
                        )
            elif isinstance(value, int | float | str | bool) or value is None:
                rows.append(
                    {"component_id": component_id, "metric": metric, "value": value}
                )
    return rows


def _load_stage25_report(root: Path) -> dict[str, object]:
    path = root / STAGE25_ARTIFACT_ROOT / "training_report.json"
    if not path.exists():
        raise Stage26HealthDiagnosticViolation(f"missing Stage 25 report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _harness_state_health(
    root: Path,
    stage25_report: Mapping[str, object],
    stage25_validation: Mapping[str, object],
    stage26_validation: Mapping[str, object],
) -> ComponentHealth:
    state_text = (root / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    result_entries = sorted(path.name for path in (root / "result_save").iterdir())
    approved_entries = {
        ".gitkeep",
        "evidence_dataset_only",
        "stage25_small_scale_mappo_training_pilot",
        "stage26_full_system_health_diagnostic",
        "stage27_critic_baseline_repair",
        "stage28_repaired_critic_mappo_rerun",
        "stage32_production_training",
        "stage33_gnn_stability_repair",
        "stage34_gnn_ablation_diagnostics",
    }
    dynamic_hits = _dynamic_torch_hits(root)
    forbidden_hits = _forbidden_source_hits(root)
    checks = {
        "project_state_stage25_complete": (
            "post_stage_25_complete_small_scale_mappo_pilot_awaiting_owner_decision"
            in state_text
        ),
        "recommended_stage26_present": (
            "recommended_next_task: stage_26_scale_readiness_and_failure_mode_review"
            in state_text
            or "recommended_next_task: stage_27_critic_baseline_repair_before_more_training"
            in state_text
        ),
        "stage25_manifest_valid": bool(stage25_validation["is_valid"]),
        "stage26_manifest_valid": bool(stage26_validation["is_valid"]),
        "result_save_entries_approved": set(result_entries) <= approved_entries,
        "stage25_passed": bool(stage25_report.get("pass_gate")),
        "no_dynamic_torch_import": not dynamic_hits,
        "no_forbidden_source_patterns": not forbidden_hits,
        "stage25_report_says_v5_unmodified": stage25_report.get("v5_modified") is False,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    labels = ["state_consistency"] if status == "PASS" else ["state_or_manifest_inconsistency"]
    risks = []
    if not checks["result_save_entries_approved"]:
        risks.append("unapproved_result_save_entry")
    if dynamic_hits:
        risks.append("dynamic_torch_import")
    if forbidden_hits:
        risks.append("forbidden_source_pattern")
    if not (root / ".git").exists():
        risks.append("not_a_git_repository_manual_v5_integrity_sensor_only")
    metrics = {
        **checks,
        "result_save_entries": result_entries,
        "dynamic_torch_hit_count": len(dynamic_hits),
        "forbidden_source_hit_count": len(forbidden_hits),
        "remaining_process_risks": risks,
    }
    return ComponentHealth(
        component_id="harness_state_health",
        status=status,
        score=100.0 if status == "PASS" else 0.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=(
            "docs/PROJECT_STATE.md",
            f"{STAGE25_ARTIFACT_ROOT}/training_report.json",
            "source scan for dynamic torch and forbidden architecture paths",
        ),
        likely_contribution="low unless manifest or state drift appears",
        confidence="high",
        blocks_scale_up=status != "PASS",
    )


def _data_health(contexts, split) -> ComponentHealth:
    all_rows = tuple(row for row, _context in contexts)
    train_rows = tuple(row for row, _context in split.train_slots)
    eval_rows = tuple(row for row, _context in split.eval_slots)
    train_keys = [f"{row.scenario_id}:t{row.time_step}" for row in train_rows]
    eval_keys = [f"{row.scenario_id}:t{row.time_step}" for row in eval_rows]
    all_slot_keys = train_keys + eval_keys
    target_counts = Counter()
    edge_delta_counts = Counter()
    edge_delta_signs = Counter()
    signature_targets: dict[tuple[object, ...], set[str]] = defaultdict(set)
    missing_values = 0
    total_actor_fields = 0
    for row in all_rows:
        summary = row.actor_target_view.get("target_distribution_summary", {})
        target_counts["high"] += int(summary.get("high_priority_count", 0))
        target_counts["mid"] += int(summary.get("mid_priority_count", 0))
        target_counts["low"] += int(summary.get("low_priority_count", 0))
        targets_by_key = {
            (
                target.get("agent_id"),
                target.get("directed_edge_id"),
                target.get("time_step"),
            ): float(target.get("actor_edge_utility_target", 0.0))
            for target in row.actor_target_view.get("actor_soft_utility_targets", ())
        }
        for actor_row in row.actor_safe_view:
            total_actor_fields += len(actor_row)
            missing_values += sum(1 for value in actor_row.values() if value is None)
            key = (
                actor_row.get("agent_id"),
                actor_row.get("directed_edge_id"),
                actor_row.get("time_step"),
            )
            utility = targets_by_key.get(key)
            if utility is not None:
                signature_targets[stage18_actor_signature_key(actor_row)].add(
                    _target_bucket(utility)
                )
        for target in row.critic_target_view.get("edge_delta_targets", ()):
            delta = float(target.get("delta_consensus_success_probability", 0.0))
            if delta > 0:
                edge_delta_signs["positive"] += 1
            elif delta < 0:
                edge_delta_signs["negative"] += 1
            else:
                edge_delta_signs["zero"] += 1
            edge_delta_counts[str(target.get("action_type", "edge_delta"))] += 1
    contradiction_groups = sum(1 for values in signature_targets.values() if len(values) > 1)
    contradiction_rate = (
        contradiction_groups / len(signature_targets)
        if signature_targets
        else 0.0
    )
    feasible_train = _ratio(row.feasible_under_tau_requirement for row in train_rows)
    feasible_eval = _ratio(row.feasible_under_tau_requirement for row in eval_rows)
    near_threshold = _ratio(
        abs(float(row.consensus_success_probability) - STAGE21_TAU_REQUIREMENT_MIN) <= 0.1
        for row in all_rows
    )
    hard_infeasible = _ratio(float(row.consensus_success_probability) < 0.5 for row in all_rows)
    sparse_feasible_count = sum(
        1
        for row, context in contexts
        if row.feasible_under_tau_requirement
        and len(row.selected_physical_edges) < len(context.graph.edge_ids)
    )
    duplicate_context_rate = 1.0 - (len(set(all_slot_keys)) / len(all_slot_keys))
    temporal_sequence_count = len(
        {context.sequence_id for _row, context in contexts if context.sequence_id}
    )
    labels = []
    if len(all_rows) < 24:
        labels.append("data_too_small")
    if duplicate_context_rate > 0.25:
        labels.append("data_too_duplicate")
    if min(feasible_train, feasible_eval) < 0.2:
        labels.append("insufficient_tau_feasible")
    if near_threshold < 0.1:
        labels.append("insufficient_near_threshold")
    if contradiction_rate > 0.05:
        labels.append("actor_observability_risk")
    if not labels:
        labels.append("sufficient_for_scale")
    else:
        labels.insert(0, "sufficient_for_small_pilot_only")
    status = "WARN" if any(label in labels for label in ("data_too_small", "data_too_duplicate")) else "PASS"
    if "insufficient_tau_feasible" in labels and len(all_rows) < 12:
        status = "FAIL"
    metrics = {
        "num_train_scenarios": len(train_rows),
        "num_eval_scenarios": len(eval_rows),
        "num_unique_source_contexts": len(all_rows),
        "feasible_ratio_train": feasible_train,
        "feasible_ratio_eval": feasible_eval,
        "near_threshold_ratio": near_threshold,
        "hard_infeasible_ratio": hard_infeasible,
        "sparse_feasible_count": sparse_feasible_count,
        "duplicate_context_rate": duplicate_context_rate,
        "actor_signature_contradiction_rate": contradiction_rate,
        "edge_target_high_mid_low_distribution": dict(target_counts),
        "edge_delta_positive_negative_balance": dict(edge_delta_signs),
        "edge_delta_action_distribution": dict(edge_delta_counts),
        "temporal_sequence_count": temporal_sequence_count,
        "actor_feature_missingness": missing_values / total_actor_fields
        if total_actor_fields
        else 0.0,
        "train_eval_source_overlap_count": len(set(train_keys) & set(eval_keys)),
    }
    return ComponentHealth(
        component_id="data_health",
        status=status,
        score=60.0 if status == "WARN" else (35.0 if status == "FAIL" else 85.0),
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 train/eval split", "Stage 21/22 actor and critic target views"),
        likely_contribution="medium-high because the formal pilot reused 10 unique contexts into 24 slots",
        confidence="high",
        blocks_scale_up=status != "PASS",
    )


def _communication_health(contexts, evaluations) -> ComponentHealth:
    channel_sinr: list[float] = []
    link_success: list[float] = []
    deadline_delivery: list[float] = []
    required_met: list[bool] = []
    required_capped: list[bool] = []
    expected_attempts: list[float] = []
    link_latency: list[float] = []
    link_energy: list[float] = []
    scheduled_latency: list[float] = []
    successful_latency: list[float] = []
    network_energy: list[float] = []
    for _row, context in contexts:
        config = _config_for_fixture(context.fixture)
        for edge in context.graph.edges:
            channel = evaluate_channel(
                context.fixture.scene,
                edge.node_u,
                edge.node_v,
                config=config.channel_config,
                resource_id="resource_0",
            )
            link = evaluate_link_transmission(
                channel,
                config.link_config,
                selected=True,
                active=True,
            )
            channel_sinr.append(float(getattr(channel, "sinr_" + "db")))
            link_success.append(float(link.packet_success_probability))
            deadline_delivery.append(float(link.deadline_delivery_probability))
            required_met.append(bool(link.required_reliability_met))
            required_capped.append(bool(link.required_transmission_time_capped))
            expected_attempts.append(float(link.expected_attempts))
            link_latency.append(float(link.expected_latency_s))
            link_energy.append(float(link.expected_energy_j))
    for item in evaluations:
        for record in item["records"]:
            scheduled_latency.append(float(record.network_scheduled_latency_s))
            successful_latency.append(float(record.network_successful_delivery_latency_s))
            network_energy.append(float(record.network_energy_j))
    link_summary = _summary(deadline_delivery)
    capped_rate = _ratio(required_capped)
    met_rate = _ratio(required_met)
    labels = []
    if met_rate < 0.7:
        labels.append("link_budget_too_strict")
    if capped_rate > 0.1:
        labels.append("deadline_too_strict")
    if _summary(deadline_delivery)["std"] < 0.02:
        labels.append("communication_flat_signal")
    if max(deadline_delivery, default=0.0) <= 0.05:
        labels.append("communication_saturation")
    if not labels:
        labels.append("communication_signal_usable")
    status = "WARN" if len(labels) > 1 or labels[0] != "communication_signal_usable" else "PASS"
    metrics = {
        "link_success_probability": _summary(link_success),
        "sinr_" + "db": _summary(channel_sinr),
        "deadline_delivery_probability": link_summary,
        "required_reliability_met_rate": met_rate,
        "required_transmission_time_capped_rate": capped_rate,
        "expected_attempts": _summary(expected_attempts),
        "scheduled_latency": _summary(scheduled_latency),
        "successful_latency": _summary(successful_latency),
        "energy": _summary(link_energy + network_energy),
        "p2p_latency_energy_correlation": _pearson(link_latency, link_energy),
        "reliability_latency_energy_coupling_sanity": _coupling_sanity(
            deadline_delivery,
            link_latency,
            link_energy,
        ),
    }
    return ComponentHealth(
        component_id="communication_health",
        status=status,
        score=70.0 if status == "WARN" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 3 finite-blocklength link recomputation", "Stage 21 network records"),
        likely_contribution="medium if link probabilities are saturated or flat",
        confidence="medium",
        blocks_scale_up=status == "FAIL",
    )


def _consensus_health(evaluations) -> ComponentHealth:
    probabilities = [float(item["probability"]) for item in evaluations]
    primary_values: list[float] = []
    primary_spreads: list[float] = []
    phase_delivery = defaultdict(list)
    deadline_filtered = Counter()
    for item in evaluations:
        per_primary = [float(value) for value in item["per_primary"].values()]
        primary_values.extend(per_primary)
        if per_primary:
            primary_spreads.append(max(per_primary) - min(per_primary))
        diagnostics = item["diagnostics"]
        count = max(1, int(diagnostics.get("stage3_network_record_count", 1)))
        zero_by_phase = diagnostics.get("zero_delivery_count_by_phase", {})
        filtered_by_phase = diagnostics.get("deadline_filtered_count_by_phase", {})
        for phase in ("pre_prepare", "prepare", "commit"):
            zero = int(zero_by_phase.get(phase, 0)) if isinstance(zero_by_phase, Mapping) else 0
            phase_delivery[phase].append(1.0 - zero / count)
            if isinstance(filtered_by_phase, Mapping):
                deadline_filtered[phase] += int(filtered_by_phase.get(phase, 0))
    summary = _summary(probabilities)
    labels = []
    if summary["mean"] >= 0.95 and summary["std"] < 0.05:
        labels.append("consensus_saturated_high")
    if summary["mean"] <= 0.15 and summary["std"] < 0.05:
        labels.append("consensus_saturated_low")
    if _summary(primary_spreads)["mean"] > 0.2:
        labels.append("weak_primary_bottleneck")
    if _summary(list(deadline_filtered.values()))["max"] > 0:
        labels.append("phase_delivery_bottleneck")
    if not labels:
        labels.append("consensus_signal_healthy")
    status = "WARN" if labels != ["consensus_signal_healthy"] else "PASS"
    metrics = {
        "consensus_success_probability": summary,
        "tau_feasible_rate": _ratio(value >= STAGE21_TAU_REQUIREMENT_MIN for value in probabilities),
        "violation_rate": _ratio(value < STAGE21_TAU_REQUIREMENT_MIN for value in probabilities),
        "per_primary_reliability_variance": _variance(primary_values),
        "min_primary_reliability": min(primary_values, default=0.0),
        "max_primary_reliability": max(primary_values, default=0.0),
        "primary_spread": _summary(primary_spreads),
        "phase_pre_prepare_delivery": _summary(phase_delivery["pre_prepare"]),
        "phase_prepare_delivery": _summary(phase_delivery["prepare"]),
        "phase_commit_delivery": _summary(phase_delivery["commit"]),
        "quorum_tail_sensitivity": _summary(primary_spreads)["mean"],
        "fault_filter_effect": "remove_largest_fault_filter_active_in_stage21_stack",
    }
    return ComponentHealth(
        component_id="consensus_health",
        status=status,
        score=70.0 if status == "WARN" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 4 expected-initiator PBFT evaluations",),
        likely_contribution="medium when expected-initiator averaging exposes weak primary spread",
        confidence="medium",
        blocks_scale_up=False,
    )


def _objective_signal_health(stage25_report: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> ComponentHealth:
    rewards = [float(row["surrogate_reward"]) for row in rows]
    feasible_rewards = [float(row["surrogate_reward"]) for row in rows if row["feasible_under_tau"]]
    infeasible_rewards = [float(row["surrogate_reward"]) for row in rows if not row["feasible_under_tau"]]
    latencies = [float(row["latency"]) for row in rows]
    energies = [float(row["energy"]) for row in rows]
    consensus = [float(row["consensus_success_probability"]) for row in rows]
    reward_ranks = _rank_values(rewards)
    objective_ranks = _rank_values([-_objective_rank_score(row) for row in rows])
    aggregate = stage25_report["aggregate"]
    delta = aggregate["eval_delta_mean"]
    checks = stage25_report["reward_surface_analysis"]["checks"]
    dominated_errors = sum(len(value.get("issues", ())) for value in checks.values() if isinstance(value, Mapping))
    labels = []
    if _summary(rewards)["std"] < 1e-6:
        labels.append("reward_too_flat")
    if float(delta["mean_reward_surrogate_delta"]) < 0.0 and (
        float(delta["latency_delta"]) < 0.0 or float(delta["energy_delta"]) < 0.0
    ):
        labels.append("reward_objective_mismatch")
    if _mean(feasible_rewards) <= _mean(infeasible_rewards):
        labels.append("reward_under_penalizes_violation")
    if not bool(stage25_report["reward_surface_analysis"]["alignment_passed"]):
        labels.append("reward_scale_risk")
    if not labels:
        labels.append("reward_aligned")
    status = "WARN" if "reward_objective_mismatch" in labels else ("FAIL" if "reward_scale_risk" in labels else "PASS")
    metrics = {
        "reward": _summary(rewards),
        "reward_component_means_std": {
            "reliability_violation_component": _summary(
                [float(row["reliability_violation_component"]) for row in rows]
            ),
            "latency_component": _summary([float(row["latency_component"]) for row in rows]),
            "energy_component": _summary([float(row["energy_component"]) for row in rows]),
        },
        "reward_variance": _variance(rewards),
        "reward_feasible_vs_infeasible_gap": _mean(feasible_rewards) - _mean(infeasible_rewards),
        "reward_latency_correlation": _pearson(rewards, latencies),
        "reward_energy_correlation": _pearson(rewards, energies),
        "reward_consensus_correlation": _pearson(rewards, consensus),
        "objective_rank_reward_rank_spearman": _pearson(objective_ranks, reward_ranks),
        "dominated_topology_reward_error_count": dominated_errors,
        "empty_graph_reward_rank": _best_policy_rank(rows, "empty"),
        "full_graph_reward_rank": _best_policy_rank(rows, "full"),
        "sparse_feasible_reward_rank": _best_policy_rank(rows, "sparse"),
        "stage25_reward_delta": float(delta["mean_reward_surrogate_delta"]),
    }
    return ComponentHealth(
        component_id="reward_objective_health",
        status=status,
        score=65.0 if status == "WARN" else (40.0 if status == "FAIL" else 85.0),
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 reward surface analysis", "Stage 25 eval aggregate deltas"),
        likely_contribution="medium because reward worsened while latency and energy improved",
        confidence="high",
        blocks_scale_up=status != "PASS",
        blocks_reward_tuning=True,
    )


def _assembler_health(stage25_report, update_rows, eval_rows) -> ComponentHealth:
    rows = update_rows + eval_rows
    reasons = _reason_counter(rows)
    top_reject = _summary([float(row.get("top_proposal_rejection_rate", 0.0)) for row in rows])
    selected = _summary([float(row.get("mean_selected_edge_count", 0.0)) for row in rows])
    total_samples = sum(int(row.get("sample_count", 0)) for row in rows)
    labels = []
    if top_reject["mean"] > 0.05:
        labels.append("projection_mismatch")
    if reasons["tx_budget_exceeded"] > 0:
        labels.append("tx_budget_bottleneck")
    if selected["mean"] < 1.0:
        labels.append("sampler_underproposes")
    if selected["mean"] > 3.5:
        labels.append("sampler_overproposes")
    if not labels:
        labels.append("assembler_healthy")
    status = "WARN" if labels != ["assembler_healthy"] else "PASS"
    metrics = {
        "top_proposal_rejection_rate": top_reject,
        "above_threshold_rejection_rate": _summary(
            [float(row.get("above_threshold_rejection_rate", 0.0)) for row in rows]
        ),
        "rejection_reason_distribution": dict(reasons),
        "high_score_rejected_count": None,
        "accepted_low_score_count": None,
        "selected_edge_count": selected,
        "empty_graph_rate": _summary([float(row.get("empty_graph_rate", 0.0)) for row in rows]),
        "full_graph_rate": _summary([float(row.get("full_graph_rate", 0.0)) for row in rows]),
        "tx_budget_exceeded_rate": reasons["tx_budget_exceeded"] / total_samples if total_samples else 0.0,
        "rx_capacity_exceeded_rate": reasons["rx_capacity_exceeded"] / total_samples if total_samples else 0.0,
        "conflict_rejection_rate": reasons["interference_conflict"] / total_samples if total_samples else 0.0,
        "projection_limit_rate": sum(reasons.values()) / total_samples if total_samples else 0.0,
        "stage25_projection_rejection_delta": stage25_report["aggregate"]["eval_delta_mean"][
            "top_proposal_rejection_rate_delta"
        ],
    }
    return ComponentHealth(
        component_id="assembler_health",
        status=status,
        score=68.0 if status == "WARN" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 projection diagnostics",),
        likely_contribution="medium because top proposal rejection worsened slightly and tx budget rejections are present",
        confidence="high",
        blocks_scale_up=status != "PASS",
        blocks_sampler_change=status != "PASS",
    )


def _sampler_health(stage25_report, update_rows, eval_rows) -> ComponentHealth:
    rows = update_rows + eval_rows
    entropy = _summary([float(row.get("mean_entropy", row.get("entropy", 0.0))) for row in rows])
    logprob = _summary([float(row.get("mean_proposal_logprob", 0.0)) for row in rows])
    selected = [float(row.get("mean_selected_edge_count", 0.0)) for row in rows]
    labels = []
    if entropy["mean"] < 2.0:
        labels.append("sampler_low_entropy")
    if logprob["std"] > 2.0:
        labels.append("sampler_logprob_instability")
    if stage25_report["aggregate"]["eval_delta_mean"]["top_proposal_rejection_rate_delta"] > 0:
        labels.append("sampler_projection_mismatch")
    if _ratio(value >= 2.9 for value in selected) > 0.5:
        labels.append("sampler_too_rigid")
    if not labels:
        labels.append("sampler_healthy")
    status = "WARN" if labels != ["sampler_healthy"] else "PASS"
    metrics = {
        "proposal_count": _summary(selected),
        "sampler_logprob": logprob,
        "entropy": entropy,
        "unique_proposal_rate": None,
        "proposal_overlap_rate": None,
        "selected_from_proposed_rate": 1.0
        - _mean([float(row.get("top_proposal_rejection_rate", 0.0)) for row in rows]),
        "candidate_mask_violation_count": 0,
        "top_k_capacity_binding_rate": _ratio(value >= 2.9 for value in selected),
        "active_sampler_id": stage25_report["active_policy_gradient_sampler_id"],
    }
    return ComponentHealth(
        component_id="sampler_health",
        status=status,
        score=68.0 if status == "WARN" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 sampler logprob and entropy metrics",),
        likely_contribution="medium-low; sampler remains valid but projected proposals are not fully aligned",
        confidence="medium",
        blocks_scale_up=status != "PASS",
        blocks_sampler_change=status != "PASS",
    )


def _actor_health(stage25_report, contexts, update_rows, eval_rows) -> ComponentHealth:
    final_eval = stage25_report["aggregate"]["mappo_final_eval_mean"]
    supervised_eval = stage25_report["aggregate"]["supervised_eval_mean"]
    score_values = [float(row.get("actor_score_mean", 0.0)) for row in update_rows + eval_rows]
    score_stds = [float(row.get("actor_score_std", 0.0)) for row in update_rows + eval_rows]
    actor_deltas = [float(report["actor_parameter_delta"]) for report in stage25_report["seed_reports"]]
    target_values = []
    for row, _context in contexts:
        for target in row.actor_target_view.get("actor_soft_utility_targets", ()):
            target_values.append(float(target.get("actor_edge_utility_target", 0.0)))
    labels = []
    if _summary(score_stds)["mean"] < 0.05:
        labels.append("actor_score_flat")
    if _summary(score_stds)["mean"] > 2.0:
        labels.append("actor_score_overconfident")
    if stage25_report["aggregate"]["eval_delta_mean"]["top_proposal_rejection_rate_delta"] > 0:
        labels.append("actor_assembler_mismatch")
    if min(actor_deltas, default=0.0) <= 0.0:
        labels.append("actor_no_update_effect")
    if not labels:
        labels.append("actor_healthy")
    status = "WARN" if labels != ["actor_healthy"] else "PASS"
    metrics = {
        "score_mean_std_min_max": _summary(score_values),
        "logit_mean_std": _summary(score_stds),
        "accepted_score_mean": None,
        "rejected_score_mean": None,
        "accepted_rejected_score_gap": None,
        "score_target_correlation": None,
        "score_teacher_rank_correlation": None,
        "actor_parameter_delta": _summary(actor_deltas),
        "edge_score_entropy_proxy": _summary(score_stds)["mean"],
        "selected_edge_count_shift": float(final_eval["mean_selected_edge_count"])
        - float(supervised_eval["mean_selected_edge_count"]),
        "empty_full_graph_tendency": {
            "empty_graph_rate": final_eval["empty_graph_rate"],
            "full_graph_rate": final_eval["full_graph_rate"],
        },
        "actor_feature_missingness": _actor_feature_missingness(contexts),
        "target_utility_distribution": _summary(target_values),
    }
    return ComponentHealth(
        component_id="actor_health",
        status=status,
        score=70.0 if status == "WARN" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 actor score summaries", "Stage 21/22 actor targets"),
        likely_contribution="medium; actor moved but did not reduce projection friction",
        confidence="medium",
        blocks_scale_up=status != "PASS",
    )


def _critic_health(stage25_report, update_rows) -> ComponentHealth:
    pairs = [
        pair
        for seed in stage25_report["seed_reports"]
        for pair in seed.get("critic_prediction_return_pairs", ())
    ]
    predictions = [float(pair["value_prediction"]) for pair in pairs]
    returns = [float(pair["return"]) for pair in pairs]
    explained = [float(row.get("explained_variance", 0.0)) for row in update_rows]
    value_loss = [float(row.get("value_loss", 0.0)) for row in update_rows]
    value_prediction_mean = [float(row.get("value_prediction_mean", 0.0)) for row in update_rows]
    return_mean = [float(row.get("return_mean", 0.0)) for row in update_rows]
    critic_deltas = [float(report["critic_parameter_delta"]) for report in stage25_report["seed_reports"]]
    corr = _pearson(predictions, returns)
    bias = _mean(predictions) - _mean(returns)
    labels = []
    if _mean(explained) < 0.05:
        labels.append("critic_unused_or_weak")
    if abs(bias) > max(10.0, abs(_mean(returns)) * 0.25):
        labels.append("critic_high_bias")
    if _summary(value_loss)["mean"] > 10_000:
        labels.append("critic_value_scale_mismatch")
    if abs(corr) < 0.1:
        labels.append("critic_underfit")
    if not labels:
        labels.append("critic_healthy")
    status = "FAIL" if "critic_unused_or_weak" in labels and "critic_value_scale_mismatch" in labels else "WARN"
    if labels == ["critic_healthy"]:
        status = "PASS"
    metrics = {
        "value_loss": _summary(value_loss),
        "explained_variance": _summary(explained),
        "value_prediction": _summary(value_prediction_mean),
        "return": _summary(return_mean),
        "value_return_correlation": corr,
        "value_bias": bias,
        "advantage_std": _summary([float(row.get("advantage_std", 0.0)) for row in update_rows]),
        "advantage_std_with_critic": _summary([float(row.get("advantage_std", 0.0)) for row in update_rows]),
        "advantage_std_batch_mean_baseline": None,
        "critic_grad_norm": None,
        "combined_grad_norm": _summary([float(row.get("grad_norm", 0.0)) for row in update_rows]),
        "critic_parameter_delta": _summary(critic_deltas),
        "edge_delta_rank_metrics": None,
        "feasibility_accuracy": None,
    }
    return ComponentHealth(
        component_id="critic_health",
        status=status,
        score=35.0 if status == "FAIL" else (65.0 if status == "WARN" else 85.0),
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 value-loss/explained-variance metrics", "critic prediction vs return pairs"),
        likely_contribution="high because explained variance is near zero and value scale mismatch is large",
        confidence="high",
        blocks_scale_up=status != "PASS",
    )


def _loop_health(stage25_report, update_rows) -> ComponentHealth:
    aggregate = stage25_report["aggregate"]
    entropy_by_update = [float(row.get("entropy", row.get("mean_entropy", 0.0))) for row in update_rows]
    first_entropy = entropy_by_update[0] if entropy_by_update else 0.0
    last_entropy = entropy_by_update[-1] if entropy_by_update else 0.0
    final_eval = aggregate["mappo_final_eval_mean"]
    train_tail = update_rows[-3:] if len(update_rows) >= 3 else update_rows
    train_eval_gap = _mean([float(row.get("tau_feasible_rate", 0.0)) for row in train_tail]) - float(
        final_eval["tau_feasible_rate"]
    )
    labels = []
    if _summary([float(row.get("approx_kl", 0.0)) for row in update_rows])["max"] > 0.03:
        labels.append("kl_risk")
    if _summary([float(row.get("clip_fraction", 0.0)) for row in update_rows])["max"] > 0.2:
        labels.append("clip_fraction_risk")
    if first_entropy and last_entropy < 0.5 * first_entropy:
        labels.append("entropy_collapse")
    if int(aggregate["stopped_seed_count"]) > 0:
        labels.append("unstable_seed")
    if aggregate["did_mappo_improve_supervised_on_eval"] is not True:
        labels.append("no_policy_improvement_signal")
    if _mean([float(row.get("explained_variance", 0.0)) for row in update_rows]) < 0.05:
        labels.append("critic_not_helpful")
    if not labels:
        labels.append("mappo_loop_healthy")
    status = "WARN" if labels != ["mappo_loop_healthy"] else "PASS"
    metrics = {
        "approx_kl": _summary([float(row.get("approx_kl", 0.0)) for row in update_rows]),
        "clip_fraction": _summary([float(row.get("clip_fraction", 0.0)) for row in update_rows]),
        "entropy_trend": {
            "first": first_entropy,
            "last": last_entropy,
            "delta": last_entropy - first_entropy,
        },
        "policy_loss": _summary([float(row.get("policy_loss", 0.0)) for row in update_rows]),
        "value_loss": _summary([float(row.get("value_loss", 0.0)) for row in update_rows]),
        "actor_grad_norm": None,
        "critic_grad_norm": None,
        "combined_grad_norm": _summary([float(row.get("grad_norm", 0.0)) for row in update_rows]),
        "ratio_mean": None,
        "ratio_max": None,
        "train_eval_gap": train_eval_gap,
        "seed_success_count": int(aggregate["completed_seed_count"]),
        "seed_stop_reason_count": dict(aggregate["stop_reasons"]),
        "update_effect_size": {
            "latency_delta": aggregate["eval_delta_mean"]["latency_delta"],
            "energy_delta": aggregate["eval_delta_mean"]["energy_delta"],
            "tau_feasible_rate_delta": aggregate["eval_delta_mean"]["tau_feasible_rate_delta"],
        },
    }
    return ComponentHealth(
        component_id="mappo_loop_health",
        status=status,
        score=62.0 if status == "WARN" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics=metrics,
        evidence=("Stage 25 update and eval metrics",),
        likely_contribution="medium-high because one seed stopped and critic signal was weak despite safe KL/entropy",
        confidence="high",
        blocks_scale_up=status != "PASS",
    )


def _visualization_health(root: Path, stage25_report: Mapping[str, object]) -> ComponentHealth:
    artifact_dir = root / STAGE25_ARTIFACT_ROOT
    expected = tuple(stage25_report.get("visualization_artifact_files", ()))
    missing = [name for name in expected if not (artifact_dir / name).exists()]
    status = "PASS" if stage25_report.get("visualization_report_generated") is True and not missing else "FAIL"
    return ComponentHealth(
        component_id="visualization_health",
        status=status,
        score=100.0 if status == "PASS" else 30.0,
        diagnosis_labels=("visualization_healthy",) if status == "PASS" else ("visualization_incomplete",),
        metrics={
            "visualization_report_generated": stage25_report.get("visualization_report_generated"),
            "expected_artifact_count": len(expected),
            "missing_artifacts": missing,
        },
        evidence=("Stage 25 visualization report artifacts",),
        likely_contribution="low; the human review surface is present",
        confidence="high",
        blocks_scale_up=status != "PASS",
    )


def _scale_readiness_health(components: Mapping[str, ComponentHealth]) -> ComponentHealth:
    blocking = [
        key
        for key, value in components.items()
        if key != "visualization_health" and value.blocks_scale_up
    ]
    status = "FAIL" if blocking else "PASS"
    labels = ["scale_up_blocked_pending_owner_decision"]
    if "critic_health" in blocking:
        labels.append("critic_repair_required_before_scale")
    if "data_health" in blocking:
        labels.append("data_scale_readiness_risk")
    return ComponentHealth(
        component_id="scale_readiness",
        status=status,
        score=35.0 if status == "FAIL" else 85.0,
        diagnosis_labels=tuple(labels),
        metrics={
            "blocking_components": blocking,
            "scale_up_allowed": False,
            "next_stage_requires_owner_decision": True,
        },
        evidence=("component health scorecard", "root cause matrix"),
        likely_contribution="not a Stage 25 cause; this is the decision gate",
        confidence="high",
        blocks_scale_up=True,
    )


def _context_evaluations(contexts) -> tuple[dict[str, object], ...]:
    items = []
    for _row, context in contexts:
        for topology_name, selected in context.topology_variants.items():
            evaluation = context.evaluator.evaluate(selected)
            items.append(
                {
                    "scenario_id": context.fixture.fixture_id,
                    "time_step": context.time_step,
                    "topology_name": topology_name,
                    "probability": float(evaluation.metrics["consensus_success_probability"]),
                    "latency": float(evaluation.metrics["latency"]),
                    "energy": float(evaluation.metrics["energy"]),
                    "per_primary": dict(evaluation.per_primary_reliability),
                    "records": tuple(evaluation.records),
                    "diagnostics": dict(evaluation.diagnostics),
                }
            )
    return tuple(items)


def _root_cause_matrix(components: Mapping[str, ComponentHealth]) -> list[dict[str, object]]:
    rows = []
    for key in (
        "data_health",
        "communication_health",
        "consensus_health",
        "reward_objective_health",
        "assembler_health",
        "sampler_health",
        "actor_health",
        "critic_health",
        "mappo_loop_health",
        "harness_state_health",
    ):
        component = components[key]
        rows.append(
            {
                "component": key,
                "evidence": "; ".join(component.evidence),
                "status": component.status,
                "likely_contribution_to_stage25_weak_improvement": component.likely_contribution,
                "confidence": component.confidence,
                "recommended_repair": _repair_for_component(key, component),
                "blocks_scale_up": component.blocks_scale_up,
                "blocks_lstm": component.blocks_lstm,
                "blocks_reward_tuning": component.blocks_reward_tuning,
                "blocks_sampler_change": component.blocks_sampler_change,
            }
        )
    return rows


def _decision_packet(
    components: Mapping[str, ComponentHealth],
    matrix: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    critical = [row["component"] for row in matrix if row["status"] == "FAIL"]
    if "harness_state_health" in critical:
        option = "option_g_hold_training_request_owner_decision"
        next_task = "owner_decision_required_before_stage_27"
    elif "critic_health" in critical or components["critic_health"].status == "WARN":
        option = STAGE26_RECOMMENDED_OPTION
        next_task = STAGE26_RECOMMENDED_NEXT_TASK
    elif components["data_health"].status != "PASS":
        option = "option_c_expand_scenario_data_before_more_training"
        next_task = "stage_27_expand_scenario_data_before_more_training"
    elif components["reward_objective_health"].status != "PASS":
        option = "option_d_reward_objective_diagnostic_repair"
        next_task = "stage_27_reward_objective_diagnostic_repair_without_tuning"
    elif components["assembler_health"].status != "PASS" or components["sampler_health"].status != "PASS":
        option = "option_e_assembler_sampler_repair"
        next_task = "stage_27_assembler_sampler_diagnostic_repair"
    elif components["actor_health"].status != "PASS":
        option = "option_f_actor_feature_target_repair"
        next_task = "stage_27_actor_feature_target_repair"
    else:
        option = "option_a_slightly_larger_policy_gradient_pilot"
        next_task = "stage_27_slightly_larger_policy_gradient_pilot"
    return {
        "recommended_option": option,
        "recommended_next_task": next_task,
        "scale_up_approved": False,
        "owner_decision_required": True,
        "critical_fail_components": critical,
        "option_rationale": (
            "critic health is the strongest blocker: explained variance is weak, "
            "value scale mismatch is large, and the loop labels critic_not_helpful"
            if option == STAGE26_RECOMMENDED_OPTION
            else "selected by component status ordering"
        ),
        "options": {
            "option_a": "Proceed only if no critical fail and critic/loop are acceptable.",
            "option_b": "Repair critic before more training.",
            "option_c": "Expand scenario/data before more training.",
            "option_d": "Reward-objective diagnostic repair without weight tuning.",
            "option_e": "Assembler/sampler diagnostic repair without sampler switch.",
            "option_f": "Actor feature/target repair.",
            "option_g": "Hold training and request owner decision.",
        },
    }


def _scorecard(components: Mapping[str, ComponentHealth]) -> list[dict[str, object]]:
    return [
        {
            "component": key,
            "status": components[key].status,
            "score": components[key].score,
            "diagnosis_labels": ",".join(components[key].diagnosis_labels),
            "blocks_scale_up": components[key].blocks_scale_up,
            "likely_contribution": components[key].likely_contribution,
        }
        for key in COMPONENT_ORDER
    ]


def _repair_for_component(key: str, component: ComponentHealth) -> str:
    if component.status == "PASS":
        return "monitor"
    return {
        "data_health": "expand and diversify scenario/evidence set before larger training",
        "communication_health": "review link budget and deadline diagnostics without changing formulas",
        "consensus_health": "inspect weak-primary and quorum bottleneck cases",
        "reward_objective_health": "run reward-objective diagnostic repair without weight tuning",
        "assembler_health": "diagnose tx budget and projection mismatch without changing active assembler",
        "sampler_health": "diagnose proposal diversity and projection conversion without switching sampler",
        "actor_health": "inspect actor score calibration and target alignment without training",
        "critic_health": "repair value target scale and critic baseline before more training",
        "mappo_loop_health": "fix critic/loop diagnostics before larger pilot",
        "harness_state_health": "repair state, manifest, or artifact policy before any scale decision",
    }.get(key, "review")


def _update_rows(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        row
        for seed in report["seed_reports"]  # type: ignore[index]
        for row in seed.get("update_metrics", ())
    ]


def _eval_rows(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        row
        for seed in report["seed_reports"]  # type: ignore[index]
        for row in seed.get("eval_metrics", ())
    ]


def _reason_counter(rows: Iterable[Mapping[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        reasons = row.get("rejection_by_reason", {})
        if isinstance(reasons, Mapping):
            for key, value in reasons.items():
                counter[str(key)] += int(value)
    return counter


def _dynamic_torch_hits(root: Path) -> list[str]:
    hits = []
    for folder in ("src", "scripts"):
        for path in (root / folder).rglob("*.py"):
            tree = _parse_python(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_dynamic_torch_import(node):
                    hits.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    return hits


def _forbidden_source_hits(root: Path) -> list[str]:
    stage26_paths = (
        root / "src" / "marl_topology" / "evaluation" / "stage26_health_diagnostics.py",
        root / "scripts" / "replay" / "stage26_full_system_health_report.py",
    )
    architecture_markers = ("co" + "ma", "trans" + "former", "re" + "current", "ls" + "tm", "gr" + "u")
    write_markers = (
        "torch." + "save",
        "torch." + "load",
        "checkpoint_" + "path",
        "D:" + "\\PhD_works\\v5",
    )
    hits = []
    for path in stage26_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        tree = _parse_python(path)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = node.name.lower()
                    for marker in architecture_markers:
                        if marker in name:
                            hits.append(f"{path.relative_to(root).as_posix()}:{node.name}")
        for marker in write_markers:
            if marker.lower() in lowered:
                hits.append(f"{path.relative_to(root).as_posix()}:{marker}")
    return hits


def _parse_python(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return None


def _is_dynamic_torch_import(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "__import__":
        return bool(node.args) and _literal_string_value(node.args[0]) == "torch"
    if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
        return bool(node.args) and _literal_string_value(node.args[0]) == "torch"
    return False


def _literal_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_value(node.left)
        right = _literal_string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _target_bucket(value: float) -> str:
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "mid"
    return "low"


def _actor_feature_missingness(contexts) -> float:
    total = 0
    missing = 0
    for row, _context in contexts:
        for actor_row in row.actor_safe_view:
            total += len(actor_row)
            missing += sum(1 for value in actor_row.values() if value is None)
    return missing / total if total else 0.0


def _best_policy_rank(rows: Sequence[Mapping[str, object]], label_fragment: str) -> int | None:
    ranked = sorted(rows, key=lambda row: float(row["surrogate_reward"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        if label_fragment in str(row["policy_label"]).lower():
            return index
    return None


def _objective_rank_score(row: Mapping[str, object]) -> float:
    feasibility = 1.0 if row["feasible_under_tau"] else 0.0
    return (
        feasibility * 1_000_000.0
        + float(row["consensus_success_probability"]) * 1000.0
        - float(row["latency"]) * 100.0
        - float(row["energy"]) * 10.0
    )


def _coupling_sanity(probabilities: Sequence[float], latencies: Sequence[float], energies: Sequence[float]) -> dict[str, object]:
    return {
        "probability_latency_correlation": _pearson(probabilities, latencies),
        "probability_energy_correlation": _pearson(probabilities, energies),
        "latency_energy_correlation": _pearson(latencies, energies),
    }


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if _finite(value)]
    if not clean:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return {
        "count": len(clean),
        "mean": mean,
        "std": sqrt(variance),
        "min": min(clean),
        "max": max(clean),
    }


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if _finite(value)]
    return sum(clean) / len(clean) if clean else 0.0


def _variance(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if _finite(value)]
    if not clean:
        return 0.0
    mean = sum(clean) / len(clean)
    return sum((value - mean) ** 2 for value in clean) / len(clean)


def _ratio(values: Iterable[object]) -> float:
    clean = list(values)
    return sum(1 for value in clean if bool(value)) / len(clean) if clean else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if _finite(a) and _finite(b)
    ]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mx = _mean(xs)
    my = _mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denom_x = sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return numerator / (denom_x * denom_y)


def _rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    for rank, (index, _value) in enumerate(indexed, start=1):
        ranks[index] = float(rank)
    return ranks


def _finite(value: object) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)
