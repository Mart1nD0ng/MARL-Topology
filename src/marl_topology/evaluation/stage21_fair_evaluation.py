"""Stage 21 fair same-assembler supervised rerun evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from marl_topology.data.stage21_assembler_aware_targets import (
    aggregate_proposal_rejection_diagnostics,
    build_stage21_conflict_aware_assembler,
    candidate_constraints_from_actor_safe_view,
    edge_score_records_from_actor_safe_view,
    proposal_rejection_diagnostics,
    selected_physical_edges_from_assembly,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE20_BEST_ACTOR_PROJECTION_REJECTION_RATE,
    STAGE20_BEST_ACTOR_TAU_FEASIBLE_RATE,
    STAGE21_EVALUATOR_ID,
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    STAGE21_TAU_REQUIREMENT_MIN,
    build_stage21_objective_stack_contexts,
    build_stage21_objective_stack_evidence_report,
    stage21_oracle_diagnostic_edges,
)
from marl_topology.policies.edge_scores import EdgeScoreRecord
from marl_topology.training.stage19_supervised_actor_stack import _edge_logits
from marl_topology.training.stage21_supervised_actor_rerun import (
    Stage21SupervisedRerunConfig,
    Stage21TrainedActorRerun,
    train_stage21_supervised_actor_models,
)
import marl_topology.training.torch_backend as _torch_backend

torch = _torch_backend.torch


STAGE21_EVALUATION_ID = "stage21_fair_same_assembler_supervised_rerun_evaluation"
STAGE21_PASS_VERDICT = "stage21_pass_policy_gradient_pilot_allowed"
STAGE21_FAIL_VERDICT = "stage21_failed_policy_gradient_blocked"


@dataclass(frozen=True, slots=True)
class Stage21FairEvaluationConfig:
    seed: int = 21
    supervised_epochs: int = 100
    pilot_update_epochs: int = 2
    pilot_learning_rate: float = 0.001
    pilot_clip_epsilon: float = 0.2
    run_policy_gradient_if_gates_pass: bool = True


def run_stage21_objective_stack_aligned_assembler_aware_rerun(
    *,
    config: Stage21FairEvaluationConfig | None = None,
) -> dict[str, object]:
    """Run Stage 21 B-E gates and conditional policy-gradient pilot."""

    cfg = config or Stage21FairEvaluationConfig()
    evidence_build = build_stage21_objective_stack_evidence_report()
    trained = train_stage21_supervised_actor_models(
        config=Stage21SupervisedRerunConfig(
            seed=cfg.seed,
            epochs=cfg.supervised_epochs,
        )
    )
    evaluation = evaluate_stage21_supervised_actors_and_baselines(trained)
    gates = _stage21_gates(
        evidence_report=evidence_build.report,
        supervised_report=trained.report,
        evaluation_report=evaluation,
    )
    all_gates_passed = all(bool(gate["passed"]) for gate in gates.values())
    pilot_report = None
    failure_review = None
    if all_gates_passed and cfg.run_policy_gradient_if_gates_pass:
        pilot_report = run_stage21_controlled_policy_gradient_pilot(
            trained,
            best_actor_label=str(evaluation["best_actor"]),
            config=cfg,
        )
    elif not all_gates_passed:
        failure_review = _failure_review(gates)
    return {
        "stage": STAGE21_EVALUATION_ID,
        "verdict": STAGE21_PASS_VERDICT if all_gates_passed else STAGE21_FAIL_VERDICT,
        "evidence_report": evidence_build.report,
        "supervised_rerun_report": trained.report,
        "fair_evaluation_report": evaluation,
        "stage21_gates": gates,
        "stage21_passed": all_gates_passed,
        "policy_gradient_pilot_ran": pilot_report is not None,
        "policy_gradient_pilot_report": pilot_report,
        "failure_review": failure_review,
        "policy_gradient_allowed": all_gates_passed,
        "coma_allowed": False,
        "transformer_allowed": False,
        "reward_weight_tuning_performed": False,
        "final_tau_selected": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "v5_modified": False,
    }


def evaluate_stage21_supervised_actors_and_baselines(
    trained: Stage21TrainedActorRerun,
) -> dict[str, object]:
    """Evaluate MLP/GNN and baselines through the same assembler constraints."""

    contexts = _context_lookup()
    model_scores = _score_records_by_model_and_row(trained)
    actor_rows: dict[str, list[dict[str, object]]] = {"MLP": [], "GNN": []}
    baseline_rows: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row_index, row in enumerate(trained.dataset.rows):
        context = contexts[(str(row.diagnostics_view["source_fixture"]), int(row.time_step))]
        for model_label in ("MLP", "GNN"):
            actor_rows[model_label].append(
                _evaluate_projected_scores(
                    row=row,
                    context=context,
                    policy_label=f"{model_label.lower()}_actor_projected",
                    score_records=model_scores[model_label].get(row_index, ()),
                    projected=True,
                    actor_policy=True,
                )
            )
        for baseline_label, selected_edges in _raw_baseline_edges(context).items():
            baseline_rows[baseline_label].append(
                _evaluate_raw_edges(
                    row=row,
                    context=context,
                    policy_label=baseline_label,
                    selected_edges=selected_edges,
                )
            )
        for baseline_label, selected_edges in _projected_baseline_edges(context).items():
            score_records = edge_score_records_from_actor_safe_view(
                row.actor_safe_view,
                score_source=f"stage21_{baseline_label}",
                proposed_edge_ids=set(selected_edges),
                include_non_proposed=False,
            )
            baseline_rows[baseline_label].append(
                _evaluate_projected_scores(
                    row=row,
                    context=context,
                    policy_label=baseline_label,
                    score_records=score_records,
                    projected=True,
                    actor_policy=False,
                )
            )
        oracle_edges = stage21_oracle_diagnostic_edges(context.evaluator)
        if oracle_edges is not None:
            baseline_rows["oracle_diagnostic"].append(
                _evaluate_raw_edges(
                    row=row,
                    context=context,
                    policy_label="oracle_diagnostic",
                    selected_edges=oracle_edges,
                )
            )

    model_summaries = {
        label: _summarize_policy_rows(rows)
        for label, rows in actor_rows.items()
    }
    baseline_summaries = {
        label: _summarize_policy_rows(tuple(rows))
        for label, rows in sorted(baseline_rows.items())
    }
    best_actor = _select_best_actor(model_summaries)
    return {
        "stage": STAGE21_EVALUATION_ID,
        "evaluator_id": STAGE21_EVALUATOR_ID,
        "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
        "objective_contract_id": STAGE21_OBJECTIVE_CONTRACT_ID,
        "tau_requirement_min": STAGE21_TAU_REQUIREMENT_MIN,
        "main_comparison_uses_projected_baselines": True,
        "raw_baselines_are_diagnostic_only": True,
        "assembler_type": "ConflictAwareGreedyAssembler",
        "same_assembler_used_for_actors_and_projected_baselines": True,
        "actor_rows": actor_rows,
        "baseline_rows": dict(baseline_rows),
        "model_summaries": model_summaries,
        "baseline_summaries": baseline_summaries,
        "best_actor": best_actor,
        "best_actor_summary": model_summaries[best_actor],
        "model_comparison": _model_comparison(model_summaries, baseline_summaries, best_actor),
        "fairness_rule": (
            "Actor-vs-baseline main comparison uses projected baselines under "
            "the same ConflictAwareGreedyAssembler constraints; raw baselines "
            "are diagnostics only."
        ),
        "full_graph_resource_dominance_cases": _full_graph_resource_dominance_cases(
            baseline_rows
        ),
        "sparse_vs_full_tradeoff_cases": _sparse_vs_full_tradeoff_cases(baseline_rows),
        "actor_outputs_consumable_by_assembler": True,
        "artifact_written": False,
        "checkpoint_written": False,
    }


def run_stage21_controlled_policy_gradient_pilot(
    trained: Stage21TrainedActorRerun,
    *,
    best_actor_label: str,
    config: Stage21FairEvaluationConfig,
) -> dict[str, object]:
    """Run a small clipped proposal-policy update after Stage 21 gates pass."""

    torch.manual_seed(config.seed)
    model = trained.gnn_model if best_actor_label == "GNN" else trained.mlp_model
    pilot_row_indices = tuple(
        index
        for index, row in enumerate(trained.dataset.rows)
        if row.topology_name == "greedy_reliability_raw"
    )[:3]
    before = _pilot_policy_snapshot(
        trained,
        model,
        pilot_row_indices=pilot_row_indices,
        sample=True,
        seed=config.seed,
    )
    old_log_prob = torch.tensor(float(before["proposal_log_prob"]), dtype=torch.float32)
    sampled_actions = tuple(float(item) for item in before["sampled_actions"])
    signal = torch.tensor(float(before["pilot_training_signal"]), dtype=torch.float32)
    update_rule = torch.optim.AdamW(model.parameters(), lr=config.pilot_learning_rate)
    last_loss = torch.tensor(0.0)
    last_ratio = torch.tensor(1.0)
    action_indices = tuple(int(item) for item in before["sample_indices"])
    action_tensor = torch.tensor(sampled_actions, dtype=torch.float32)
    for _ in range(config.pilot_update_epochs):
        update_rule.zero_grad()
        logits = _edge_logits(model, trained.batch)[list(action_indices)]
        distribution = torch.distributions.Bernoulli(logits=logits)
        new_log_prob = distribution.log_prob(action_tensor).sum()
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped = ratio * signal
        clipped = torch.clamp(
            ratio,
            1.0 - config.pilot_clip_epsilon,
            1.0 + config.pilot_clip_epsilon,
        ) * signal
        last_loss = -torch.minimum(unclipped, clipped)
        last_loss.backward()
        update_rule.step()
        last_ratio = ratio.detach()
    after = _pilot_policy_snapshot(
        trained,
        model,
        pilot_row_indices=pilot_row_indices,
        sample=True,
        seed=config.seed + 1,
    )
    violation_worsened = (
        float(after["violation_rate"]) > float(before["violation_rate"]) + 0.05
    )
    collapsed = bool(after["mean_selected_edge_count"] in {0.0, after["mean_candidate_edge_count"]})
    stop_reason = None
    completed = True
    if violation_worsened:
        stop_reason = "violation_rate_worsened"
        completed = False
    elif collapsed:
        stop_reason = "actor_collapsed_to_full_or_empty_graph"
        completed = False
    elif float(after["top_proposal_rejection_rate"]) > 0.95:
        stop_reason = "assembler_rejected_most_high_score_proposals"
        completed = False
    return {
        "stage": "stage21_controlled_policy_gradient_pilot",
        "best_supervised_actor_initialization": best_actor_label,
        "completed": completed,
        "stop_reason": stop_reason,
        "before": before,
        "after": after,
        "policy_action_semantics": (
            "policy action is the sampled directed-edge proposal set; environment "
            "transition uses the assembler-projected topology"
        ),
        "projected_topology_logprob_claimed_exact": False,
        "policy_loss": float(last_loss.detach().item()),
        "value_loss": float(float(before["pilot_training_signal"]) ** 2),
        "kl_proxy": abs(float(old_log_prob.item()) - float(after["proposal_log_prob"])),
        "entropy": float(after["proposal_entropy"]),
        "clip_fraction": float((torch.abs(last_ratio - 1.0) > config.pilot_clip_epsilon).float().item()),
        "violation_rate_worsened": violation_worsened,
        "actor_collapsed": collapsed,
        "reward_weight_tuning_performed": False,
        "final_tau_selected": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "coma_allowed": False,
        "transformer_allowed": False,
    }


def _score_records_by_model_and_row(
    trained: Stage21TrainedActorRerun,
) -> dict[str, dict[int, tuple[EdgeScoreRecord, ...]]]:
    result: dict[str, dict[int, list[EdgeScoreRecord]]] = {
        "MLP": defaultdict(list),
        "GNN": defaultdict(list),
    }
    with torch.no_grad():
        logits_by_model = {
            "MLP": trained.mlp_model(trained.batch.features),
            "GNN": trained.gnn_model.forward_with_groups(
                trained.batch.features,
                trained.batch.group_ids,
            ),
        }
    for model_label, logits in logits_by_model.items():
        for index, sample in enumerate(trained.batch.samples):
            result[model_label][sample.row_index].append(
                _edge_score_from_sample(sample, float(logits[index].item()), model_label)
            )
    return {
        label: {row_index: tuple(scores) for row_index, scores in rows.items()}
        for label, rows in result.items()
    }


def _edge_score_from_sample(sample, logit: float, model_label: str) -> EdgeScoreRecord:
    neighbor_id = sample.directed_edge_id.split("->", 1)[1]
    return EdgeScoreRecord(
        agent_id=sample.agent_id,
        neighbor_id=neighbor_id,
        edge_id=sample.edge_id,
        directed_edge_id=sample.directed_edge_id,
        score=logit,
        probability=float(torch.sigmoid(torch.tensor(logit)).item()),
        score_source=f"stage21_{model_label.lower()}_supervised_actor_score",
        time_step=sample.time_step,
    )


def _evaluate_projected_scores(
    *,
    row,
    context,
    policy_label: str,
    score_records: Iterable[EdgeScoreRecord],
    projected: bool,
    actor_policy: bool,
) -> dict[str, object]:
    scores = tuple(score_records)
    constraints = candidate_constraints_from_actor_safe_view(row.actor_safe_view)
    assembler = build_stage21_conflict_aware_assembler(row.actor_safe_view)
    assembled = assembler.assemble(scores, constraints, metadata={})
    selected_edges = selected_physical_edges_from_assembly(scores, assembled.selected_directed_edges)
    evaluation = context.evaluator.evaluate(
        selected_edges,
        topology_id=f"stage21:{policy_label}:{row.evidence_id}",
    )
    diagnostics = proposal_rejection_diagnostics(scores, assembled)
    return {
        "policy_label": policy_label,
        "row_evidence_id": row.evidence_id,
        "scenario_id": row.scenario_id,
        "topology_name": row.topology_name,
        "projected": projected,
        "actor_policy": actor_policy,
        "raw_diagnostic_only": False,
        "selected_edge_ids": list(selected_edges),
        "selected_edge_count": len(selected_edges),
        "pre_projection_edge_count": assembled.pre_projection_edge_count,
        "post_projection_edge_count": assembled.post_projection_edge_count,
        "rejected_edge_count": len(assembled.rejected_edges),
        "rejection_by_reason": dict(assembled.diagnostics["rejection_reason_counts"]),
        "projection_diagnostics": diagnostics,
        **_metric_summary(evaluation),
        "candidate_edge_count": len(context.graph.edge_ids),
    }


def _evaluate_raw_edges(
    *,
    row,
    context,
    policy_label: str,
    selected_edges: Iterable[str],
) -> dict[str, object]:
    selected = tuple(sorted(set(selected_edges)))
    evaluation = context.evaluator.evaluate(
        selected,
        topology_id=f"stage21:{policy_label}:{row.evidence_id}",
    )
    return {
        "policy_label": policy_label,
        "row_evidence_id": row.evidence_id,
        "scenario_id": row.scenario_id,
        "topology_name": row.topology_name,
        "projected": False,
        "actor_policy": False,
        "raw_diagnostic_only": policy_label != "oracle_diagnostic",
        "selected_edge_ids": list(selected),
        "selected_edge_count": len(selected),
        "pre_projection_edge_count": len(selected),
        "post_projection_edge_count": len(selected),
        "rejected_edge_count": 0,
        "rejection_by_reason": {},
        "projection_diagnostics": {},
        **_metric_summary(evaluation),
        "candidate_edge_count": len(context.graph.edge_ids),
    }


def _metric_summary(evaluation) -> dict[str, object]:
    probability = float(evaluation.metrics["consensus_success_probability"])
    return {
        "tau_feasible": probability >= STAGE21_TAU_REQUIREMENT_MIN,
        "consensus_success_probability": probability,
        "per_primary_reliability": dict(evaluation.per_primary_reliability),
        "latency": float(evaluation.metrics["latency"]),
        "energy": float(evaluation.metrics["energy"]),
        "topology_diagnostics": dict(evaluation.metrics["topology_diagnostics"]),
    }


def _summarize_policy_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = tuple(rows)
    if not items:
        return {
            "row_count": 0,
            "tau_feasible_rate": 0.0,
            "mean_consensus_success_probability": 0.0,
            "mean_latency": 0.0,
            "mean_energy": 0.0,
            "mean_selected_edge_count": 0.0,
            "sparse_feasible_rate": 0.0,
            "top_proposal_rejection_rate": 0.0,
            "above_threshold_rejection_rate": 0.0,
            "rejection_by_reason": {},
            "accepted_rejected_score_separation": 0.0,
        }
    feasible = [bool(row["tau_feasible"]) for row in items]
    projection_diagnostics = [
        row["projection_diagnostics"]
        for row in items
        if isinstance(row.get("projection_diagnostics"), Mapping)
        and row.get("projection_diagnostics")
    ]
    rejection_counts = Counter(
        reason
        for row in items
        for reason, count in dict(row.get("rejection_by_reason", {})).items()
        for _ in range(int(count))
    )
    accepted_means = [
        float(row["projection_diagnostics"]["accepted_edge_score_distribution"]["mean"])
        for row in items
        if isinstance(row.get("projection_diagnostics"), Mapping)
        and row["projection_diagnostics"].get("accepted_edge_score_distribution", {}).get("mean")
        is not None
    ]
    rejected_means = [
        float(row["projection_diagnostics"]["rejected_edge_score_distribution"]["mean"])
        for row in items
        if isinstance(row.get("projection_diagnostics"), Mapping)
        and row["projection_diagnostics"].get("rejected_edge_score_distribution", {}).get("mean")
        is not None
    ]
    return {
        "row_count": len(items),
        "tau_feasible_count": sum(feasible),
        "tau_feasible_rate": sum(feasible) / len(items),
        "mean_consensus_success_probability": _mean(
            float(row["consensus_success_probability"]) for row in items
        ),
        "mean_latency": _mean(float(row["latency"]) for row in items),
        "mean_energy": _mean(float(row["energy"]) for row in items),
        "mean_selected_edge_count": _mean(float(row["selected_edge_count"]) for row in items),
        "sparse_feasible_rate": sum(
            bool(row["tau_feasible"]) and int(row["selected_edge_count"]) < int(row["candidate_edge_count"])
            for row in items
        )
        / len(items),
        "top_proposal_rejection_rate": aggregate_proposal_rejection_diagnostics(
            projection_diagnostics
        )["top_proposal_rejection_rate"],
        "above_threshold_rejection_rate": aggregate_proposal_rejection_diagnostics(
            projection_diagnostics
        )["above_threshold_rejection_rate"],
        "rejection_by_reason": dict(rejection_counts),
        "accepted_rejected_score_separation": (
            _mean(accepted_means) - _mean(rejected_means)
            if accepted_means and rejected_means
            else 0.0
        ),
        "collapsed_to_empty_rate": sum(int(row["selected_edge_count"]) == 0 for row in items)
        / len(items),
        "collapsed_to_full_rate": sum(
            int(row["selected_edge_count"]) == int(row["candidate_edge_count"])
            for row in items
        )
        / len(items),
    }


def _raw_baseline_edges(context) -> dict[str, tuple[str, ...]]:
    return {
        "full_graph_raw": tuple(context.graph.edge_ids),
        "greedy_reliability_raw": tuple(context.topology_variants["greedy_reliability_raw"]),
        "random_raw": tuple(context.topology_variants["random_raw"]),
    }


def _projected_baseline_edges(context) -> dict[str, tuple[str, ...]]:
    return {
        "full_graph_projected": tuple(context.graph.edge_ids),
        "greedy_reliability_projected": tuple(context.topology_variants["greedy_reliability_raw"]),
        "random_projected": tuple(context.topology_variants["random_raw"]),
        "sparse_heuristic_projected": tuple(context.topology_variants["sparse_quorum_raw"]),
    }


def _select_best_actor(summaries: Mapping[str, Mapping[str, object]]) -> str:
    return max(
        ("MLP", "GNN"),
        key=lambda label: (
            float(summaries[label]["tau_feasible_rate"]),
            float(summaries[label]["mean_consensus_success_probability"]),
            -float(summaries[label]["mean_latency"]),
            -float(summaries[label]["mean_energy"]),
        ),
    )


def _model_comparison(
    model_summaries: Mapping[str, Mapping[str, object]],
    baseline_summaries: Mapping[str, Mapping[str, object]],
    best_actor: str,
) -> dict[str, object]:
    best = model_summaries[best_actor]
    greedy = baseline_summaries.get("greedy_reliability_projected", {})
    full = baseline_summaries.get("full_graph_projected", {})
    return {
        "best_actor": best_actor,
        "gnn_remained_better_than_mlp_by_tau": (
            float(model_summaries["GNN"]["tau_feasible_rate"])
            >= float(model_summaries["MLP"]["tau_feasible_rate"])
        ),
        "best_actor_tau_minus_projected_greedy": float(best["tau_feasible_rate"])
        - float(greedy.get("tau_feasible_rate", 0.0)),
        "best_actor_tau_minus_full_graph_projected": float(best["tau_feasible_rate"])
        - float(full.get("tau_feasible_rate", 0.0)),
        "best_actor_improved_over_stage20_best_actor": (
            float(best["tau_feasible_rate"]) > STAGE20_BEST_ACTOR_TAU_FEASIBLE_RATE
        ),
        "projection_mismatch_reduced_vs_stage20": (
            float(best["top_proposal_rejection_rate"])
            < STAGE20_BEST_ACTOR_PROJECTION_REJECTION_RATE
            or float(best["above_threshold_rejection_rate"])
            < STAGE20_BEST_ACTOR_PROJECTION_REJECTION_RATE
            or float(best["accepted_rejected_score_separation"]) > 0.0
        ),
        "actor_approached_projected_greedy_baseline": (
            float(best["tau_feasible_rate"])
            >= 0.8 * float(greedy.get("tau_feasible_rate", 0.0))
        ),
    }


def _stage21_gates(
    *,
    evidence_report: Mapping[str, object],
    supervised_report: Mapping[str, object],
    evaluation_report: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    target = evidence_report["target_distribution"]
    best = evaluation_report["best_actor_summary"]
    baseline = evaluation_report["baseline_summaries"]
    full_projected = baseline.get("full_graph_projected", {})
    comparison = evaluation_report["model_comparison"]
    gates = {
        "objective_stack_alignment": {
            "passed": bool(
                evidence_report["all_rows_use_final_objective_stack"]
                and not evidence_report["silent_fallback_to_simple_link_or_min_link"]
            ),
            "evidence": {
                "main_evaluator_id": evidence_report["main_evaluator_id"],
                "partial_readiness_flag": evidence_report["partial_readiness_flag"],
            },
        },
        "actor_target_quality": {
            "passed": bool(
                target["has_high_mid_low_priority"]
                and int(target["ranking_pair_count"]) > 0
                and not target["uniformly_high"]
                and evidence_report["critic_target_view_is_critic_only"]
            ),
            "evidence": target,
        },
        "fair_assembler_evaluation": {
            "passed": bool(
                evaluation_report["main_comparison_uses_projected_baselines"]
                and evaluation_report["same_assembler_used_for_actors_and_projected_baselines"]
                and evaluation_report["raw_baselines_are_diagnostic_only"]
            ),
            "evidence": evaluation_report["fairness_rule"],
        },
        "projection_mismatch_improved": {
            "passed": bool(comparison["projection_mismatch_reduced_vs_stage20"]),
            "evidence": {
                "stage20_projection_rejection_baseline": STAGE20_BEST_ACTOR_PROJECTION_REJECTION_RATE,
                "best_actor_top_proposal_rejection_rate": best["top_proposal_rejection_rate"],
                "best_actor_above_threshold_rejection_rate": best["above_threshold_rejection_rate"],
                "accepted_rejected_score_separation": best["accepted_rejected_score_separation"],
            },
        },
        "actor_performance_sufficient_for_pilot": {
            "passed": bool(
                (
                    float(best["tau_feasible_rate"])
                    >= float(full_projected.get("tau_feasible_rate", 1.0))
                )
                and bool(comparison["best_actor_improved_over_stage20_best_actor"])
                and float(best["collapsed_to_empty_rate"]) < 0.95
                and float(best["collapsed_to_full_rate"]) < 0.95
                and int(best["tau_feasible_count"]) > 0
            ),
            "evidence": {
                "best_actor_tau_feasible_rate": best["tau_feasible_rate"],
                "full_graph_projected_tau_feasible_rate": full_projected.get("tau_feasible_rate"),
                "stage20_best_actor_tau_feasible_rate": STAGE20_BEST_ACTOR_TAU_FEASIBLE_RATE,
                "collapsed_to_empty_rate": best["collapsed_to_empty_rate"],
                "collapsed_to_full_rate": best["collapsed_to_full_rate"],
                "tau_feasible_count": best["tau_feasible_count"],
            },
        },
        "boundary_safety": {
            "passed": bool(
                supervised_report["no_actor_leakage"]
                and not supervised_report["critic_global_targets_actor_input"]
                and not supervised_report["checkpoint_written"]
                and not supervised_report["reward_weight_tuning_performed"]
                and not supervised_report["final_tau_selected"]
            ),
            "evidence": {
                "no_actor_leakage": supervised_report["no_actor_leakage"],
                "checkpoint_written": supervised_report["checkpoint_written"],
                "reward_weight_tuning_performed": supervised_report["reward_weight_tuning_performed"],
                "final_tau_selected": supervised_report["final_tau_selected"],
                "v5_modified": supervised_report["v5_modified"],
            },
        },
    }
    return gates


def _failure_review(gates: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    failed = [name for name, gate in gates.items() if not bool(gate["passed"])]
    repair = "actor_feature_or_target_repair"
    if "objective_stack_alignment" in failed:
        repair = "evidence_stack_repair"
    elif "actor_target_quality" in failed:
        repair = "target_repair"
    elif "fair_assembler_evaluation" in failed:
        repair = "baseline_fairness_repair"
    elif "projection_mismatch_improved" in failed:
        repair = "assembler_or_actor_feature_repair"
    elif "actor_performance_sufficient_for_pilot" in failed:
        repair = "actor_feature_or_supervised_training_repair"
    return {
        "failed_gates": failed,
        "policy_gradient_skipped": True,
        "recommended_repair": repair,
        "root_cause_summary": (
            "Stage 21 policy-gradient is blocked because at least one hard "
            "evidence, target, fairness, projection, performance, or boundary "
            "gate failed."
        ),
    }


def _pilot_policy_snapshot(
    trained: Stage21TrainedActorRerun,
    model: torch.nn.Module,
    *,
    pilot_row_indices: tuple[int, ...],
    sample: bool,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    contexts = _context_lookup()
    logits = _edge_logits(model, trained.batch)
    sample_indices = [
        index
        for index, sample_item in enumerate(trained.batch.samples)
        if sample_item.row_index in pilot_row_indices
    ]
    selected_logits = logits[sample_indices]
    distribution = torch.distributions.Bernoulli(logits=selected_logits)
    actions = distribution.sample() if sample else (torch.sigmoid(selected_logits) >= 0.5).float()
    by_row: dict[int, list[EdgeScoreRecord]] = defaultdict(list)
    for action, sample_index in zip(actions.tolist(), sample_indices):
        if float(action) < 0.5:
            continue
        sample_item = trained.batch.samples[sample_index]
        by_row[sample_item.row_index].append(
            _edge_score_from_sample(sample_item, float(logits[sample_index].item()), "pilot")
        )
    policy_rows = []
    for row_index in pilot_row_indices:
        row = trained.dataset.rows[row_index]
        context = contexts[(str(row.diagnostics_view["source_fixture"]), int(row.time_step))]
        policy_rows.append(
            _evaluate_projected_scores(
                row=row,
                context=context,
                policy_label="stage21_policy_gradient_pilot_sample",
                score_records=tuple(by_row.get(row_index, ())),
                projected=True,
                actor_policy=True,
            )
        )
    summary = _summarize_policy_rows(policy_rows)
    log_prob = distribution.log_prob(actions).sum()
    entropy = distribution.entropy().mean()
    signal = (
        float(summary["mean_consensus_success_probability"])
        - STAGE21_TAU_REQUIREMENT_MIN
        - 0.01 * float(summary["mean_latency"]) / 0.01
        - 0.01 * float(summary["mean_energy"]) / 0.1
    )
    return {
        **summary,
        "violation_rate": 1.0 - float(summary["tau_feasible_rate"]),
        "proposal_log_prob": float(log_prob.detach().item()),
        "proposal_entropy": float(entropy.detach().item()),
        "sampled_actions": tuple(float(item) for item in actions.detach().tolist()),
        "sample_indices": tuple(sample_indices),
        "pilot_training_signal": signal,
        "mean_candidate_edge_count": _mean(float(row["candidate_edge_count"]) for row in policy_rows),
        "policy_rows": policy_rows,
    }


def _context_lookup() -> dict[tuple[str, int], object]:
    return {
        (context.fixture.fixture_id, context.time_step): context
        for context in build_stage21_objective_stack_contexts()
    }


def _full_graph_resource_dominance_cases(
    rows_by_policy: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, object]:
    full = tuple(rows_by_policy.get("full_graph_raw", ()))
    greedy = tuple(rows_by_policy.get("greedy_reliability_raw", ()))
    cases = []
    for full_row, greedy_row in zip(full, greedy):
        if (
            float(full_row["energy"]) > float(greedy_row["energy"])
            and float(full_row["consensus_success_probability"])
            <= float(greedy_row["consensus_success_probability"]) + 1e-12
        ):
            cases.append(
                {
                    "row_evidence_id": full_row["row_evidence_id"],
                    "full_energy": full_row["energy"],
                    "greedy_energy": greedy_row["energy"],
                    "full_probability": full_row["consensus_success_probability"],
                    "greedy_probability": greedy_row["consensus_success_probability"],
                }
            )
    return {"case_count": len(cases), "cases": cases[:10]}


def _sparse_vs_full_tradeoff_cases(
    rows_by_policy: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, object]:
    sparse = tuple(rows_by_policy.get("sparse_heuristic_projected", ()))
    full = tuple(rows_by_policy.get("full_graph_projected", ()))
    cases = []
    for sparse_row, full_row in zip(sparse, full):
        if (
            bool(sparse_row["tau_feasible"])
            and float(sparse_row["latency"]) <= float(full_row["latency"])
            and float(sparse_row["energy"]) <= float(full_row["energy"])
        ):
            cases.append(
                {
                    "row_evidence_id": sparse_row["row_evidence_id"],
                    "sparse_latency": sparse_row["latency"],
                    "full_latency": full_row["latency"],
                    "sparse_energy": sparse_row["energy"],
                    "full_energy": full_row["energy"],
                }
            )
    return {"case_count": len(cases), "cases": cases[:10]}


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if isfinite(float(value))]
    return sum(items) / len(items) if items else 0.0
