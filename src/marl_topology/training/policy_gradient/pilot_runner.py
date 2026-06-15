"""Stage 23 selected-physical policy-gradient micro-pilot runner."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import torch
import torch.nn.functional as F

from marl_topology.data.actor_feature_rebuild import actor_safe_view_has_no_forbidden_fields
from marl_topology.data.stage21_assembler_aware_targets import (
    candidate_constraints_from_actor_safe_view,
    rx_capacity_from_actor_safe_view,
    tx_capacity_from_actor_safe_view,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_EVALUATOR_ID,
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PHYSICS_REGIME_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    STAGE21_TAU_REQUIREMENT_MIN,
    Stage21EvaluationContext,
    build_stage21_objective_stack_contexts,
)
from marl_topology.data.stage22_action_semantics_evidence import (
    build_stage22_action_semantics_evidence,
)
from marl_topology.evaluation import (
    STAGE5_0L_RANGE_REVIEW_STAGE_ID,
    build_stage5_0l_stage3_backed_sweep_range_review,
)
from marl_topology.models import (
    ACTOR_EDGE_FEATURE_FIELDS,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalGNNEdgeScorer,
    build_model_registry,
    tensorize_actor_history_sequence,
    tensorize_actor_policy_inputs,
)
from marl_topology.objectives import (
    NormalizationReferenceConfig,
    SurrogateSignalConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
    select_normalization_references,
)
from marl_topology.objectives.surrogate_signal import (
    SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
)
from marl_topology.policies import (
    ACTIVE_ACTION_SEMANTICS_ID,
    PHYSICAL_LINK_ASSEMBLER_ID,
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    ActorPolicyInput,
    AssemblerConfig,
    EdgeScoreRecord,
    PhysicalLinkConflictAwareAssembler,
    aggregate_endpoint_scores_to_physical_links,
    build_active_action_semantics_registry,
)
from marl_topology.training.run_manifest_validator import (
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)

from .policy_gradient_losses import PolicyGradientLossInputs, clipped_reinforce_loss
from .samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    ARCHIVED_TRIAL_SAMPLER_IDS,
    FORBIDDEN_SAMPLER_INPUT_FIELDS,
    PhysicalProposalSampler,
    ProposalSample,
    ProposalSamplerConfig,
    build_active_policy_gradient_sampler_registry,
    build_stage23_trial_sampler_registry,
    sampler_cleanup_report,
)


STAGE23_SELECTED_PHYSICAL_POLICY_GRADIENT_STAGE_ID = (
    "stage_23_selected_physical_policy_gradient_landing"
)
STAGE23_PASS_VERDICT = "stage23_pass_pg_pilot_landed_sampler_promoted"
STAGE23_FAIL_VERDICT = "stage23_fail_pg_pilot_blocked_root_cause_review_required"
STAGE23_RECOMMENDED_NEXT_TASK_PASS = (
    "stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review"
)


class Stage23PolicyGradientViolation(ValueError):
    """Raised when Stage 23 crosses a declared policy-gradient boundary."""


@dataclass(frozen=True, slots=True)
class Stage23PolicyGradientConfig:
    seed: int = 2301
    sampler_seed_offsets: tuple[int, ...] = (0, 1)
    scenario_limit: int = 3
    supervised_epochs: int = 30
    supervised_learning_rate: float = 0.006
    policy_updates: int = 5
    policy_learning_rate: float = 0.00005
    clip_epsilon: float = 0.2
    top_k: int = 3
    endpoint_budget: int = 1
    # Route B / variable proposal size: when True the actor proposes a variable
    # number of edges (size bounded by per-node kind-aware budgets, not a fixed
    # top_k), so it can match the per-scenario feasible star degree (5-7) instead
    # of over-/under-selecting. Default False preserves the fixed-top_k behaviour.
    variable_proposal_size: bool = False
    tau_requirement_min: float = STAGE21_TAU_REQUIREMENT_MIN
    material_tau_drop: float = 0.05
    material_violation_worsening: float = 0.05
    material_projection_worsening: float = 0.25
    max_approx_kl: float = 0.75
    min_entropy: float = 1e-6

    def __post_init__(self) -> None:
        if self.scenario_limit < 1:
            raise Stage23PolicyGradientViolation("scenario_limit must be positive")
        if self.policy_updates < 1:
            raise Stage23PolicyGradientViolation("policy_updates must be positive")
        if not self.sampler_seed_offsets:
            raise Stage23PolicyGradientViolation("at least one sampler seed is required")
        if self.tau_requirement_min != STAGE21_TAU_REQUIREMENT_MIN:
            raise Stage23PolicyGradientViolation("Stage 23 must keep tau_requirement_min fixed at 0.9")


@dataclass(frozen=True, slots=True)
class PhysicalLogitView:
    physical_edge_ids: tuple[str, ...]
    physical_logits: torch.Tensor
    mask: torch.Tensor
    directed_records: tuple[EdgeScoreRecord, ...]
    config: ProposalSamplerConfig


@dataclass(frozen=True, slots=True)
class RolloutItem:
    row_index: int
    seed: int
    sample: ProposalSample
    training_signal_value: float
    record: Mapping[str, object]


def run_stage23_preflight(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Run execution preflight for the selected-physical Stage 23 pilot."""

    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    action_registry = build_active_action_semantics_registry()
    model_registry = build_model_registry()
    evidence = build_stage22_action_semantics_evidence()
    selected_dataset = evidence.datasets[UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID]
    gnn_boundary = LocalGNNEdgeScorer().boundary_report()
    manifest = build_stage23_manifest("stage23_preflight_manifest")
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    reward_config = build_stage23_reward_surrogate_config()
    discarded_semantics = "directed_" + "outgoing_v1"
    active_ids = set(action_registry)
    sample_actor_rows = selected_dataset.rows[0].actor_safe_view if selected_dataset.rows else ()
    gates = {
        "active_action_semantics_selected_physical": {
            "passed": (
                ACTIVE_ACTION_SEMANTICS_ID == UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
                and active_ids == {UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID}
            ),
            "active_action_semantics_id": ACTIVE_ACTION_SEMANTICS_ID,
            "active_registry_ids": sorted(active_ids),
        },
        "discarded_action_semantics_not_active": {
            "passed": discarded_semantics not in active_ids,
            "discarded_semantics_active": discarded_semantics in active_ids,
        },
        "full_gnn_v2_actor_active": {
            "passed": (
                LOCAL_GNN_EDGE_SCORER_MODEL_ID in model_registry
                and model_registry[LOCAL_GNN_EDGE_SCORER_MODEL_ID].family
                == "local_message_passing_gnn"
                and bool(gnn_boundary["outputs_edge_scores_only"])
                and bool(gnn_boundary["full_message_passing_gnn"])
            ),
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "boundary_report": gnn_boundary,
        },
        "actor_safe_input_boundary": {
            "passed": (
                bool(sample_actor_rows)
                and actor_safe_view_has_no_forbidden_fields(sample_actor_rows)
            ),
            "actor_feature_fields": list(ACTOR_EDGE_FEATURE_FIELDS),
            "forbidden_sampler_input_fields": sorted(FORBIDDEN_SAMPLER_INPUT_FIELDS),
        },
        "physical_link_assembler_active": {
            "passed": (
                PhysicalLinkConflictAwareAssembler().recommended_for_deployment is True
            ),
            "assembler_id": PHYSICAL_LINK_ASSEMBLER_ID,
        },
        "stage3_stage4_objective_evaluator_active": {
            "passed": (
                selected_dataset.readiness["uses_final_objective_stack"] is True
                and selected_dataset.readiness["fallback_used"] is False
                and STAGE21_EVALUATOR_ID == "stage21_stage3_urlcc_stage4_expected_initiator_pbft_objective_stack_v1"
            ),
            "evaluator_id": STAGE21_EVALUATOR_ID,
            "physics_regime_id": STAGE21_PHYSICS_REGIME_ID,
            "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
            "objective_contract_id": STAGE21_OBJECTIVE_CONTRACT_ID,
        },
        "reward_surrogate_config_read_only": {
            "passed": reward_config.tau == STAGE21_TAU_REQUIREMENT_MIN,
            "reward_config_fingerprint": _surrogate_config_fingerprint(reward_config),
            "weight_tuning_performed": False,
        },
        "run_manifest_validator_available": {
            "passed": validation.is_valid and validation.writes_performed is False,
            "validator_result": validation.to_dict(),
        },
    }
    return {
        "stage": STAGE23_SELECTED_PHYSICAL_POLICY_GRADIENT_STAGE_ID,
        "preflight_passed": all(bool(gate["passed"]) for gate in gates.values()),
        "gates": gates,
        "selected_action_semantics": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
        "v5_modified": False,
    }


def run_stage23_selected_physical_policy_gradient_pilot(
    *,
    config: Stage23PolicyGradientConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Run the controlled Stage 23 sampler A/B/C micro-pilot."""

    cfg = config or Stage23PolicyGradientConfig()
    torch.manual_seed(cfg.seed)
    preflight = run_stage23_preflight(project_root=project_root)
    reward_config_before = build_stage23_reward_surrogate_config()
    if not preflight["preflight_passed"]:
        return _blocked_report(preflight, reward_config_before)

    contexts = build_stage21_objective_stack_contexts()
    evidence = build_stage22_action_semantics_evidence()
    dataset = evidence.datasets[UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID]
    row_contexts = tuple(zip(dataset.rows, contexts))[: cfg.scenario_limit]
    actor = _train_stage22_supervised_full_gnn_actor(
        (row for row, _context in row_contexts),
        cfg,
    )
    supervised_state = deepcopy(actor.state_dict())
    baseline_comparison = _baseline_comparison(row_contexts, cfg, reward_config_before)
    samplers = build_stage23_trial_sampler_registry()
    sampler_reports: dict[str, dict[str, object]] = {}
    for sampler_id, sampler in samplers.items():
        sampler_actor = LocalGNNEdgeScorer()
        sampler_actor.load_state_dict(deepcopy(supervised_state))
        sampler_reports[sampler_id] = _run_sampler_micro_pilot(
            sampler_id=sampler_id,
            sampler=sampler,
            actor=sampler_actor,
            row_contexts=row_contexts,
            config=cfg,
            reward_config=reward_config_before,
        )

    selection = _select_sampler(sampler_reports)
    reward_config_after = build_stage23_reward_surrogate_config()
    reward_config_unchanged = (
        _surrogate_config_fingerprint(reward_config_before)
        == _surrogate_config_fingerprint(reward_config_after)
    )
    cleanup = sampler_cleanup_report()
    selected_sampler_id = selection.get("selected_sampler_id")
    pass_gate = bool(selected_sampler_id) and selected_sampler_id == ACTIVE_POLICY_GRADIENT_SAMPLER_ID
    pass_gate = pass_gate and reward_config_unchanged
    pass_gate = pass_gate and cleanup["active_sampler_count"] == 1
    pass_gate = pass_gate and all(
        bool(report["ran"]) for report in sampler_reports.values()
    )
    verdict = STAGE23_PASS_VERDICT if pass_gate else STAGE23_FAIL_VERDICT
    selected_report = sampler_reports.get(str(selected_sampler_id), {}) if selected_sampler_id else {}
    return {
        "stage": STAGE23_SELECTED_PHYSICAL_POLICY_GRADIENT_STAGE_ID,
        "verdict": verdict,
        "pass_gate": pass_gate,
        "preflight": preflight,
        "selected_action_semantics": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
        "policy_action_semantics": {
            "policy_action": "stochastic_proposal_over_undirected_physical_link_candidates",
            "environment_transition": (
                "proposal_set_to_selected_physical_topology_via_environment_assembler"
            ),
            "proposal_logprob_claims_final_projected_topology_probability": False,
            "policy_gradient_approximate_under_projection": True,
        },
        "samplers_tested": sorted(sampler_reports),
        "sampler_reports": sampler_reports,
        "sampler_selection": selection,
        "active_policy_gradient_sampler_id": selected_sampler_id if pass_gate else None,
        "active_sampler_registry": cleanup,
        "baseline_comparison": baseline_comparison,
        "winner_before_after": {
            "before": selected_report.get("before"),
            "after": selected_report.get("after"),
        },
        "reward_config_before": _surrogate_config_fingerprint(reward_config_before),
        "reward_config_after": _surrogate_config_fingerprint(reward_config_after),
        "reward_config_unchanged": reward_config_unchanged,
        "reward_weight_tuning_performed": False,
        "final_tau_selected": False,
        "coma_introduced": False,
        "transformer_introduced": False,
        "new_gnn_or_recurrent_architecture_introduced": False,
        "scale_up_training_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "uncontrolled_artifact_written": False,
        "v5_modified": False,
        "recommended_next_task": STAGE23_RECOMMENDED_NEXT_TASK_PASS
        if pass_gate
        else selection.get("recommended_repair_task"),
        "owner_decision_required": True,
    }


def build_stage23_manifest(run_id: str) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save/stage23_selected_physical_policy_gradient_pilot",
        overrides={
            "run_id": run_id,
            "stage_id": STAGE23_SELECTED_PHYSICAL_POLICY_GRADIENT_STAGE_ID,
            "owner_approval_id": "owner_approved_stage23_selected_physical_pg_landing",
            "config_id": "stage23_selected_physical_pg_micro_pilot_config_v1",
            "scenario_set_id": "stage22_selected_physical_micro_contexts",
            "split_id": "stage23_micro_fixed_seed_sampler_trial",
            "seed": 2301,
            "seed_group_id": "stage23_sampler_ab_c_trial_seed_group",
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "action_semantics_id": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
            "sampler_trial_ids": sorted(build_stage23_trial_sampler_registry()),
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
            "objective_id": STAGE21_OBJECTIVE_CONTRACT_ID,
            "reward_id": "stage5_3_surrogate_config_with_selected_references",
            "architecture_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
        },
    )


def build_stage23_reward_surrogate_config() -> SurrogateSignalConfig:
    # The active reward is the owner-approved feasibility-first BARRIER (Stage 31),
    # not the old flat soft-penalty (reliability_weight=100). The flat sum had a verified
    # ordering inversion: a slightly-infeasible cheap topology (psucc=0.89) could
    # outscore a comfortably-feasible expensive one (psucc=0.95), so the policy was
    # rewarded for sitting just below tau. The barrier guarantees every feasible
    # topology strictly outranks every infeasible one (offset = feasibility_margin +
    # (latency+energy weights)*clip_max), reliability plateaus above tau (zero
    # gradient), and latency+energy are the only gradient among feasible topologies.
    # The ordering guarantee is independent of the normalization references, so the
    # selected Stage 5.0L references are kept unchanged; only the structure/weights
    # change. config_id is kept stable so the reward stays a drop-in for every
    # consumer (rollout, critic-pretrain dataset, warm-start eval) that reads this
    # one function. Weights are the fixed owner-approved Stage 31 values (1/1/1,
    # linear reliability, clip_max 4.0, margin 1.0) — not tuned per run.
    references = select_normalization_references(
        build_stage5_0l_stage3_backed_sweep_range_review()["sweep_rows"],
        NormalizationReferenceConfig(source_stage_id=STAGE5_0L_RANGE_REVIEW_STAGE_ID),
    )
    return SurrogateSignalConfig(
        tau=STAGE21_TAU_REQUIREMENT_MIN,
        reliability_weight=1.0,
        latency_weight=1.0,
        energy_weight=1.0,
        latency_reference_s=references.latency_reference_s,
        energy_reference_j=references.energy_reference_j,
        clip_min=0.0,
        clip_max=4.0,
        reliability_penalty_power=1.0,
        structure=SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
        feasibility_margin=1.0,
        config_id="stage5_3_surrogate_config_with_selected_references",
    )


def _run_sampler_micro_pilot(
    *,
    sampler_id: str,
    sampler: PhysicalProposalSampler,
    actor: LocalGNNEdgeScorer,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage23PolicyGradientConfig,
    reward_config: SurrogateSignalConfig,
) -> dict[str, object]:
    before = _evaluate_sampler_policy(
        actor=actor,
        sampler=sampler,
        row_contexts=row_contexts,
        config=config,
        reward_config=reward_config,
        seed_base=config.seed,
        label=f"{sampler_id}:before",
    )
    update_rule = torch.optim.AdamW(actor.parameters(), lr=config.policy_learning_rate)
    loss_history: list[dict[str, float]] = []
    stop_reason = None
    for update_index in range(config.policy_updates):
        rollout = _collect_rollout(
            actor=actor,
            sampler=sampler,
            row_contexts=row_contexts,
            config=config,
            reward_config=reward_config,
            seed_base=config.seed + 101 * (update_index + 1),
            label=f"{sampler_id}:update{update_index}",
        )
        signal_values = torch.tensor(
            [item.training_signal_value for item in rollout],
            dtype=torch.float32,
        )
        if not torch.isfinite(signal_values).all().item():
            stop_reason = "nonfinite_reward"
            break
        baseline = signal_values.mean().expand_as(signal_values)
        old_logprob = torch.stack(
            [item.sample.logprob.detach().reshape(()) for item in rollout]
        )
        new_logprob_values = []
        for item in rollout:
            row, _context = row_contexts[item.row_index]
            view = _actor_physical_logit_view(actor, row, config)
            new_logprob_values.append(
                sampler.logprob_of(
                    view.physical_logits,
                    view.mask,
                    view.config,
                    item.sample.raw_sample_data,
                ).reshape(())
            )
        new_logprob = torch.stack(new_logprob_values)
        loss_result = clipped_reinforce_loss(
            PolicyGradientLossInputs(
                new_logprob=new_logprob,
                old_logprob=old_logprob,
                training_signal=signal_values,
                baseline=baseline,
                clip_epsilon=config.clip_epsilon,
                normalize_advantage=True,
            )
        )
        if not torch.isfinite(loss_result.loss).all().item():
            stop_reason = "nonfinite_policy_loss"
            break
        update_rule.zero_grad()
        loss_result.loss.backward()
        update_rule.step()
        loss_history.append(loss_result.to_payload())
        if loss_result.approximate_kl > config.max_approx_kl:
            stop_reason = "kl_explosion"
            break
    after = _evaluate_sampler_policy(
        actor=actor,
        sampler=sampler,
        row_contexts=row_contexts,
        config=config,
        reward_config=reward_config,
        seed_base=config.seed,
        label=f"{sampler_id}:after",
    )
    safety = _safety_checks(before["summary"], after["summary"], config, stop_reason)
    return {
        "sampler_id": sampler_id,
        "ran": True,
        "before": before["summary"],
        "after": after["summary"],
        "loss_history": loss_history,
        "representative_records": after["records"][: min(3, len(after["records"]))],
        "safety": safety,
        "pass_gate": safety["passed"],
        "active_after_stage23": sampler_id == ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
        "archived_after_stage23": sampler_id in ARCHIVED_TRIAL_SAMPLER_IDS,
    }


def _collect_rollout(
    *,
    actor: LocalGNNEdgeScorer,
    sampler: PhysicalProposalSampler,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage23PolicyGradientConfig,
    reward_config: SurrogateSignalConfig,
    seed_base: int,
    label: str,
) -> tuple[RolloutItem, ...]:
    items: list[RolloutItem] = []
    for seed_offset in config.sampler_seed_offsets:
        rng = torch.Generator().manual_seed(seed_base + seed_offset)
        for row_index, (row, context) in enumerate(row_contexts):
            view = _actor_physical_logit_view(actor, row, config)
            sample = sampler.sample(view.physical_logits, view.mask, view.config, rng)
            record = _evaluate_proposal_sample(
                row=row,
                context=context,
                view=view,
                sample=sample,
                reward_config=reward_config,
                config=config,
                topology_label=f"{label}:seed{seed_base + seed_offset}:row{row_index}",
            )
            items.append(
                RolloutItem(
                    row_index=row_index,
                    seed=seed_base + seed_offset,
                    sample=sample,
                    training_signal_value=float(record["reward_surrogate"]),
                    record=record,
                )
            )
    return tuple(items)


def _evaluate_sampler_policy(
    *,
    actor: LocalGNNEdgeScorer,
    sampler: PhysicalProposalSampler,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage23PolicyGradientConfig,
    reward_config: SurrogateSignalConfig,
    seed_base: int,
    label: str,
) -> dict[str, object]:
    rollout = _collect_rollout(
        actor=actor,
        sampler=sampler,
        row_contexts=row_contexts,
        config=config,
        reward_config=reward_config,
        seed_base=seed_base,
        label=label,
    )
    records = [dict(item.record) for item in rollout]
    return {"summary": _summarize_records(records), "records": records}


def _evaluate_proposal_sample(
    *,
    row: object,
    context: Stage21EvaluationContext,
    view: PhysicalLogitView,
    sample: ProposalSample,
    reward_config: SurrogateSignalConfig,
    config: Stage23PolicyGradientConfig,
    topology_label: str,
) -> dict[str, object]:
    assembly = _assemble_physical_proposal(row.actor_safe_view, view, sample)
    evaluation = context.evaluator.evaluate(
        assembly.selected_physical_edges,
        topology_id=f"stage23:{topology_label}",
    )
    signal = evaluate_reward_surrogate(
        SurrogateSignalInput(
            consensus_success_probability=float(
                evaluation.metrics["consensus_success_probability"]
            ),
            latency=float(evaluation.metrics["latency"]),
            energy=float(evaluation.metrics["energy"]),
            topology_diagnostics=evaluation.metrics["topology_diagnostics"],
        ),
        reward_config,
    )
    projection = _projection_diagnostics(view, sample, assembly)
    consensus = float(evaluation.metrics["consensus_success_probability"])
    return {
        "pre_projection_proposals": list(sample.proposed_physical_edges),
        "proposal_logprob": float(sample.logprob.detach().cpu().item()),
        "proposal_entropy": float(sample.entropy.detach().cpu().item()),
        "post_projection_selected_physical_edges": list(assembly.selected_physical_edges),
        "projection_rejection_reasons": {
            edge_id: [reason.value for reason in reasons]
            for edge_id, reasons in assembly.rejection_reasons.items()
        },
        "top_proposal_rejection_rate": projection["top_proposal_rejection_rate"],
        "above_threshold_rejection_rate": projection["above_threshold_rejection_rate"],
        "rejection_by_reason": projection["rejection_by_reason"],
        "selected_edge_count": len(assembly.selected_physical_edges),
        "candidate_physical_edge_count": len(view.physical_edge_ids),
        "reward_surrogate": float(signal.training_signal_value),
        "consensus_success_probability": consensus,
        "latency": float(evaluation.metrics["latency"]),
        "energy": float(evaluation.metrics["energy"]),
        "tau_feasible": consensus >= config.tau_requirement_min,
        "violation_indicator": int(consensus < config.tau_requirement_min),
        "topology_diagnostics": dict(evaluation.metrics["topology_diagnostics"]),
        "sampler_id": sample.sampler_id,
        "selected_action_semantics": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
        "policy_action": "proposal",
        "environment_transition": "projected_topology",
    }


def _assemble_physical_proposal(
    actor_safe_view: Iterable[Mapping[str, object]],
    view: PhysicalLogitView,
    sample: ProposalSample,
):
    proposed = set(sample.proposed_physical_edges)
    directed_records = tuple(
        record for record in view.directed_records if record.edge_id in proposed
    )
    physical_scores = aggregate_endpoint_scores_to_physical_links(directed_records)
    actor_rows = tuple(actor_safe_view)
    assembler = PhysicalLinkConflictAwareAssembler(
        AssemblerConfig(
            assembler_id=PHYSICAL_LINK_ASSEMBLER_ID,
            mode="physical_link_conflict_aware_greedy",
            tx_capacity=tx_capacity_from_actor_safe_view(actor_rows),
            rx_capacity=rx_capacity_from_actor_safe_view(actor_rows),
            deterministic=True,
        )
    )
    return assembler.assemble(
        physical_scores,
        candidate_constraints_from_actor_safe_view(actor_rows),
        metadata={},
    )


def _actor_history_window(
    frames: object,
    step_index: int,
    window: int,
) -> torch.Tensor | None:
    """Leakage-safe ``[E, W, F]`` history for ``step_index`` from a per-scenario A3 frame
    sequence (``[(row, context), ...]`` oldest->present). Built ONLY from already-validated
    actor edge tensors of past+present frames (A4), aligned to the current frame's directed
    edge order. ``window <= 1`` (single frame) or no frames -> None (the static v3 path)."""
    if frames is None or window <= 1:
        return None
    low = max(0, step_index - window + 1)
    frames_inputs = [
        [
            _policy_input_from_stage22_actor_row(actor_row)
            for actor_row in frames[time_step][0].actor_safe_view
        ]
        for time_step in range(low, step_index + 1)
    ]
    return tensorize_actor_history_sequence(frames_inputs, window=window)


def _actor_physical_logit_view(
    actor: LocalGNNEdgeScorer,
    row: object,
    config: Stage23PolicyGradientConfig,
    history: torch.Tensor | None = None,
) -> PhysicalLogitView:
    actor_inputs = tuple(
        _policy_input_from_stage22_actor_row(actor_row)
        for actor_row in row.actor_safe_view
    )
    batch = tensorize_actor_policy_inputs(actor_inputs)
    if history is not None:
        # Read-only temporal window carried alongside the (unchanged) current-frame edge
        # tensor; only a temporal actor consumes it, the v3 GNN / MLP ignore it. When
        # absent the batch and scoring are byte-identical to the static path.
        batch = replace(batch, history=history)
    directed_logits = actor.score_tensor_batch(batch)
    directed_records = batch.to_edge_score_batch(
        directed_logits,
        batch_id="stage23_actor_endpoint_logits",
        source="stage23_full_gnn_actor_endpoint_score",
    ).edge_scores
    by_physical: dict[str, list[int]] = defaultdict(list)
    for index, ref in enumerate(batch.records):
        by_physical[ref.edge_id].append(index)
    physical_ids = tuple(sorted(by_physical))
    physical_logits: list[torch.Tensor] = []
    for edge_id in physical_ids:
        indices = torch.tensor(by_physical[edge_id], dtype=torch.long, device=directed_logits.device)
        physical_logits.append(directed_logits[indices].max())
    logits = torch.stack(physical_logits) if physical_logits else directed_logits.new_empty((0,))
    mask = _physical_candidate_mask(row.actor_safe_view, physical_ids, logits)
    if getattr(config, "variable_proposal_size", False):
        # Variable proposal size (the Stage 31/32 mechanism, restored): set the
        # per-scenario proposal size to N-1 (the feasible RSU-star degree) derived
        # from the node count, instead of a fixed global top_k, then let the normal
        # Plackett-Luce sampler pick exactly that many top-scored edges. This matches
        # each scene's feasible size (n6->5, n7->6, n8->7). NB: budget-saturation
        # (sequential sampler) over-connects and per-edge Bernoulli is too noisy;
        # deterministic N-1 is the validated mechanism. Both the rollout and the
        # update recompute call this same builder, so the log-prob factorization
        # stays identical (PPO-ratio consistent).
        node_ids: set[str] = set()
        for actor_row in row.actor_safe_view:
            node_ids.add(str(actor_row["agent_id"]))
            node_ids.add(str(actor_row["neighbor_id"]))
        top_k = max(1, min(len(node_ids) - 1, len(physical_ids)))
    else:
        top_k = min(config.top_k, len(physical_ids))
    sampler_config = ProposalSamplerConfig(
        physical_edge_ids=physical_ids,
        top_k=top_k,
        endpoint_budget=config.endpoint_budget,
    )
    return PhysicalLogitView(
        physical_edge_ids=physical_ids,
        physical_logits=logits,
        mask=mask,
        directed_records=directed_records,
        config=sampler_config,
    )


def _physical_candidate_mask(
    actor_safe_view: Iterable[Mapping[str, object]],
    physical_ids: tuple[str, ...],
    logits: torch.Tensor,
) -> torch.Tensor:
    constraints_by_physical: dict[str, list[object]] = defaultdict(list)
    for constraint in candidate_constraints_from_actor_safe_view(actor_safe_view):
        constraints_by_physical[constraint.edge_id].append(constraint)
    values = []
    for edge_id in physical_ids:
        constraints = constraints_by_physical.get(edge_id, ())
        values.append(
            bool(constraints)
            and all(constraint.valid_candidate and constraint.role_allowed for constraint in constraints)
        )
    return torch.tensor(values, dtype=torch.bool, device=logits.device)


def _projection_diagnostics(
    view: PhysicalLogitView,
    sample: ProposalSample,
    assembly,
) -> dict[str, object]:
    rejected = set(assembly.rejected_physical_edges)
    proposed_indices = [
        index
        for index, edge_id in enumerate(view.physical_edge_ids)
        if edge_id in set(sample.proposed_physical_edges)
    ]
    ranked = sorted(
        proposed_indices,
        key=lambda index: (-float(view.physical_logits[index].detach().cpu().item()), view.physical_edge_ids[index]),
    )
    top_count = max(1, min(5, len(ranked))) if ranked else 0
    top_rejected = (
        sum(1 for index in ranked[:top_count] if view.physical_edge_ids[index] in rejected)
        if top_count
        else 0
    )
    probabilities = torch.sigmoid(view.physical_logits.detach()).cpu()
    above = [
        index
        for index in proposed_indices
        if float(probabilities[index].item()) >= 0.5
    ]
    above_rejected = sum(1 for index in above if view.physical_edge_ids[index] in rejected)
    return {
        "top_proposal_rejection_rate": top_rejected / top_count if top_count else 0.0,
        "above_threshold_rejection_rate": above_rejected / len(above) if above else 0.0,
        "rejection_by_reason": dict(assembly.diagnostics["rejection_reason_counts"]),
    }


def _train_stage22_supervised_full_gnn_actor(
    rows: Iterable[object],
    config: Stage23PolicyGradientConfig,
) -> LocalGNNEdgeScorer:
    row_tuple = tuple(rows)
    actor = LocalGNNEdgeScorer()
    batch, targets = _stage22_actor_batch_and_targets(row_tuple)
    update_rule = torch.optim.AdamW(actor.parameters(), lr=config.supervised_learning_rate)
    for _epoch in range(config.supervised_epochs):
        update_rule.zero_grad()
        logits = actor.score_tensor_batch(batch)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        update_rule.step()
    return actor


def _stage22_actor_batch_and_targets(rows: tuple[object, ...]):
    policy_inputs: list[ActorPolicyInput] = []
    targets_by_sample_edge: dict[tuple[int, str], float] = {}
    for row in rows:
        target_by_agent_edge = {
            (str(target["agent_id"]), str(target["edge_id"])): float(
                target["actor_edge_utility_target"]
            )
            for target in row.actor_target_view["actor_soft_utility_targets"]
        }
        for actor_row in row.actor_safe_view:
            policy_input = _policy_input_from_stage22_actor_row(actor_row)
            sample_index = len(policy_inputs)
            for neighbor in policy_input.local_neighbor_observations:
                edge_id = str(_neighbor_value(neighbor, "edge_id"))
                key = (policy_input.agent_id, edge_id)
                if key not in target_by_agent_edge:
                    raise Stage23PolicyGradientViolation(f"missing actor target for {key}")
                targets_by_sample_edge[(sample_index, edge_id)] = target_by_agent_edge[key]
            policy_inputs.append(policy_input)
    batch = tensorize_actor_policy_inputs(policy_inputs)
    targets = torch.tensor(
        [
            targets_by_sample_edge[(ref.sample_index, ref.edge_id)]
            for ref in batch.records
        ],
        dtype=torch.float32,
    )
    return batch, targets


def _baseline_comparison(
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage23PolicyGradientConfig,
    reward_config: SurrogateSignalConfig,
) -> dict[str, object]:
    records: dict[str, list[dict[str, object]]] = {
        "objective_aware_teacher": [],
        "projected_greedy_baseline": [],
        "projected_full_graph_baseline": [],
    }
    for row_index, (row, context) in enumerate(row_contexts):
        records["objective_aware_teacher"].append(
            _evaluate_fixed_edges(
                context,
                row.selected_physical_edges,
                reward_config,
                config,
                topology_id=f"stage23:teacher:row{row_index}",
            )
        )
        records["projected_greedy_baseline"].append(
            _evaluate_projected_fixed_proposal(
                row.actor_safe_view,
                context,
                context.topology_variants["greedy_reliability_raw"],
                reward_config,
                config,
                topology_id=f"stage23:projected_greedy:row{row_index}",
            )
        )
        records["projected_full_graph_baseline"].append(
            _evaluate_projected_fixed_proposal(
                row.actor_safe_view,
                context,
                context.graph.edge_ids,
                reward_config,
                config,
                topology_id=f"stage23:projected_full_graph:row{row_index}",
            )
        )
    return {name: _summarize_records(items) for name, items in records.items()}


def _evaluate_projected_fixed_proposal(
    actor_safe_view: Iterable[Mapping[str, object]],
    context: Stage21EvaluationContext,
    proposed_edge_ids: Iterable[str],
    reward_config: SurrogateSignalConfig,
    config: Stage23PolicyGradientConfig,
    *,
    topology_id: str,
) -> dict[str, object]:
    actor_rows = tuple(actor_safe_view)
    proposed = set(proposed_edge_ids)
    scores: list[EdgeScoreRecord] = []
    for row in actor_rows:
        edge_id = str(row["edge_id"])
        if edge_id not in proposed:
            continue
        neighbor = tuple(row["local_neighbor_observations"])[0]
        scores.append(
            EdgeScoreRecord(
                agent_id=str(row["agent_id"]),
                neighbor_id=str(row["neighbor_id"]),
                edge_id=edge_id,
                directed_edge_id=str(row["directed_edge_id"]),
                score=float(neighbor["link_success_probability"]),
                probability=float(neighbor["link_success_probability"]),
                score_source="stage23_fixed_baseline_physical_projection",
                time_step=int(row["time_step"]),
            )
        )
    physical_scores = aggregate_endpoint_scores_to_physical_links(scores)
    assembler = PhysicalLinkConflictAwareAssembler(
        AssemblerConfig(
            assembler_id=PHYSICAL_LINK_ASSEMBLER_ID,
            mode="physical_link_conflict_aware_greedy",
            tx_capacity=tx_capacity_from_actor_safe_view(actor_rows),
            rx_capacity=rx_capacity_from_actor_safe_view(actor_rows),
            deterministic=True,
        )
    )
    assembly = assembler.assemble(
        physical_scores,
        candidate_constraints_from_actor_safe_view(actor_rows),
        metadata={},
    )
    record = _evaluate_fixed_edges(
        context,
        assembly.selected_physical_edges,
        reward_config,
        config,
        topology_id=topology_id,
    )
    record.update(
        {
            "pre_projection_proposals": list(proposed),
            "post_projection_selected_physical_edges": list(assembly.selected_physical_edges),
            "projection_rejection_reasons": {
                edge_id: [reason.value for reason in reasons]
                for edge_id, reasons in assembly.rejection_reasons.items()
            },
            "rejection_by_reason": dict(assembly.diagnostics["rejection_reason_counts"]),
        }
    )
    return record


def _evaluate_fixed_edges(
    context: Stage21EvaluationContext,
    selected_edges: Iterable[str],
    reward_config: SurrogateSignalConfig,
    config: Stage23PolicyGradientConfig,
    *,
    topology_id: str,
) -> dict[str, object]:
    evaluation = context.evaluator.evaluate(selected_edges, topology_id=topology_id)
    consensus = float(evaluation.metrics["consensus_success_probability"])
    signal = evaluate_reward_surrogate(
        SurrogateSignalInput(
            consensus_success_probability=consensus,
            latency=float(evaluation.metrics["latency"]),
            energy=float(evaluation.metrics["energy"]),
            topology_diagnostics=evaluation.metrics["topology_diagnostics"],
        ),
        reward_config,
    )
    return {
        "consensus_success_probability": consensus,
        "latency": float(evaluation.metrics["latency"]),
        "energy": float(evaluation.metrics["energy"]),
        "selected_edge_count": len(tuple(selected_edges)),
        "reward_surrogate": float(signal.training_signal_value),
        "tau_feasible": consensus >= config.tau_requirement_min,
        "violation_indicator": int(consensus < config.tau_requirement_min),
    }


def _summarize_records(records: list[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        return {}
    edge_counts = [float(row.get("selected_edge_count", 0.0)) for row in records]
    candidate_counts = [
        int(row.get(
            "candidate_physical_edge_count",
            len(row.get("post_projection_selected_physical_edges", ()))
            + len(row.get("projection_rejection_reasons", {})),
        ))
        for row in records
        if "post_projection_selected_physical_edges" in row
    ]
    rejection_counter: Counter[str] = Counter()
    for row in records:
        rejection_counter.update(dict(row.get("rejection_by_reason", {})))
    full_rates = [
        1.0 if candidate_count > 1 and edge_count >= candidate_count else 0.0
        for edge_count, candidate_count in zip(edge_counts, candidate_counts)
    ]
    empty_rates = [1.0 if edge_count == 0 else 0.0 for edge_count in edge_counts]
    return {
        "sample_count": len(records),
        "tau_feasible_rate": _mean(float(row["tau_feasible"]) for row in records),
        "violation_rate": _mean(float(row["violation_indicator"]) for row in records),
        "mean_consensus_success_probability": _mean(
            float(row["consensus_success_probability"]) for row in records
        ),
        "mean_latency": _mean(float(row["latency"]) for row in records),
        "mean_energy": _mean(float(row["energy"]) for row in records),
        "mean_selected_edge_count": _mean(edge_counts),
        "selected_edge_count_std": _std(edge_counts),
        "top_proposal_rejection_rate": _mean(
            float(row.get("top_proposal_rejection_rate", 0.0)) for row in records
        ),
        "above_threshold_rejection_rate": _mean(
            float(row.get("above_threshold_rejection_rate", 0.0)) for row in records
        ),
        "rejection_by_reason": dict(rejection_counter),
        "mean_entropy": _mean(float(row.get("proposal_entropy", 0.0)) for row in records),
        "mean_proposal_logprob": _mean(float(row.get("proposal_logprob", 0.0)) for row in records),
        "mean_reward_surrogate": _mean(float(row["reward_surrogate"]) for row in records),
        "empty_graph_rate": _mean(empty_rates),
        "full_graph_rate": _mean(full_rates) if full_rates else 0.0,
    }


def _safety_checks(
    before: Mapping[str, object],
    after: Mapping[str, object],
    config: Stage23PolicyGradientConfig,
    stop_reason: str | None,
) -> dict[str, object]:
    tau_drop = float(before["tau_feasible_rate"]) - float(after["tau_feasible_rate"])
    violation_worsening = float(after["violation_rate"]) - float(before["violation_rate"])
    projection_worsening = (
        float(after["top_proposal_rejection_rate"])
        - float(before["top_proposal_rejection_rate"])
    )
    collapse = (
        float(after["empty_graph_rate"]) >= 0.5
        or float(after["full_graph_rate"]) > 0.5
        or float(after["selected_edge_count_std"]) <= 1e-9
        and float(after["mean_selected_edge_count"]) in {0.0}
    )
    entropy_collapse = float(after["mean_entropy"]) < config.min_entropy
    issues = []
    if stop_reason:
        issues.append(stop_reason)
    if tau_drop > config.material_tau_drop:
        issues.append("tau_feasible_rate_degraded")
    if violation_worsening > config.material_violation_worsening:
        issues.append("violation_rate_worsened")
    if projection_worsening > config.material_projection_worsening:
        issues.append("projection_mismatch_worsened")
    if collapse:
        issues.append("actor_collapse")
    if entropy_collapse:
        issues.append("entropy_collapse")
    return {
        "passed": not issues,
        "issues": issues,
        "tau_feasible_drop": tau_drop,
        "violation_worsening": violation_worsening,
        "projection_worsening": projection_worsening,
        "collapse_detected": collapse,
        "entropy_collapse": entropy_collapse,
        "early_stop_reason": stop_reason,
    }


def _select_sampler(sampler_reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    eligible = []
    for sampler_id, report in sampler_reports.items():
        if not report.get("pass_gate"):
            continue
        after = report["after"]
        eligible.append(
            (
                float(after["top_proposal_rejection_rate"]),
                -float(after["mean_reward_surrogate"]),
                float(after["violation_rate"]),
                float(after["mean_latency"]) + float(after["mean_energy"]),
                _sampler_complexity_rank(sampler_id),
                sampler_id,
            )
        )
    if not eligible:
        return {
            "selected_sampler_id": None,
            "selection_reason": "no sampler passed safety gates",
            "policy_gradient_blocked": True,
            "recommended_repair_task": "stage_24_sampler_logprob_or_projection_repair",
        }
    selected = min(eligible)[-1]
    return {
        "selected_sampler_id": selected,
        "selection_reason": (
            "selected from safety-passing samplers by projection mismatch, "
            "reward surrogate, violation rate, resource cost, and simplicity"
        ),
        "policy_gradient_blocked": False,
        "eligible_sampler_ids": [item[-1] for item in sorted(eligible)],
        "low_entropy_promotion": {
            "active_registry_expected_count": 1,
            "archived_trial_sampler_ids": list(ARCHIVED_TRIAL_SAMPLER_IDS),
        },
    }


def _sampler_complexity_rank(sampler_id: str) -> int:
    ranks = {
        "physical_plackett_luce_top_k_sampler": 0,
        "physical_bernoulli_proposal_sampler": 1,
        "endpoint_budgeted_physical_proposal_sampler": 2,
    }
    return ranks.get(sampler_id, 99)


def _blocked_report(
    preflight: Mapping[str, object],
    reward_config: SurrogateSignalConfig,
) -> dict[str, object]:
    return {
        "stage": STAGE23_SELECTED_PHYSICAL_POLICY_GRADIENT_STAGE_ID,
        "verdict": STAGE23_FAIL_VERDICT,
        "pass_gate": False,
        "preflight": dict(preflight),
        "samplers_tested": [],
        "sampler_selection": {
            "selected_sampler_id": None,
            "selection_reason": "preflight failed",
            "policy_gradient_blocked": True,
            "recommended_repair_task": "stage_24_preflight_blocker_repair",
        },
        "reward_config_before": _surrogate_config_fingerprint(reward_config),
        "reward_config_unchanged": True,
        "checkpoint_written": False,
        "artifact_written": False,
        "v5_modified": False,
        "owner_decision_required": True,
    }


def _surrogate_config_fingerprint(config: SurrogateSignalConfig) -> dict[str, object]:
    return {
        "tau": config.tau,
        "reliability_weight": config.reliability_weight,
        "latency_weight": config.latency_weight,
        "energy_weight": config.energy_weight,
        "latency_reference_s": config.latency_reference_s,
        "energy_reference_j": config.energy_reference_j,
        "clip_min": config.clip_min,
        "clip_max": config.clip_max,
        "reliability_penalty_power": config.reliability_penalty_power,
        "config_id": config.config_id,
        "model_id": config.model_id,
    }


def _neighbor_value(neighbor: object, name: str) -> object:
    if isinstance(neighbor, Mapping):
        return neighbor[name]
    return getattr(neighbor, name)


def _policy_input_from_stage22_actor_row(
    row: Mapping[str, object],
) -> ActorPolicyInput:
    """Drop Stage 22 training diagnostics before deployment actor tensorization."""

    return ActorPolicyInput(
        agent_id=str(row["agent_id"]),
        agent_kind=str(row["agent_kind"]),
        time_step=int(row["time_step"]),
        local_position_m=tuple(row["local_position_m"]),  # type: ignore[arg-type]
        local_neighbor_observations=tuple(row["local_neighbor_observations"]),  # type: ignore[arg-type]
        local_messages=tuple(row["local_messages"]),  # type: ignore[arg-type]
        local_history=dict(row["local_history"]),  # type: ignore[arg-type]
    )


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    mean = _mean(items)
    return (sum((value - mean) ** 2 for value in items) / len(items)) ** 0.5


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None
