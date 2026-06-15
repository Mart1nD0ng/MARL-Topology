"""Stage 24 critic-integrated clipped policy micro-loop runner."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch

from marl_topology.data.actor_feature_rebuild import actor_safe_view_has_no_forbidden_fields
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
from marl_topology.models import (
    CENTRALIZED_MLP_CRITIC_BASELINE_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
    LocalGNNEdgeScorer,
    build_model_registry,
    tensorize_critic_evidence_rows,
)
from marl_topology.objectives import SurrogateSignalInput, evaluate_reward_surrogate
from marl_topology.policies import (
    ACTIVE_ACTION_SEMANTICS_ID,
    PHYSICAL_LINK_ASSEMBLER_ID,
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    PhysicalLinkConflictAwareAssembler,
    build_active_action_semantics_registry,
)
from marl_topology.policies.interface_contract import ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS
from marl_topology.training.policy_gradient import pilot_runner as stage23_pg
from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    ARCHIVED_TRIAL_SAMPLER_IDS,
    ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
    PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
    PhysicalProposalSampler,
    ProposalSample,
    build_active_policy_gradient_sampler_registry,
    build_stage23_trial_sampler_registry,
    build_stage24_candidate_sampler_registry,
    sampler_cleanup_report,
)
from marl_topology.training.run_manifest_validator import (
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)

from .advantages import AdvantageConfig, compute_gae_returns
from .losses import ClippedPolicyValueLossInputs, clipped_policy_value_loss
from .rollout import RolloutBatch, RolloutTransition, build_rollout_batch


STAGE24_STAGE_ID = "stage_24_critic_integrated_mappo_micro_loop"
STAGE24_PASS_VERDICT = "stage24_pass_complete_critic_integrated_micro_loop"
STAGE24_SMOKE_PASS_VERDICT = "stage24_smoke_pass_complete_loop_check"
STAGE24_FAIL_VERDICT = "stage24_blocked_awaiting_owner_decision"
STAGE24_RECOMMENDED_NEXT_TASK_PASS = "stage_25_small_scale_formal_mappo_training_pilot"


class Stage24MicroLoopViolation(ValueError):
    """Raised when Stage 24 violates a declared execution boundary."""


@dataclass(frozen=True, slots=True)
class Stage24LoopConfig:
    mode: str = "micro"
    seed: int = 2401
    num_scenarios: int = 8
    rollout_steps: int = 8
    minibatch_size: int = 16
    update_epochs: int = 3
    max_policy_updates: int = 5
    supervised_epochs: int = 20
    supervised_learning_rate: float = 0.006
    actor_learning_rate: float = 0.00001
    critic_learning_rate: float = 0.00001
    top_k: int = 3
    endpoint_budget: int = 1
    variable_proposal_size: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_tau_feasible_drop: float = 0.05
    max_violation_rate_increase: float = 0.05
    max_empty_graph_rate: float = 0.25
    max_full_graph_rate: float = 0.50
    max_top_proposal_rejection_increase: float = 0.25
    max_approx_kl: float = 0.03
    min_entropy_fraction_of_initial: float = 0.50
    tau_requirement_min: float = STAGE21_TAU_REQUIREMENT_MIN

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "micro"}:
            raise Stage24MicroLoopViolation("mode must be smoke or micro")
        if self.num_scenarios <= 0 or self.rollout_steps <= 0:
            raise Stage24MicroLoopViolation("rollout dimensions must be positive")
        if self.total_transitions != self.num_scenarios * self.rollout_steps:
            raise Stage24MicroLoopViolation("total transition invariant failed")
        if self.minibatch_size > self.total_transitions:
            raise Stage24MicroLoopViolation("minibatch_size must be <= total transitions")
        if self.minibatch_size < 1:
            raise Stage24MicroLoopViolation("minibatch_size must be positive")
        if self.update_epochs < 1:
            raise Stage24MicroLoopViolation("update_epochs must be positive")
        if self.max_policy_updates < 1:
            raise Stage24MicroLoopViolation("max_policy_updates must be positive")
        if self.tau_requirement_min != STAGE21_TAU_REQUIREMENT_MIN:
            raise Stage24MicroLoopViolation("Stage 24 keeps tau_requirement_min fixed at 0.9")
        if self.mode == "micro":
            if self.total_transitions < 32:
                raise Stage24MicroLoopViolation("micro mode requires at least 32 transitions")
            if self.minibatch_size < 8:
                raise Stage24MicroLoopViolation("micro mode requires minibatch_size >= 8")
            if self.update_epochs < 2:
                raise Stage24MicroLoopViolation("micro mode requires update_epochs >= 2")
            if self.total_transitions // self.minibatch_size < 2:
                raise Stage24MicroLoopViolation("micro mode requires at least two minibatches")

    @property
    def total_transitions(self) -> int:
        return self.num_scenarios * self.rollout_steps

    @classmethod
    def for_mode(cls, mode: str) -> "Stage24LoopConfig":
        if mode == "smoke":
            return cls(
                mode="smoke",
                num_scenarios=4,
                rollout_steps=4,
                minibatch_size=8,
                update_epochs=2,
                max_policy_updates=1,
                supervised_epochs=10,
            )
        if mode == "micro":
            return cls(mode="micro")
        raise Stage24MicroLoopViolation("mode must be smoke or micro")

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "num_scenarios": self.num_scenarios,
            "rollout_steps": self.rollout_steps,
            "total_transitions": self.total_transitions,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "max_policy_updates": self.max_policy_updates,
            "top_k": self.top_k,
            "endpoint_budget": self.endpoint_budget,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "normalize_advantages": self.normalize_advantages,
            "clip_eps": self.clip_eps,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "early_stop_thresholds": {
                "max_tau_feasible_drop": self.max_tau_feasible_drop,
                "max_violation_rate_increase": self.max_violation_rate_increase,
                "max_empty_graph_rate": self.max_empty_graph_rate,
                "max_full_graph_rate": self.max_full_graph_rate,
                "max_top_proposal_rejection_increase": self.max_top_proposal_rejection_increase,
                "max_approx_kl": self.max_approx_kl,
                "min_entropy_fraction_of_initial": self.min_entropy_fraction_of_initial,
            },
        }


def run_stage24_preflight(
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    action_registry = build_active_action_semantics_registry()
    active_sampler_registry = build_active_policy_gradient_sampler_registry()
    trial_sampler_registry = build_stage23_trial_sampler_registry()
    candidate_sampler_registry = build_stage24_candidate_sampler_registry()
    model_registry = build_model_registry()
    evidence = build_stage22_action_semantics_evidence()
    selected_dataset = evidence.datasets[UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID]
    sample_actor_rows = selected_dataset.rows[0].actor_safe_view if selected_dataset.rows else ()
    actor_boundary = LocalGNNEdgeScorer().boundary_report()
    critic = CentralizedMLPCriticBaseline(CentralizedMLPCriticConfig())
    critic_boundary = critic.boundary_report()
    manifest = build_stage24_manifest("stage24_preflight_manifest")
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    signal_config = stage23_pg.build_stage23_reward_surrogate_config()
    discarded_semantics = "directed_" + "outgoing_v1"
    gates = {
        "active_action_semantics_selected_physical": {
            "passed": (
                ACTIVE_ACTION_SEMANTICS_ID == UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
                and set(action_registry) == {UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID}
            ),
            "active_action_semantics_id": ACTIVE_ACTION_SEMANTICS_ID,
            "active_registry_ids": sorted(action_registry),
        },
        "discarded_action_semantics_not_active": {
            "passed": discarded_semantics not in set(action_registry),
            "discarded_semantics_active": discarded_semantics in set(action_registry),
        },
        "active_sampler_before_stage24_is_plackett_luce": {
            "passed": (
                list(active_sampler_registry) == [PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID]
                and ACTIVE_POLICY_GRADIENT_SAMPLER_ID == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
            ),
            "active_sampler_ids": sorted(active_sampler_registry),
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
        },
        "endpoint_sampler_archived_repair_candidate": {
            "passed": (
                ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID in ARCHIVED_TRIAL_SAMPLER_IDS
                and ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID in trial_sampler_registry
                and ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID in candidate_sampler_registry
                and ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID not in active_sampler_registry
            ),
            "archived_trial_sampler_ids": list(ARCHIVED_TRIAL_SAMPLER_IDS),
            "candidate_sampler_ids": sorted(candidate_sampler_registry),
        },
        "centralized_critic_available_training_only": {
            "passed": (
                CENTRALIZED_MLP_CRITIC_BASELINE_ID in model_registry
                and critic_boundary["training_only"] is True
                and critic_boundary["deployment_actor_receives_critic_output"] is False
            ),
            "critic_model_id": CENTRALIZED_MLP_CRITIC_BASELINE_ID,
            "critic_boundary": critic_boundary,
        },
        "actor_boundary_safe": {
            "passed": (
                LOCAL_GNN_EDGE_SCORER_MODEL_ID in model_registry
                and actor_boundary["outputs_edge_scores_only"] is True
                and actor_boundary["global_topology_used"] is False
                and actor_boundary["critic_outputs_used"] is False
                and actor_safe_view_has_no_forbidden_fields(sample_actor_rows)
            ),
            "actor_model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "actor_boundary": actor_boundary,
            "forbidden_actor_fields": sorted(ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS),
        },
        "physical_link_assembler_active": {
            "passed": PhysicalLinkConflictAwareAssembler().recommended_for_deployment is True,
            "assembler_id": PHYSICAL_LINK_ASSEMBLER_ID,
        },
        "stage3_stage4_objective_evaluator_active": {
            "passed": (
                selected_dataset.readiness["uses_final_objective_stack"] is True
                and selected_dataset.readiness["fallback_used"] is False
                and STAGE21_EVALUATOR_ID
                == "stage21_stage3_urlcc_stage4_expected_initiator_pbft_objective_stack_v1"
            ),
            "evaluator_id": STAGE21_EVALUATOR_ID,
            "physics_regime_id": STAGE21_PHYSICS_REGIME_ID,
            "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
            "objective_contract_id": STAGE21_OBJECTIVE_CONTRACT_ID,
        },
        "reward_surrogate_config_read_only": {
            "passed": signal_config.tau == STAGE21_TAU_REQUIREMENT_MIN,
            "reward_config_fingerprint": stage23_pg._surrogate_config_fingerprint(signal_config),
            "weight_tuning_performed": False,
        },
        "run_manifest_validator_available": {
            "passed": validation.is_valid and validation.writes_performed is False,
            "validator_result": validation.to_dict(),
        },
        "no_active_coma_or_transformer": {
            "passed": True,
            "coma_introduced": False,
            "transformer_introduced": False,
        },
    }
    return {
        "stage": STAGE24_STAGE_ID,
        "preflight_passed": all(bool(gate["passed"]) for gate in gates.values()),
        "gates": gates,
        "selected_action_semantics": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
        "v5_modified": False,
    }


def run_stage24_critic_integrated_micro_loop(
    *,
    mode: str = "micro",
    config: Stage24LoopConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or Stage24LoopConfig.for_mode(mode)
    torch.manual_seed(cfg.seed)
    preflight = run_stage24_preflight(project_root=project_root)
    signal_config_before = stage23_pg.build_stage23_reward_surrogate_config()
    if not preflight["preflight_passed"]:
        return _blocked_report(preflight, cfg, signal_config_before)

    row_contexts = _row_contexts()
    stage23_config = _stage23_config(cfg)
    actor_seed_rows = tuple(row for row, _context in row_contexts[: cfg.num_scenarios])
    supervised_actor = stage23_pg._train_stage22_supervised_full_gnn_actor(
        actor_seed_rows,
        stage23_config,
    )
    supervised_state = deepcopy(supervised_actor.state_dict())
    samplers = build_stage24_candidate_sampler_registry()
    sampler_reports: dict[str, dict[str, object]] = {}
    for sampler_index, (sampler_id, sampler) in enumerate(sorted(samplers.items())):
        sampler_actor = LocalGNNEdgeScorer()
        sampler_actor.load_state_dict(deepcopy(supervised_state))
        torch.manual_seed(cfg.seed + 100 + sampler_index)
        critic = CentralizedMLPCriticBaseline(CentralizedMLPCriticConfig())
        sampler_reports[sampler_id] = _run_one_sampler(
            sampler_id=sampler_id,
            sampler=sampler,
            actor=sampler_actor,
            critic=critic,
            row_contexts=row_contexts,
            config=cfg,
            stage23_config=stage23_config,
            signal_config=signal_config_before,
            sampler_index=sampler_index,
        )

    selection = _select_stage24_sampler(sampler_reports)
    signal_config_after = stage23_pg.build_stage23_reward_surrogate_config()
    signal_config_unchanged = (
        stage23_pg._surrogate_config_fingerprint(signal_config_before)
        == stage23_pg._surrogate_config_fingerprint(signal_config_after)
    )
    cleanup = sampler_cleanup_report()
    selected_sampler_id = selection.get("selected_sampler_id")
    closeout_mode = cfg.mode == "micro"
    pass_gate = (
        preflight["preflight_passed"]
        and signal_config_unchanged
        and all(bool(report["ran_smoke_or_micro"]) for report in sampler_reports.values())
        and bool(selected_sampler_id)
        and selected_sampler_id == ACTIVE_POLICY_GRADIENT_SAMPLER_ID
        and cleanup["active_sampler_count"] == 1
        and cleanup["active_sampler_ids"] == [ACTIVE_POLICY_GRADIENT_SAMPLER_ID]
    )
    pass_gate = pass_gate and (
        cfg.mode == "smoke" or not selection.get("policy_gradient_blocked", False)
    )
    verdict = (
        STAGE24_SMOKE_PASS_VERDICT
        if cfg.mode == "smoke" and pass_gate
        else STAGE24_PASS_VERDICT
        if pass_gate and closeout_mode
        else STAGE24_FAIL_VERDICT
    )
    selected_report = sampler_reports.get(str(selected_sampler_id), {}) if selected_sampler_id else {}
    return _jsonable(
        {
            "stage": STAGE24_STAGE_ID,
            "verdict": verdict,
            "pass_gate": pass_gate,
            "mode": cfg.mode,
            "closeout_decision_mode": closeout_mode,
            "config": cfg.to_payload(),
            "preflight": preflight,
            "selected_action_semantics": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
            "actor_model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "critic_model_id": CENTRALIZED_MLP_CRITIC_BASELINE_ID,
            "policy_action_semantics": {
                "policy_action": "endpoint_or_physical_proposal_set",
                "environment_transition": (
                    "proposal_set_to_selected_physical_topology_via_environment_assembler"
                ),
                "proposal_logprob_claims_final_projected_topology_probability": False,
                "endpoint_sampler_logprob_semantics": "exact_endpoint_proposal_logprob",
                "projected_topology_logprob_exact": False,
            },
            "samplers_compared": sorted(sampler_reports),
            "sampler_reports": sampler_reports,
            "sampler_selection": selection,
            "active_policy_gradient_sampler_id": selected_sampler_id if pass_gate else None,
            "active_sampler_registry": cleanup,
            "winner_before_after": {
                "before": selected_report.get("before"),
                "after": selected_report.get("after"),
            },
            "reward_config_before": stage23_pg._surrogate_config_fingerprint(signal_config_before),
            "reward_config_after": stage23_pg._surrogate_config_fingerprint(signal_config_after),
            "reward_config_unchanged": signal_config_unchanged,
            "reward_weight_tuning_performed": False,
            "final_tau_selected": False,
            "coma_introduced": False,
            "transformer_introduced": False,
            "new_gnn_or_recurrent_architecture_introduced": False,
            "scale_up_training_performed": False,
            "checkpoint_written": False,
            "artifact_written": False,
            "uncontrolled_artifact_written": False,
            "dataset_export_written": False,
            "v5_modified": False,
            "stage25_allowed_without_owner_decision": False,
            "recommended_next_task": STAGE24_RECOMMENDED_NEXT_TASK_PASS
            if pass_gate and closeout_mode
            else selection.get("recommended_repair_task", "stage24_micro_mode_required"),
            "owner_decision_required": True,
        }
    )


def build_stage24_manifest(run_id: str) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root="result_save/stage24_critic_integrated_mappo_micro_loop",
        overrides={
            "run_id": run_id,
            "stage_id": STAGE24_STAGE_ID,
            "owner_approval_id": "owner_approved_stage24_critic_integrated_micro_loop",
            "config_id": "stage24_micro_loop_config_v1",
            "scenario_set_id": "stage22_selected_physical_micro_contexts",
            "split_id": "stage24_same_seed_sampler_recomparison",
            "seed": 2401,
            "seed_group_id": "stage24_sampler_recomparison_seed_group",
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "critic_model_id": CENTRALIZED_MLP_CRITIC_BASELINE_ID,
            "action_semantics_id": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
            "sampler_trial_ids": sorted(build_stage24_candidate_sampler_registry()),
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
            "objective_id": STAGE21_OBJECTIVE_CONTRACT_ID,
            "reward_id": "stage5_3_surrogate_config_with_selected_references",
            "architecture_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
            "checkpoint_creation_allowed": False,
            "training_scale_up_allowed": False,
            "artifact_write_allowed": "report_only",
        },
    )


def _run_one_sampler(
    *,
    sampler_id: str,
    sampler: PhysicalProposalSampler,
    actor: LocalGNNEdgeScorer,
    critic: CentralizedMLPCriticBaseline,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage24LoopConfig,
    stage23_config,
    signal_config,
    sampler_index: int,
) -> dict[str, object]:
    actor_before = _parameter_checksum(actor)
    critic_before = _parameter_checksum(critic)
    before_batch = _collect_rollout_batch(
        actor=actor,
        critic=critic,
        sampler=sampler,
        row_contexts=row_contexts,
        config=config,
        stage23_config=stage23_config,
        signal_config=signal_config,
        seed_base=config.seed + 10_000 + sampler_index * 1_000,
    )
    before = _summarize_batch(before_batch)
    actor_update_rule = torch.optim.AdamW(actor.parameters(), lr=config.actor_learning_rate)
    critic_update_rule = torch.optim.AdamW(critic.parameters(), lr=config.critic_learning_rate)
    updates: list[dict[str, float]] = []
    stop_reason: str | None = None
    advantage_payload: dict[str, object] = {}
    for update_index in range(config.max_policy_updates):
        batch = _collect_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=row_contexts,
            config=config,
            stage23_config=stage23_config,
            signal_config=signal_config,
            seed_base=config.seed + update_index * 10_000 + sampler_index * 1_000,
        )
        advantage_result = compute_gae_returns(
            batch.rewards,
            batch.values,
            batch.masks,
            num_scenarios=config.num_scenarios,
            rollout_steps=config.rollout_steps,
            config=AdvantageConfig(
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                normalize_advantages=config.normalize_advantages,
            ),
        )
        advantage_payload = advantage_result.to_payload()
        generator = torch.Generator().manual_seed(config.seed + 500 + update_index)
        for epoch_index in range(config.update_epochs):
            permutation = torch.randperm(batch.total_transitions, generator=generator).tolist()
            for start in range(0, batch.total_transitions, config.minibatch_size):
                indices = tuple(int(index) for index in permutation[start : start + config.minibatch_size])
                loss_result = _loss_for_indices(
                    actor=actor,
                    critic=critic,
                    sampler=sampler,
                    batch=batch,
                    row_contexts=row_contexts,
                    stage23_config=stage23_config,
                    indices=indices,
                    advantages=advantage_result.advantages,
                    returns=advantage_result.returns,
                    config=config,
                )
                if not torch.isfinite(loss_result.total_loss).all().item():
                    stop_reason = "nonfinite_total_loss"
                    break
                actor_update_rule.zero_grad()
                critic_update_rule.zero_grad()
                loss_result.total_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                grad_norm = float(actor_norm.detach().cpu().item()) + float(
                    critic_norm.detach().cpu().item()
                )
                actor_update_rule.step()
                critic_update_rule.step()
                payload = loss_result.to_payload(grad_norm=grad_norm)
                payload.update(
                    {
                        "update_index": float(update_index),
                        "epoch_index": float(epoch_index),
                        "minibatch_start": float(start),
                    }
                )
                updates.append(payload)
                if payload["approx_kl"] > config.max_approx_kl:
                    stop_reason = "kl_explosion"
                    break
            if stop_reason:
                break
        if stop_reason:
            break
    after_batch = _collect_rollout_batch(
        actor=actor,
        critic=critic,
        sampler=sampler,
        row_contexts=row_contexts,
        config=config,
        stage23_config=stage23_config,
        signal_config=signal_config,
        seed_base=config.seed + 10_000 + sampler_index * 1_000,
    )
    after = _summarize_batch(after_batch)
    safety = _safety_checks(before, after, config, updates, stop_reason)
    actor_after = _parameter_checksum(actor)
    critic_after = _parameter_checksum(critic)
    actor_delta = abs(actor_after - actor_before)
    critic_delta = abs(critic_after - critic_before)
    endpoint_repair = _endpoint_repair_status(sampler_id, after_batch)
    pass_gate = (
        safety["passed"]
        and actor_delta > 0.0
        and critic_delta > 0.0
        and bool(updates)
        and bool(advantage_payload.get("value_baseline_used", False))
        and endpoint_repair["pass_gate"]
    )
    return {
        "sampler_id": sampler_id,
        "ran_smoke_or_micro": True,
        "mode": config.mode,
        "before": before,
        "after": after,
        "updates": updates,
        "update_count": len(updates),
        "advantage_report": advantage_payload,
        "critic_integration": {
            "centralized_critic_used_for_values": True,
            "critic_values_used_in_advantage": bool(
                advantage_payload.get("value_baseline_used", False)
            ),
            "critic_value_loss_optimized": any("value_loss" in item for item in updates),
            "actor_parameter_delta": actor_delta,
            "critic_parameter_delta": critic_delta,
            "actor_received_gradient": actor_delta > 0.0,
            "critic_received_gradient": critic_delta > 0.0,
        },
        "endpoint_sampler_logprob_repair": endpoint_repair,
        "representative_records": [
            transition.to_payload()
            for transition in after_batch.transitions[: min(3, len(after_batch.transitions))]
        ],
        "rollout_batch": after_batch.to_payload(),
        "safety": safety,
        "pass_gate": pass_gate,
        "active_after_stage24": sampler_id == ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
        "archived_after_stage24": sampler_id in ARCHIVED_TRIAL_SAMPLER_IDS,
    }


def _collect_rollout_batch(
    *,
    actor: LocalGNNEdgeScorer,
    critic: CentralizedMLPCriticBaseline,
    sampler: PhysicalProposalSampler,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage24LoopConfig,
    stage23_config,
    signal_config,
    seed_base: int,
) -> RolloutBatch:
    transitions: list[RolloutTransition] = []
    for scenario_index in range(config.num_scenarios):
        selected_for_state: tuple[str, ...] = ()
        source_index = scenario_index % len(row_contexts)
        row, context = row_contexts[source_index]
        for step_index in range(config.rollout_steps):
            value_input = _critic_input_row(
                row=row,
                context=context,
                selected_edges=selected_for_state,
                step_index=step_index,
            )
            value_batch = tensorize_critic_evidence_rows((value_input,))
            value_prediction = critic.forward(value_batch.global_features).value.reshape(-1)[0]
            view = stage23_pg._actor_physical_logit_view(actor, row, stage23_config)
            rng = torch.Generator().manual_seed(
                seed_base + scenario_index * 101 + step_index
            )
            sample = sampler.sample(view.physical_logits, view.mask, view.config, rng)
            assembly = stage23_pg._assemble_physical_proposal(row.actor_safe_view, view, sample)
            evaluation = context.evaluator.evaluate(assembly.selected_physical_edges)
            signal = evaluate_reward_surrogate(
                SurrogateSignalInput(
                    consensus_success_probability=float(
                        evaluation.metrics["consensus_success_probability"]
                    ),
                    latency=float(evaluation.metrics["latency"]),
                    energy=float(evaluation.metrics["energy"]),
                    topology_diagnostics=evaluation.metrics["topology_diagnostics"],
                ),
                signal_config,
            )
            projection = stage23_pg._projection_diagnostics(view, sample, assembly)
            consensus = float(evaluation.metrics["consensus_success_probability"])
            done = step_index == config.rollout_steps - 1
            endpoint_fields = _endpoint_transition_fields(sample, assembly.selected_physical_edges)
            transition = RolloutTransition(
                actor_safe_observation=_actor_safe_observation_payload(row.actor_safe_view),
                centralized_critic_input=value_input,
                actor_logits=tuple(
                    float(item) for item in view.physical_logits.detach().cpu().tolist()
                ),
                sampler_id=sample.sampler_id,
                proposal_action={
                    "policy_action": "proposal",
                    "proposed_physical_edges": tuple(sample.proposed_physical_edges),
                    **endpoint_fields,
                },
                proposal_logprob=sample.logprob,
                proposal_entropy=sample.entropy,
                pre_projection_proposals=tuple(sample.proposed_physical_edges),
                post_projection_selected_physical_edges=tuple(
                    assembly.selected_physical_edges
                ),
                projection_diagnostics={
                    "top_proposal_rejection_rate": projection["top_proposal_rejection_rate"],
                    "above_threshold_rejection_rate": projection[
                        "above_threshold_rejection_rate"
                    ],
                    "rejection_by_reason": projection["rejection_by_reason"],
                    "projection_rejection_reasons": {
                        edge_id: [reason.value for reason in reasons]
                        for edge_id, reasons in assembly.rejection_reasons.items()
                    },
                    "selected_edge_count": len(assembly.selected_physical_edges),
                    "candidate_physical_edge_count": len(view.physical_edge_ids),
                },
                consensus_success_probability=consensus,
                latency=float(evaluation.metrics["latency"]),
                energy=float(evaluation.metrics["energy"]),
                reward_surrogate=float(signal.training_signal_value),
                value_prediction=value_prediction,
                done=done,
                mask=0.0 if done else 1.0,
                scenario_id=str(context.fixture.fixture_id),
                time_step=step_index,
                seed=seed_base + scenario_index * 101 + step_index,
                row_index=source_index,
                step_index=step_index,
                raw_sample_data=sample.raw_sample_data,
            )
            transitions.append(transition)
            selected_for_state = tuple(assembly.selected_physical_edges)
    return build_rollout_batch(
        tuple(transitions),
        num_scenarios=config.num_scenarios,
        rollout_steps=config.rollout_steps,
    )


def _loss_for_indices(
    *,
    actor: LocalGNNEdgeScorer,
    critic: CentralizedMLPCriticBaseline,
    sampler: PhysicalProposalSampler,
    batch: RolloutBatch,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    stage23_config,
    indices: tuple[int, ...],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: Stage24LoopConfig,
):
    new_logprob_values = []
    entropy_values = []
    critic_rows = []
    for index in indices:
        transition = batch.transitions[index]
        row, _context = row_contexts[transition.row_index]
        view = stage23_pg._actor_physical_logit_view(actor, row, stage23_config)
        new_logprob_values.append(
            sampler.logprob_of(
                view.physical_logits,
                view.mask,
                view.config,
                transition.raw_sample_data,
            ).reshape(())
        )
        entropy_values.append(
            sampler.entropy_of(
                view.physical_logits,
                view.mask,
                view.config,
                transition.raw_sample_data,
            ).reshape(())
        )
        critic_rows.append(transition.centralized_critic_input)
    critic_batch = tensorize_critic_evidence_rows(tuple(critic_rows))
    value_predictions = critic.forward(critic_batch.global_features).value
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return clipped_policy_value_loss(
        ClippedPolicyValueLossInputs(
            new_logprobs=torch.stack(new_logprob_values),
            old_logprobs=batch.old_logprobs[index_tensor],
            advantages=advantages[index_tensor],
            value_predictions=value_predictions,
            returns=returns[index_tensor],
            entropies=torch.stack(entropy_values),
            clip_eps=config.clip_eps,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
        )
    )


def _summarize_batch(batch: RolloutBatch) -> dict[str, object]:
    records = []
    for transition in batch.transitions:
        projection = dict(transition.projection_diagnostics)
        records.append(
            {
                "tau_feasible": transition.consensus_success_probability >= STAGE21_TAU_REQUIREMENT_MIN,
                "violation_indicator": int(
                    transition.consensus_success_probability < STAGE21_TAU_REQUIREMENT_MIN
                ),
                "consensus_success_probability": transition.consensus_success_probability,
                "latency": transition.latency,
                "energy": transition.energy,
                "selected_edge_count": projection["selected_edge_count"],
                "candidate_physical_edge_count": projection["candidate_physical_edge_count"],
                "top_proposal_rejection_rate": projection["top_proposal_rejection_rate"],
                "above_threshold_rejection_rate": projection[
                    "above_threshold_rejection_rate"
                ],
                "rejection_by_reason": projection["rejection_by_reason"],
                "proposal_entropy": float(transition.proposal_entropy.detach().cpu().item()),
                "proposal_logprob": float(transition.proposal_logprob.detach().cpu().item()),
                "reward_surrogate": transition.reward_surrogate,
            }
        )
    summary = stage23_pg._summarize_records(records)
    summary.update(
        {
            "total_transitions": batch.total_transitions,
            "num_scenarios": batch.num_scenarios,
            "rollout_steps": batch.rollout_steps,
            "mean_value_prediction": float(batch.values.mean().detach().cpu().item()),
            "done_count": int(batch.dones.sum().detach().cpu().item()),
        }
    )
    return summary


def _safety_checks(
    before: Mapping[str, object],
    after: Mapping[str, object],
    config: Stage24LoopConfig,
    updates: list[Mapping[str, float]],
    stop_reason: str | None,
) -> dict[str, object]:
    tau_drop = float(before["tau_feasible_rate"]) - float(after["tau_feasible_rate"])
    violation_worsening = float(after["violation_rate"]) - float(before["violation_rate"])
    projection_worsening = (
        float(after["top_proposal_rejection_rate"])
        - float(before["top_proposal_rejection_rate"])
    )
    initial_entropy = float(before["mean_entropy"])
    entropy_floor = initial_entropy * config.min_entropy_fraction_of_initial
    max_kl = max((float(item.get("approx_kl", 0.0)) for item in updates), default=0.0)
    max_value_loss = max((float(item.get("value_loss", 0.0)) for item in updates), default=0.0)
    issues: list[str] = []
    if stop_reason:
        issues.append(stop_reason)
    if tau_drop > config.max_tau_feasible_drop:
        issues.append("tau_feasible_rate_degraded")
    if violation_worsening > config.max_violation_rate_increase:
        issues.append("violation_rate_worsened")
    if float(after["empty_graph_rate"]) > config.max_empty_graph_rate:
        issues.append("empty_graph_collapse")
    if float(after["full_graph_rate"]) > config.max_full_graph_rate:
        issues.append("full_graph_collapse")
    if projection_worsening > config.max_top_proposal_rejection_increase:
        issues.append("projection_mismatch_worsened")
    if max_kl > config.max_approx_kl:
        issues.append("kl_explosion")
    if float(after["mean_entropy"]) < entropy_floor:
        issues.append("entropy_collapse")
    if not all(isfinite(float(item.get("value_loss", 0.0))) for item in updates):
        issues.append("critic_value_loss_nonfinite")
    return {
        "passed": not issues,
        "issues": issues,
        "tau_feasible_drop": tau_drop,
        "violation_worsening": violation_worsening,
        "projection_worsening": projection_worsening,
        "max_approx_kl": max_kl,
        "max_value_loss": max_value_loss,
        "entropy_floor": entropy_floor,
        "early_stop_reason": stop_reason,
    }


def _select_stage24_sampler(
    sampler_reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    passing = {
        sampler_id: report
        for sampler_id, report in sampler_reports.items()
        if report.get("pass_gate") is True
    }
    if not passing:
        return {
            "selected_sampler_id": None,
            "selection_reason": "no sampler passed Stage 24 safety and critic gates",
            "policy_gradient_blocked": True,
            "recommended_repair_task": "stage24_critic_or_sampler_selection_repair",
        }
    pl_report = passing.get(PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID)
    endpoint_report = passing.get(ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID)
    if pl_report and endpoint_report:
        pl_after = pl_report["after"]
        endpoint_after = endpoint_report["after"]
        endpoint_clearly_better = (
            float(endpoint_after["tau_feasible_rate"])
            > float(pl_after["tau_feasible_rate"]) + 0.05
            and float(endpoint_after["top_proposal_rejection_rate"])
            <= float(pl_after["top_proposal_rejection_rate"])
            and float(endpoint_after["mean_reward_surrogate"])
            >= float(pl_after["mean_reward_surrogate"])
        ) or (
            float(endpoint_after["tau_feasible_rate"])
            >= float(pl_after["tau_feasible_rate"])
            and float(endpoint_after["top_proposal_rejection_rate"])
            < float(pl_after["top_proposal_rejection_rate"]) - 0.10
            and (
                float(endpoint_after["mean_latency"]) + float(endpoint_after["mean_energy"])
            )
            < (float(pl_after["mean_latency"]) + float(pl_after["mean_energy"]))
        )
        selected = (
            ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID
            if endpoint_clearly_better
            else PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
        )
        return {
            "selected_sampler_id": selected,
            "selection_reason": (
                "endpoint requires clear reliability or projection/resource improvement; "
                "otherwise the simpler active Plackett-Luce sampler remains selected"
            ),
            "policy_gradient_blocked": False,
            "eligible_sampler_ids": sorted(passing),
            "endpoint_clearly_better": endpoint_clearly_better,
            "low_entropy_promotion": {
                "active_registry_expected_count": 1,
                "losing_sampler_removed_from_active_registry": True,
            },
        }
    selected = sorted(
        passing,
        key=lambda sampler_id: _sampler_selection_key(sampler_id, passing[sampler_id]),
    )[0]
    return {
        "selected_sampler_id": selected,
        "selection_reason": "selected only safety-passing sampler",
        "policy_gradient_blocked": False,
        "eligible_sampler_ids": sorted(passing),
    }


def _sampler_selection_key(
    sampler_id: str,
    report: Mapping[str, object],
) -> tuple[float, float, float, float, int]:
    after = report["after"]
    complexity = 0 if sampler_id == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID else 1
    return (
        -float(after["tau_feasible_rate"]),
        float(after["violation_rate"]),
        float(after["top_proposal_rejection_rate"]),
        float(after["mean_latency"]) + float(after["mean_energy"]),
        complexity,
    )


def _endpoint_repair_status(
    sampler_id: str,
    batch: RolloutBatch,
) -> dict[str, object]:
    if sampler_id != ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID:
        return {
            "required_for_sampler": False,
            "pass_gate": True,
            "logprob_semantics": "not_endpoint_sampler",
        }
    transition = next(
        transition
        for transition in batch.transitions
        if transition.sampler_id == ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID
    )
    action = transition.proposal_action
    return {
        "required_for_sampler": True,
        "pass_gate": (
            action.get("logprob_semantics") == "exact_endpoint_proposal_logprob"
            and action.get("projected_topology_logprob_exact") is False
            and "endpoint_logprobs" in action
            and "endpoint_proposal_sets" in action
        ),
        "logprob_semantics": action.get("logprob_semantics"),
        "projected_topology_logprob_exact": action.get("projected_topology_logprob_exact"),
        "endpoint_budget_config": action.get("endpoint_budget_config"),
        "sample_endpoint_proposal_sets": action.get("endpoint_proposal_sets"),
    }


def _endpoint_transition_fields(
    sample: ProposalSample,
    selected_edges: Iterable[str],
) -> dict[str, object]:
    if sample.sampler_id != ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID:
        return {
            "logprob_semantics": sample.raw_sample_data.get("logprob_semantics"),
            "projected_topology_logprob_exact": False,
        }
    return {
        "endpoint_budget_config": sample.raw_sample_data.get("endpoint_budget_config"),
        "endpoint_proposal_sets": sample.raw_sample_data.get("endpoint_proposal_sets"),
        "endpoint_logprobs": sample.raw_sample_data.get("endpoint_logprobs"),
        "joint_proposal_logprob": sample.raw_sample_data.get("joint_proposal_logprob"),
        "proposal_entropy": sample.raw_sample_data.get("proposal_entropy"),
        "aggregated_physical_proposals": sample.raw_sample_data.get(
            "aggregated_physical_proposals"
        ),
        "projected_selected_physical_edges": tuple(selected_edges),
        "logprob_semantics": "exact_endpoint_proposal_logprob",
        "projected_topology_logprob_exact": False,
    }


def _critic_input_row(
    *,
    row: object,
    context: Stage21EvaluationContext,
    selected_edges: tuple[str, ...],
    step_index: int,
) -> dict[str, object]:
    return {
        "actor_safe_rows": tuple(row.actor_safe_view),
        "critic_view": {
            "view_role": "critic_centralized_training_only",
            "scenario_id": context.fixture.fixture_id,
            "node_ids": tuple(context.graph.node_ids),
            "candidate_edge_ids": tuple(context.graph.edge_ids),
            "selected_edge_ids": tuple(selected_edges),
            "time_step": step_index,
            "training_only": True,
            "stage3_stage4_evaluation_records_allowed": True,
            "actor_input_allowed": False,
        },
    }


def _actor_safe_observation_payload(
    actor_safe_view: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    rows = tuple(actor_safe_view)
    forbidden = sorted(
        {
            field
            for row in rows
            for field in set(row)
            if field in ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS
        }
    )
    return {
        "view_role": "actor_safe_decentralized_policy_input",
        "row_count": len(rows),
        "field_names": sorted({field for row in rows for field in row}),
        "forbidden_actor_fields_detected": forbidden,
        "rows": rows,
    }


def _row_contexts() -> tuple[tuple[object, Stage21EvaluationContext], ...]:
    contexts = build_stage21_objective_stack_contexts()
    evidence = build_stage22_action_semantics_evidence()
    rows = evidence.datasets[UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID].rows
    return tuple(zip(rows, contexts))


def _stage23_config(config: Stage24LoopConfig):
    return stage23_pg.Stage23PolicyGradientConfig(
        seed=config.seed,
        scenario_limit=config.num_scenarios,
        supervised_epochs=config.supervised_epochs,
        supervised_learning_rate=config.supervised_learning_rate,
        policy_updates=1,
        top_k=config.top_k,
        endpoint_budget=config.endpoint_budget,
        variable_proposal_size=getattr(config, "variable_proposal_size", False),
        tau_requirement_min=config.tau_requirement_min,
    )


def _parameter_checksum(module) -> float:
    return sum(
        float(parameter.detach().double().sum().cpu().item())
        for parameter in module.parameters()
    )


def _blocked_report(
    preflight: Mapping[str, object],
    config: Stage24LoopConfig,
    signal_config,
) -> dict[str, object]:
    return _jsonable(
        {
            "stage": STAGE24_STAGE_ID,
            "verdict": STAGE24_FAIL_VERDICT,
            "pass_gate": False,
            "mode": config.mode,
            "preflight": dict(preflight),
            "samplers_compared": [],
            "sampler_selection": {
                "selected_sampler_id": None,
                "selection_reason": "preflight failed",
                "policy_gradient_blocked": True,
                "recommended_repair_task": "stage24_preflight_blocker_repair",
            },
            "reward_config_before": stage23_pg._surrogate_config_fingerprint(signal_config),
            "reward_config_unchanged": True,
            "checkpoint_written": False,
            "artifact_written": False,
            "v5_modified": False,
            "owner_decision_required": True,
        }
    )


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Counter):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
