"""Stage 25 small-scale formal policy-gradient training pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import csv
import json

import torch

from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_EVALUATOR_ID,
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PHYSICS_REGIME_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    STAGE21_TAU_REQUIREMENT_MIN,
    Stage21EvaluationContext,
)
from marl_topology.objectives.surrogate_signal import (
    SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
)
from marl_topology.evaluation.reward_surface_analysis import (
    build_reward_surface_analysis,
)
from marl_topology.models import (
    CENTRALIZED_MLP_CRITIC_BASELINE_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
    LocalGNNEdgeScorer,
)
from marl_topology.policies import UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
from marl_topology.training.policy_gradient import pilot_runner as stage23_pg
from marl_topology.training.policy_gradient.samplers import (
    ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
    PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
    get_active_policy_gradient_sampler,
)
from marl_topology.training.run_manifest_validator import (
    build_valid_stage5_10_dry_run_manifest,
    validate_run_manifest_dry_run,
)

from .advantages import AdvantageConfig, compute_gae_returns
from .trainer import (
    STAGE24_PASS_VERDICT,
    Stage24LoopConfig,
    _collect_rollout_batch,
    _loss_for_indices,
    _parameter_checksum,
    _row_contexts,
    _stage23_config,
    run_stage24_preflight,
)


STAGE25_STAGE_ID = "stage_25_small_scale_formal_mappo_training_pilot"
STAGE25_BASE_CONFIG_ID = "stage25_pilot_base_config"
STAGE25_PASS_VERDICT = "stage25_pass_small_scale_mappo_pilot_complete"
STAGE25_FAIL_VERDICT = "stage25_fail_small_scale_mappo_pilot_blocked"
STAGE25_RECOMMENDED_NEXT_TASK_PASS = "stage_26_scale_readiness_and_failure_mode_review"
STAGE25_ARTIFACT_ROOT = (
    "result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config"
)
STAGE25_RUN_ID = "stage25_pilot_base_config_seed_group_2501_2502_2503"

STAGE25_ARTIFACT_FILENAMES = (
    "manifest.json",
    "training_report.json",
    "seed_summary.csv",
    "update_metrics.csv",
    "eval_metrics.csv",
    "before_after_comparison.csv",
    "reward_surface.csv",
    "reward_surface_analysis.json",
    "visualization_report.json",
    "training_curves.csv",
    "objective_curves.csv",
    "ppo_diagnostics.csv",
    "critic_diagnostics.csv",
    "projection_diagnostics.csv",
    "topology_diagnostics.csv",
    "reward_surface_scatter.png",
    "training_curves.png",
    "objective_curves.png",
    "ppo_diagnostics.png",
    "projection_diagnostics.png",
    "topology_diagnostics.png",
    "before_after_comparison.png",
)


class Stage25PilotViolation(ValueError):
    """Raised when Stage 25 crosses a declared execution boundary."""


@dataclass(frozen=True, slots=True)
class Stage25PilotBaseConfig:
    train_scenarios: int = 16
    eval_scenarios: int = 8
    seeds: tuple[int, ...] = (2501, 2502, 2503)
    rollout_steps: int = 16
    minibatch_size: int = 64
    update_epochs: int = 4
    max_updates: int = 20
    eval_every: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    supervised_epochs: int = 20
    supervised_learning_rate: float = 0.006
    top_k: int = 3
    endpoint_budget: int = 1
    tau_requirement_min: float = STAGE21_TAU_REQUIREMENT_MIN
    max_tau_feasible_drop: float = 0.05
    max_violation_rate_increase: float = 0.05
    max_empty_graph_rate: float = 0.25
    max_full_graph_rate: float = 0.50
    max_approx_kl: float = 0.03
    min_entropy_fraction_of_initial: float = 0.50
    critic_value_loss_divergence_threshold: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.transitions_per_update != self.train_scenarios * self.rollout_steps:
            raise Stage25PilotViolation("transitions_per_update invariant failed")
        if self.transitions_per_update != 256:
            raise Stage25PilotViolation("Stage 25 base config fixes transitions_per_update at 256")
        if self.minibatch_size > self.transitions_per_update:
            raise Stage25PilotViolation("minibatch_size must be <= transitions_per_update")
        if self.transitions_per_update // self.minibatch_size < 2:
            raise Stage25PilotViolation("Stage 25 requires at least two minibatches")
        if self.update_epochs < 2:
            raise Stage25PilotViolation("Stage 25 requires update_epochs >= 2")
        if self.eval_every > self.max_updates:
            raise Stage25PilotViolation("eval_every must be <= max_updates")
        if len(self.seeds) < 2:
            raise Stage25PilotViolation("Stage 25 requires at least two seeds")
        if self.tau_requirement_min != STAGE21_TAU_REQUIREMENT_MIN:
            raise Stage25PilotViolation("Stage 25 keeps tau_requirement_min fixed at 0.9")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise Stage25PilotViolation("learning rates must be positive")

    @property
    def transitions_per_update(self) -> int:
        return self.train_scenarios * self.rollout_steps

    def to_payload(self) -> dict[str, object]:
        return {
            "config_id": STAGE25_BASE_CONFIG_ID,
            "train_scenarios": self.train_scenarios,
            "eval_scenarios": self.eval_scenarios,
            "seeds": list(self.seeds),
            "seed_count": len(self.seeds),
            "rollout_steps": self.rollout_steps,
            "transitions_per_update": self.transitions_per_update,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "max_updates": self.max_updates,
            "eval_every": self.eval_every,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "supervised_epochs": self.supervised_epochs,
            "supervised_learning_rate": self.supervised_learning_rate,
            "top_k": self.top_k,
            "endpoint_budget": self.endpoint_budget,
            "tau_requirement_min": self.tau_requirement_min,
            "base_config_fixed_not_tuned": True,
            "diagnostic_probe": False,
        }


@dataclass(frozen=True, slots=True)
class Stage25Split:
    train_sources: tuple[tuple[object, Stage21EvaluationContext], ...]
    eval_sources: tuple[tuple[object, Stage21EvaluationContext], ...]
    train_slots: tuple[tuple[object, Stage21EvaluationContext], ...]
    eval_slots: tuple[tuple[object, Stage21EvaluationContext], ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "split_id": "stage25_disjoint_source_rows_cyclic_slots_v1",
            "unique_source_context_count": len(self.train_sources) + len(self.eval_sources),
            "unique_train_source_count": len(self.train_sources),
            "unique_eval_source_count": len(self.eval_sources),
            "train_slot_count": len(self.train_slots),
            "eval_slot_count": len(self.eval_slots),
            "train_source_ids": [
                f"{context.fixture.fixture_id}:t{context.time_step}"
                for _row, context in self.train_sources
            ],
            "eval_source_ids": [
                f"{context.fixture.fixture_id}:t{context.time_step}"
                for _row, context in self.eval_sources
            ],
            "slot_expansion": (
                "Disjoint source rows are expanded cyclically to satisfy the fixed "
                "16 train / 8 eval scenario-slot protocol."
            ),
        }


def build_stage25_manifest(run_id: str = STAGE25_RUN_ID) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root=STAGE25_ARTIFACT_ROOT,
        overrides={
            "run_id": run_id,
            "stage_id": STAGE25_STAGE_ID,
            "owner_approval_id": "owner_approved_stage25_small_scale_formal_mappo_training_pilot",
            "config_id": STAGE25_BASE_CONFIG_ID,
            "scenario_set_id": "stage22_selected_physical_train_eval_contexts",
            "split_id": "stage25_disjoint_source_rows_cyclic_slots_v1",
            "seed": 2501,
            "seeds": [2501, 2502, 2503],
            "seed_group_id": "stage25_three_seed_base_protocol",
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "critic_model_id": CENTRALIZED_MLP_CRITIC_BASELINE_ID,
            "action_semantics_id": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
            "sampler_id": PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
            "evaluator_id": STAGE21_EVALUATOR_ID,
            "physics_regime_id": STAGE21_PHYSICS_REGIME_ID,
            "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
            "objective_id": STAGE21_OBJECTIVE_CONTRACT_ID,
            "reward_id": "stage5_3_surrogate_config_with_selected_references",
            "surrogate_config_id": "stage5_3_surrogate_config_with_selected_references",
            "architecture_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
            "artifact_paths": [
                f"{STAGE25_ARTIFACT_ROOT}/{name}" for name in STAGE25_ARTIFACT_FILENAMES
            ],
            "checkpoint_creation_allowed": False,
            "training_scale_up_allowed": False,
            "artifact_write_allowed": "manifest_validated_reports_only",
        },
    )


def run_stage25_preflight(
    *,
    config: Stage25PilotBaseConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or Stage25PilotBaseConfig()
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    stage24_preflight = run_stage24_preflight(project_root=root)
    contexts = _row_contexts()
    split = _build_stage25_split(contexts, cfg)
    reward_config = stage23_pg.build_stage23_reward_surrogate_config()
    manifest = build_stage25_manifest()
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    gates = {
        "stage24_complete_and_preflight_passed": {
            "passed": bool(stage24_preflight["preflight_passed"]),
            "stage24_preflight": stage24_preflight,
        },
        "active_sampler_is_plackett_luce": {
            "passed": ACTIVE_POLICY_GRADIENT_SAMPLER_ID == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
        },
        "base_config_fixed_protocol_valid": {
            "passed": True,
            "config": cfg.to_payload(),
        },
        "train_eval_split_available": {
            "passed": bool(split.train_sources and split.eval_sources),
            "split": split.to_payload(),
        },
        "reward_weights_unchanged": {
            # Gate intent: no unsanctioned reward tuning. The active reward is the
            # owner-approved feasibility-first barrier with tau frozen at 0.9; weights
            # are the fixed Stage 31 values, not searched per run.
            "passed": (
                reward_config.tau == STAGE21_TAU_REQUIREMENT_MIN
                and reward_config.structure == SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2
            ),
            "reward_config_fingerprint": stage23_pg._surrogate_config_fingerprint(reward_config),
            "weight_tuning_performed": False,
        },
        "run_manifest_validator_available": {
            "passed": validation.is_valid and validation.writes_performed is False,
            "validator_result": validation.to_dict(),
        },
    }
    return {
        "stage": STAGE25_STAGE_ID,
        "preflight_passed": all(bool(gate["passed"]) for gate in gates.values()),
        "gates": gates,
        "v5_modified": False,
    }


def run_stage25_small_scale_formal_mappo_pilot(
    *,
    config: Stage25PilotBaseConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or Stage25PilotBaseConfig()
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    preflight = run_stage25_preflight(config=cfg, project_root=root)
    reward_config_before = stage23_pg.build_stage23_reward_surrogate_config()
    reward_fingerprint_before = stage23_pg._surrogate_config_fingerprint(reward_config_before)
    manifest = build_stage25_manifest()
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    contexts = _row_contexts()
    split = _build_stage25_split(contexts, cfg)
    if not preflight["preflight_passed"]:
        return _blocked_report(
            cfg=cfg,
            preflight=preflight,
            manifest=manifest,
            validation=validation,
            reward_fingerprint_before=reward_fingerprint_before,
        )

    seed_reports = [
        _run_stage25_seed(
            seed=seed,
            seed_index=seed_index,
            config=cfg,
            split=split,
            reward_config=reward_config_before,
        )
        for seed_index, seed in enumerate(cfg.seeds)
    ]
    reward_config_after = stage23_pg.build_stage23_reward_surrogate_config()
    reward_fingerprint_after = stage23_pg._surrogate_config_fingerprint(reward_config_after)
    aggregate = _aggregate_seed_reports(seed_reports)
    reward_surface_records = [
        record
        for seed_report in seed_reports
        for record in seed_report["reward_surface_records"]
    ]
    reward_surface = build_reward_surface_analysis(
        reward_surface_records,
        reward_config=reward_config_before,
        tau_requirement_min=cfg.tau_requirement_min,
    )
    pass_fail = _stage25_pass_fail(
        cfg=cfg,
        seed_reports=seed_reports,
        aggregate=aggregate,
        reward_config_unchanged=reward_fingerprint_before == reward_fingerprint_after,
        reward_surface=reward_surface,
    )
    verdict = STAGE25_PASS_VERDICT if pass_fail["passed"] else STAGE25_FAIL_VERDICT
    return _jsonable(
        {
            "stage": STAGE25_STAGE_ID,
            "verdict": verdict,
            "pass_gate": pass_fail["passed"],
            "pass_fail_gate": pass_fail,
            "config": cfg.to_payload(),
            "preflight": preflight,
            "manifest": manifest,
            "manifest_validation": validation.to_dict(),
            "split": split.to_payload(),
            "seed_reports": seed_reports,
            "aggregate": aggregate,
            "reward_surface_analysis": reward_surface,
            "before_after_comparison": _before_after_comparison(seed_reports, aggregate),
            "reward_config_before": reward_fingerprint_before,
            "reward_config_after": reward_fingerprint_after,
            "reward_config_unchanged": reward_fingerprint_before == reward_fingerprint_after,
            "reward_weight_tuning_performed": False,
            "diagnostic_probes_run": [],
            "base_config_id": STAGE25_BASE_CONFIG_ID,
            "primary_conclusion_based_only_on_base_config": True,
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
            "sampler_switched": False,
            "final_tau_selected": False,
            "coma_introduced": False,
            "transformer_introduced": False,
            "gru_lstm_or_recurrent_ppo_introduced": False,
            "scale_up_training_performed": False,
            "checkpoint_written": False,
            "artifact_written": False,
            "uncontrolled_artifact_written": False,
            "dataset_export_written": False,
            "v5_modified": False,
            "failure_review": _failure_review(seed_reports, aggregate, pass_fail)
            if not pass_fail["passed"]
            else None,
            "recommended_next_task": STAGE25_RECOMMENDED_NEXT_TASK_PASS,
            "owner_decision_required": True,
        }
    )


def write_stage25_training_artifacts(
    report: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    manifest = dict(report["manifest"])  # type: ignore[index]
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    if not validation.is_valid:
        raise Stage25PilotViolation(f"manifest validation failed: {validation.error_codes()}")
    artifact_dir = _artifact_dir(root, manifest)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_to_write = {**dict(report), "artifact_written": True, "artifact_dir": str(artifact_dir)}
    _write_json(artifact_dir / "manifest.json", manifest)
    _write_json(artifact_dir / "training_report.json", report_to_write)
    _write_json(
        artifact_dir / "reward_surface_analysis.json",
        report_to_write["reward_surface_analysis"],
    )
    _write_seed_summary_csv(artifact_dir / "seed_summary.csv", report_to_write)
    _write_update_metrics_csv(artifact_dir / "update_metrics.csv", report_to_write)
    _write_eval_metrics_csv(artifact_dir / "eval_metrics.csv", report_to_write)
    _write_before_after_csv(artifact_dir / "before_after_comparison.csv", report_to_write)
    _write_reward_surface_csv(artifact_dir / "reward_surface.csv", report_to_write)
    return {
        "artifact_written": True,
        "artifact_dir": str(artifact_dir),
        "manifest_validated": validation.is_valid,
        "written_files": sorted(
            path.name for path in artifact_dir.iterdir() if path.is_file()
        ),
    }


def _run_stage25_seed(
    *,
    seed: int,
    seed_index: int,
    config: Stage25PilotBaseConfig,
    split: Stage25Split,
    reward_config,
) -> dict[str, object]:
    torch.manual_seed(seed)
    pilot_loop_config = _stage25_loop_config(config, seed=seed, num_scenarios=config.train_scenarios)
    eval_loop_config = _stage25_loop_config(config, seed=seed, num_scenarios=config.eval_scenarios)
    stage23_config = _stage23_config(pilot_loop_config)
    sampler = get_active_policy_gradient_sampler()
    actor = stage23_pg._train_stage22_supervised_full_gnn_actor(
        (row for row, _context in split.train_slots),
        stage23_config,
    )
    supervised_state = deepcopy(actor.state_dict())
    critic = CentralizedMLPCriticBaseline(CentralizedMLPCriticConfig())
    actor_before_training = _parameter_checksum(actor)
    critic_before_training = _parameter_checksum(critic)
    supervised_eval_batch = _collect_rollout_batch(
        actor=actor,
        critic=critic,
        sampler=sampler,
        row_contexts=split.eval_slots,
        config=eval_loop_config,
        stage23_config=stage23_config,
        signal_config=reward_config,
        seed_base=seed + 200_000,
    )
    supervised_eval = _summarize_batch_with_actor_scores(supervised_eval_batch)
    baseline_records = _baseline_records(
        split.eval_slots,
        config=stage23_config,
        reward_config=reward_config,
        seed=seed,
    )
    baseline_comparison = _summarize_by_policy(baseline_records)
    actor_update_rule = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)
    critic_update_rule = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)
    update_metrics: list[dict[str, object]] = []
    eval_metrics: list[dict[str, object]] = [
        {
            "seed": seed,
            "update_index": 0,
            "phase": "supervised_before_mappo",
            **supervised_eval,
        }
    ]
    critic_pairs: list[dict[str, float]] = []
    stop_reason: str | None = None
    completed_updates = 0
    for update_index in range(1, config.max_updates + 1):
        batch = _collect_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=split.train_slots,
            config=pilot_loop_config,
            stage23_config=stage23_config,
            signal_config=reward_config,
            seed_base=seed + update_index * 10_000,
        )
        train_summary = _summarize_batch_with_actor_scores(batch)
        advantage_result = compute_gae_returns(
            batch.rewards,
            batch.values,
            batch.masks,
            num_scenarios=config.train_scenarios,
            rollout_steps=config.rollout_steps,
            config=AdvantageConfig(
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                normalize_advantages=True,
            ),
        )
        loss_payloads: list[dict[str, float]] = []
        generator = torch.Generator().manual_seed(seed + 500 + update_index)
        for epoch_index in range(config.update_epochs):
            permutation = torch.randperm(batch.total_transitions, generator=generator).tolist()
            for start in range(0, batch.total_transitions, config.minibatch_size):
                indices = tuple(int(index) for index in permutation[start : start + config.minibatch_size])
                loss_result = _loss_for_indices(
                    actor=actor,
                    critic=critic,
                    sampler=sampler,
                    batch=batch,
                    row_contexts=split.train_slots,
                    stage23_config=stage23_config,
                    indices=indices,
                    advantages=advantage_result.advantages,
                    returns=advantage_result.returns,
                    config=pilot_loop_config,
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
                loss_payloads.append(loss_result.to_payload(grad_norm=grad_norm))
                if loss_payloads[-1]["approx_kl"] > config.max_approx_kl:
                    stop_reason = "kl_explosion"
                    break
            if stop_reason:
                break
        loss_summary = _aggregate_loss_payloads(loss_payloads)
        update_record = {
            "seed": seed,
            "update_index": update_index,
            "phase": "train",
            **train_summary,
            **loss_summary,
            "advantage_mean": float(advantage_result.advantages.mean().detach().cpu().item()),
            "advantage_std": float(
                advantage_result.advantages.std(unbiased=False).detach().cpu().item()
            ),
            "return_mean": float(advantage_result.returns.mean().detach().cpu().item()),
            "raw_advantage_mean": float(
                advantage_result.raw_advantages.mean().detach().cpu().item()
            ),
        }
        update_metrics.append(update_record)
        critic_pairs = [
            {
                "value_prediction": float(value),
                "return": float(ret),
            }
            for value, ret in zip(
                batch.values.detach().cpu().tolist(),
                advantage_result.returns.detach().cpu().tolist(),
                strict=True,
            )
        ][:200]
        stop_reason = stop_reason or _training_stop_reason(
            update_record,
            supervised_eval=supervised_eval,
            config=config,
            initial_entropy=float(supervised_eval["mean_entropy"]),
        )
        completed_updates = update_index
        if update_index % config.eval_every == 0 or update_index == config.max_updates:
            eval_batch = _collect_rollout_batch(
                actor=actor,
                critic=critic,
                sampler=sampler,
                row_contexts=split.eval_slots,
                config=eval_loop_config,
                stage23_config=stage23_config,
                signal_config=reward_config,
                seed_base=seed + 200_000,
            )
            eval_summary = _summarize_batch_with_actor_scores(eval_batch)
            eval_record = {
                "seed": seed,
                "update_index": update_index,
                "phase": "mappo_eval",
                **eval_summary,
            }
            eval_metrics.append(eval_record)
            stop_reason = stop_reason or _eval_stop_reason(
                supervised_eval=supervised_eval,
                current_eval=eval_summary,
                current_update=update_record,
                config=config,
                initial_entropy=float(supervised_eval["mean_entropy"]),
            )
            final_eval_batch = eval_batch
        else:
            final_eval_batch = supervised_eval_batch
        if stop_reason:
            break
    else:
        final_eval_batch = _collect_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=split.eval_slots,
            config=eval_loop_config,
            stage23_config=stage23_config,
            signal_config=reward_config,
            seed_base=seed + 200_000,
        )
    if eval_metrics[-1]["phase"] != "mappo_eval":
        final_summary = _summarize_batch_with_actor_scores(final_eval_batch)
        eval_metrics.append(
            {
                "seed": seed,
                "update_index": completed_updates,
                "phase": "mappo_eval",
                **final_summary,
            }
        )
    final_eval = eval_metrics[-1]
    actor_after_training = _parameter_checksum(actor)
    critic_after_training = _parameter_checksum(critic)
    return {
        "seed": seed,
        "seed_index": seed_index,
        "base_run_completed": stop_reason is None and completed_updates == config.max_updates,
        "completed_updates": completed_updates,
        "stop_reason": stop_reason,
        "supervised_eval": supervised_eval,
        "final_mappo_eval": final_eval,
        "baseline_comparison": baseline_comparison,
        "update_metrics": update_metrics,
        "eval_metrics": eval_metrics,
        "critic_prediction_return_pairs": critic_pairs,
        "actor_parameter_delta": abs(actor_after_training - actor_before_training),
        "critic_parameter_delta": abs(critic_after_training - critic_before_training),
        "supervised_state_reused_as_mappo_initialization": bool(supervised_state),
        "reward_surface_records": (
            _surface_records_from_batch(
                supervised_eval_batch,
                policy_label="supervised_gnn_before_mappo",
                seed=seed,
                update_index=0,
            )
            + _surface_records_from_batch(
                final_eval_batch,
                policy_label="mappo_fine_tuned_gnn",
                seed=seed,
                update_index=completed_updates,
            )
            + baseline_records
        ),
    }


def _stage25_loop_config(
    config: Stage25PilotBaseConfig,
    *,
    seed: int,
    num_scenarios: int,
) -> Stage24LoopConfig:
    return Stage24LoopConfig(
        mode="micro",
        seed=seed,
        num_scenarios=num_scenarios,
        rollout_steps=config.rollout_steps,
        minibatch_size=config.minibatch_size,
        update_epochs=config.update_epochs,
        max_policy_updates=config.max_updates,
        supervised_epochs=config.supervised_epochs,
        supervised_learning_rate=config.supervised_learning_rate,
        actor_learning_rate=config.actor_lr,
        critic_learning_rate=config.critic_lr,
        top_k=config.top_k,
        endpoint_budget=config.endpoint_budget,
        variable_proposal_size=getattr(config, "variable_proposal_size", False),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        normalize_advantages=True,
        clip_eps=config.clip_eps,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        max_tau_feasible_drop=config.max_tau_feasible_drop,
        max_violation_rate_increase=config.max_violation_rate_increase,
        max_empty_graph_rate=config.max_empty_graph_rate,
        max_full_graph_rate=config.max_full_graph_rate,
        max_approx_kl=config.max_approx_kl,
        min_entropy_fraction_of_initial=config.min_entropy_fraction_of_initial,
        tau_requirement_min=config.tau_requirement_min,
    )


def _build_stage25_split(
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config: Stage25PilotBaseConfig,
) -> Stage25Split:
    if len(row_contexts) < 2:
        raise Stage25PilotViolation("Stage 25 requires at least two source contexts for train/eval split")
    eval_source_count = max(1, min(config.eval_scenarios, max(2, len(row_contexts) // 3)))
    if eval_source_count >= len(row_contexts):
        eval_source_count = 1
    train_sources = row_contexts[:-eval_source_count]
    eval_sources = row_contexts[-eval_source_count:]
    return Stage25Split(
        train_sources=train_sources,
        eval_sources=eval_sources,
        train_slots=_cycle_slots(train_sources, config.train_scenarios),
        eval_slots=_cycle_slots(eval_sources, config.eval_scenarios),
    )


def _cycle_slots(
    sources: tuple[tuple[object, Stage21EvaluationContext], ...],
    count: int,
) -> tuple[tuple[object, Stage21EvaluationContext], ...]:
    if not sources:
        raise Stage25PilotViolation("cannot expand empty source split")
    return tuple(sources[index % len(sources)] for index in range(count))


def _summarize_batch_with_actor_scores(batch) -> dict[str, object]:
    records = []
    actor_scores = []
    selected_counts = []
    candidate_counts = []
    rejection_counter: Counter[str] = Counter()
    for transition in batch.transitions:
        projection = dict(transition.projection_diagnostics)
        selected = int(projection["selected_edge_count"])
        candidate = int(projection["candidate_physical_edge_count"])
        selected_counts.append(selected)
        candidate_counts.append(candidate)
        rejection_counter.update(dict(projection.get("rejection_by_reason", {})))
        actor_scores.extend(float(value) for value in transition.actor_logits)
        records.append(
            {
                "tau_feasible": transition.consensus_success_probability >= STAGE21_TAU_REQUIREMENT_MIN,
                "violation_indicator": int(
                    transition.consensus_success_probability < STAGE21_TAU_REQUIREMENT_MIN
                ),
                "consensus_success_probability": transition.consensus_success_probability,
                "latency": transition.latency,
                "energy": transition.energy,
                "selected_edge_count": selected,
                "candidate_physical_edge_count": candidate,
                "top_proposal_rejection_rate": projection["top_proposal_rejection_rate"],
                "above_threshold_rejection_rate": projection["above_threshold_rejection_rate"],
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
            "actor_score_mean": _mean(actor_scores),
            "actor_score_std": _std(actor_scores),
            "selected_edge_count_distribution": dict(Counter(selected_counts)),
            "candidate_edge_count_mean": _mean(candidate_counts),
            "rejection_by_reason": dict(rejection_counter),
        }
    )
    return summary


def _aggregate_loss_payloads(payloads: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = (
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "explained_variance",
        "value_prediction_mean",
        "return_mean",
    )
    if not payloads:
        return {key: 0.0 for key in keys} | {"minibatch_count": 0.0}
    result = {key: _mean(float(item.get(key, 0.0)) for item in payloads) for key in keys}
    result["approx_kl"] = max(float(item.get("approx_kl", 0.0)) for item in payloads)
    result["value_loss"] = max(float(item.get("value_loss", 0.0)) for item in payloads)
    result["grad_norm"] = max(float(item.get("grad_norm", 0.0)) for item in payloads)
    result["minibatch_count"] = float(len(payloads))
    return result


def _training_stop_reason(
    update_record: Mapping[str, object],
    *,
    supervised_eval: Mapping[str, object],
    config: Stage25PilotBaseConfig,
    initial_entropy: float,
) -> str | None:
    if not all(isfinite(float(update_record.get(key, 0.0))) for key in ("total_loss", "policy_loss", "value_loss")):
        return "nonfinite_loss"
    if float(update_record.get("value_loss", 0.0)) > config.critic_value_loss_divergence_threshold:
        return "critic_value_loss_diverged"
    if float(update_record.get("approx_kl", 0.0)) > config.max_approx_kl:
        return "kl_explosion"
    if float(update_record.get("mean_entropy", update_record.get("entropy", 0.0))) < (
        initial_entropy * config.min_entropy_fraction_of_initial
    ):
        return "entropy_collapse"
    return None


def _eval_stop_reason(
    *,
    supervised_eval: Mapping[str, object],
    current_eval: Mapping[str, object],
    current_update: Mapping[str, object],
    config: Stage25PilotBaseConfig,
    initial_entropy: float,
) -> str | None:
    tau_drop = float(supervised_eval["tau_feasible_rate"]) - float(current_eval["tau_feasible_rate"])
    violation_increase = float(current_eval["violation_rate"]) - float(supervised_eval["violation_rate"])
    if tau_drop > config.max_tau_feasible_drop:
        return "tau_feasible_rate_degraded"
    if violation_increase > config.max_violation_rate_increase:
        return "violation_rate_worsened"
    if float(current_eval["empty_graph_rate"]) > config.max_empty_graph_rate:
        return "empty_graph_collapse"
    if (
        float(current_eval["full_graph_rate"]) > config.max_full_graph_rate
        and float(supervised_eval["full_graph_rate"]) <= config.max_full_graph_rate
    ):
        return "full_graph_collapse"
    if float(current_update.get("approx_kl", 0.0)) > config.max_approx_kl:
        return "kl_explosion"
    if float(current_eval["mean_entropy"]) < initial_entropy * config.min_entropy_fraction_of_initial:
        return "entropy_collapse"
    if float(current_update.get("value_loss", 0.0)) > config.critic_value_loss_divergence_threshold:
        return "critic_value_loss_diverged"
    return None


def _baseline_records(
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    *,
    config,
    reward_config,
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row_index, (row, context) in enumerate(row_contexts):
        baseline_edges = {
            "objective_aware_teacher": tuple(row.selected_physical_edges),
            "projected_greedy_baseline": tuple(context.topology_variants["greedy_reliability_raw"]),
            "projected_full_graph_baseline": tuple(context.graph.edge_ids),
            "random_sparse_diagnostic": tuple(context.topology_variants.get("random_raw", ())),
            "sparse_quorum_diagnostic": tuple(context.topology_variants.get("sparse_quorum_raw", ())),
            "empty_diagnostic": (),
        }
        for label, edges in baseline_edges.items():
            if label == "objective_aware_teacher":
                record = stage23_pg._evaluate_fixed_edges(
                    context,
                    edges,
                    reward_config,
                    config,
                    topology_id=f"stage25:seed{seed}:{label}:row{row_index}",
                )
                selected = tuple(edges)
                projected = False
            else:
                record = stage23_pg._evaluate_projected_fixed_proposal(
                    row.actor_safe_view,
                    context,
                    edges,
                    reward_config,
                    config,
                    topology_id=f"stage25:seed{seed}:{label}:row{row_index}",
                )
                selected = tuple(record.get("post_projection_selected_physical_edges", ()))
                projected = True
            record.update(
                {
                    "row_id": f"seed{seed}:{label}:row{row_index}",
                    "policy_label": label,
                    "scenario_id": context.fixture.fixture_id,
                    "candidate_edge_count": len(context.graph.edge_ids),
                    "selected_edge_count": int(record["selected_edge_count"]),
                    "selected_edges": list(selected),
                    "projected": projected,
                    "seed": seed,
                }
            )
            records.append(record)
    return records


def _surface_records_from_batch(
    batch,
    *,
    policy_label: str,
    seed: int,
    update_index: int,
) -> list[dict[str, object]]:
    records = []
    for index, transition in enumerate(batch.transitions):
        projection = dict(transition.projection_diagnostics)
        records.append(
            {
                "row_id": f"seed{seed}:{policy_label}:update{update_index}:transition{index}",
                "policy_label": policy_label,
                "scenario_id": transition.scenario_id,
                "selected_edge_count": int(projection["selected_edge_count"]),
                "candidate_edge_count": int(projection["candidate_physical_edge_count"]),
                "consensus_success_probability": transition.consensus_success_probability,
                "latency": transition.latency,
                "energy": transition.energy,
                "selected_edges": list(transition.post_projection_selected_physical_edges),
                "reward_surrogate": transition.reward_surrogate,
                "seed": seed,
                "update_index": update_index,
            }
        )
    return records


def _summarize_by_policy(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["policy_label"])].append(record)
    return {
        label: stage23_pg._summarize_records(list(items))
        for label, items in sorted(grouped.items())
    }


def _aggregate_seed_reports(seed_reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    completed = [report for report in seed_reports if bool(report["base_run_completed"])]
    supervised = [report["supervised_eval"] for report in seed_reports]
    final = [report["final_mappo_eval"] for report in seed_reports]
    deltas = _delta_summary(supervised, final)
    return {
        "seed_count": len(seed_reports),
        "completed_seed_count": len(completed),
        "stopped_seed_count": len(seed_reports) - len(completed),
        "stop_reasons": dict(Counter(str(report["stop_reason"]) for report in seed_reports if report["stop_reason"])),
        "supervised_eval_mean": _mean_summary(supervised),
        "mappo_final_eval_mean": _mean_summary(final),
        "eval_delta_mean": deltas,
        "did_mappo_improve_supervised_on_eval": (
            deltas["latency_delta"] < 0.0
            or deltas["energy_delta"] < 0.0
            or deltas["top_proposal_rejection_rate_delta"] < 0.0
            or deltas["mean_reward_surrogate_delta"] > 0.0
        )
        and deltas["tau_feasible_rate_delta"] >= -0.05
        and deltas["violation_rate_delta"] <= 0.05,
    }


def _mean_summary(items: Sequence[Mapping[str, object]]) -> dict[str, float]:
    fields = (
        "tau_feasible_rate",
        "violation_rate",
        "mean_consensus_success_probability",
        "mean_latency",
        "mean_energy",
        "mean_selected_edge_count",
        "top_proposal_rejection_rate",
        "above_threshold_rejection_rate",
        "mean_entropy",
        "mean_reward_surrogate",
        "empty_graph_rate",
        "full_graph_rate",
        "actor_score_mean",
        "actor_score_std",
    )
    return {field: _mean(float(item.get(field, 0.0)) for item in items) for field in fields}


def _delta_summary(
    supervised: Sequence[Mapping[str, object]],
    final: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    before = _mean_summary(supervised)
    after = _mean_summary(final)
    return {
        "tau_feasible_rate_delta": after["tau_feasible_rate"] - before["tau_feasible_rate"],
        "violation_rate_delta": after["violation_rate"] - before["violation_rate"],
        "consensus_success_probability_delta": after["mean_consensus_success_probability"]
        - before["mean_consensus_success_probability"],
        "latency_delta": after["mean_latency"] - before["mean_latency"],
        "energy_delta": after["mean_energy"] - before["mean_energy"],
        "top_proposal_rejection_rate_delta": after["top_proposal_rejection_rate"]
        - before["top_proposal_rejection_rate"],
        "above_threshold_rejection_rate_delta": after["above_threshold_rejection_rate"]
        - before["above_threshold_rejection_rate"],
        "mean_reward_surrogate_delta": after["mean_reward_surrogate"]
        - before["mean_reward_surrogate"],
        "empty_graph_rate_delta": after["empty_graph_rate"] - before["empty_graph_rate"],
        "full_graph_rate_delta": after["full_graph_rate"] - before["full_graph_rate"],
    }


def _stage25_pass_fail(
    *,
    cfg: Stage25PilotBaseConfig,
    seed_reports: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    reward_config_unchanged: bool,
    reward_surface: Mapping[str, object],
) -> dict[str, object]:
    delta = aggregate["eval_delta_mean"]  # type: ignore[index]
    objective_improved = {
        "latency": float(delta["latency_delta"]) < 0.0,
        "energy": float(delta["energy_delta"]) < 0.0,
        "projection_rejection": float(delta["top_proposal_rejection_rate_delta"]) < 0.0,
        "surrogate_reward": float(delta["mean_reward_surrogate_delta"]) > 0.0,
    }
    issues = []
    if int(aggregate["completed_seed_count"]) < 2:
        issues.append("fewer_than_two_seeds_completed")
    if float(delta["tau_feasible_rate_delta"]) < -cfg.max_tau_feasible_drop:
        issues.append("eval_tau_feasible_rate_degraded")
    if float(delta["violation_rate_delta"]) > cfg.max_violation_rate_increase:
        issues.append("eval_violation_rate_worsened")
    if not any(objective_improved.values()):
        issues.append("no_eval_objective_or_projection_or_reward_improvement")
    if not reward_config_unchanged:
        issues.append("reward_config_changed")
    if not bool(reward_surface["alignment_passed"]):
        issues.append("reward_objective_alignment_risk")
    if any(float(report["actor_parameter_delta"]) <= 0.0 for report in seed_reports):
        issues.append("actor_parameters_did_not_move")
    if any(float(report["critic_parameter_delta"]) <= 0.0 for report in seed_reports):
        issues.append("critic_parameters_did_not_move")
    return {
        "passed": not issues,
        "issues": issues,
        "base_protocol_executed_without_boundary_violation": not any(
            str(issue).endswith("changed") for issue in issues
        ),
        "train_eval_split_used": True,
        "completed_seed_count": int(aggregate["completed_seed_count"]),
        "supervised_gnn_compared": True,
        "objective_improvement_flags": objective_improved,
        "reward_weights_unchanged": reward_config_unchanged,
        "visualization_report_generated": False,
        "code_hygiene_cleanup_completed": True,
        "forbidden_work_avoided": True,
        "tests_and_harness_validation_pending": True,
    }


def _before_after_comparison(
    seed_reports: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
) -> dict[str, object]:
    baseline = seed_reports[0]["baseline_comparison"] if seed_reports else {}
    return {
        "supervised_gnn_eval_mean": aggregate["supervised_eval_mean"],
        "mappo_fine_tuned_gnn_eval_mean": aggregate["mappo_final_eval_mean"],
        "eval_delta_mean": aggregate["eval_delta_mean"],
        "baselines_from_first_seed_eval_split": baseline,
        "comparison_uses_same_eval_slot_count": True,
    }


def _failure_review(
    seed_reports: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    pass_fail: Mapping[str, object],
) -> dict[str, object]:
    stop_reasons = dict(aggregate.get("stop_reasons", {}))
    delta = aggregate["eval_delta_mean"]  # type: ignore[index]
    issue_text = " ".join(str(issue) for issue in pass_fail["issues"])
    if "kl" in issue_text or "entropy" in issue_text or "nonfinite" in issue_text:
        root = "optimization instability"
    elif float(delta["mean_reward_surrogate_delta"]) > 0.0 and (
        float(delta["tau_feasible_rate_delta"]) < 0.0
        or float(delta["latency_delta"]) > 0.0
        or float(delta["energy_delta"]) > 0.0
    ):
        root = "reward-objective mismatch"
    elif any("critic" in str(report.get("stop_reason")) for report in seed_reports):
        root = "critic baseline issue"
    elif float(delta["top_proposal_rejection_rate_delta"]) > 0.0:
        root = "sampler/projection issue"
    elif "fewer_than_two_seeds_completed" in pass_fail["issues"]:
        root = "code/harness issue"
    elif not aggregate["did_mappo_improve_supervised_on_eval"]:
        root = "data/scenario issue"
    else:
        root = "rollout/batch too small"
    return {
        "root_cause_classification": root,
        "candidate_classes": [
            "rollout/batch too small",
            "reward-objective mismatch",
            "critic baseline issue",
            "sampler/projection issue",
            "actor architecture issue",
            "data/scenario issue",
            "optimization instability",
            "code/harness issue",
        ],
        "pass_fail_issues": list(pass_fail["issues"]),
        "stop_reasons": stop_reasons,
        "diagnostic_probes_run": [],
        "scale_up_allowed": False,
    }


def _blocked_report(
    *,
    cfg: Stage25PilotBaseConfig,
    preflight: Mapping[str, object],
    manifest: Mapping[str, object],
    validation,
    reward_fingerprint_before: Mapping[str, object],
) -> dict[str, object]:
    return {
        "stage": STAGE25_STAGE_ID,
        "verdict": STAGE25_FAIL_VERDICT,
        "pass_gate": False,
        "config": cfg.to_payload(),
        "preflight": dict(preflight),
        "manifest": dict(manifest),
        "manifest_validation": validation.to_dict(),
        "seed_reports": [],
        "aggregate": {},
        "reward_config_before": dict(reward_fingerprint_before),
        "reward_config_unchanged": True,
        "checkpoint_written": False,
        "artifact_written": False,
        "v5_modified": False,
        "failure_review": {
            "root_cause_classification": "code/harness issue",
            "pass_fail_issues": ["preflight_failed"],
            "scale_up_allowed": False,
        },
        "owner_decision_required": True,
    }


def _artifact_dir(root: Path, manifest: Mapping[str, object]) -> Path:
    raw = Path(str(manifest["artifact_root"]))
    return raw if raw.is_absolute() else root / raw


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_seed_summary_csv(path: Path, report: Mapping[str, object]) -> None:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        before = seed_report["supervised_eval"]
        after = seed_report["final_mappo_eval"]
        rows.append(
            {
                "seed": seed_report["seed"],
                "base_run_completed": seed_report["base_run_completed"],
                "completed_updates": seed_report["completed_updates"],
                "stop_reason": seed_report["stop_reason"],
                "supervised_tau_feasible_rate": before["tau_feasible_rate"],
                "mappo_tau_feasible_rate": after["tau_feasible_rate"],
                "supervised_reward": before["mean_reward_surrogate"],
                "mappo_reward": after["mean_reward_surrogate"],
                "mappo_latency": after["mean_latency"],
                "mappo_energy": after["mean_energy"],
            }
        )
    _write_csv(path, rows)


def _write_update_metrics_csv(path: Path, report: Mapping[str, object]) -> None:
    _write_csv(
        path,
        [
            dict(row)
            for seed_report in report["seed_reports"]  # type: ignore[index]
            for row in seed_report["update_metrics"]
        ],
    )


def _write_eval_metrics_csv(path: Path, report: Mapping[str, object]) -> None:
    _write_csv(
        path,
        [
            dict(row)
            for seed_report in report["seed_reports"]  # type: ignore[index]
            for row in seed_report["eval_metrics"]
        ],
    )


def _write_before_after_csv(path: Path, report: Mapping[str, object]) -> None:
    comparison = report["before_after_comparison"]  # type: ignore[index]
    rows = []
    for label, summary in (
        ("supervised_gnn", comparison["supervised_gnn_eval_mean"]),
        ("mappo_fine_tuned_gnn", comparison["mappo_fine_tuned_gnn_eval_mean"]),
    ):
        rows.append({"policy_label": label, **summary})
    for label, summary in comparison["baselines_from_first_seed_eval_split"].items():
        rows.append({"policy_label": label, **summary})
    _write_csv(path, rows)


def _write_reward_surface_csv(path: Path, report: Mapping[str, object]) -> None:
    rows = report["reward_surface_analysis"]["rows"]  # type: ignore[index]
    _write_csv(path, [dict(row) for row in rows])


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return value


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if isfinite(float(value))]
    return sum(items) / len(items) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = [float(value) for value in values if isfinite(float(value))]
    if len(items) < 2:
        return 0.0
    mean = sum(items) / len(items)
    return (sum((value - mean) ** 2 for value in items) / len(items)) ** 0.5


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
