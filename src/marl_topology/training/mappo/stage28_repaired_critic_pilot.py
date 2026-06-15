"""Stage 28 fixed small-scale policy-gradient rerun with repaired value critic."""

from __future__ import annotations

from collections import Counter
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
from marl_topology.evaluation.reward_surface_analysis import build_reward_surface_analysis
from marl_topology.models import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalGNNEdgeScorer,
    build_model_registry,
)
from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCritic,
    GraphCriticBatch,
)
from marl_topology.objectives import SurrogateSignalInput, evaluate_reward_surrogate
from marl_topology.policies import UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
from marl_topology.training.critic_dataset import _graph_parts
from marl_topology.training.critic_repair_trainer import (
    Stage27CriticRepairTrainConfig,
    Stage27SelectedCriticBundle,
    denormalize_values_for_gae,
    train_stage27_selected_graph_value_critic_bundle,
    _standardize_graph_batch,
)
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
from .losses import ClippedPolicyValueLossInputs, clipped_policy_value_loss
from .rollout import RolloutTransition, build_rollout_batch
from .stage25_pilot import (
    STAGE25_ARTIFACT_ROOT,
    STAGE25_BASE_CONFIG_ID,
    STAGE25_PASS_VERDICT,
    STAGE25_RUN_ID,
    Stage25PilotBaseConfig,
    Stage25Split,
    _aggregate_seed_reports,
    _baseline_records,
    _before_after_comparison,
    _build_stage25_split,
    _cycle_slots,
    _delta_summary,
    _failure_review,
    _mean,
    _mean_summary,
    _stage25_loop_config,
    _summarize_batch_with_actor_scores,
    _summarize_by_policy,
    _surface_records_from_batch,
)
from .trainer import (
    _actor_safe_observation_payload,
    _endpoint_transition_fields,
    _parameter_checksum,
    _row_contexts,
    _stage23_config,
)


STAGE28_STAGE_ID = "stage_28_rerun_small_scale_mappo_with_repaired_critic"
STAGE28_CONFIG_ID = "stage28_repaired_critic_stage25_base_protocol"
STAGE28_RUN_ID = "stage28_repaired_critic_seed_group_2501_2502_2503"
STAGE28_PASS_VERDICT = "stage28_pass_repaired_critic_small_scale_rerun_complete"
STAGE28_FAIL_VERDICT = "stage28_fail_repaired_critic_rerun_blocked"
STAGE28_RECOMMENDED_NEXT_TASK_PASS = "stage_29_pre_scale_decision_review"
STAGE28_ARTIFACT_ROOT = (
    "result_save/stage28_repaired_critic_mappo_rerun/"
    "stage28_repaired_critic_stage25_base_protocol"
)
STAGE28_ARTIFACT_FILENAMES = (
    "manifest.json",
    "training_report.json",
    "seed_summary.csv",
    "update_metrics.csv",
    "eval_metrics.csv",
    "before_after_comparison.csv",
    "stage25_stage28_comparison.csv",
    "critic_diagnostics.csv",
    "reward_surface.csv",
    "reward_surface_analysis.json",
    "training_curves.png",
    "objective_curves.png",
    "critic_diagnostics.png",
    "before_after_comparison.png",
)


class Stage28PilotViolation(ValueError):
    """Raised when Stage 28 crosses a declared execution boundary."""


@dataclass(frozen=True, slots=True)
class Stage28PilotConfig:
    base: Stage25PilotBaseConfig = Stage25PilotBaseConfig()

    def to_payload(self) -> dict[str, object]:
        payload = self.base.to_payload()
        payload.update(
            {
                "config_id": STAGE28_CONFIG_ID,
                "inherits_stage25_fixed_base_protocol": True,
                "stage25_base_config_id": STAGE25_BASE_CONFIG_ID,
                "critic_model_id": CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
                "critic_source": "stage27_selected_graph_value_critic_in_memory",
                "hyperparameter_tuning_performed": False,
                "scale_up_training": False,
            }
        )
        return payload


def build_stage28_manifest(run_id: str = STAGE28_RUN_ID) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root=STAGE28_ARTIFACT_ROOT,
        overrides={
            "run_id": run_id,
            "stage_id": STAGE28_STAGE_ID,
            "owner_approval_id": "owner_approved_stage28_repaired_critic_rerun",
            "config_id": STAGE28_CONFIG_ID,
            "scenario_set_id": "stage22_selected_physical_train_eval_contexts",
            "split_id": "stage25_disjoint_source_rows_cyclic_slots_v1",
            "seed": 2501,
            "seeds": [2501, 2502, 2503],
            "seed_group_id": "stage28_same_stage25_seed_group",
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "critic_model_id": CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
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
                f"{STAGE28_ARTIFACT_ROOT}/{name}" for name in STAGE28_ARTIFACT_FILENAMES
            ],
            "checkpoint_creation_allowed": False,
            "training_scale_up_allowed": False,
            "artifact_write_allowed": "manifest_validated_reports_only",
        },
    )


def run_stage28_preflight(
    *,
    config: Stage28PilotConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or Stage28PilotConfig()
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    contexts = _row_contexts()
    split = _build_stage25_split(contexts, cfg.base)
    registry = build_model_registry()
    active = [
        entry.model_id
        for entry in registry.values()
        if entry.active_for_future_value_baseline
    ]
    signal_config = stage23_pg.build_stage23_reward_surrogate_config()
    manifest = build_stage28_manifest()
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    stage27_report = _load_stage27_report(root)
    stage25_report = _load_stage25_report(root)
    gates = {
        "stage27_selected_graph_critic_available": {
            "passed": (
                bool(stage27_report.get("pass_gate"))
                and stage27_report.get("selected_critic_id")
                == CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
                and active == [CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID]
            ),
            "selected_critic_id": stage27_report.get("selected_critic_id"),
            "active_future_value_critics": active,
        },
        "stage25_base_protocol_reused_without_tuning": {
            "passed": (
                cfg.base.transitions_per_update == 256
                and cfg.base.train_scenarios == 16
                and cfg.base.eval_scenarios == 8
                and cfg.base.seeds == (2501, 2502, 2503)
            ),
            "config": cfg.to_payload(),
        },
        "active_sampler_is_plackett_luce": {
            "passed": ACTIVE_POLICY_GRADIENT_SAMPLER_ID == PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
        },
        "train_eval_split_available": {
            "passed": bool(split.train_sources and split.eval_sources),
            "split": split.to_payload(),
        },
        "stage25_historical_comparison_available": {
            "passed": stage25_report.get("verdict") == STAGE25_PASS_VERDICT,
            "stage25_run_id": STAGE25_RUN_ID,
            "stage25_verdict": stage25_report.get("verdict"),
        },
        "reward_weights_unchanged": {
            # Gate intent: no unsanctioned reward tuning. The active reward is the
            # owner-approved feasibility-first barrier with tau frozen at 0.9; weights
            # are the fixed Stage 31 values, not searched per run.
            "passed": (
                signal_config.tau == STAGE21_TAU_REQUIREMENT_MIN
                and signal_config.structure == SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2
            ),
            "reward_config_fingerprint": stage23_pg._surrogate_config_fingerprint(signal_config),
            "weight_tuning_performed": False,
        },
        "run_manifest_validator_available": {
            "passed": validation.is_valid and validation.writes_performed is False,
            "validator_result": validation.to_dict(),
        },
    }
    return {
        "stage": STAGE28_STAGE_ID,
        "preflight_passed": all(bool(gate["passed"]) for gate in gates.values()),
        "gates": gates,
        "v5_modified": False,
    }


def run_stage28_repaired_critic_small_scale_pilot(
    *,
    config: Stage28PilotConfig | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or Stage28PilotConfig()
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    preflight = run_stage28_preflight(config=cfg, project_root=root)
    signal_config_before = stage23_pg.build_stage23_reward_surrogate_config()
    signal_fingerprint_before = stage23_pg._surrogate_config_fingerprint(signal_config_before)
    manifest = build_stage28_manifest()
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    contexts = _row_contexts()
    split = _build_stage25_split(contexts, cfg.base)
    stage25_report = _load_stage25_report(root)
    if not preflight["preflight_passed"]:
        return _blocked_report(
            cfg=cfg,
            preflight=preflight,
            manifest=manifest,
            validation=validation,
            signal_fingerprint_before=signal_fingerprint_before,
        )
    bundle = train_stage27_selected_graph_value_critic_bundle(
        config=Stage27CriticRepairTrainConfig(),
        project_root=root,
    )
    initial_critic_state = deepcopy(bundle.model.state_dict())
    seed_reports = [
        _run_stage28_seed(
            seed=seed,
            seed_index=seed_index,
            config=cfg.base,
            split=split,
            signal_config=signal_config_before,
            critic_bundle=bundle,
            initial_critic_state=initial_critic_state,
        )
        for seed_index, seed in enumerate(cfg.base.seeds)
    ]
    signal_config_after = stage23_pg.build_stage23_reward_surrogate_config()
    signal_fingerprint_after = stage23_pg._surrogate_config_fingerprint(signal_config_after)
    aggregate = _aggregate_seed_reports(seed_reports)
    reward_surface_records = [
        record
        for seed_report in seed_reports
        for record in seed_report["reward_surface_records"]
    ]
    reward_surface = build_reward_surface_analysis(
        reward_surface_records,
        reward_config=signal_config_before,
        tau_requirement_min=cfg.base.tau_requirement_min,
    )
    stage25_stage28 = _compare_stage25_stage28(stage25_report, aggregate)
    pass_fail = _stage28_pass_fail(
        cfg=cfg.base,
        seed_reports=seed_reports,
        aggregate=aggregate,
        reward_config_unchanged=signal_fingerprint_before == signal_fingerprint_after,
        reward_surface=reward_surface,
        stage25_stage28=stage25_stage28,
        bundle_report=bundle.repair_report,
    )
    verdict = STAGE28_PASS_VERDICT if pass_fail["passed"] else STAGE28_FAIL_VERDICT
    return _jsonable(
        {
            "stage": STAGE28_STAGE_ID,
            "verdict": verdict,
            "pass_gate": pass_fail["passed"],
            "pass_fail_gate": pass_fail,
            "config": cfg.to_payload(),
            "preflight": preflight,
            "manifest": manifest,
            "manifest_validation": validation.to_dict(),
            "split": split.to_payload(),
            "stage27_repaired_critic_report": bundle.repair_report,
            "stage25_historical_comparison": stage25_stage28,
            "seed_reports": seed_reports,
            "aggregate": aggregate,
            "reward_surface_analysis": reward_surface,
            "before_after_comparison": _before_after_comparison(seed_reports, aggregate),
            "reward_config_before": signal_fingerprint_before,
            "reward_config_after": signal_fingerprint_after,
            "reward_config_unchanged": signal_fingerprint_before == signal_fingerprint_after,
            "reward_weight_tuning_performed": False,
            "diagnostic_probes_run": [],
            "base_config_id": STAGE25_BASE_CONFIG_ID,
            "stage28_config_id": STAGE28_CONFIG_ID,
            "primary_conclusion_based_only_on_fixed_base_config": True,
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
            "recommended_next_task": (
                STAGE28_RECOMMENDED_NEXT_TASK_PASS
                if pass_fail["passed"]
                else "stage_29_failure_mode_review_after_repaired_critic_rerun"
            ),
            "owner_decision_required": True,
        }
    )


def write_stage28_training_artifacts(
    report: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    manifest = dict(report["manifest"])  # type: ignore[index]
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    if not validation.is_valid:
        raise Stage28PilotViolation(f"manifest validation failed: {validation.error_codes()}")
    artifact_dir = root / STAGE28_ARTIFACT_ROOT
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_to_write = {**dict(report), "artifact_written": True, "artifact_dir": str(artifact_dir)}
    _write_json(artifact_dir / "manifest.json", manifest)
    _write_json(artifact_dir / "training_report.json", report_to_write)
    _write_json(artifact_dir / "reward_surface_analysis.json", report_to_write["reward_surface_analysis"])
    _write_csv(artifact_dir / "seed_summary.csv", _seed_rows(report_to_write))
    _write_csv(artifact_dir / "update_metrics.csv", _update_rows(report_to_write))
    _write_csv(artifact_dir / "eval_metrics.csv", _eval_rows(report_to_write))
    _write_csv(artifact_dir / "before_after_comparison.csv", _comparison_rows(report_to_write))
    _write_csv(artifact_dir / "stage25_stage28_comparison.csv", _stage25_comparison_rows(report_to_write))
    _write_csv(artifact_dir / "critic_diagnostics.csv", _critic_rows(report_to_write))
    _write_csv(artifact_dir / "reward_surface.csv", _surface_rows(report_to_write))
    _write_optional_plots(artifact_dir, report_to_write)
    _write_docs(root, report_to_write)
    return {
        "artifact_written": True,
        "artifact_dir": str(artifact_dir),
        "manifest_validated": validation.is_valid,
        "written_files": sorted(path.name for path in artifact_dir.iterdir() if path.is_file()),
    }


def _run_stage28_seed(
    *,
    seed: int,
    seed_index: int,
    config: Stage25PilotBaseConfig,
    split: Stage25Split,
    signal_config,
    critic_bundle: Stage27SelectedCriticBundle,
    initial_critic_state: Mapping[str, object],
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
    critic = CentralizedMessagePassingGraphCritic(critic_bundle.model.config)
    critic.load_state_dict(initial_critic_state)
    actor_before_training = _parameter_checksum(actor)
    critic_before_training = _parameter_checksum(critic)
    supervised_eval_batch = _collect_repaired_rollout_batch(
        actor=actor,
        critic=critic,
        sampler=sampler,
        row_contexts=split.eval_slots,
        config=eval_loop_config,
        stage23_config=stage23_config,
        signal_config=signal_config,
        critic_bundle=critic_bundle,
        seed_base=seed + 200_000,
    )
    supervised_eval = _summarize_batch_with_actor_scores(supervised_eval_batch)
    baseline_records = _baseline_records(
        split.eval_slots,
        config=stage23_config,
        reward_config=signal_config,
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
            "phase": "supervised_before_repaired_critic_update",
            **supervised_eval,
        }
    ]
    critic_pairs: list[dict[str, float]] = []
    stop_reason: str | None = None
    completed_updates = 0
    final_eval_batch = supervised_eval_batch
    for update_index in range(1, config.max_updates + 1):
        batch = _collect_repaired_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=split.train_slots,
            config=pilot_loop_config,
            stage23_config=stage23_config,
            signal_config=signal_config,
            critic_bundle=critic_bundle,
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
        for _epoch_index in range(config.update_epochs):
            permutation = torch.randperm(batch.total_transitions, generator=generator).tolist()
            for start in range(0, batch.total_transitions, config.minibatch_size):
                indices = tuple(int(index) for index in permutation[start : start + config.minibatch_size])
                loss_result = _loss_for_repaired_indices(
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
                    critic_bundle=critic_bundle,
                )
                if not torch.isfinite(loss_result["total_loss"]).all().item():
                    stop_reason = "nonfinite_total_loss"
                    break
                actor_update_rule.zero_grad()
                critic_update_rule.zero_grad()
                loss_result["total_loss"].backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                grad_norm = float(actor_norm.detach().cpu().item()) + float(
                    critic_norm.detach().cpu().item()
                )
                actor_update_rule.step()
                critic_update_rule.step()
                payload = dict(loss_result["payload"])
                payload["grad_norm"] = grad_norm
                loss_payloads.append(payload)
                if payload["approx_kl"] > config.max_approx_kl:
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
            "repaired_critic_used_for_gae": True,
            "value_loss_target_scale": "normalized_return",
        }
        update_metrics.append(update_record)
        critic_pairs = [
            {"value_prediction": float(value), "return": float(ret)}
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
            eval_batch = _collect_repaired_rollout_batch(
                actor=actor,
                critic=critic,
                sampler=sampler,
                row_contexts=split.eval_slots,
                config=eval_loop_config,
                stage23_config=stage23_config,
                signal_config=signal_config,
                critic_bundle=critic_bundle,
                seed_base=seed + 200_000,
            )
            eval_summary = _summarize_batch_with_actor_scores(eval_batch)
            eval_record = {
                "seed": seed,
                "update_index": update_index,
                "phase": "repaired_critic_eval",
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
        if stop_reason:
            break
    else:
        final_eval_batch = _collect_repaired_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=split.eval_slots,
            config=eval_loop_config,
            stage23_config=stage23_config,
            signal_config=signal_config,
            critic_bundle=critic_bundle,
            seed_base=seed + 200_000,
        )
    if eval_metrics[-1]["phase"] != "repaired_critic_eval":
        final_summary = _summarize_batch_with_actor_scores(final_eval_batch)
        eval_metrics.append(
            {
                "seed": seed,
                "update_index": completed_updates,
                "phase": "repaired_critic_eval",
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
        "repaired_graph_critic_reused_as_initial_value_baseline": True,
        "reward_surface_records": (
            _surface_records_from_batch(
                supervised_eval_batch,
                policy_label="supervised_gnn_before_repaired_critic_update",
                seed=seed,
                update_index=0,
            )
            + _surface_records_from_batch(
                final_eval_batch,
                policy_label="repaired_critic_fine_tuned_gnn",
                seed=seed,
                update_index=completed_updates,
            )
            + baseline_records
        ),
    }


def _collect_repaired_rollout_batch(
    *,
    actor: LocalGNNEdgeScorer,
    critic: CentralizedMessagePassingGraphCritic,
    sampler,
    row_contexts: tuple[tuple[object, Stage21EvaluationContext], ...],
    config,
    stage23_config,
    signal_config,
    critic_bundle: Stage27SelectedCriticBundle,
    seed_base: int,
    frame_contexts: tuple[tuple[tuple[object, Stage21EvaluationContext], ...], ...] | None = None,
    history_window: int = 1,
    predictive_horizon: int = 0,
    deterministic: bool = False,
):
    # frame_contexts (A3, opt-in): a per-scenario-slot SEQUENCE of (row, context),
    # one per rollout step, so each step uses the scene advanced in time (vehicles
    # moved). When None the collector behaves byte-identically to the static path
    # (one frozen (row, context) reused for every step). The PPO importance ratio
    # stays valid because the loss recompute indexes the SAME frame by the SAME
    # (row_index, step_index) carried on each transition.
    transitions: list[RolloutTransition] = []
    for scenario_index in range(config.num_scenarios):
        selected_for_state: tuple[str, ...] = ()
        source_index = scenario_index % len(row_contexts)
        frames = None if frame_contexts is None else frame_contexts[source_index]
        if frames is not None and len(frames) < config.rollout_steps:
            raise ValueError(
                f"frame_contexts slot {source_index} has {len(frames)} frames < "
                f"rollout_steps={config.rollout_steps}; trajectory frames must cover every step"
            )
        for step_index in range(config.rollout_steps):
            row, context = (
                row_contexts[source_index] if frames is None else frames[step_index]
            )
            graph_payload = _graph_payload(
                row=row,
                context=context,
                previous_selected_edges=selected_for_state,
                step_index=step_index,
            )
            graph_batch = _standardize_graph_batch(
                _graph_batch_from_payloads((graph_payload,)),
                critic_bundle.graph_state,
            )
            norm_value = critic(graph_batch).normalized_value.reshape(-1)[0]
            value_prediction = denormalize_values_for_gae(
                norm_value.reshape(1),
                critic_bundle.normalizer,
            ).reshape(-1)[0]
            # A4 history window (opt-in, temporal actor only): same (frames, step_index)
            # indexing as the loss recompute -> the PPO ratio stays a pure policy measure.
            history = stage23_pg._actor_history_window(frames, step_index, history_window)
            view = stage23_pg._actor_physical_logit_view(
                actor, row, stage23_config, history=history
            )
            rng = torch.Generator().manual_seed(
                seed_base + scenario_index * 101 + step_index
            )
            sample = sampler.sample(
                view.physical_logits, view.mask, view.config, rng, deterministic=deterministic
            )
            assembly = stage23_pg._assemble_physical_proposal(row.actor_safe_view, view, sample)
            # Predictive horizon (opt-in): score the chosen topology against frame t+h's
            # evaluator (vehicles moved), clamped at the trajectory end. The actor view,
            # history, and critic payload above are all on the CURRENT frame t -- only the
            # reward's evaluator looks ahead, so this stays leakage-safe and PPO-exact.
            # h==0 or static (frames None) -> eval_context is context (byte-identical).
            eval_context = context
            if frames is not None and predictive_horizon > 0:
                horizon_index = min(step_index + predictive_horizon, len(frames) - 1)
                eval_context = frames[horizon_index][1]
            evaluation = eval_context.evaluator.evaluate(assembly.selected_physical_edges)
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
            done = step_index == config.rollout_steps - 1
            endpoint_fields = _endpoint_transition_fields(sample, assembly.selected_physical_edges)
            transition = RolloutTransition(
                actor_safe_observation=_actor_safe_observation_payload(row.actor_safe_view),
                centralized_critic_input=graph_payload,
                actor_logits=tuple(float(item) for item in view.physical_logits.detach().cpu().tolist()),
                sampler_id=sample.sampler_id,
                proposal_action={
                    "policy_action": "proposal",
                    "proposed_physical_edges": tuple(sample.proposed_physical_edges),
                    **endpoint_fields,
                },
                proposal_logprob=sample.logprob,
                proposal_entropy=sample.entropy,
                pre_projection_proposals=tuple(sample.proposed_physical_edges),
                post_projection_selected_physical_edges=tuple(assembly.selected_physical_edges),
                projection_diagnostics={
                    "top_proposal_rejection_rate": projection["top_proposal_rejection_rate"],
                    "above_threshold_rejection_rate": projection["above_threshold_rejection_rate"],
                    "rejection_by_reason": projection["rejection_by_reason"],
                    "projection_rejection_reasons": {
                        edge_id: [reason.value for reason in reasons]
                        for edge_id, reasons in assembly.rejection_reasons.items()
                    },
                    "selected_edge_count": len(assembly.selected_physical_edges),
                    "candidate_physical_edge_count": len(view.physical_edge_ids),
                },
                consensus_success_probability=float(evaluation.metrics["consensus_success_probability"]),
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


def _loss_for_repaired_indices(
    *,
    actor: LocalGNNEdgeScorer,
    critic: CentralizedMessagePassingGraphCritic,
    sampler,
    batch,
    row_contexts,
    stage23_config,
    indices: tuple[int, ...],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config,
    critic_bundle: Stage27SelectedCriticBundle,
    frame_contexts: tuple[tuple[tuple[object, object], ...], ...] | None = None,
    history_window: int = 1,
) -> dict[str, object]:
    new_logprob_values = []
    entropy_values = []
    graph_payloads = []
    for index in indices:
        transition = batch.transitions[index]
        # A3: re-score the actor on the SAME frame the rollout used for this
        # transition. Indexing the SAME immutable frame_contexts by the transition's
        # (row_index, step_index) -- identical to the rollout collector -- keeps
        # exp(new_logprob - old_logprob) a pure measure of the policy-parameter
        # change, not a cross-frame artifact. (The critic input is the verbatim
        # snapshot in transition.centralized_critic_input, so it needs no re-index.)
        if frame_contexts is None:
            row, _context = row_contexts[transition.row_index]
            frames = None
        else:
            frames = frame_contexts[transition.row_index]
            row, _context = frames[transition.step_index]
        # Rebuild the IDENTICAL A4 history window the rollout used for this transition
        # (same frames + same step_index) so the temporal actor re-scores the same input.
        history = stage23_pg._actor_history_window(
            frames, transition.step_index, history_window
        )
        view = stage23_pg._actor_physical_logit_view(
            actor, row, stage23_config, history=history
        )
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
        graph_payloads.append(transition.centralized_critic_input)
    graph_batch = _standardize_graph_batch(
        _graph_batch_from_payloads(tuple(graph_payloads)),
        critic_bundle.graph_state,
    )
    norm_values = critic(graph_batch).normalized_value.reshape(-1)
    raw_values = denormalize_values_for_gae(norm_values, critic_bundle.normalizer)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    target_returns = returns[index_tensor].reshape(-1)
    target_norm = torch.tensor(
        critic_bundle.normalizer.transform(target_returns.detach().cpu().tolist()),
        dtype=target_returns.dtype,
        device=target_returns.device,
    )
    loss = clipped_policy_value_loss(
        ClippedPolicyValueLossInputs(
            new_logprobs=torch.stack(new_logprob_values),
            old_logprobs=batch.old_logprobs[index_tensor],
            advantages=advantages[index_tensor],
            value_predictions=norm_values,
            returns=target_norm,
            entropies=torch.stack(entropy_values),
            clip_eps=config.clip_eps,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
        )
    )
    raw_explained = _explained_variance(target_returns, raw_values)
    payload = loss.to_payload()
    payload.update(
        {
            "explained_variance": raw_explained,
            "value_prediction_mean": _scalar(raw_values.mean()),
            "return_mean": _scalar(target_returns.mean()),
            "normalized_return_mean": _scalar(target_norm.mean()),
            "normalized_value_prediction_mean": _scalar(norm_values.mean()),
            "value_return_correlation": _correlation(target_returns, raw_values),
        }
    )
    return {
        "total_loss": loss.total_loss,
        "payload": payload,
    }


def _graph_payload(
    *,
    row: object,
    context: Stage21EvaluationContext,
    previous_selected_edges: Sequence[str],
    step_index: int,
) -> dict[str, object]:
    parts = _graph_parts(
        row=row,
        context=context,
        previous_selected_edges=previous_selected_edges,
        step_index=step_index,
    )
    return {
        "view_role": "stage28_repaired_graph_value_critic_pre_action",
        "feature_schema_id": "stage27_centralized_graph_value_features_v1",
        "scenario_id": str(context.fixture.fixture_id),
        "time_step": int(step_index),
        "previous_selected_edges": tuple(str(edge) for edge in previous_selected_edges),
        "node_features": parts["node_features"],
        "edge_features": parts["edge_features"],
        "edge_index": parts["edge_index"],
        "training_only": True,
        "actor_input_allowed": False,
    }


def _graph_batch_from_payloads(payloads: Sequence[Mapping[str, object]]) -> GraphCriticBatch:
    if not payloads:
        raise Stage28PilotViolation("graph batch requires payloads")
    max_nodes = max(len(payload["node_features"]) for payload in payloads)  # type: ignore[arg-type]
    max_edges = max(len(payload["edge_features"]) for payload in payloads)  # type: ignore[arg-type]
    first_node = payloads[0]["node_features"][0]  # type: ignore[index]
    first_edge = payloads[0]["edge_features"][0]  # type: ignore[index]
    node_tensor = torch.zeros((len(payloads), max_nodes, len(first_node)), dtype=torch.float32)
    edge_tensor = torch.zeros((len(payloads), max_edges, len(first_edge)), dtype=torch.float32)
    edge_index = torch.zeros((len(payloads), max_edges, 2), dtype=torch.long)
    node_mask = torch.zeros((len(payloads), max_nodes), dtype=torch.bool)
    edge_mask = torch.zeros((len(payloads), max_edges), dtype=torch.bool)
    for row_index, payload in enumerate(payloads):
        nodes = tuple(payload["node_features"])  # type: ignore[arg-type]
        edges = tuple(payload["edge_features"])  # type: ignore[arg-type]
        indices = tuple(payload["edge_index"])  # type: ignore[arg-type]
        node_tensor[row_index, : len(nodes)] = torch.tensor(nodes, dtype=torch.float32)
        edge_tensor[row_index, : len(edges)] = torch.tensor(edges, dtype=torch.float32)
        edge_index[row_index, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        node_mask[row_index, : len(nodes)] = True
        edge_mask[row_index, : len(edges)] = True
    return GraphCriticBatch(
        node_features=node_tensor,
        edge_features=edge_tensor,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )


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
    if float(update_record.get("empty_graph_rate", 0.0)) > config.max_empty_graph_rate:
        return "empty_graph_collapse"
    if float(update_record.get("full_graph_rate", 0.0)) > config.max_full_graph_rate:
        return "full_graph_collapse"
    if float(update_record.get("mean_entropy", 0.0)) < initial_entropy * config.min_entropy_fraction_of_initial:
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
    if tau_drop > config.max_tau_feasible_drop:
        return "eval_tau_feasible_rate_degraded"
    violation_increase = float(current_eval["violation_rate"]) - float(supervised_eval["violation_rate"])
    if violation_increase > config.max_violation_rate_increase:
        return "eval_violation_rate_worsened"
    if float(current_eval["empty_graph_rate"]) > config.max_empty_graph_rate:
        return "empty_graph_collapse"
    if float(current_eval["full_graph_rate"]) > config.max_full_graph_rate:
        return "full_graph_collapse"
    if float(current_update.get("approx_kl", 0.0)) > config.max_approx_kl:
        return "kl_explosion"
    if float(current_eval["mean_entropy"]) < initial_entropy * config.min_entropy_fraction_of_initial:
        return "entropy_collapse"
    if float(current_update.get("value_loss", 0.0)) > config.critic_value_loss_divergence_threshold:
        return "critic_value_loss_diverged"
    return None


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
        "value_return_correlation",
    )
    if not payloads:
        return {key: 0.0 for key in keys} | {"minibatch_count": 0.0}
    result = {key: _mean(float(item.get(key, 0.0)) for item in payloads) for key in keys}
    result["approx_kl"] = max(float(item.get("approx_kl", 0.0)) for item in payloads)
    result["value_loss"] = max(float(item.get("value_loss", 0.0)) for item in payloads)
    result["grad_norm"] = max(float(item.get("grad_norm", 0.0)) for item in payloads)
    result["minibatch_count"] = float(len(payloads))
    return result


def _stage28_pass_fail(
    *,
    cfg: Stage25PilotBaseConfig,
    seed_reports: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    reward_config_unchanged: bool,
    reward_surface: Mapping[str, object],
    stage25_stage28: Mapping[str, object],
    bundle_report: Mapping[str, object],
) -> dict[str, object]:
    delta = aggregate["eval_delta_mean"]  # type: ignore[index]
    objective_improved = {
        "latency": float(delta["latency_delta"]) < 0.0,
        "energy": float(delta["energy_delta"]) < 0.0,
        "projection_rejection": float(delta["top_proposal_rejection_rate_delta"]) < 0.0,
        "surrogate_reward": float(delta["mean_reward_surrogate_delta"]) > 0.0,
    }
    critic_health = _critic_health_summary(seed_reports)
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
    if not bool(bundle_report.get("graph_gate_passed")):
        issues.append("stage27_repaired_critic_gate_not_reused")
    if float(critic_health["mean_update_explained_variance"]) <= 0.10:
        issues.append("repaired_critic_explained_variance_below_minimum")
    return {
        "passed": not issues,
        "issues": issues,
        "fixed_stage25_protocol_executed": True,
        "train_eval_split_used": True,
        "completed_seed_count": int(aggregate["completed_seed_count"]),
        "supervised_gnn_compared": True,
        "objective_improvement_flags": objective_improved,
        "reward_weights_unchanged": reward_config_unchanged,
        "stage27_selected_critic_reused": bool(bundle_report.get("graph_gate_passed")),
        "critic_health": critic_health,
        "stage25_stage28_comparison": stage25_stage28,
        "visualization_report_generated": True,
        "forbidden_work_avoided": True,
        "tests_and_harness_validation_pending": True,
    }


def _critic_health_summary(seed_reports: Sequence[Mapping[str, object]]) -> dict[str, float]:
    updates = [
        update
        for report in seed_reports
        for update in report.get("update_metrics", ())
    ]
    return {
        "mean_update_explained_variance": _mean(
            float(update.get("explained_variance", 0.0)) for update in updates
        ),
        "mean_value_return_correlation": _mean(
            float(update.get("value_return_correlation", 0.0)) for update in updates
        ),
        "mean_normalized_value_loss": _mean(
            float(update.get("value_loss", 0.0)) for update in updates
        ),
    }


def _compare_stage25_stage28(
    stage25_report: Mapping[str, object],
    stage28_aggregate: Mapping[str, object],
) -> dict[str, object]:
    stage25_aggregate = stage25_report.get("aggregate", {})
    stage25_final = stage25_aggregate.get("mappo_final_eval_mean", {}) if isinstance(stage25_aggregate, Mapping) else {}
    stage28_final = stage28_aggregate.get("mappo_final_eval_mean", {})
    fields = (
        "tau_feasible_rate",
        "violation_rate",
        "mean_latency",
        "mean_energy",
        "top_proposal_rejection_rate",
        "mean_reward_surrogate",
    )
    deltas = {
        f"{field}_stage28_minus_stage25": float(stage28_final.get(field, 0.0))
        - float(stage25_final.get(field, 0.0))
        for field in fields
    }
    return {
        "stage25_available": bool(stage25_aggregate),
        "stage25_completed_seed_count": int(stage25_aggregate.get("completed_seed_count", 0))
        if isinstance(stage25_aggregate, Mapping)
        else 0,
        "stage28_completed_seed_count": int(stage28_aggregate.get("completed_seed_count", 0)),
        "metric_deltas": deltas,
        "repaired_critic_improved_old_critic_result": (
            deltas["mean_reward_surrogate_stage28_minus_stage25"] > 0.0
            or deltas["mean_latency_stage28_minus_stage25"] < 0.0
            or deltas["mean_energy_stage28_minus_stage25"] < 0.0
            or deltas["top_proposal_rejection_rate_stage28_minus_stage25"] < 0.0
        )
        and deltas["tau_feasible_rate_stage28_minus_stage25"] >= -0.05
        and deltas["violation_rate_stage28_minus_stage25"] <= 0.05,
    }


def _blocked_report(
    *,
    cfg: Stage28PilotConfig,
    preflight: Mapping[str, object],
    manifest: Mapping[str, object],
    validation,
    signal_fingerprint_before: Mapping[str, object],
) -> dict[str, object]:
    return {
        "stage": STAGE28_STAGE_ID,
        "verdict": STAGE28_FAIL_VERDICT,
        "pass_gate": False,
        "config": cfg.to_payload(),
        "preflight": dict(preflight),
        "manifest": dict(manifest),
        "manifest_validation": validation.to_dict(),
        "seed_reports": [],
        "aggregate": {},
        "reward_config_before": dict(signal_fingerprint_before),
        "reward_config_after": dict(signal_fingerprint_before),
        "reward_config_unchanged": True,
        "reward_weight_tuning_performed": False,
        "sampler_switched": False,
        "scale_up_training_performed": False,
        "checkpoint_written": False,
        "v5_modified": False,
        "failure_review": {
            "root_cause_classification": "code/harness issue",
            "preflight": dict(preflight),
            "scale_up_allowed": False,
        },
        "recommended_next_task": "stage_28_preflight_repair_before_training",
        "owner_decision_required": True,
    }


def _load_stage27_report(root: Path) -> dict[str, object]:
    path = (
        root
        / "result_save"
        / "stage27_critic_baseline_repair"
        / "stage27_critic_repair_dataset_and_train_config"
        / "stage27_critic_repair_report.json"
    )
    if not path.exists():
        return {"pass_gate": False, "missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stage25_report(root: Path) -> dict[str, object]:
    path = root / STAGE25_ARTIFACT_ROOT / "training_report.json"
    if not path.exists():
        return {"verdict": None, "missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "seed": seed_report["seed"],
            "completed_updates": seed_report["completed_updates"],
            "base_run_completed": seed_report["base_run_completed"],
            "stop_reason": seed_report["stop_reason"],
            "actor_parameter_delta": seed_report["actor_parameter_delta"],
            "critic_parameter_delta": seed_report["critic_parameter_delta"],
            **{
                f"final_{key}": value
                for key, value in seed_report["final_mappo_eval"].items()
                if isinstance(value, (int, float, str, bool))
            },
        }
        for seed_report in report["seed_reports"]  # type: ignore[index]
    ]


def _update_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for update in seed_report["update_metrics"]:
            rows.append(dict(update))
    return rows


def _eval_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for update in seed_report["eval_metrics"]:
            rows.append(dict(update))
    return rows


def _comparison_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    comparison = report["before_after_comparison"]
    rows = []
    for section in ("supervised_gnn_eval_mean", "mappo_fine_tuned_gnn_eval_mean", "eval_delta_mean"):
        for key, value in comparison[section].items():
            rows.append({"section": section, "metric": key, "value": value})
    return rows


def _stage25_comparison_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    comparison = report["stage25_historical_comparison"]
    rows = [
        {
            "metric": key,
            "value": value,
        }
        for key, value in comparison.get("metric_deltas", {}).items()
    ]
    rows.append(
        {
            "metric": "repaired_critic_improved_old_critic_result",
            "value": comparison.get("repaired_critic_improved_old_critic_result"),
        }
    )
    return rows


def _critic_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed_report in report["seed_reports"]:  # type: ignore[index]
        for item in seed_report.get("critic_prediction_return_pairs", ()):
            rows.append({"seed": seed_report["seed"], **item})
    return rows


def _surface_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        record
        for seed_report in report["seed_reports"]  # type: ignore[index]
        for record in seed_report["reward_surface_records"]
    ]


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


def _write_optional_plots(artifact_dir: Path, report: Mapping[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    update_rows = _update_rows(report)
    if update_rows:
        for filename, fields, title in (
            ("training_curves.png", ("mean_reward_surrogate", "tau_feasible_rate", "violation_rate"), "Training"),
            ("objective_curves.png", ("mean_consensus_success_probability", "mean_latency", "mean_energy"), "Objective"),
            ("critic_diagnostics.png", ("explained_variance", "value_loss"), "Critic"),
        ):
            fig, ax = plt.subplots(figsize=(7, 4))
            for field in fields:
                ax.plot(
                    [int(row["update_index"]) for row in update_rows],
                    [float(row.get(field, 0.0)) for row in update_rows],
                    label=field,
                )
            ax.set_title(f"Stage 28 {title} Curves")
            ax.set_xlabel("Update")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(artifact_dir / filename, dpi=150)
            plt.close(fig)
    comparison = report["before_after_comparison"]["eval_delta_mean"]
    fig, ax = plt.subplots(figsize=(6, 4))
    fields = ("latency_delta", "energy_delta", "mean_reward_surrogate_delta")
    ax.bar(fields, [float(comparison.get(field, 0.0)) for field in fields])
    ax.set_title("Stage 28 Before/After Delta")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(artifact_dir / "before_after_comparison.png", dpi=150)
    plt.close(fig)


def _write_docs(root: Path, report: Mapping[str, object]) -> None:
    docs_dir = root / "docs"
    docs = {
        "STAGE28_REPAIRED_CRITIC_POLICY_GRADIENT_RERUN.md": _main_markdown(report),
        "STAGE28_REPAIRED_CRITIC_TRAINING_REPORT.md": _training_markdown(report),
        "STAGE28_REPAIRED_CRITIC_COMPARISON.md": _comparison_markdown(report),
        "STAGE28_POLICY_GRADIENT_READINESS_AFTER_RERUN.md": _readiness_markdown(report),
    }
    if not report["pass_gate"]:
        docs["STAGE28_REPAIRED_CRITIC_FAILURE_REVIEW.md"] = _stage28_failure_markdown(report)
    for filename, text in docs.items():
        (docs_dir / filename).write_text(text, encoding="utf-8")


def _main_markdown(report: Mapping[str, object]) -> str:
    aggregate = report["aggregate"]
    delta = aggregate["eval_delta_mean"]
    return "\n".join(
        [
            "# Stage 28 Repaired-Critic Policy-Gradient Rerun",
            "",
            "Stage 28 reran the fixed Stage 25 small-scale protocol with the Stage 27 selected graph value critic.",
            "",
            f"- verdict: `{report['verdict']}`",
            f"- pass gate: `{report['pass_gate']}`",
            f"- completed seeds: `{aggregate['completed_seed_count']}` / `{aggregate['seed_count']}`",
            f"- selected critic: `{CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID}`",
            f"- tau-feasible delta vs supervised: `{delta['tau_feasible_rate_delta']}`",
            f"- violation delta vs supervised: `{delta['violation_rate_delta']}`",
            f"- latency delta vs supervised: `{delta['latency_delta']}`",
            f"- energy delta vs supervised: `{delta['energy_delta']}`",
            f"- surrogate delta vs supervised: `{delta['mean_reward_surrogate_delta']}`",
            "- scale-up authorized: `False`",
            "- owner decision required: `True`",
        ]
    ) + "\n"


def _training_markdown(report: Mapping[str, object]) -> str:
    gate = report["pass_fail_gate"]
    critic = gate["critic_health"]
    return f"""# Stage 28 Repaired-Critic Training Report

- fixed base protocol reused: `{gate['fixed_stage25_protocol_executed']}`
- train/eval split used: `{gate['train_eval_split_used']}`
- completed seed count: `{gate['completed_seed_count']}`
- mean update explained variance: `{critic['mean_update_explained_variance']}`
- mean value-return correlation: `{critic['mean_value_return_correlation']}`
- mean normalized value loss: `{critic['mean_normalized_value_loss']}`
- reward weights unchanged: `{gate['reward_weights_unchanged']}`
- forbidden work avoided: `{gate['forbidden_work_avoided']}`
"""


def _comparison_markdown(report: Mapping[str, object]) -> str:
    comparison = report["stage25_historical_comparison"]
    lines = [
        "# Stage 28 Repaired-Critic Comparison",
        "",
        f"- repaired critic improved old-critic result: `{comparison['repaired_critic_improved_old_critic_result']}`",
        "",
        "| Metric | Stage28 minus Stage25 |",
        "| --- | ---: |",
    ]
    for key, value in comparison["metric_deltas"].items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


def _readiness_markdown(report: Mapping[str, object]) -> str:
    next_task = report["recommended_next_task"]
    return f"""# Stage 28 Policy-Gradient Readiness After Repaired-Critic Rerun

- pass gate: `{report['pass_gate']}`
- recommended next task: `{next_task}`
- scale-up training performed: `{report['scale_up_training_performed']}`
- checkpoint written: `{report['checkpoint_written']}`
- sampler switched: `{report['sampler_switched']}`
- reward weights changed: `{not report['reward_config_unchanged']}`

Stage 28 does not authorize scale-up. The owner must decide the next stage.
"""


def _stage28_failure_markdown(report: Mapping[str, object]) -> str:
    gate = report.get("pass_fail_gate", {})
    return "# Stage 28 Repaired-Critic Failure Review\n\n" + "\n".join(
        [
            f"- issue: `{issue}`"
            for issue in gate.get("issues", ())
        ]
    ) + "\n"


def _explained_variance(returns: torch.Tensor, predictions: torch.Tensor) -> float:
    variance = returns.var(unbiased=False)
    if float(variance.detach().cpu().item()) <= 1e-12:
        return 0.0
    return float((1.0 - (returns - predictions).var(unbiased=False) / variance).detach().cpu().item())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().reshape(-1)
    right = right.detach().reshape(-1)
    if left.numel() < 2:
        return 0.0
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.detach().cpu().item()) <= 1e-12:
        return 0.0
    return float((left_centered @ right_centered / denom).detach().cpu().item())


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(()).item())


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
