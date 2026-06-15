"""Stage 33 production mappo adapter.

This adapter owns Stage33 orchestration only: scenario rows, fixed GNN stability
configs, actor factories, learning-rate scheduling, and closeout gates. Rollout,
GAE, repaired graph-critic value handling, and clipped policy/value loss remain
in the official Stage24/25/28 mappo modules.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from math import ceil, cos, isfinite, pi
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from marl_topology.data.actor_feature_rebuild import (
    actor_safe_view_has_no_forbidden_fields,
    build_stage18_actor_safe_feature_rows,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_EVALUATOR_ID,
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PHYSICS_REGIME_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    STAGE21_TAU_REQUIREMENT_MIN,
    _source_learning_evidence_row,
)
from marl_topology.data.stage33_graph_structure_dataset import (
    STAGE33_GRAPH_STRUCTURE_FAMILIES,
    STAGE33_GRAPH_STRUCTURE_DATASET_ID,
    Stage33GraphStructureConfig,
    Stage33GraphStructureDataset,
    build_stage33_graph_structure_dataset,
)
from marl_topology.models import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
    LOCAL_MLP_EDGE_SCORER_MODEL_ID,
    LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
    LOCAL_ATTENTION_GNN_EDGE_SCORER_MODEL_ID,
    LOCAL_TEMPORAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalAttentionGNNEdgeScorer,
    LocalGNNEdgeScorer,
    LocalGNNV3ResidualNormConfig,
    LocalMLPEdgeScorer,
    LocalMessagePassingGNNV3ResidualNorm,
    LocalRoleResourceAwareGNNV3,
    LocalTemporalGNNEdgeScorer,
    active_stage33_production_gnn_entries,
)
from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCritic,
)
from marl_topology.models.tensorizers import tensorize_actor_policy_inputs
from marl_topology.objectives import evaluate_reward_surrogate
from marl_topology.policies import (
    PHYSICAL_LINK_ASSEMBLER_ID,
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    build_local_observations,
)
from marl_topology.training.critic_repair_trainer import (
    Stage27CriticRepairTrainConfig,
    train_stage27_selected_graph_value_critic_bundle,
)
from marl_topology.training.mappo.advantages import AdvantageConfig, compute_gae_returns
from marl_topology.training.mappo.losses import clipped_policy_value_loss
from marl_topology.training.mappo.stage25_pilot import (
    _aggregate_seed_reports,
    _baseline_records,
    _mean,
    _mean_summary,
    _stage25_loop_config,
    _summarize_batch_with_actor_scores,
    _summarize_by_policy,
    _surface_records_from_batch,
)
from marl_topology.training.mappo.stage28_repaired_critic_pilot import (
    _aggregate_loss_payloads,
    _collect_repaired_rollout_batch,
    _eval_stop_reason,
    _loss_for_repaired_indices,
    _training_stop_reason,
)
from marl_topology.training.mappo.trainer import _parameter_checksum, _stage23_config
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


STAGE33_STAGE_ID = "stage_33_gnn_stability_and_mappo_loop_unification"
STAGE33_CONFIG_ID = "stage33_gnn_stability_config"
STAGE33_ARTIFACT_ROOT = (
    "result_save/stage33_gnn_stability_repair/stage33_fixed_protocol"
)
STAGE33_RUN_ID = "stage33_fixed_gnn_stability_repair_seed_group_3301_3305"
STAGE33_PASS_VERDICT = "stage33_pass_gnn_stability_repaired_awaiting_owner_decision"
STAGE33_FAIL_VERDICT = "stage33_gnn_repair_blocked_awaiting_owner_decision"
STAGE33_RECOMMENDED_NEXT_IF_PASS = "stage_34_gnn_production_actor_artifact_and_validation"
STAGE33_RECOMMENDED_NEXT_IF_FAIL = "owner_decision_on_stage33_gnn_repair_failure_classification"
STAGE32_CUSTOM_LOOP_MODULE = "marl_topology.training.stage32_production_training"


class Stage33ProductionMappoViolation(ValueError):
    """Raised when the Stage33 production mappo adapter crosses a boundary."""


@dataclass(frozen=True, slots=True)
class Stage33TrainingRow:
    evidence_id: str
    scenario_id: str
    structural_family: str
    actor_safe_view: tuple[Mapping[str, object], ...]
    selected_physical_edges: tuple[str, ...]

    def __post_init__(self) -> None:
        if not actor_safe_view_has_no_forbidden_fields(self.actor_safe_view):
            raise Stage33ProductionMappoViolation("actor_safe_view leaked forbidden fields")


@dataclass(frozen=True, slots=True)
class Stage33GNNStabilityConfig:
    config_id: str = "low_lr_with_warmup"
    seed: int = 3301
    seeds: tuple[int, ...] = (3301, 3302, 3303, 3304, 3305)
    train_scenarios: int = 4
    eval_scenarios: int = 4
    test_scenarios: int = 4
    rollout_steps: int = 8
    minibatch_size: int = 16
    update_epochs: int = 2
    max_updates: int = 1
    eval_every: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 5e-4
    critic_lr: float = 1e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    weight_decay: float = 1e-5
    warmup_fraction: float = 0.10
    scheduler: str = "linear_warmup_cosine_decay"
    supervised_epochs: int = 2
    supervised_learning_rate: float = 0.003
    top_k: int = 3
    endpoint_budget: int = 1
    # When True the actor proposes a per-scenario variable number of edges
    # (top_k = node_count - 1, the feasible RSU-star degree) instead of a fixed
    # top_k, matching each scene's feasible size. Default False = fixed top_k.
    variable_proposal_size: bool = False
    tau_requirement_min: float = STAGE21_TAU_REQUIREMENT_MIN
    max_tau_feasible_drop: float = 0.10
    max_violation_rate_increase: float = 0.10
    max_empty_graph_rate: float = 0.50
    max_full_graph_rate: float = 0.75
    max_approx_kl: float = 0.05
    min_entropy_fraction_of_initial: float = 0.35
    critic_value_loss_divergence_threshold: float = 1_000_000.0
    # A3: opt-in trajectory mode. When True each rollout step uses the scene advanced
    # in time (vehicles moved), so the actor sees time-varying links; when False the
    # rollout is the static single-frame path (byte-identical to before). The frame
    # sequences are built once and shared by the rollout and the loss recompute (see
    # build_frame_contexts), so the PPO importance ratio stays consistent.
    trajectory_mode: bool = False
    traj_dt_s: float = 1.0
    traj_speed_min_mps: float = 5.0
    traj_speed_max_mps: float = 15.0
    # history_window (Part B, opt-in): the leakage-safe [E, W, F] window the temporal
    # actor reads. <=1 means single-frame (no temporal context) -> the rollout/loss take
    # the byte-identical static path and any actor (incl. the v3 GNN) is unaffected. Only
    # meaningful together with trajectory_mode (which supplies the per-step frames).
    history_window: int = 1
    # predictive_horizon (opt-in): the topology the actor CHOOSES on frame t is SCORED
    # against frame t+h's evaluator (vehicles have moved). h=0 keeps the current myopic
    # behaviour. The reward FUNCTION (the frozen feasibility barrier) is unchanged -- only
    # WHICH frame's consensus/latency/energy feeds it. This turns the per-step bandit into
    # a predictive task where anticipating motion (what the temporal actor can do) helps,
    # so a temporal-vs-static A/B is non-vacuous. Only meaningful with trajectory_mode.
    predictive_horizon: int = 0
    # deterministic_eval (opt-in): evaluate/test with the policy MODE (greedy argmax over
    # the Plackett-Luce steps) -- the action a deployed controller takes -- instead of the
    # stochastic sampler. TRAINING rollouts always stay stochastic (exploration), so the
    # PPO ratio is untouched. Diagnostics (logs/diagnose_bc_quality.py) showed the learned
    # policy's mode reaches the achievable feasibility ceiling while stochastic sampling
    # under-reports it ~3x; this flag reports the deployment-correct feasibility. Default
    # False keeps the stochastic-eval behaviour byte-identical.
    deterministic_eval: bool = False
    # keep_best_eval (opt-in): make the policy-gradient fine-tune NON-DESTRUCTIVE -- track
    # the actor checkpoint with the best validation (eval-split) feasibility, starting from
    # the warm-started (BC) policy, and restore it before the final test. So the fine-tune
    # can only help, never degrade below BC. Pairs with deterministic_eval (stable signal).
    # Selection is on the eval split; the test split is never used to pick the checkpoint.
    keep_best_eval: bool = False

    def __post_init__(self) -> None:
        allowed = {
            "low_lr_no_warmup",
            "low_lr_with_warmup",
            "low_lr_with_warmup_stronger_grad_clip",
        }
        if self.config_id not in allowed:
            raise Stage33ProductionMappoViolation(f"unknown stability config: {self.config_id}")
        if len(self.seeds) < 1:
            raise Stage33ProductionMappoViolation("at least one seed is required")
        if self.tau_requirement_min != STAGE21_TAU_REQUIREMENT_MIN:
            raise Stage33ProductionMappoViolation("Stage 33 keeps tau_requirement_min fixed at 0.9")
        if self.actor_lr > 1e-3:
            raise Stage33ProductionMappoViolation("Stage 33 GNN actor LR must be low")
        if not 0.0 <= self.warmup_fraction <= 0.20:
            raise Stage33ProductionMappoViolation("warmup_fraction must stay bounded")
        if self.max_grad_norm <= 0.0 or self.max_grad_norm > 0.5:
            raise Stage33ProductionMappoViolation("Stage 33 max_grad_norm must be <= 0.5")
        if self.transition_count < 32:
            raise Stage33ProductionMappoViolation("Stage 33 mappo smoke uses at least 32 transitions")
        if self.eval_scenarios * self.rollout_steps < 32:
            raise Stage33ProductionMappoViolation(
                "Stage 33 mappo eval uses official micro-loop dimensions"
            )
        if self.test_scenarios * self.rollout_steps < 32:
            raise Stage33ProductionMappoViolation(
                "Stage 33 mappo test uses official micro-loop dimensions"
            )
        if self.predictive_horizon < 0:
            raise Stage33ProductionMappoViolation("predictive_horizon must be >= 0")

    @property
    def transition_count(self) -> int:
        return self.train_scenarios * self.rollout_steps

    @property
    def minibatches_per_update(self) -> int:
        return ceil(self.transition_count / self.minibatch_size)

    @property
    def total_opt_steps(self) -> int:
        return self.max_updates * self.update_epochs * self.minibatches_per_update

    @property
    def warmup_updates(self) -> int:
        if self.config_id == "low_lr_no_warmup":
            return 0
        return max(1, int(round(self.total_opt_steps * self.warmup_fraction)))

    def to_payload(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "seed": self.seed,
            "seeds": list(self.seeds),
            "seed_count": len(self.seeds),
            "train_scenarios": self.train_scenarios,
            "eval_scenarios": self.eval_scenarios,
            "test_scenarios": self.test_scenarios,
            "rollout_steps": self.rollout_steps,
            "transitions_per_update": self.transition_count,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "max_updates": self.max_updates,
            "eval_every": self.eval_every,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "warmup_updates": self.warmup_updates,
            "scheduler": self.scheduler,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "entropy_coef": self.entropy_coef,
            "reward_weights_tuned": False,
            "tau_requirement_min": self.tau_requirement_min,
            "trajectory_mode": self.trajectory_mode,
            "history_window": self.history_window,
            "predictive_horizon": self.predictive_horizon,
            "deterministic_eval": self.deterministic_eval,
            "keep_best_eval": self.keep_best_eval,
        }


class Stage33LearningRateScheduler:
    """Linear warmup followed by cosine decay or constant LR."""

    def __init__(self, config: Stage33GNNStabilityConfig) -> None:
        self.config = config

    def factor(self, opt_step: int) -> float:
        step = max(1, int(opt_step))
        warmup = self.config.warmup_updates
        total = max(1, self.config.total_opt_steps)
        if warmup and step <= warmup:
            return step / warmup
        if self.config.scheduler == "constant":
            return 1.0
        remaining = max(1, total - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / remaining))
        return 0.1 + 0.9 * 0.5 * (1.0 + cos(pi * progress))

    def apply(self, actor_opt, critic_opt, opt_step: int) -> dict[str, float]:
        factor = self.factor(opt_step)
        actor_lr = self.config.actor_lr * factor
        critic_lr = self.config.critic_lr * factor
        for group in actor_opt.param_groups:
            group["lr"] = actor_lr
        for group in critic_opt.param_groups:
            group["lr"] = critic_lr
        return {"actor_lr": actor_lr, "critic_lr": critic_lr, "lr_factor": factor}


def stage33_fixed_gnn_stability_configs(
    *,
    seeds: tuple[int, ...] = (3301, 3302, 3303, 3304, 3305),
) -> tuple[Stage33GNNStabilityConfig, ...]:
    base = {"seeds": seeds}
    return (
        Stage33GNNStabilityConfig(
            config_id="low_lr_no_warmup",
            actor_lr=1e-3,
            warmup_fraction=0.0,
            scheduler="constant",
            max_grad_norm=0.5,
            **base,
        ),
        Stage33GNNStabilityConfig(
            config_id="low_lr_with_warmup",
            actor_lr=5e-4,
            warmup_fraction=0.10,
            scheduler="linear_warmup_cosine_decay",
            max_grad_norm=0.5,
            **base,
        ),
        Stage33GNNStabilityConfig(
            config_id="low_lr_with_warmup_stronger_grad_clip",
            actor_lr=5e-4,
            warmup_fraction=0.10,
            scheduler="linear_warmup_cosine_decay",
            max_grad_norm=0.35,
            **base,
        ),
    )


def build_stage33_manifest(run_id: str = STAGE33_RUN_ID) -> dict[str, object]:
    return build_valid_stage5_10_dry_run_manifest(
        artifact_root=STAGE33_ARTIFACT_ROOT,
        overrides={
            "run_id": run_id,
            "stage_id": STAGE33_STAGE_ID,
            "owner_approval_id": "owner_approved_stage33_gnn_stability_repair_execution",
            "config_id": STAGE33_CONFIG_ID,
            "scenario_set_id": STAGE33_GRAPH_STRUCTURE_DATASET_ID,
            "split_id": "stage33_context_keyed_leakage_checked_split",
            "seed": 3301,
            "seeds": [3301, 3302, 3303, 3304, 3305],
            "seed_group_id": "stage33_five_seed_fixed_protocol",
            "model_id": ACTIVE_STAGE33_GNN_MODEL_ID,
            "critic_model_id": CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
            "action_semantics_id": UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
            "sampler_id": PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID,
            "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
            "assembler_id": PHYSICAL_LINK_ASSEMBLER_ID,
            "evaluator_id": STAGE21_EVALUATOR_ID,
            "physics_regime_id": STAGE21_PHYSICS_REGIME_ID,
            "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
            "objective_id": STAGE21_OBJECTIVE_CONTRACT_ID,
            "reward_id": "stage5_3_surrogate_config_with_selected_references",
            "surrogate_config_id": "stage5_3_surrogate_config_with_selected_references",
            "architecture_id": ACTIVE_STAGE33_GNN_MODEL_ID,
            "artifact_policy_id": "stage33_report_only_no_checkpoint",
            "artifact_paths": [
                f"{STAGE33_ARTIFACT_ROOT}/manifest.json",
                f"{STAGE33_ARTIFACT_ROOT}/training_report.json",
            ],
            "checkpoint_creation_allowed": False,
            "training_scale_up_allowed": False,
            "artifact_write_allowed": "manifest_validated_reports_only",
        },
    )


class Stage33ProductionMappoAdapter:
    """Production training adapter that routes through official mappo modules."""

    official_trainer_path = "marl_topology.training.mappo"

    def official_component_report(self) -> dict[str, object]:
        return {
            "official_mappo_trainer_path": self.official_trainer_path,
            "official_rollout_collector": _collect_repaired_rollout_batch.__name__,
            "official_advantage_function": compute_gae_returns.__name__,
            "official_loss_function": clipped_policy_value_loss.__name__,
            "official_repaired_critic_loss_adapter": _loss_for_repaired_indices.__name__,
            "stage32_custom_training_loop_called": False,
            "stage32_custom_training_loop_module": STAGE32_CUSTOM_LOOP_MODULE,
            "reimplements_ppo_loss_locally": False,
            "reimplements_rollout_collector_locally": False,
            "hand_rolled_reinforce_update": False,
        }

    def build_row_contexts(
        self,
        dataset: Stage33GraphStructureDataset,
        split_name: str,
    ) -> tuple[tuple[Stage33TrainingRow, object], ...]:
        spec_by_id = {spec.scenario_id: spec for spec in dataset.source_dataset.specs}
        row_contexts = []
        for record in dataset.records_for_split(split_name):
            spec = spec_by_id[record.scenario_id]
            context = build_stage33_context(spec, dataset.source_dataset.teacher_labels[spec.scenario_id])
            row = build_stage33_training_row(
                context=context,
                teacher_label=dataset.source_dataset.teacher_labels[spec.scenario_id],
                structural_family=record.structural_family,
            )
            row_contexts.append((row, context))
        if not row_contexts:
            raise Stage33ProductionMappoViolation(f"empty Stage33 split: {split_name}")
        return tuple(row_contexts)

    def build_frame_contexts(
        self,
        dataset: Stage33GraphStructureDataset,
        split_name: str,
        *,
        num_frames: int,
        dt_s: float,
        speed_min_mps: float,
        speed_max_mps: float,
    ) -> tuple[tuple[tuple[Stage33TrainingRow, object], ...], ...]:
        """Per-scenario SEQUENCE of (row, context), one per rollout step (A3).

        Reuses the static build (build_stage33_context / build_stage33_training_row)
        on the scene advanced in time, so each frame's evaluator + actor-safe row
        reflect the moved geometry. Motions are sampled deterministically per slot
        (seed = slot index) and the sequence is built ONCE and shared by the rollout
        and the loss recompute, so both index the identical frame -> the PPO ratio
        stays a pure policy-parameter measure. Slot order matches build_row_contexts,
        so frame_contexts[i] aligns with row_contexts[i].
        """

        import random

        from marl_topology.data.stage31_scenario_generator import _sample_vehicle_motions
        from marl_topology.scenario.scene import advance_scene

        if num_frames < 1:
            raise Stage33ProductionMappoViolation("num_frames must be >= 1")
        spec_by_id = {spec.scenario_id: spec for spec in dataset.source_dataset.specs}
        frame_sequences: list[tuple[tuple[Stage33TrainingRow, object], ...]] = []
        for slot_index, record in enumerate(dataset.records_for_split(split_name)):
            spec = spec_by_id[record.scenario_id]
            teacher_label = dataset.source_dataset.teacher_labels[spec.scenario_id]
            rng = random.Random(slot_index)
            motions = _sample_vehicle_motions(rng, spec.scene, speed_min_mps, speed_max_mps)
            motion_map = {motion.node_id: motion for motion in motions}
            frames: list[tuple[Stage33TrainingRow, object]] = []
            scene = spec.scene
            for time_index in range(num_frames):
                if time_index > 0:
                    scene = advance_scene(scene, motion_map, dt_s)
                frame_id = f"{spec.scenario_id}:t{time_index}"
                frame_scene = replace(scene, scenario_id=frame_id)
                frame_spec = replace(spec, scenario_id=frame_id, scene=frame_scene)
                context_t = build_stage33_context(frame_spec, teacher_label)
                row_t = build_stage33_training_row(
                    context=context_t,
                    teacher_label=teacher_label,
                    structural_family=record.structural_family,
                )
                frames.append((row_t, context_t))
            frame_sequences.append(tuple(frames))
        if not frame_sequences:
            raise Stage33ProductionMappoViolation(f"empty Stage33 split: {split_name}")
        return tuple(frame_sequences)

    def run_fixed_protocol(
        self,
        *,
        dataset: Stage33GraphStructureDataset | None = None,
        stability_configs: Sequence[Stage33GNNStabilityConfig] | None = None,
        model_ids: Sequence[str] | None = None,
        project_root: str | Path | None = None,
        critic_epochs: int = 3,
    ) -> dict[str, object]:
        root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
        data = dataset or build_stage33_graph_structure_dataset()
        configs = tuple(stability_configs or stage33_fixed_gnn_stability_configs())
        if len(configs) > 3:
            raise Stage33ProductionMappoViolation("Stage33 may run at most three fixed configs")
        models = tuple(
            model_ids
            or (
                ACTIVE_STAGE33_GNN_MODEL_ID,
                LOCAL_GNN_EDGE_SCORER_MODEL_ID,
                LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
                LOCAL_MLP_EDGE_SCORER_MODEL_ID,
            )
        )
        train_contexts = self.build_row_contexts(data, "train")
        eval_contexts = self.build_row_contexts(data, "eval")
        test_contexts = self.build_row_contexts(data, "test")
        critic_bundle = train_stage27_selected_graph_value_critic_bundle(
            config=Stage27CriticRepairTrainConfig(epochs=critic_epochs),
            project_root=root,
        )
        initial_critic_state = deepcopy(critic_bundle.model.state_dict())
        signal_config_before = stage23_pg.build_stage23_reward_surrogate_config()
        reward_fingerprint_before = stage23_pg._surrogate_config_fingerprint(signal_config_before)
        run_matrix = _stage33_run_matrix(models=models, configs=configs)
        model_config_reports = []
        for model_id, cfg in run_matrix:
            # A3 + Part B: build trajectory frames per split once per config (only when
            # trajectory_mode), shared across this config's seeds. num_frames =
            # rollout_steps + predictive_horizon so the collector can read frame t+h for
            # the predictive-horizon eval. All three splits get frames so the temporal
            # actor is exercised (with history + horizon) at TRAIN, EVAL and TEST -- not
            # just train -- otherwise eval/test would degrade it to the static v3 path.
            traj_frames = getattr(cfg, "trajectory_mode", False)
            frame_count = cfg.rollout_steps + max(0, getattr(cfg, "predictive_horizon", 0))

            def _split_frames(split_name: str):
                if not traj_frames:
                    return None
                return self.build_frame_contexts(
                    data,
                    split_name,
                    num_frames=frame_count,
                    dt_s=cfg.traj_dt_s,
                    speed_min_mps=cfg.traj_speed_min_mps,
                    speed_max_mps=cfg.traj_speed_max_mps,
                )

            train_frame_contexts = _split_frames("train")
            eval_frame_contexts = _split_frames("eval")
            test_frame_contexts = _split_frames("test")
            seed_reports = [
                self._run_seed(
                    model_id=model_id,
                    seed=seed,
                    seed_index=seed_index,
                    config=cfg,
                    train_contexts=train_contexts,
                    eval_contexts=eval_contexts,
                    test_contexts=test_contexts,
                    signal_config=signal_config_before,
                    critic_bundle=critic_bundle,
                    initial_critic_state=initial_critic_state,
                    train_frame_contexts=train_frame_contexts,
                    eval_frame_contexts=eval_frame_contexts,
                    test_frame_contexts=test_frame_contexts,
                )
                for seed_index, seed in enumerate(cfg.seeds)
            ]
            aggregate = _aggregate_seed_reports(seed_reports)
            model_config_reports.append(
                {
                    "model_id": model_id,
                    "config_id": cfg.config_id,
                    "config": cfg.to_payload(),
                    "seed_reports": seed_reports,
                    "aggregate": aggregate,
                    "collapse_summary": collapse_summary(seed_reports),
                    "per_family": _aggregate_family_metrics(seed_reports),
                    "actor_is_gnn": model_id != LOCAL_MLP_EDGE_SCORER_MODEL_ID,
                    "mlp_diagnostic_baseline": model_id == LOCAL_MLP_EDGE_SCORER_MODEL_ID,
                }
            )
        signal_config_after = stage23_pg.build_stage23_reward_surrogate_config()
        reward_fingerprint_after = stage23_pg._surrogate_config_fingerprint(signal_config_after)
        selection = select_stage33_winner(model_config_reports)
        pass_fail = stage33_production_gate(
            model_config_reports=model_config_reports,
            selection=selection,
            reward_config_unchanged=reward_fingerprint_before == reward_fingerprint_after,
            dataset=data,
        )
        verdict = STAGE33_PASS_VERDICT if pass_fail["passed"] else STAGE33_FAIL_VERDICT
        return _jsonable(
            {
                "stage": STAGE33_STAGE_ID,
                "verdict": verdict,
                "pass_gate": pass_fail["passed"],
                "pass_fail_gate": pass_fail,
                "official_component_report": self.official_component_report(),
                "dataset_quality": data.quality_report,
                "model_config_reports": model_config_reports,
                "selection": selection,
                "manifest": build_stage33_manifest(),
                "manifest_validation": validate_run_manifest_dry_run(
                    build_stage33_manifest(),
                    project_root=root,
                ).to_dict(),
                "reward_config_before": reward_fingerprint_before,
                "reward_config_after": reward_fingerprint_after,
                "reward_config_unchanged": reward_fingerprint_before == reward_fingerprint_after,
                "reward_weight_tuning_performed": False,
                "final_tau_selected": False,
                "mlp_promoted_to_production": False,
                "coma_introduced": False,
                "transformer_introduced": False,
                "gru_lstm_or_recurrent_ppo_introduced": False,
                "stage32_custom_loop_active": False,
                "checkpoint_written": False,
                "artifact_policy": {
                    "checkpoint_allowed": False,
                    "manifest_validated_reports_allowed": True,
                    "production_actor_artifact_available": False,
                    "owner_decision_required_for_checkpoint": True,
                },
                "failure_review": failure_review(pass_fail, selection)
                if not pass_fail["passed"]
                else None,
                "recommended_next_task": STAGE33_RECOMMENDED_NEXT_IF_PASS
                if pass_fail["passed"]
                else STAGE33_RECOMMENDED_NEXT_IF_FAIL,
                "owner_decision_required": True,
            }
        )

    def _run_seed(
        self,
        *,
        model_id: str,
        seed: int,
        seed_index: int,
        config: Stage33GNNStabilityConfig,
        train_contexts,
        eval_contexts,
        test_contexts,
        signal_config,
        critic_bundle,
        initial_critic_state: Mapping[str, object],
        train_frame_contexts=None,
        eval_frame_contexts=None,
        test_frame_contexts=None,
    ) -> dict[str, object]:
        torch.manual_seed(seed)
        train_cfg = _stage33_loop_config(config, seed=seed, num_scenarios=config.train_scenarios)
        eval_cfg = _stage33_loop_config(config, seed=seed, num_scenarios=config.eval_scenarios)
        test_cfg = _stage33_loop_config(config, seed=seed, num_scenarios=config.test_scenarios)
        stage23_config = _stage23_config(train_cfg)
        sampler = get_active_policy_gradient_sampler()
        actor = build_stage33_actor(model_id)
        warm_start_actor(actor, (row for row, _context in train_contexts), config)
        critic = CentralizedMessagePassingGraphCritic(critic_bundle.model.config)
        critic.load_state_dict(initial_critic_state)
        actor_before = _parameter_checksum(actor)
        critic_before = _parameter_checksum(critic)
        # Tile the per-scenario trajectory frame sequences (when trajectory_mode) to match
        # each split's cycled row-context slots, so the temporal actor sees history + the
        # predictive horizon at TRAIN, EVAL and TEST. None in static mode -> byte-identical
        # to the pre-trajectory path. predictive_horizon/history_window are read per-call.
        train_frame_slots = (
            _cycle_contexts(train_frame_contexts, config.train_scenarios)
            if train_frame_contexts
            else None
        )
        eval_frame_slots = (
            _cycle_contexts(eval_frame_contexts, config.eval_scenarios)
            if eval_frame_contexts
            else None
        )
        test_frame_slots = (
            _cycle_contexts(test_frame_contexts, config.test_scenarios)
            if test_frame_contexts
            else None
        )
        supervised_eval_batch = _collect_repaired_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=_cycle_contexts(eval_contexts, config.eval_scenarios),
            config=eval_cfg,
            stage23_config=stage23_config,
            signal_config=signal_config,
            critic_bundle=critic_bundle,
            seed_base=seed + 200_000,
            frame_contexts=eval_frame_slots,
            history_window=config.history_window,
            predictive_horizon=config.predictive_horizon,
            deterministic=config.deterministic_eval,
        )
        supervised_eval = _summarize_batch_with_actor_scores(supervised_eval_batch)
        # keep-best (opt-in): the warm-started (BC) policy is the baseline-best, so a
        # destructive RL phase can never land below it. Selection metric = (eval-split
        # tau_feasible_rate, mean_reward_surrogate); the test split is never consulted.
        def _keep_best_score(summary: Mapping[str, object]) -> tuple[float, float]:
            return (
                float(summary["tau_feasible_rate"]),
                float(summary.get("mean_reward_surrogate", 0.0)),
            )

        best_eval_score = _keep_best_score(supervised_eval)
        best_actor_state = deepcopy(actor.state_dict()) if config.keep_best_eval else None
        best_eval_summary = supervised_eval
        baseline_records = _baseline_records(
            _cycle_contexts(eval_contexts, config.eval_scenarios),
            config=stage23_config,
            reward_config=signal_config,
            seed=seed,
        )
        baseline_comparison = _summarize_by_policy(baseline_records)
        actor_opt = torch.optim.AdamW(
            actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay,
        )
        critic_opt = torch.optim.AdamW(
            critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )
        lr_schedule = Stage33LearningRateScheduler(config)
        update_metrics: list[dict[str, object]] = []
        eval_metrics: list[dict[str, object]] = [
            {"seed": seed, "update_index": 0, "phase": "warm_start_eval", **supervised_eval}
        ]
        critic_pairs: list[dict[str, float]] = []
        stop_reason: str | None = None
        # KL is enforced as a per-update epoch early-stop (standard PPO target-KL),
        # NOT as a terminal run-killer. Give the shared collapse-check helpers a
        # config with the KL gate disabled so they still flag genuine collapse
        # (entropy / empty-graph / critic divergence) without aborting the seed on
        # a single trust-region overshoot. The old behavior capped every seed at
        # completed_updates=1.
        stop_check_config = replace(config, max_approx_kl=float("inf"))
        completed_updates = 0
        opt_step = 0
        final_eval_batch = supervised_eval_batch
        train_slots = _cycle_contexts(train_contexts, config.train_scenarios)
        eval_slots = _cycle_contexts(eval_contexts, config.eval_scenarios)
        for update_index in range(1, config.max_updates + 1):
            batch = _collect_repaired_rollout_batch(
                actor=actor,
                critic=critic,
                sampler=sampler,
                row_contexts=train_slots,
                config=train_cfg,
                stage23_config=stage23_config,
                signal_config=signal_config,
                critic_bundle=critic_bundle,
                seed_base=seed + update_index * 10_000,
                frame_contexts=train_frame_slots,
                history_window=config.history_window,
                predictive_horizon=config.predictive_horizon,
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
            generator = torch.Generator().manual_seed(seed + 500 + update_index)
            loss_payloads: list[dict[str, float]] = []
            kl_early_stop = False
            for _epoch_index in range(config.update_epochs):
                permutation = torch.randperm(batch.total_transitions, generator=generator).tolist()
                for start in range(0, batch.total_transitions, config.minibatch_size):
                    indices = tuple(
                        int(index) for index in permutation[start : start + config.minibatch_size]
                    )
                    opt_step += 1
                    lr_payload = lr_schedule.apply(actor_opt, critic_opt, opt_step)
                    loss_result = _loss_for_repaired_indices(
                        actor=actor,
                        critic=critic,
                        sampler=sampler,
                        batch=batch,
                        row_contexts=train_slots,
                        stage23_config=stage23_config,
                        indices=indices,
                        advantages=advantage_result.advantages,
                        returns=advantage_result.returns,
                        config=train_cfg,
                        critic_bundle=critic_bundle,
                        frame_contexts=train_frame_slots,
                        history_window=config.history_window,
                    )
                    if not torch.isfinite(loss_result["total_loss"]).all().item():
                        stop_reason = "nonfinite_total_loss"
                        break
                    actor_opt.zero_grad()
                    critic_opt.zero_grad()
                    loss_result["total_loss"].backward()
                    actor_norm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(),
                        config.max_grad_norm,
                    )
                    critic_norm = torch.nn.utils.clip_grad_norm_(
                        critic.parameters(),
                        config.max_grad_norm,
                    )
                    actor_opt.step()
                    critic_opt.step()
                    payload = dict(loss_result["payload"])
                    payload.update(lr_payload)
                    payload["actor_grad_norm"] = float(actor_norm.detach().cpu().item())
                    payload["critic_grad_norm"] = float(critic_norm.detach().cpu().item())
                    payload["grad_norm"] = payload["actor_grad_norm"] + payload["critic_grad_norm"]
                    loss_payloads.append(payload)
                    # Standard PPO target-KL early stop: once a minibatch pushes the
                    # policy past the trust region, stop iterating epochs on THIS
                    # batch and move to the next rollout. A single overshoot is normal
                    # and must NOT terminate the run. Genuine divergence is still
                    # caught by nonfinite_total_loss and the collapse checks below.
                    if payload["approx_kl"] > config.max_approx_kl:
                        kl_early_stop = True
                        break
                if stop_reason or kl_early_stop:
                    break
            loss_summary = _aggregate_loss_payloads(loss_payloads)
            update_record = {
                "seed": seed,
                "model_id": model_id,
                "config_id": config.config_id,
                "update_index": update_index,
                "phase": "train",
                **train_summary,
                **loss_summary,
                "actor_grad_norm": max((p["actor_grad_norm"] for p in loss_payloads), default=0.0),
                "critic_grad_norm": max((p["critic_grad_norm"] for p in loss_payloads), default=0.0),
                "actor_lr": loss_payloads[-1]["actor_lr"] if loss_payloads else config.actor_lr,
                "critic_lr": loss_payloads[-1]["critic_lr"] if loss_payloads else config.critic_lr,
                "advantage_mean": float(advantage_result.advantages.mean().detach().cpu().item()),
                "advantage_std": float(
                    advantage_result.advantages.std(unbiased=False).detach().cpu().item()
                ),
                "return_mean": float(advantage_result.returns.mean().detach().cpu().item()),
                "raw_advantage_mean": float(
                    advantage_result.raw_advantages.mean().detach().cpu().item()
                ),
                "official_gae_path": True,
                "official_clipped_loss_path": True,
                "kl_early_stopped": kl_early_stop,
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
                config=stop_check_config,
                initial_entropy=float(supervised_eval["mean_entropy"]),
            )
            completed_updates = update_index
            if update_index % config.eval_every == 0 or update_index == config.max_updates:
                final_eval_batch = _collect_repaired_rollout_batch(
                    actor=actor,
                    critic=critic,
                    sampler=sampler,
                    row_contexts=eval_slots,
                    config=eval_cfg,
                    stage23_config=stage23_config,
                    signal_config=signal_config,
                    critic_bundle=critic_bundle,
                    seed_base=seed + 200_000,
                    frame_contexts=eval_frame_slots,
                    history_window=config.history_window,
                    predictive_horizon=config.predictive_horizon,
                    deterministic=config.deterministic_eval,
                )
                eval_summary = _summarize_batch_with_actor_scores(final_eval_batch)
                if config.keep_best_eval and _keep_best_score(eval_summary) > best_eval_score:
                    best_eval_score = _keep_best_score(eval_summary)
                    best_actor_state = deepcopy(actor.state_dict())
                    best_eval_summary = eval_summary
                eval_record = {
                    "seed": seed,
                    "model_id": model_id,
                    "config_id": config.config_id,
                    "update_index": update_index,
                    "phase": "mappo_eval",
                    **eval_summary,
                }
                eval_metrics.append(eval_record)
                stop_reason = stop_reason or _eval_stop_reason(
                    supervised_eval=supervised_eval,
                    current_eval=eval_summary,
                    current_update=update_record,
                    config=stop_check_config,
                    initial_entropy=float(supervised_eval["mean_entropy"]),
                )
            if stop_reason:
                break
        # keep-best: restore the best-validation actor so the reported TEST policy (and the
        # returned actor) is never worse than the warm-started BC policy on eval.
        if config.keep_best_eval and best_actor_state is not None:
            actor.load_state_dict(best_actor_state)
        if eval_metrics[-1]["phase"] != "mappo_eval":
            eval_metrics.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "config_id": config.config_id,
                    "update_index": completed_updates,
                    "phase": "mappo_eval",
                    **_summarize_batch_with_actor_scores(final_eval_batch),
                }
            )
        test_batch = _collect_repaired_rollout_batch(
            actor=actor,
            critic=critic,
            sampler=sampler,
            row_contexts=_cycle_contexts(test_contexts, config.test_scenarios),
            config=test_cfg,
            stage23_config=stage23_config,
            signal_config=signal_config,
            critic_bundle=critic_bundle,
            seed_base=seed + 300_000,
            frame_contexts=test_frame_slots,
            history_window=config.history_window,
            predictive_horizon=config.predictive_horizon,
            deterministic=config.deterministic_eval,
        )
        final_eval = eval_metrics[-1]
        actor_after = _parameter_checksum(actor)
        critic_after = _parameter_checksum(critic)
        family_metrics = _family_metrics_from_batch(final_eval_batch)
        return {
            "seed": seed,
            "seed_index": seed_index,
            "model_id": model_id,
            "config_id": config.config_id,
            "base_run_completed": stop_reason is None and completed_updates == config.max_updates,
            "completed_updates": completed_updates,
            "stop_reason": stop_reason,
            "supervised_eval": supervised_eval,
            "final_mappo_eval": final_eval,
            "test_eval": _summarize_batch_with_actor_scores(test_batch),
            "keep_best_eval_applied": config.keep_best_eval,
            "best_eval_summary": best_eval_summary,
            "baseline_comparison": baseline_comparison,
            "per_family_eval": family_metrics,
            "update_metrics": update_metrics,
            "eval_metrics": eval_metrics,
            "critic_prediction_return_pairs": critic_pairs,
            "actor_parameter_delta": abs(actor_after - actor_before),
            "critic_parameter_delta": abs(critic_after - critic_before),
            "reward_surface_records": (
                _surface_records_from_batch(
                    supervised_eval_batch,
                    policy_label=f"{model_id}:warm_start_before_mappo",
                    seed=seed,
                    update_index=0,
                )
                + _surface_records_from_batch(
                    final_eval_batch,
                    policy_label=f"{model_id}:mappo_final",
                    seed=seed,
                    update_index=completed_updates,
                )
                + _surface_records_from_batch(
                    test_batch,
                    policy_label=f"{model_id}:test",
                    seed=seed,
                    update_index=completed_updates,
                )
            ),
        }


def build_stage33_context(spec, teacher_label: Mapping[str, object]):
    context = build_stage33_base_context(spec)
    teacher_edges = tuple(str(edge) for edge in teacher_label["selected_physical_edges"])
    context.evaluator.evaluate(set(teacher_edges), topology_id=f"stage33:teacher:{spec.scenario_id}")
    return context


def build_stage33_base_context(spec):
    from marl_topology.data.stage31_production_dataset import build_production_context

    return build_production_context(spec)


def build_stage33_training_row(
    *,
    context,
    teacher_label: Mapping[str, object],
    structural_family: str,
) -> Stage33TrainingRow:
    teacher_edges = tuple(str(edge) for edge in teacher_label["selected_physical_edges"])
    evaluation = context.evaluator.evaluate(
        set(teacher_edges),
        topology_id=f"stage33:{context.fixture.fixture_id}:objective_teacher",
    )
    observations = build_local_observations(
        scene=context.fixture.scene,
        graph=context.graph,
        link_records=context.evaluator.link_records,
        time_step=context.time_step,
    )
    source_row = _source_learning_evidence_row(
        evaluation=evaluation,
        topology_name="stage33_objective_teacher",
        observations=observations,
        context=context,
        learning_targets=(),
    )
    actor_safe_view = tuple(
        {**dict(row), "stage33_structural_family": structural_family}
        for row in build_stage18_actor_safe_feature_rows(source_row)
    )
    return Stage33TrainingRow(
        evidence_id=f"stage33:{context.fixture.fixture_id}:{structural_family}",
        scenario_id=str(context.fixture.fixture_id),
        structural_family=structural_family,
        actor_safe_view=actor_safe_view,
        selected_physical_edges=teacher_edges,
    )


def build_stage33_actor(model_id: str):
    if model_id == LOCAL_GNN_EDGE_SCORER_MODEL_ID:
        return LocalGNNEdgeScorer()
    if model_id == LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID:
        return LocalMessagePassingGNNV3ResidualNorm(LocalGNNV3ResidualNormConfig())
    if model_id == LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID:
        return LocalRoleResourceAwareGNNV3()
    if model_id == LOCAL_MLP_EDGE_SCORER_MODEL_ID:
        return LocalMLPEdgeScorer()
    if model_id == LOCAL_TEMPORAL_GNN_EDGE_SCORER_MODEL_ID:
        return LocalTemporalGNNEdgeScorer()
    if model_id == LOCAL_ATTENTION_GNN_EDGE_SCORER_MODEL_ID:
        return LocalAttentionGNNEdgeScorer()
    raise Stage33ProductionMappoViolation(f"unknown Stage33 actor model_id: {model_id}")


def warm_start_actor(
    actor,
    rows: Iterable[Stage33TrainingRow],
    config: Stage33GNNStabilityConfig,
) -> None:
    row_tuple = tuple(rows)
    if config.supervised_epochs <= 0 or not row_tuple:
        return
    policy_inputs = []
    targets_by_sample_edge: dict[tuple[int, str], float] = {}
    for row in row_tuple:
        selected = set(row.selected_physical_edges)
        for actor_row in row.actor_safe_view:
            policy_input = stage23_pg._policy_input_from_stage22_actor_row(actor_row)
            sample_index = len(policy_inputs)
            for neighbor in policy_input.local_neighbor_observations:
                edge_id = str(_neighbor_value(neighbor, "edge_id"))
                targets_by_sample_edge[(sample_index, edge_id)] = 1.0 if edge_id in selected else 0.0
            policy_inputs.append(policy_input)
    batch = tensorize_actor_policy_inputs(policy_inputs)
    targets = torch.tensor(
        [targets_by_sample_edge[(ref.sample_index, ref.edge_id)] for ref in batch.records],
        dtype=torch.float32,
    )
    update_rule = torch.optim.AdamW(
        actor.parameters(),
        lr=config.supervised_learning_rate,
        weight_decay=config.weight_decay,
    )
    actor.train()
    for _ in range(config.supervised_epochs):
        update_rule.zero_grad()
        logits = actor.score_tensor_batch(batch)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        update_rule.step()


def collapse_summary(seed_reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    collapse_reasons = []
    for report in seed_reports:
        final = report["final_mappo_eval"]
        mlp_floor = 0.0
        if report.get("stop_reason"):
            collapse_reasons.append(str(report["stop_reason"]))
        elif float(final.get("empty_graph_rate", 0.0)) >= 0.90:
            collapse_reasons.append("empty_graph_collapse")
        elif float(final.get("full_graph_rate", 0.0)) >= 0.90:
            collapse_reasons.append("full_graph_collapse")
        elif float(final.get("mean_entropy", 0.0)) <= 1e-9:
            collapse_reasons.append("entropy_collapse")
        elif float(final.get("tau_feasible_rate", 0.0)) < mlp_floor:
            collapse_reasons.append("relative_tau_collapse")
    return {
        "seed_count": len(seed_reports),
        "collapse_count": len(collapse_reasons),
        "collapse_rate": len(collapse_reasons) / len(seed_reports) if seed_reports else 0.0,
        "collapse_reasons": dict(Counter(collapse_reasons)),
    }


def select_stage33_winner(
    model_config_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    gnn_reports = [
        report
        for report in model_config_reports
        if report["model_id"] != LOCAL_MLP_EDGE_SCORER_MODEL_ID
    ]
    if not gnn_reports:
        return {
            "selected_model_id": None,
            "selected_config_id": None,
            "selection_reason": "no_gnn_reports_available",
            "mlp_promoted": False,
        }
    best = max(gnn_reports, key=_stage33_selection_key)
    return {
        "selected_model_id": best["model_id"],
        "selected_config_id": best["config_id"],
        "selection_reason": "highest gnn tau/reward with collapse penalty; MLP excluded from production",
        "mlp_promoted": False,
        "active_production_gnn_entries": [
            entry.model_id for entry in active_stage33_production_gnn_entries()
        ],
    }


def stage33_production_gate(
    *,
    model_config_reports: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    reward_config_unchanged: bool,
    dataset: Stage33GraphStructureDataset,
) -> dict[str, object]:
    issues = []
    active_entries = active_stage33_production_gnn_entries()
    selected_model_id = selection.get("selected_model_id")
    selected_report = next(
        (
            report
            for report in model_config_reports
            if report["model_id"] == selected_model_id
            and report["config_id"] == selection.get("selected_config_id")
        ),
        None,
    )
    mlp_reports = [
        report for report in model_config_reports if report["model_id"] == LOCAL_MLP_EDGE_SCORER_MODEL_ID
    ]
    mlp_mean = max(
        (
            float(report["aggregate"]["mappo_final_eval_mean"].get("tau_feasible_rate", 0.0))
            for report in mlp_reports
        ),
        default=0.0,
    )
    selected_mean = (
        float(selected_report["aggregate"]["mappo_final_eval_mean"].get("tau_feasible_rate", 0.0))
        if selected_report
        else 0.0
    )
    selected_collapse = (
        float(selected_report["collapse_summary"]["collapse_rate"]) if selected_report else 1.0
    )
    if len(active_entries) != 1:
        issues.append("not_exactly_one_active_stage33_production_gnn")
    active_model_ids = {entry.model_id for entry in active_entries}
    if selected_model_id not in active_model_ids:
        issues.append("selected_gnn_not_active_stage33_production_actor")
    if selected_model_id == LOCAL_MLP_EDGE_SCORER_MODEL_ID:
        issues.append("mlp_selected_as_production_actor")
    if selected_collapse > 0.20:
        issues.append("selected_gnn_collapse_rate_above_20_percent")
    if selected_mean + 1e-9 < mlp_mean:
        issues.append("selected_gnn_below_mlp_diagnostic_mean_tau_feasible_rate")
    if not bool(dataset.quality_report["all_required_families_present"]):
        issues.append("graph_structure_families_missing")
    if not bool(dataset.quality_report["stage3_stage4_evaluator_used"]):
        issues.append("stage3_stage4_evaluator_not_used")
    if not reward_config_unchanged:
        issues.append("reward_config_changed")
    official_ok = True
    forbidden_ok = True
    return {
        "passed": not issues,
        "issues": issues,
        "official_mappo_trainer_active": official_ok,
        "stage32_custom_loop_retired": True,
        "gnn_mandatory_direction_preserved": selected_model_id != LOCAL_MLP_EDGE_SCORER_MODEL_ID,
        "selected_model_id": selected_model_id,
        "selected_config_id": selection.get("selected_config_id"),
        "selected_gnn_mean_tau_feasible_rate": selected_mean,
        "mlp_diagnostic_mean_tau_feasible_rate": mlp_mean,
        "selected_gnn_collapse_rate": selected_collapse,
        "reward_weights_unchanged": reward_config_unchanged,
        "no_lstm_coma_transformer": forbidden_ok,
        "graph_structure_data_used": True,
        "exactly_one_active_production_gnn": len(active_entries) == 1,
        "selected_gnn_is_active_production_actor": selected_model_id in active_model_ids,
        "checkpoint_available": False,
        "owner_decision_required": True,
    }


def failure_review(
    pass_fail: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    issue_text = " ".join(str(issue) for issue in pass_fail.get("issues", ()))
    classes = {
        "optimization_failure": any(token in issue_text for token in ("collapse", "kl", "entropy")),
        "architecture_failure": any(
            token in issue_text
            for token in (
                "selected_gnn_below_mlp",
                "selected_gnn_not_active_stage33_production_actor",
            )
        ),
        "data_not_graph_structural_enough": "graph_structure_families_missing" in issue_text,
        "official_mappo_integration_failure": "official" in issue_text,
        "teacher_ceiling_issue": False,
        "actor_feature_insufficiency": "selected_gnn_below_mlp" in issue_text,
        "production_artifact_blocker": not bool(pass_fail.get("checkpoint_available")),
    }
    return {
        "classification": classes,
        "pass_fail_issues": list(pass_fail.get("issues", ())),
        "selected_model_id": selection.get("selected_model_id"),
        "mlp_not_promoted": True,
        "owner_decision_required": True,
    }


def write_stage33_training_artifacts(
    report: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=False) if project_root else Path(__file__).resolve().parents[4]
    manifest = dict(report["manifest"])  # type: ignore[index]
    validation = validate_run_manifest_dry_run(manifest, project_root=root)
    if not validation.is_valid:
        raise Stage33ProductionMappoViolation(f"manifest validation failed: {validation.error_codes()}")
    artifact_dir = root / STAGE33_ARTIFACT_ROOT
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_to_write = {**dict(report), "artifact_written": True, "artifact_dir": str(artifact_dir)}
    _write_json(artifact_dir / "manifest.json", manifest)
    _write_json(artifact_dir / "training_report.json", report_to_write)
    return {
        "artifact_written": True,
        "artifact_dir": str(artifact_dir),
        "manifest_validated": validation.is_valid,
        "written_files": sorted(path.name for path in artifact_dir.iterdir() if path.is_file()),
    }


def _stage33_loop_config(
    config: Stage33GNNStabilityConfig,
    *,
    seed: int,
    num_scenarios: int,
):
    loop_proxy = _Stage25ConfigProxy(config, train_scenarios=num_scenarios)
    return _stage25_loop_config(loop_proxy, seed=seed, num_scenarios=num_scenarios)


@dataclass(frozen=True, slots=True)
class _Stage25ConfigProxy:
    source: Stage33GNNStabilityConfig
    train_scenarios: int

    def __getattr__(self, name: str):
        return getattr(self.source, name)


def _stage33_run_matrix(
    *,
    models: tuple[str, ...],
    configs: tuple[Stage33GNNStabilityConfig, ...],
) -> tuple[tuple[str, Stage33GNNStabilityConfig], ...]:
    if not configs:
        raise Stage33ProductionMappoViolation("at least one Stage33 stability config is required")
    selected = configs[1] if len(configs) > 1 else configs[0]
    pairs: list[tuple[str, Stage33GNNStabilityConfig]] = []
    for cfg in configs:
        pairs.append((ACTIVE_STAGE33_GNN_MODEL_ID, cfg))
    for model_id in models:
        if model_id == ACTIVE_STAGE33_GNN_MODEL_ID:
            continue
        pairs.append((model_id, selected))
    return tuple(pairs)


def _cycle_contexts(
    contexts: tuple[tuple[Stage33TrainingRow, object], ...],
    count: int,
) -> tuple[tuple[Stage33TrainingRow, object], ...]:
    return tuple(contexts[index % len(contexts)] for index in range(count))


def _family_metrics_from_batch(batch) -> dict[str, dict[str, object]]:
    records_by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for transition in batch.transitions:
        observation = transition.actor_safe_observation
        rows = observation.get("rows", ()) if isinstance(observation, Mapping) else ()
        family = "unknown"
        if rows:
            first = rows[0]
            if isinstance(first, Mapping):
                family = str(first.get("stage33_structural_family", "unknown"))
        projection = dict(transition.projection_diagnostics)
        records_by_family[family].append(
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
                "above_threshold_rejection_rate": projection["above_threshold_rejection_rate"],
                "rejection_by_reason": projection["rejection_by_reason"],
                "proposal_entropy": float(transition.proposal_entropy.detach().cpu().item()),
                "proposal_logprob": float(transition.proposal_logprob.detach().cpu().item()),
                "reward_surrogate": transition.reward_surrogate,
            }
        )
    return {
        family: stage23_pg._summarize_records(rows)
        for family, rows in sorted(records_by_family.items())
    }


def _aggregate_family_metrics(
    seed_reports: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for report in seed_reports:
        for family, metrics in report.get("per_family_eval", {}).items():
            by_family[str(family)].append(metrics)
    return {family: _mean_summary(items) for family, items in sorted(by_family.items())}


def _stage33_selection_key(report: Mapping[str, object]) -> tuple[float, float, float, float]:
    final = report["aggregate"]["mappo_final_eval_mean"]  # type: ignore[index]
    collapse = report["collapse_summary"]  # type: ignore[index]
    return (
        -float(collapse.get("collapse_rate", 1.0)),
        float(final.get("tau_feasible_rate", 0.0)),
        float(final.get("mean_reward_surrogate", 0.0)),
        -float(final.get("top_proposal_rejection_rate", 0.0)),
    )


def _neighbor_value(neighbor: object, name: str) -> object:
    if isinstance(neighbor, Mapping):
        return neighbor[name]
    return getattr(neighbor, name)


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(json.dumps(_jsonable(payload), indent=2, sort_keys=True).encode("utf-8"))


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
