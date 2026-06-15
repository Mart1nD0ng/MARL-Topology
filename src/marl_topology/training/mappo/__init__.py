"""Critic-integrated clipped policy micro-loop utilities."""

from .advantages import AdvantageConfig, AdvantageResult, compute_gae_returns
from .losses import (
    ClippedPolicyValueLossInputs,
    ClippedPolicyValueLossResult,
    clipped_policy_value_loss,
)
from .rollout import RolloutBatch, RolloutTransition, build_rollout_batch
from .trainer import (
    STAGE24_PASS_VERDICT,
    STAGE24_STAGE_ID,
    Stage24LoopConfig,
    run_stage24_critic_integrated_micro_loop,
    run_stage24_preflight,
)
from .stage25_pilot import (
    STAGE25_BASE_CONFIG_ID,
    STAGE25_FAIL_VERDICT,
    STAGE25_PASS_VERDICT,
    STAGE25_STAGE_ID,
    Stage25PilotBaseConfig,
    build_stage25_manifest,
    run_stage25_preflight,
    run_stage25_small_scale_formal_mappo_pilot,
    write_stage25_training_artifacts,
)

__all__ = [
    "AdvantageConfig",
    "AdvantageResult",
    "ClippedPolicyValueLossInputs",
    "ClippedPolicyValueLossResult",
    "RolloutBatch",
    "RolloutTransition",
    "STAGE24_PASS_VERDICT",
    "STAGE24_STAGE_ID",
    "STAGE25_BASE_CONFIG_ID",
    "STAGE25_FAIL_VERDICT",
    "STAGE25_PASS_VERDICT",
    "STAGE25_STAGE_ID",
    "Stage24LoopConfig",
    "Stage25PilotBaseConfig",
    "build_rollout_batch",
    "build_stage25_manifest",
    "clipped_policy_value_loss",
    "compute_gae_returns",
    "run_stage24_critic_integrated_micro_loop",
    "run_stage24_preflight",
    "run_stage25_preflight",
    "run_stage25_small_scale_formal_mappo_pilot",
    "write_stage25_training_artifacts",
]
