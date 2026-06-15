"""Bug B fix: a KL trust-region overshoot is a per-update epoch early-stop (standard
PPO target-KL), NOT a terminal run-killer.

The stage33 MAPPO loop (Stage33ProductionMappoAdapter._run_seed) previously capped
EVERY seed at completed_updates=1 because a single minibatch crossing max_approx_kl
set stop_reason='kl_explosion' and aborted the whole seed. The fix:

* the inner minibatch loop early-stops the epoch (kl_early_stop) instead of
  setting a terminal stop_reason, and
* the shared collapse helpers are called with a KL-disabled config
  (``replace(config, max_approx_kl=inf)``) so they still flag genuine collapse
  (entropy / empty-graph / critic divergence) without the KL verdict.

These tests pin that semantics at the helper level (fast, no training loop).
"""

from dataclasses import replace

from marl_topology.training.mappo.stage28_repaired_critic_pilot import (
    _eval_stop_reason,
    _training_stop_reason,
)
from marl_topology.training.production_mappo_adapter import Stage33GNNStabilityConfig

_SUPERVISED = {"mean_entropy": 1.0, "tau_feasible_rate": 0.5, "violation_rate": 0.5}


def _update_record(**overrides) -> dict[str, float]:
    record = {
        "total_loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "approx_kl": 99.0,  # a massive trust-region overshoot
        "empty_graph_rate": 0.0,
        "full_graph_rate": 0.0,
        "mean_entropy": 1.0,
    }
    record.update(overrides)
    return record


def test_default_config_still_flags_a_kl_overshoot() -> None:
    # Control: under the unmodified config the overshoot IS a kl_explosion. This
    # proves the knob we change (max_approx_kl) is the one that gates it.
    config = Stage33GNNStabilityConfig()
    assert (
        _training_stop_reason(
            _update_record(),
            supervised_eval=_SUPERVISED,
            config=config,
            initial_entropy=1.0,
        )
        == "kl_explosion"
    )


def test_kl_disabled_config_does_not_terminate_on_overshoot() -> None:
    # The fix path: with max_approx_kl=inf, even a huge KL is no longer a stop
    # reason -> the seed keeps training instead of dying at update 1.
    stop_check = replace(Stage33GNNStabilityConfig(), max_approx_kl=float("inf"))
    assert (
        _training_stop_reason(
            _update_record(),
            supervised_eval=_SUPERVISED,
            config=stop_check,
            initial_entropy=1.0,
        )
        is None
    )


def test_genuine_collapse_still_caught_with_kl_disabled() -> None:
    # Disabling the KL verdict must NOT blind the helper to real collapse: an
    # entropy collapse (and an empty-graph collapse) are still terminal even when
    # the KL is huge.
    stop_check = replace(Stage33GNNStabilityConfig(), max_approx_kl=float("inf"))
    assert (
        _training_stop_reason(
            _update_record(mean_entropy=0.0),
            supervised_eval=_SUPERVISED,
            config=stop_check,
            initial_entropy=1.0,
        )
        == "entropy_collapse"
    )
    assert (
        _training_stop_reason(
            _update_record(empty_graph_rate=0.99),
            supervised_eval=_SUPERVISED,
            config=stop_check,
            initial_entropy=1.0,
        )
        == "empty_graph_collapse"
    )


def test_eval_stop_reason_also_drops_kl_but_keeps_real_signals() -> None:
    stop_check = replace(Stage33GNNStabilityConfig(), max_approx_kl=float("inf"))
    healthy_eval = {
        "tau_feasible_rate": 0.5,
        "violation_rate": 0.5,
        "empty_graph_rate": 0.0,
        "full_graph_rate": 0.0,
        "mean_entropy": 1.0,
    }
    # huge KL alone -> not a stop reason under the disabled config
    assert (
        _eval_stop_reason(
            supervised_eval=_SUPERVISED,
            current_eval=healthy_eval,
            current_update=_update_record(),
            config=stop_check,
            initial_entropy=1.0,
        )
        is None
    )
    # a real tau-feasible degradation is still caught
    degraded_eval = {**healthy_eval, "tau_feasible_rate": 0.0}
    assert (
        _eval_stop_reason(
            supervised_eval=_SUPERVISED,
            current_eval=degraded_eval,
            current_update=_update_record(),
            config=stop_check,
            initial_entropy=1.0,
        )
        == "eval_tau_feasible_rate_degraded"
    )
