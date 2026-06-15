"""Stage 31 Phase F: integrated readiness training, eval, and blocker scorecard."""

from __future__ import annotations

import pytest

from marl_topology.data.stage31_production_dataset import build_production_dataset
from marl_topology.data.stage31_scenario_generator import ProductionScenarioConfig
from marl_topology.training.stage31_readiness import (
    Stage31ReadinessConfig,
    build_readiness_scorecard,
    train_readiness,
)


@pytest.fixture(scope="module")
def readiness():
    dataset = build_production_dataset(
        ProductionScenarioConfig(seed=31, scenario_count=100)
    )
    result = train_readiness(
        dataset, Stage31ReadinessConfig(seed=31, warm_start_epochs=15, epochs=8)
    )
    scorecard = build_readiness_scorecard(result, dataset)
    return dataset, result, scorecard


def test_projection_friction_is_eliminated(readiness) -> None:
    _dataset, result, _scorecard = readiness
    assert result.after_eval["projection_rejection_rate"] == 0.0
    assert result.test_eval["projection_rejection_rate"] == 0.0


def test_training_improves_feasibility_over_random_init(readiness) -> None:
    _dataset, result, _scorecard = readiness
    assert result.after_eval["tau_feasible_rate"] > result.before_eval["tau_feasible_rate"]
    # Surrogate signal improves (less negative) after training.
    assert result.after_eval["mean_surrogate_signal"] > result.before_eval["mean_surrogate_signal"]


def test_keep_best_makes_fine_tune_non_destructive(readiness) -> None:
    _dataset, result, _scorecard = readiness
    # Policy-gradient fine-tune never lands below the warm-started policy.
    assert (
        result.after_eval["mean_surrogate_signal"]
        >= result.warm_start_eval["mean_surrogate_signal"] - 1e-9
    )


def test_reliability_is_not_traded_for_resources(readiness) -> None:
    _dataset, result, _scorecard = readiness
    assert result.after_eval["violation_rate"] <= result.before_eval["violation_rate"] + 1e-9


def test_scorecard_resolves_all_blockers(readiness) -> None:
    _dataset, _result, scorecard = readiness
    assert scorecard["all_stage26_30_blockers_resolved"] is True
    assert scorecard["verdict"] == "ready_for_production_training_scale_up"
    for key in (
        "B0_tau_feasibility_data",
        "B1_reward_objective_alignment",
        "B2_data_scale",
        "B3_projection_friction",
        "B4_reliability_margin",
        "learning_signal",
    ):
        assert scorecard["components"][key]["status"] == "RESOLVED"


def test_surrogate_alignment_is_tight(readiness) -> None:
    _dataset, _result, scorecard = readiness
    assert scorecard["inversion_rate"] < 0.06
