import json
from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import (
    STAGE30_TAU_REQUIREMENT_MIN,
    reliability_margin_repair,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE28_REPORT = (
    ROOT
    / "result_save"
    / "stage28_repaired_critic_mappo_rerun"
    / "stage28_repaired_critic_stage25_base_protocol"
    / "training_report.json"
)


def _stage28_report() -> dict[str, object]:
    return json.loads(STAGE28_REPORT.read_text(encoding="utf-8"))


def test_stage30_reliability_margin_monitor_keeps_tau_fixed() -> None:
    repair = reliability_margin_repair(_stage28_report())

    assert repair["repair_class"] == "reliability_margin_monitor"
    assert repair["tau_requirement_min"] == STAGE30_TAU_REQUIREMENT_MIN
    assert repair["tau_changed"] is False
    assert repair["safety_buffer_threshold_for_monitoring_only"] > STAGE30_TAU_REQUIREMENT_MIN


def test_stage30_reliability_margin_blocks_scale_readiness_on_stricter_gate() -> None:
    repair = reliability_margin_repair(_stage28_report())

    assert repair["tau_feasible_rate_delta"] < -0.02
    assert repair["violation_rate_delta"] > 0.02
    assert repair["readiness_margin_passed"] is False


def test_stage30_reliability_margin_metrics_are_bounded() -> None:
    repair = reliability_margin_repair(_stage28_report())

    assert -1.0 <= repair["min_margin"] <= 1.0
    assert 0.0 <= repair["near_zero_margin_ratio"] <= 1.0
    assert 0.0 <= repair["negative_margin_ratio"] <= 1.0
    assert repair["margin_std"] >= 0.0
