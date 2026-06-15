import json
from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import (
    STAGE30_TAU_REQUIREMENT_MIN,
    surrogate_objective_alignment_repair,
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


def test_stage30_surrogate_objective_candidate_improves_rank_alignment_without_weight_sweep() -> None:
    repair = surrogate_objective_alignment_repair(_stage28_report())

    assert repair["repair_class"] == "surrogate_objective_alignment"
    assert repair["alignment_improved"] is True
    assert repair["after_inversion_count"] <= repair["before_inversion_count"]
    assert repair["weight_sweep_performed"] is False
    assert repair["tau_requirement_min"] == STAGE30_TAU_REQUIREMENT_MIN


def test_stage30_surrogate_objective_candidate_is_not_active_training_change() -> None:
    repair = surrogate_objective_alignment_repair(_stage28_report())

    assert repair["formula_structure_changed"] is True
    assert repair["active_training_surrogate_changed"] is False
    assert repair["owner_activation_required"] is True


def test_stage30_surrogate_objective_candidate_keeps_feasible_above_infeasible_on_average() -> None:
    repair = surrogate_objective_alignment_repair(_stage28_report())
    gap = repair["feasible_infeasible_gap_after"]

    assert gap["available"] is True
    assert gap["gap"] > 0.0
