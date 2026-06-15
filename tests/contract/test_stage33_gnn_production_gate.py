from types import SimpleNamespace

from marl_topology.models import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
    LOCAL_MLP_EDGE_SCORER_MODEL_ID,
    build_model_registry,
    active_stage33_production_gnn_entries,
)
from marl_topology.training.production_mappo_adapter import (
    select_stage33_winner,
    stage33_production_gate,
)


def _report(model_id: str, tau: float, collapse: float = 0.0) -> dict[str, object]:
    return {
        "model_id": model_id,
        "config_id": "low_lr_with_warmup",
        "aggregate": {
            "mappo_final_eval_mean": {
                "tau_feasible_rate": tau,
                "mean_reward_surrogate": tau,
                "top_proposal_rejection_rate": 0.0,
            },
        },
        "collapse_summary": {"collapse_rate": collapse},
    }


def test_stage33_registry_has_exactly_one_active_production_gnn_and_mlp_is_diagnostic() -> None:
    entries = active_stage33_production_gnn_entries()
    registry = build_model_registry()

    assert len(entries) == 1
    assert entries[0].model_id == ACTIVE_STAGE33_GNN_MODEL_ID
    assert ACTIVE_STAGE33_GNN_MODEL_ID == LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
    assert registry[LOCAL_MLP_EDGE_SCORER_MODEL_ID].diagnostic_baseline_only is True
    assert registry[LOCAL_MLP_EDGE_SCORER_MODEL_ID].active_for_stage33_production is False


def test_stage33_selection_excludes_mlp_even_when_mlp_is_strong() -> None:
    reports = [
        _report(LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID, tau=0.9),
        _report(LOCAL_MLP_EDGE_SCORER_MODEL_ID, tau=1.0),
    ]

    selection = select_stage33_winner(reports)

    assert selection["selected_model_id"] == LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
    assert selection["mlp_promoted"] is False


def test_stage33_gate_passes_only_when_selected_gnn_meets_mlp_diagnostic_floor() -> None:
    dataset = SimpleNamespace(
        quality_report={
            "all_required_families_present": True,
            "stage3_stage4_evaluator_used": True,
        }
    )
    passing_reports = [
        _report(LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID, tau=1.0),
        _report(LOCAL_MLP_EDGE_SCORER_MODEL_ID, tau=1.0),
    ]
    passing_selection = select_stage33_winner(passing_reports)
    passing_gate = stage33_production_gate(
        model_config_reports=passing_reports,
        selection=passing_selection,
        reward_config_unchanged=True,
        dataset=dataset,
    )
    failing_reports = [
        _report(LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID, tau=0.5),
        _report(LOCAL_MLP_EDGE_SCORER_MODEL_ID, tau=1.0),
    ]
    failing_selection = select_stage33_winner(failing_reports)
    failing_gate = stage33_production_gate(
        model_config_reports=failing_reports,
        selection=failing_selection,
        reward_config_unchanged=True,
        dataset=dataset,
    )

    assert passing_gate["passed"] is True
    assert passing_gate["selected_model_id"] == LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
    assert passing_gate["gnn_mandatory_direction_preserved"] is True
    assert failing_gate["passed"] is False
    assert "selected_gnn_below_mlp_diagnostic_mean_tau_feasible_rate" in failing_gate["issues"]
