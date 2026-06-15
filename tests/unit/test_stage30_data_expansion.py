import json
from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import data_scale_repair


ROOT = Path(__file__).resolve().parents[2]
STAGE26_REPORT = (
    ROOT
    / "result_save"
    / "stage26_full_system_health_diagnostic"
    / "stage26_no_training_full_system_health_diagnostic_v1"
    / "stage26_full_system_health_report.json"
)


def _stage26_report() -> dict[str, object]:
    return json.loads(STAGE26_REPORT.read_text(encoding="utf-8"))


def test_stage30_data_scale_repair_reports_owner_gated_data_blocker() -> None:
    repair = data_scale_repair(_stage26_report())

    assert repair["repair_class"] == "data_scale_assessment"
    assert repair["scenario_data_expanded"] is False
    assert repair["owner_scope_required"] is True
    assert repair["sufficient_for_large_scale"] is False


def test_stage30_data_scale_repair_keeps_duplicate_context_risk_visible() -> None:
    repair = data_scale_repair(_stage26_report())

    assert repair["duplicate_context_risk"] == "high"
    assert "train/eval/test split" in repair["recommended_expansion"]
    assert repair["status_from_stage26"] in {"PASS", "WARN", "FAIL"}
