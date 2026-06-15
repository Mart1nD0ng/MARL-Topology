import json
from pathlib import Path

from marl_topology.training.mappo.stage25_pilot import STAGE25_ARTIFACT_ROOT


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / STAGE25_ARTIFACT_ROOT


def test_stage25_visualization_outputs_are_manifest_approved_and_present() -> None:
    training_report_path = ARTIFACT_DIR / "training_report.json"
    visualization_report_path = ARTIFACT_DIR / "visualization_report.json"
    manifest_path = ARTIFACT_DIR / "manifest.json"

    assert training_report_path.exists()
    assert visualization_report_path.exists()
    assert manifest_path.exists()

    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    visualization_report = json.loads(visualization_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = {Path(path).name for path in manifest["artifact_paths"]}

    assert training_report["visualization_report_generated"] is True
    assert training_report["pass_fail_gate"]["visualization_report_generated"] is True
    assert visualization_report["visualization_artifacts_written"] is True
    assert visualization_report["manifest_validation"]["is_valid"] is True
    assert all(visualization_report["required_views"].values())

    expected_csv = {
        "training_curves.csv",
        "objective_curves.csv",
        "ppo_diagnostics.csv",
        "critic_diagnostics.csv",
        "projection_diagnostics.csv",
        "topology_diagnostics.csv",
        "reward_surface.csv",
        "before_after_comparison.csv",
    }
    assert expected_csv.issubset(set(visualization_report["csv_outputs"]))
    assert expected_csv.issubset(manifest_paths)
    for filename in expected_csv:
        path = ARTIFACT_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0

    for filename in visualization_report["plot_files"]:
        path = ARTIFACT_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0
        assert filename in manifest_paths


def test_stage25_visualization_report_covers_required_human_review_views() -> None:
    report = json.loads((ARTIFACT_DIR / "visualization_report.json").read_text(encoding="utf-8"))

    required_views = {
        "training_curves",
        "objective_curves",
        "ppo_diagnostics",
        "critic_diagnostics",
        "projection_diagnostics",
        "topology_diagnostics",
        "reward_surface",
        "before_after_comparison",
    }
    assert required_views == set(report["required_views"])
    assert all(report["required_views"][view] is True for view in required_views)
