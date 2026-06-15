import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = (
    ROOT
    / "result_save"
    / "stage28_repaired_critic_mappo_rerun"
    / "stage28_repaired_critic_stage25_base_protocol"
)
REPORT_PATH = ARTIFACT_DIR / "training_report.json"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_stage28_artifact_set_covers_training_comparison_and_diagnostics() -> None:
    expected = {
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
    }
    actual = {path.name for path in ARTIFACT_DIR.iterdir() if path.is_file()}

    assert expected <= actual


def test_stage28_report_has_required_scorecards_and_comparisons() -> None:
    report = _report()

    assert report["pass_gate"] is True
    assert report["before_after_comparison"]["eval_delta_mean"]
    assert report["stage25_historical_comparison"]["metric_deltas"]
    assert report["reward_surface_analysis"]["alignment_passed"] is True
    assert report["pass_fail_gate"]["critic_health"]["mean_update_explained_variance"] > 0.10
    assert len(report["seed_reports"]) == 3
    assert all(seed["completed_updates"] == 20 for seed in report["seed_reports"])


def test_stage28_docs_and_project_state_are_synchronized_after_closeout() -> None:
    docs = {
        "STAGE28_REPAIRED_CRITIC_POLICY_GRADIENT_RERUN.md",
        "STAGE28_REPAIRED_CRITIC_TRAINING_REPORT.md",
        "STAGE28_REPAIRED_CRITIC_COMPARISON.md",
        "STAGE28_POLICY_GRADIENT_READINESS_AFTER_RERUN.md",
    }
    missing_docs = [name for name in docs if not (ROOT / "docs" / name).exists()]
    state_text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    required_terms = [
        "post_stage_28_complete_repaired_critic_mappo_rerun_awaiting_owner_decision",
        "stage_28_rerun_small_scale_mappo_with_repaired_critic",
        "stage28_repaired_critic_rerun_gate",
        "stage28_repaired_critic_health_gate",
        "stage_29_pre_scale_decision_review",
        "Scale-up training remains blocked",
        "LSTM/recurrent PPO remains blocked",
    ]
    missing_terms = [term for term in required_terms if term not in state_text]

    assert missing_docs == []
    assert missing_terms == []
