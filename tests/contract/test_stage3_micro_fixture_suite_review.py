from pathlib import Path

import yaml

from marl_topology.evaluation import (
    REQUIRED_STAGE3_MICRO_FIXTURES,
    build_stage3_micro_fixture_suite_review,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage3_micro_fixture_suite_report_checks_pass() -> None:
    report = build_stage3_micro_fixture_suite_review()

    assert report["stage"] == "stage_3_5_micro_fixture_suite_review"
    assert tuple(report["required_micro_fixture_ids"]) == REQUIRED_STAGE3_MICRO_FIXTURES
    assert report["fixture_counts"]["geometry"] >= 6
    assert report["fixture_counts"]["channel"] >= 5
    assert report["fixture_counts"]["link_transmission"] >= 5
    assert report["fixture_counts"]["network"] >= 5
    assert report["checks"]["required_micro_fixtures_covered"] is True
    assert report["checks"]["missing_required_micro_fixture_ids"] == []
    assert report["checks"]["geometry_expectations_match"] is True
    assert report["checks"]["channel_expectations_match"] is True
    assert report["checks"]["link_records_nonnegative"] is True
    assert report["checks"]["network_records_nonnegative"] is True
    assert report["checks"]["network_records_not_oracle"] is True
    assert report["checks"]["training_run"] is False
    assert report["checks"]["v5_code_migrated"] is False
    assert report["checks"]["reward_implemented"] is False
    assert report["checks"]["pbft_application_semantics_defined"] is False
    assert report["checks"]["new_stage3_registry_entries"] == []
    assert report["boundary"]["records_are_stage3_communication_only"] is True
    assert report["boundary"]["not_pbft_consensus"] is True
    assert report["boundary"]["not_reward"] is True
    assert report["boundary"]["not_training_data"] is True
    assert report["boundary"]["not_oracle"] is True


def test_stage3_micro_fixture_suite_replay_script_is_print_only() -> None:
    text = (ROOT / "scripts" / "replay" / "stage3_micro_fixture_suite_report.py").read_text(
        encoding="utf-8"
    )

    assert "build_stage3_micro_fixture_suite_review" in text
    assert "print(" in text
    banned_terms = ["to_csv", "result_save", "pickle", "torch.save", "open("]
    hits = [term for term in banned_terms if term in text]
    assert not hits, f"Stage 3.5 replay script writes outputs: {hits}"


def test_stage3_micro_fixture_suite_source_has_no_training_or_v5_dependency() -> None:
    scan_paths = [
        ROOT / "src" / "marl_topology" / "evaluation" / "stage3_fixture_suite.py",
        ROOT / "scripts" / "replay" / "stage3_micro_fixture_suite_report.py",
    ]
    banned_terms = [
        "D:\\PhD_works\\v5",
        "train_loop",
        "optimizer",
        "backward(",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "P_eff",
        "consensus_success",
        "consensus_success_probability",
    ]
    offenders: dict[str, list[str]] = {}
    for path in scan_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 3.5 fixture suite added forbidden dependency: {offenders}"
