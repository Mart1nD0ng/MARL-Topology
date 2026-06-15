from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage18_documents_define_feature_target_rebuild_and_gate() -> None:
    docs = "\n".join(
        _read(path)
        for path in (
            "docs/STAGE18_ACTOR_SAFE_FEATURE_REBUILD.md",
            "docs/STAGE18_ACTOR_TARGET_REBUILD.md",
            "docs/STAGE18_EVIDENCE_REBUILD_WITH_DISAMBIGUATED_FEATURES_AND_TARGETS.md",
            "docs/STAGE18_DATA_QUALITY_GATE.md",
        )
    )

    required = [
        "edge_active_prev",
        "edge_active_current_local",
        "tx_budget_remaining",
        "local_conflict_group_occupancy",
        "estimated_deadline_delivery_probability",
        "actor_edge_utility_target",
        "pairwise ranking",
        "hard_label_allowed_for_actor_training",
        "critic_only",
        "delta_consensus_success_probability",
        "hard-label allowed subset contradiction rate: `0.0`",
        "stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence",
    ]
    missing = [term for term in required if term not in docs]
    assert not missing, f"Stage 18 docs missing terms: {missing}"


def test_stage18_harness_task_exists_and_declares_negative_checks() -> None:
    task = yaml.safe_load(_read("harness/tasks/stage18_evidence_rebuild.yaml"))

    assert task["id"] == "stage18_evidence_rebuild"
    negative = " ".join(task["negative_checks"])
    assert "training" in negative
    assert "PPO" in negative
    assert "COMA" in negative
    assert "Transformer" in negative
    assert "oracle labels enter actor" in negative


def test_stage18_project_state_marks_stage18_complete_and_stage19_owner_gate() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_18_complete_awaiting_owner_decision_for_stage_19",
        "stage_18_evidence_rebuild_with_disambiguated_features_and_targets",
        "recommended_next_task: stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence",
        "PPO_MAPPO_rerun_until_stage19_owner_approval",
        "scale_up_training",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing Stage 18 state: {missing}"


def test_stage18_report_script_is_report_only() -> None:
    text = _read("scripts/replay/stage18_evidence_rebuild_report.py")

    required = [
        "build_stage18_evidence_rebuild_report",
        "print(json.dumps",
        "report-only",
    ]
    missing = [term for term in required if term not in text]
    assert not missing

    forbidden = [
        "optimizer.step",
        ".backward(",
        "torch.save",
        "checkpoint_path",
        "write_text(",
        "write_bytes(",
        "D:\\PhD_works\\v5",
    ]
    hits = [term for term in forbidden if term in text]
    assert not hits, f"Stage 18 report script introduced forbidden terms: {hits}"

