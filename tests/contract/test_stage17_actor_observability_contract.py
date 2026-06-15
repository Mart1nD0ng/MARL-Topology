from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage17_document_contains_observability_verdict_and_fix_options() -> None:
    text = _read("docs/STAGE17_ACTOR_OBSERVABILITY_LABEL_DISAMBIGUATION.md")

    required = [
        "Stage 17 Actor Observability and Label Disambiguation",
        "actor_safe_edge_observation_signature_v1",
        "agent_id",
        "local_neighbor_observations",
        "delta consensus_success_probability",
        "delta latency",
        "delta energy",
        "feasibility-changing",
        "missing_previous_topology_or_history",
        "target_depends_on_global_context",
        "hard_label_should_be_soft_or_ranked",
        "oracle_label_not_actor_observable",
        "true_dec_pomdp_ambiguity",
        "Actor local observations are not sufficient",
        "Rerunning Stage 11-15 is not allowed yet",
        "stage_18_evidence_rebuild_with_disambiguated_features_or_targets",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 17 document missing terms: {missing}"


def test_stage17_harness_task_exists() -> None:
    task = yaml.safe_load(
        _read("harness/tasks/stage17_actor_observability_label_disambiguation.yaml")
    )

    assert task["id"] == "stage17_actor_observability_label_disambiguation"
    negative_text = " ".join(task["negative_checks"])
    assert "Stage 11-15 is rerun" in negative_text
    assert "actor signature includes global topology" in negative_text
    assert "checkpoint" in negative_text


def test_stage17_project_state_marks_stage17_complete_and_blocks_training_rerun() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage: post_stage_17_complete_awaiting_owner_decision_for_stage_18",
        "stage_17_actor_observability_and_label_disambiguation",
        "recommended_next_task: stage_18_evidence_rebuild_with_disambiguated_features_or_targets",
        "stage_11_to_stage_15_rerun_until_stage18_rebuilds_disambiguated_evidence",
        "PPO_MAPPO_rerun_until_stage18_rebuilds_disambiguated_evidence",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing Stage 17 state: {missing}"


def test_stage17_report_script_exists_and_is_report_only() -> None:
    text = _read("scripts/replay/stage17_actor_label_disambiguation_report.py")

    required = [
        "build_stage17_actor_label_disambiguation_report",
        "print(json.dumps",
        "report-only",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Stage 17 script missing report-only terms: {missing}"

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
    assert not hits, f"Stage 17 report script introduced forbidden terms: {hits}"


def test_stage17_source_does_not_add_training_model_checkpoint_or_v5_path() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "data" / "actor_label_disambiguation.py",
        ROOT / "scripts" / "replay" / "stage17_actor_label_disambiguation_report.py",
    ]
    forbidden = [
        "import torch",
        "optimizer.step",
        ".backward(",
        "torch.save",
        "checkpoint_path",
        "PPOTrainer",
        "MAPPOTrainer",
        "class COMA",
        "class Transformer",
        "D:\\PhD_works\\v5",
    ]
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 17 source introduced forbidden terms: {hits}"
