from pathlib import Path

from marl_topology.data.learning_evidence_stage18 import build_stage18_learning_evidence_dataset


ROOT = Path(__file__).resolve().parents[2]


def test_stage18_actor_views_do_not_contain_forbidden_global_fields() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    actor_safe_text = repr([row.actor_safe_view for row in dataset.rows])
    actor_target_text = repr([row.actor_target_view for row in dataset.rows])

    actor_safe_forbidden = [
        "global_topology",
        "global_objective",
        "selected_edge_ids",
        "consensus_success_probability",
        "delta_consensus_success_probability",
        "delta_latency",
        "delta_energy",
        "oracle_label",
        "oracle_topology_membership",
        "reward_surrogate",
        "future_outcome",
    ]
    actor_target_forbidden = [
        "global_topology",
        "global_objective",
        "selected_edge_ids",
        "consensus_success_probability",
        "delta_consensus_success_probability",
        "delta_latency",
        "delta_energy",
        "oracle_topology_membership",
        "reward_surrogate",
        "future_outcome",
    ]
    assert not [term for term in actor_safe_forbidden if term in actor_safe_text]
    assert not [term for term in actor_target_forbidden if term in actor_target_text]


def test_stage18_source_does_not_add_training_model_checkpoint_or_v5_paths() -> None:
    paths = [
        ROOT / "src" / "marl_topology" / "data" / "actor_feature_rebuild.py",
        ROOT / "src" / "marl_topology" / "data" / "disambiguated_targets.py",
        ROOT / "src" / "marl_topology" / "data" / "learning_evidence_stage18.py",
        ROOT / "scripts" / "replay" / "stage18_evidence_rebuild_report.py",
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
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 18 source introduced forbidden terms: {hits}"
