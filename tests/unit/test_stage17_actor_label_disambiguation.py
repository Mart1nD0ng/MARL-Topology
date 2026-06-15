from pathlib import Path

import pytest

from marl_topology.data.actor_label_disambiguation import (
    ActorLabelDisambiguationViolation,
    ContradictionCause,
    analyze_actor_label_disambiguation,
    build_actor_observation_signature,
    build_stage17_actor_label_disambiguation_report,
)
from marl_topology.data.learning_evidence import LearningEvidenceDataset, LearningEvidenceRow
from marl_topology.env import LocalNeighborObservation


ROOT = Path(__file__).resolve().parents[2]
EDGE_ID = "agent_0--agent_1"


def _actor_row() -> dict[str, object]:
    return {
        "agent_id": "agent_0",
        "agent_kind": "vehicle",
        "time_step": 0,
        "local_position_m": (0.0, 0.0, 1.5),
        "local_neighbor_observations": (
            LocalNeighborObservation(
                neighbor_id="agent_1",
                neighbor_kind="vehicle",
                edge_id=EDGE_ID,
                distance_3d_m=10.0,
                link_success_probability=0.9,
                estimated_link_latency_s=0.001,
                estimated_link_energy_j=0.02,
            ),
        ),
        "local_messages": (),
        "local_history": {},
    }


def _target(edge_id: str, action_type: str, probability_delta: float) -> dict[str, object]:
    return {
        "topology_id": f"target:{action_type}",
        "edge_id": edge_id,
        "action_type": action_type,
        "delta_consensus_success_probability": probability_delta,
        "delta_latency": 0.001 if action_type == "add_edge" else -0.001,
        "delta_energy": 0.02 if action_type == "add_edge" else -0.02,
        "delta_feasibility": 1 if probability_delta > 0.0 else -1,
        "delta_reward_surrogate_diagnostic": {
            "delta_component_sum_diagnostic": -0.1 if probability_delta > 0.0 else 0.1
        },
        "target_role": "learning_target_only",
    }


def _keep(edge_id: str) -> dict[str, object]:
    return {
        "topology_id": "target:keep",
        "edge_id": edge_id,
        "action_type": "keep_edge",
        "delta_consensus_success_probability": 0.0,
        "delta_latency": 0.0,
        "delta_energy": 0.0,
        "delta_feasibility": 0,
        "delta_reward_surrogate_diagnostic": {},
        "target_role": "learning_target_only",
    }


def _row(
    *,
    topology_name: str,
    selected: tuple[str, ...],
    counterfactual_action: str,
    probability_delta: float,
) -> LearningEvidenceRow:
    return LearningEvidenceRow(
        scenario_id="synthetic_stage17",
        topology_id=f"synthetic:{topology_name}",
        topology_name=topology_name,
        selected_edges=selected,
        consensus_success_probability=0.5,
        latency=0.001,
        energy=0.02,
        topology_diagnostics={},
        feasible_under_tau_requirement=False,
        actor_safe_rows=(_actor_row(),),
        critic_view={"view_role": "critic_centralized_training_only"},
        learning_targets=(
            _keep(EDGE_ID),
            _target(EDGE_ID, counterfactual_action, probability_delta),
        ),
        diagnostics={
            "topology_family": "synthetic",
            "topology_label_role": "baseline_evaluation",
        },
    )


def _synthetic_conflict_dataset() -> LearningEvidenceDataset:
    return LearningEvidenceDataset(
        dataset_id="synthetic_stage17_conflict",
        rows=(
            _row(
                topology_name="empty",
                selected=(),
                counterfactual_action="add_edge",
                probability_delta=0.4,
            ),
            _row(
                topology_name="selected",
                selected=(EDGE_ID,),
                counterfactual_action="remove_edge",
                probability_delta=-0.4,
            ),
        ),
        edge_delta_targets=(),
    )


def test_stage17_detector_finds_known_synthetic_conflict() -> None:
    report = analyze_actor_label_disambiguation(_synthetic_conflict_dataset())

    assert report.contradiction_cluster_count == 1
    cluster = report.contradiction_clusters[0]
    assert cluster.affected_agent == "agent_0"
    assert cluster.affected_edge == EDGE_ID
    assert cluster.label_distribution["counterfactual_action"] == {
        "add_edge": 1,
        "remove_edge": 1,
    }
    assert ContradictionCause.MISSING_PREVIOUS_TOPOLOGY_OR_HISTORY.value in cluster.suspected_causes
    assert report.actor_local_observations_sufficient_for_current_hard_labels is False


def test_stage17_signature_rejects_forbidden_global_fields() -> None:
    actor_row = _actor_row()
    actor_row["consensus_success_probability"] = 0.9
    neighbor = actor_row["local_neighbor_observations"][0]

    with pytest.raises(ActorLabelDisambiguationViolation):
        build_actor_observation_signature(actor_row, neighbor)


def test_stage17_report_classifies_stage16_contradiction_reasons() -> None:
    report = build_stage17_actor_label_disambiguation_report()

    assert report.sample_count == 850
    assert report.signature_count > 0
    assert report.contradiction_cluster_count > 0
    assert report.contradiction_rate_by_signature > 0.0
    assert report.actor_local_observations_sufficient_for_current_hard_labels is False
    assert report.next_stage_readiness["stage11_to_stage15_rerun_allowed"] is False
    assert report.next_stage_readiness["next_stage_readiness_gate_passed"] is False
    assert ContradictionCause.MISSING_PREVIOUS_TOPOLOGY_OR_HISTORY.value in report.cause_counts
    assert ContradictionCause.TARGET_DEPENDS_ON_GLOBAL_CONTEXT.value in report.cause_counts
    assert ContradictionCause.HARD_LABEL_SHOULD_BE_SOFT_OR_RANKED.value in report.cause_counts


def test_stage17_signature_uses_only_actor_safe_fields() -> None:
    actor_row = _actor_row()
    neighbor = actor_row["local_neighbor_observations"][0]
    signature = build_actor_observation_signature(actor_row, neighbor)
    signature_text = repr(signature.to_dict())

    forbidden_terms = [
        "global_topology",
        "oracle_label",
        "reward_surrogate",
        "consensus_success_probability",
        "critic_features",
        "selected_edges",
    ]
    assert not [term for term in forbidden_terms if term in signature_text]


def test_stage17_source_keeps_training_model_checkpoint_and_v5_out() -> None:
    source = (ROOT / "src" / "marl_topology" / "data" / "actor_label_disambiguation.py").read_text(
        encoding="utf-8"
    )
    forbidden_terms = [
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
    hits = [term for term in forbidden_terms if term in source]
    assert not hits, f"Stage 17 source introduced forbidden terms: {hits}"
