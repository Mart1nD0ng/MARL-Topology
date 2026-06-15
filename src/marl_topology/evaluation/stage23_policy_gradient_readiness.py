"""Stage 23 low-entropy preflight and policy-gradient readiness review."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    STAGE21_TAU_REQUIREMENT_MIN,
)
from marl_topology.data.stage22_action_semantics_evidence import (
    build_stage22_action_semantics_evidence,
)
from marl_topology.models import (
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalGNNEdgeScorer,
    build_model_registry,
)
from marl_topology.policies import (
    ACTIVE_ACTION_SEMANTICS_ID,
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    build_active_action_semantics_registry,
)


STAGE23_READINESS_STAGE_ID = "stage_23_controlled_policy_gradient_pilot_readiness_review"
STAGE23_REVIEW_COMPLETE_VERDICT = "stage23_review_complete_policy_gradient_still_blocked"
STAGE23_RECOMMENDED_NEXT_TASK = (
    "stage_24_selected_physical_policy_gradient_pilot_harness_or_owner_decision"
)


def run_stage23_controlled_policy_gradient_pilot_readiness_review(
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Review readiness without executing policy-gradient."""

    root = project_root or Path(__file__).resolve().parents[3]
    active_registry = build_active_action_semantics_registry()
    evidence = build_stage22_action_semantics_evidence()
    selected_dataset = evidence.datasets[UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID]
    target_summary = evidence.report["target_distribution"][
        UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
    ]
    model_registry = build_model_registry()
    gnn_boundary = LocalGNNEdgeScorer().boundary_report()
    stale_hits = _stale_executable_reference_hits(root)
    removed_files = _removed_executable_ab_paths(root)
    gates = {
        "low_entropy_preflight_cleanup": {
            "passed": not stale_hits and all(item["absent"] for item in removed_files),
            "stale_reference_hits": stale_hits,
            "removed_executable_paths": removed_files,
        },
        "single_selected_action_semantics": {
            "passed": (
                ACTIVE_ACTION_SEMANTICS_ID == UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
                and set(active_registry) == {UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID}
            ),
            "active_registry": {key: value.to_dict() for key, value in active_registry.items()},
        },
        "full_message_passing_gnn_active": {
            "passed": (
                LOCAL_GNN_EDGE_SCORER_MODEL_ID in model_registry
                and model_registry[LOCAL_GNN_EDGE_SCORER_MODEL_ID].family
                == "local_message_passing_gnn"
                and bool(gnn_boundary["full_message_passing_gnn"])
            ),
            "model_id": LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            "boundary_report": gnn_boundary,
        },
        "objective_stack_selected_evidence": {
            "passed": (
                selected_dataset.readiness["uses_final_objective_stack"] is True
                and selected_dataset.readiness["fallback_used"] is False
                and selected_dataset.readiness["tau_feasible_rows"] > 0
            ),
            "objective_contract_id": STAGE21_OBJECTIVE_CONTRACT_ID,
            "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
            "tau_requirement_min": STAGE21_TAU_REQUIREMENT_MIN,
            "readiness": dict(selected_dataset.readiness),
        },
        "selected_actor_target_quality": {
            "passed": (
                target_summary["uniformly_high"] is False
                and target_summary["has_high_mid_low_priority"] is True
                and target_summary["ranking_pair_count"] > 0
                and target_summary["low_priority_abstain_signal_count"] > 0
            ),
            "target_distribution": dict(target_summary),
        },
        "policy_gradient_execution_guard": {
            "passed": True,
            "policy_gradient_performed": False,
            "checkpoint_written": False,
            "artifact_written": False,
            "reason": "Stage 23 is a readiness review only.",
        },
        "selected_physical_pilot_harness_ready": {
            "passed": False,
            "reason": (
                "No selected-physical-link policy-gradient pilot harness exists after "
                "low-entropy cleanup. The historical Stage 15 pilot must not be reused "
                "as the Stage 23 pilot because it predates the selected action semantics "
                "and objective-aware teacher repairs."
            ),
        },
    }
    cleanup_passed = bool(gates["low_entropy_preflight_cleanup"]["passed"])
    pilot_execution_allowed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "stage": STAGE23_READINESS_STAGE_ID,
        "verdict": STAGE23_REVIEW_COMPLETE_VERDICT,
        "cleanup_passed": cleanup_passed,
        "pilot_execution_allowed": pilot_execution_allowed,
        "policy_gradient_performed": False,
        "checkpoint_written": False,
        "artifact_written": False,
        "v5_modified": False,
        "gates": gates,
        "recommended_next_task": STAGE23_RECOMMENDED_NEXT_TASK,
        "owner_decision_required": True,
    }


def _removed_executable_ab_paths(root: Path) -> list[dict[str, object]]:
    paths = [
        "src/marl_topology/evaluation/stage22_action_semantics_ab_evaluation.py",
        "src/marl_topology/training/stage22_action_semantics_supervised.py",
        "scripts/train/stage22_action_semantics_ab_full_gnn_report.py",
    ]
    return [
        {"path": path, "absent": not (root / path).exists()}
        for path in paths
    ]


def _stale_executable_reference_hits(root: Path) -> list[str]:
    banned_terms = (
        "directed_" + "outgoing_v1",
        "DIRECTED_" + "OUTGOING_ACTION_SEMANTICS_ID",
        "Stage22" + "DirectedObjectiveStackEvaluator",
        "local_gnn_edge_" + "scorer_v1",
        "ARCHIVED_" + "TOY_GNN_MODEL_ID",
        "toy_edge_" + "set_aggregator",
    )
    hits: list[str] = []
    for path in _iter_executable_project_files(root):
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(root)}:{term}")
    return sorted(hits)


def _iter_executable_project_files(root: Path) -> Iterable[Path]:
    for relative in ("src", "scripts", "tests"):
        base = root / relative
        if not base.exists():
            continue
        yield from sorted(base.rglob("*.py"))
