"""Active topology action semantics registry."""

from __future__ import annotations

from dataclasses import dataclass


UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID = "undirected_physical_link_v1"
ACTIVE_ACTION_SEMANTICS_ID = UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID


@dataclass(frozen=True, slots=True)
class ActionSemanticsSpec:
    action_semantics_id: str
    topology_object: str
    actor_output_object: str
    assembler_family: str
    evaluator_consumes: str
    active: bool
    status: str
    selection_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action_semantics_id": self.action_semantics_id,
            "topology_object": self.topology_object,
            "actor_output_object": self.actor_output_object,
            "assembler_family": self.assembler_family,
            "evaluator_consumes": self.evaluator_consumes,
            "active": self.active,
            "status": self.status,
            "selection_note": self.selection_note,
        }


def build_active_action_semantics_registry() -> dict[str, ActionSemanticsSpec]:
    """Return only the selected active topology action semantics."""

    selected = _undirected_spec()
    return {selected.action_semantics_id: selected}


def get_active_action_semantics_id() -> str:
    return ACTIVE_ACTION_SEMANTICS_ID


def assert_action_semantics_active(action_semantics_id: str) -> None:
    if action_semantics_id != ACTIVE_ACTION_SEMANTICS_ID:
        raise ValueError(f"inactive action semantics: {action_semantics_id}")


def _undirected_spec() -> ActionSemanticsSpec:
    return ActionSemanticsSpec(
        action_semantics_id=UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
        topology_object="selected_physical_edges",
        actor_output_object="endpoint_local_physical_link_score",
        assembler_family="physical_link_conflict_aware_greedy_max_endpoint",
        evaluator_consumes="selected_physical_edges",
        active=True,
        status="active_selected_stage22_stage23_preflight_cleaned",
        selection_note=(
            "Stage 22 selected this semantics because it aligns the first "
            "topology action with the physical-edge evaluator. Stage 23 "
            "preflight removed the losing A/B branch from executable code."
        ),
    )
