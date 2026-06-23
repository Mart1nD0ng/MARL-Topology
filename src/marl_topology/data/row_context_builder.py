"""Critic-free shared data path: actor-safe training rows + per-split row/context pairs.

Extracted verbatim from the retired ``training/production_mappo_adapter.py`` so the
single decentralized cold-start RL trunk (``scripts/train/train_decentralized_rl.py``)
can build its (row, context) pool WITHOUT importing the deleted centralized-critic
multi-agent trunk. Imports only from ``marl_topology.data.*`` / ``.policies.*`` / ``.scenario.*``
-- no ``models.centralized_*``, no ``training.mappo``, no critic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from marl_topology.data.actor_feature_rebuild import (
    actor_safe_view_has_no_forbidden_fields,
    build_stage18_actor_safe_feature_rows,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    _source_learning_evidence_row,
)
from marl_topology.data.stage33_graph_structure_dataset import (
    Stage33GraphStructureDataset,
)
from marl_topology.policies import build_local_observations


class RowContextBuilderViolation(ValueError):
    """Raised when the shared data path crosses a critic-free boundary."""


@dataclass(frozen=True, slots=True)
class Stage33TrainingRow:
    evidence_id: str
    scenario_id: str
    structural_family: str
    actor_safe_view: tuple[Mapping[str, object], ...]
    selected_physical_edges: tuple[str, ...]

    def __post_init__(self) -> None:
        if not actor_safe_view_has_no_forbidden_fields(self.actor_safe_view):
            raise RowContextBuilderViolation("actor_safe_view leaked forbidden fields")


def build_stage33_context(spec, teacher_label: Mapping[str, object]):
    context = build_stage33_base_context(spec)
    teacher_edges = tuple(str(edge) for edge in teacher_label["selected_physical_edges"])
    context.evaluator.evaluate(set(teacher_edges), topology_id=f"stage33:teacher:{spec.scenario_id}")
    return context


def build_stage33_base_context(spec):
    from marl_topology.data.stage31_production_dataset import build_production_context

    return build_production_context(spec)


def build_stage33_training_row(
    *,
    context,
    teacher_label: Mapping[str, object],
    structural_family: str,
) -> Stage33TrainingRow:
    teacher_edges = tuple(str(edge) for edge in teacher_label["selected_physical_edges"])
    evaluation = context.evaluator.evaluate(
        set(teacher_edges),
        topology_id=f"stage33:{context.fixture.fixture_id}:objective_teacher",
    )
    observations = build_local_observations(
        scene=context.fixture.scene,
        graph=context.graph,
        link_records=context.evaluator.link_records,
        time_step=context.time_step,
    )
    source_row = _source_learning_evidence_row(
        evaluation=evaluation,
        topology_name="stage33_objective_teacher",
        observations=observations,
        context=context,
        learning_targets=(),
    )
    actor_safe_view = tuple(
        {**dict(row), "stage33_structural_family": structural_family}
        for row in build_stage18_actor_safe_feature_rows(source_row)
    )
    return Stage33TrainingRow(
        evidence_id=f"stage33:{context.fixture.fixture_id}:{structural_family}",
        scenario_id=str(context.fixture.fixture_id),
        structural_family=structural_family,
        actor_safe_view=actor_safe_view,
        selected_physical_edges=teacher_edges,
    )


def build_row_contexts(
    dataset: Stage33GraphStructureDataset,
    split: str | None = None,
) -> tuple[tuple[Stage33TrainingRow, object], ...]:
    """Per-split (row, context) pairs for the shared data path.

    Reproduces the body of the retired adapter's ``build_row_contexts`` method.
    ``split`` is the split name (e.g. ``"train"`` / ``"eval"`` / ``"test"``); kept
    keyword-optional only to mirror the prior method default semantics.
    """

    spec_by_id = {spec.scenario_id: spec for spec in dataset.source_dataset.specs}
    row_contexts = []
    for record in dataset.records_for_split(split):
        spec = spec_by_id[record.scenario_id]
        context = build_stage33_context(spec, dataset.source_dataset.teacher_labels[spec.scenario_id])
        row = build_stage33_training_row(
            context=context,
            teacher_label=dataset.source_dataset.teacher_labels[spec.scenario_id],
            structural_family=record.structural_family,
        )
        row_contexts.append((row, context))
    if not row_contexts:
        raise RowContextBuilderViolation(f"empty Stage33 split: {split}")
    return tuple(row_contexts)
