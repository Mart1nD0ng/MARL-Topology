"""Critic-free graph payload for the shared data path.

Moves ``_graph_parts`` (+ its private helpers) out of the retired
``training/critic_dataset.py`` and exposes the public ``graph_payload`` previously
provided by ``training/mappo/stage28_repaired_critic_pilot.py``. This is the node /
edge / edge-index featurization the decentralized actor reads -- it is critic-FREE in
its body. Imports only from ``marl_topology.data.*``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from marl_topology.data.stage21_objective_stack_evidence import (
    Stage21EvaluationContext,
)


def _graph_parts(
    *,
    row: object,
    context: Stage21EvaluationContext,
    previous_selected_edges: Sequence[str],
    step_index: int,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    node_ids = tuple(context.graph.node_ids)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    actor_rows = tuple(row.actor_safe_view)
    by_agent: dict[str, list[Mapping[str, object]]] = {}
    for actor_row in actor_rows:
        by_agent.setdefault(str(actor_row["agent_id"]), []).append(actor_row)
    previous_degree = Counter()
    for edge_id in previous_selected_edges:
        left, right = _split_edge(edge_id)
        previous_degree[left] += 1
        previous_degree[right] += 1
    node_features = []
    for node_id in node_ids:
        records = by_agent.get(node_id, [])
        role = _node_role(node_id)
        node_features.append(
            (
                1.0 if role == "vehicle" else 0.0,
                1.0 if role == "rsu" else 0.0,
                1.0 if role == "base_station" else 0.0,
                float(previous_degree[node_id]),
                _avg(records, "tx_budget_used"),
                _avg(records, "tx_budget_remaining"),
                _avg(records, "rx_capacity_estimate_for_neighbor"),
                float(step_index),
            )
        )
    edge_features = []
    edge_index = []
    previous_set = set(previous_selected_edges)
    for edge in context.graph.edges:
        record = context.link_records[edge.edge_id]
        left_role = _node_role(edge.node_u)
        right_role = _node_role(edge.node_v)
        edge_features.append(
            (
                float(record.link_success_probability),
                float(record.link_success_probability),
                float(record.latency_s),
                float(record.energy_j),
                1.0 if edge.edge_id in previous_set else 0.0,
                float(edge.distance_3d_m),
                _role_code(left_role),
                _role_code(right_role),
            )
        )
        edge_index.append((node_index[edge.node_u], node_index[edge.node_v]))
    return {
        "node_features": tuple(node_features),
        "edge_features": tuple(edge_features),
        "edge_index": tuple(edge_index),
    }


def _avg(records: Sequence[Mapping[str, object]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def _split_edge(edge_id: str) -> tuple[str, str]:
    parts = str(edge_id).split("--")
    if len(parts) != 2:
        return str(edge_id), ""
    return parts[0], parts[1]


def _node_role(node_id: str) -> str:
    lowered = str(node_id).lower()
    if lowered.startswith("veh"):
        return "vehicle"
    if lowered.startswith("rsu"):
        return "rsu"
    if lowered.startswith("bs") or "base" in lowered:
        return "base_station"
    return "unknown"


def _role_code(role: str) -> float:
    return {"vehicle": 0.25, "rsu": 0.5, "base_station": 0.75}.get(role, 0.0)


def graph_payload(
    *,
    row: object,
    context: Stage21EvaluationContext,
    previous_selected_edges: Sequence[str],
    step_index: int,
) -> dict[str, object]:
    parts = _graph_parts(
        row=row,
        context=context,
        previous_selected_edges=previous_selected_edges,
        step_index=step_index,
    )
    return {
        "view_role": "stage28_repaired_graph_value_critic_pre_action",
        "feature_schema_id": "stage27_centralized_graph_value_features_v1",
        "scenario_id": str(context.fixture.fixture_id),
        "time_step": int(step_index),
        "previous_selected_edges": tuple(str(edge) for edge in previous_selected_edges),
        "node_features": parts["node_features"],
        "edge_features": parts["edge_features"],
        "edge_index": parts["edge_index"],
        "training_only": True,
        "actor_input_allowed": False,
    }
