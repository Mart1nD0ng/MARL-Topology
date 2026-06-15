"""Network communication aggregation for Stage 3.4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import prod
from typing import Mapping

from marl_topology.channel import ActiveTransmission, ChannelModelConfig, evaluate_channel
from marl_topology.link import (
    LinkTransmissionConfig,
    LinkTransmissionRecord,
    evaluate_link_transmission,
)
from marl_topology.topology import CandidateGraph, canonical_edge_id


NETWORK_COMMUNICATION_REGIME_ID = "stage3_network_communication_v1"


@dataclass(frozen=True, slots=True)
class NetworkCommunicationConfig:
    network_model_id: str = NETWORK_COMMUNICATION_REGIME_ID
    primitive: str = "route"
    channel_config: ChannelModelConfig = field(default_factory=ChannelModelConfig)
    link_config: LinkTransmissionConfig = field(default_factory=LinkTransmissionConfig)
    max_hops: int | None = None
    shadowing_seed: int | None = None

    def __post_init__(self) -> None:
        if not self.network_model_id:
            raise ValueError("network_model_id must be non-empty")
        if self.primitive not in {"route", "broadcast"}:
            raise ValueError("primitive must be route or broadcast")
        if self.max_hops is not None and self.max_hops < 0:
            raise ValueError("max_hops must be nonnegative when provided")


@dataclass(frozen=True, slots=True)
class NetworkTransmissionSpec:
    edge_id: str
    tx_id: str
    rx_id: str
    resource_id: str = "resource_0"
    tx_power_dbm: float | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id must be non-empty")
        if not self.tx_id or not self.rx_id:
            raise ValueError("tx_id and rx_id must be non-empty")
        if self.tx_id == self.rx_id:
            raise ValueError("tx_id and rx_id must differ")
        if self.edge_id != canonical_edge_id(self.tx_id, self.rx_id):
            raise ValueError("edge_id must match tx/rx endpoints canonically")
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")

    @property
    def transmission_id(self) -> str:
        return f"{self.edge_id}:{self.tx_id}->{self.rx_id}:{self.resource_id}"

    def as_active_transmission(self) -> ActiveTransmission:
        return ActiveTransmission(
            tx_id=self.tx_id,
            rx_id=self.rx_id,
            resource_id=self.resource_id,
            tx_power_dbm=self.tx_power_dbm,
            active=self.active,
        )


@dataclass(frozen=True, slots=True)
class NetworkHopRecord:
    hop_index: int
    edge_id: str
    tx_id: str
    rx_id: str
    resource_id: str
    transmission_id: str
    p2p_delivery_probability: float
    p2p_packet_success_probability: float
    p2p_expected_attempts: float
    p2p_latency_s: float
    p2p_energy_j: float
    interference_tx_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hop_index < 0:
            raise ValueError("hop_index must be nonnegative")
        if not self.edge_id or not self.tx_id or not self.rx_id or not self.resource_id:
            raise ValueError("hop ids and resource_id must be non-empty")
        if not 0.0 <= self.p2p_delivery_probability <= 1.0:
            raise ValueError("p2p_delivery_probability must be in [0, 1]")
        if not 0.0 <= self.p2p_packet_success_probability <= 1.0:
            raise ValueError("p2p_packet_success_probability must be in [0, 1]")
        if self.p2p_expected_attempts < 0:
            raise ValueError("p2p_expected_attempts must be nonnegative")
        if self.p2p_latency_s < 0 or self.p2p_energy_j < 0:
            raise ValueError("p2p latency and energy must be nonnegative")


@dataclass(frozen=True, slots=True)
class NetworkCommunicationRecord:
    scenario_id: str
    network_model_id: str
    primitive: str
    source_id: str
    target_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    active_transmission_ids: tuple[str, ...]
    interference_group_ids: tuple[str, ...]
    reachable_node_ids: tuple[str, ...]
    route_node_ids: tuple[str, ...]
    route_edge_ids: tuple[str, ...]
    hop_records: tuple[NetworkHopRecord, ...]
    network_delivery_probability: float
    network_latency_s: float
    network_scheduled_latency_s: float
    network_successful_delivery_latency_s: float
    network_energy_j: float
    hop_count: int
    is_full_graph_baseline: bool
    is_oracle: bool = False
    latency_aggregation: str = "sum_for_route_max_for_broadcast"
    energy_aggregation: str = "sum"
    network_regime: str = NETWORK_COMMUNICATION_REGIME_ID

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.network_model_id:
            raise ValueError("scenario_id and network_model_id must be non-empty")
        if self.primitive not in {"route", "broadcast"}:
            raise ValueError("primitive must be route or broadcast")
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if not 0.0 <= self.network_delivery_probability <= 1.0:
            raise ValueError("network_delivery_probability must be in [0, 1]")
        if (
            self.network_latency_s < 0
            or self.network_scheduled_latency_s < 0
            or self.network_successful_delivery_latency_s < 0
            or self.network_energy_j < 0
        ):
            raise ValueError("network latency and energy must be nonnegative")
        if self.network_latency_s != self.network_successful_delivery_latency_s:
            raise ValueError("network_latency_s is the successful delivery latency alias")
        if self.network_successful_delivery_latency_s > self.network_scheduled_latency_s:
            raise ValueError("successful delivery latency cannot exceed scheduled latency")
        if self.hop_count != len(self.hop_records):
            raise ValueError("hop_count must match hop_records length")
        if self.is_oracle:
            raise ValueError("Stage 3 network records must not be labelled oracle")


def evaluate_network_communication(
    scene,
    graph: CandidateGraph,
    selected_edge_ids: set[str] | frozenset[str] | tuple[str, ...],
    source_id: str,
    target_ids: tuple[str, ...],
    config: NetworkCommunicationConfig | None = None,
    resource_assignments: Mapping[str, str] | None = None,
    background_transmissions: tuple[NetworkTransmissionSpec, ...] = (),
) -> NetworkCommunicationRecord:
    if config is None:
        config = NetworkCommunicationConfig()
    if resource_assignments is None:
        resource_assignments = {}
    if source_id not in graph.node_ids:
        raise KeyError(f"unknown source_id: {source_id}")
    unknown_targets = sorted(set(target_ids) - set(graph.node_ids))
    if unknown_targets:
        raise KeyError(f"unknown target ids: {unknown_targets}")

    selected = tuple(sorted(set(selected_edge_ids)))
    unknown_edges = sorted(set(selected) - set(graph.edge_ids))
    if unknown_edges:
        raise ValueError(f"selected unknown edge ids: {unknown_edges}")
    unknown_resources = sorted(set(resource_assignments) - set(selected))
    if unknown_resources:
        raise ValueError(f"resource assignments for unselected edges: {unknown_resources}")

    adjacency = _adjacency(graph, selected)
    reachable = _reachable_nodes(adjacency, source_id)
    route_nodes, route_edges = _primitive_trace(
        graph=graph,
        adjacency=adjacency,
        selected_edge_ids=selected,
        source_id=source_id,
        target_ids=target_ids,
        primitive=config.primitive,
        max_hops=config.max_hops,
    )
    route_specs = _transmission_specs_for_trace(route_nodes, route_edges, resource_assignments)
    concurrent_specs = background_transmissions
    if config.primitive == "broadcast":
        concurrent_specs = route_specs + background_transmissions
    active_specs = tuple(spec for spec in route_specs + background_transmissions if spec.active)
    concurrent_transmissions = tuple(
        spec.as_active_transmission() for spec in concurrent_specs if spec.active
    )

    hop_records: list[NetworkHopRecord] = []
    for index, spec in enumerate(route_specs):
        channel_record = evaluate_channel(
            scene,
            spec.tx_id,
            spec.rx_id,
            config=config.channel_config,
            active_transmissions=tuple(
                transmission
                for transmission in concurrent_transmissions
                if not (
                    transmission.tx_id == spec.tx_id
                    and transmission.rx_id == spec.rx_id
                    and transmission.resource_id == spec.resource_id
                )
                and transmission.tx_id != spec.rx_id
            ),
            resource_id=spec.resource_id,
            tx_power_dbm=spec.tx_power_dbm,
            shadowing_seed=config.shadowing_seed,
        )
        link_record = evaluate_link_transmission(
            channel_record,
            config.link_config,
            selected=spec.edge_id in selected,
            active=spec.active,
        )
        hop_records.append(_hop_record(index, spec, link_record, channel_record.interference_tx_ids))

    delivery_probability = _delivery_probability(tuple(hop_records), target_ids, route_nodes)
    scheduled_latency_s = _network_scheduled_latency(config.primitive, tuple(hop_records))
    successful_latency_s = _network_successful_delivery_latency(
        scheduled_latency_s,
        delivery_probability,
    )
    energy_j = sum(hop.p2p_energy_j for hop in hop_records)
    active_ids = tuple(sorted(spec.transmission_id for spec in active_specs))
    interference_group_ids = _interference_group_ids(tuple(hop_records))

    return NetworkCommunicationRecord(
        scenario_id=graph.scenario_id,
        network_model_id=config.network_model_id,
        primitive=config.primitive,
        source_id=source_id,
        target_ids=tuple(sorted(target_ids)),
        selected_edge_ids=selected,
        active_transmission_ids=active_ids,
        interference_group_ids=interference_group_ids,
        reachable_node_ids=tuple(sorted(reachable)),
        route_node_ids=route_nodes,
        route_edge_ids=route_edges,
        hop_records=tuple(hop_records),
        network_delivery_probability=delivery_probability,
        network_latency_s=successful_latency_s,
        network_scheduled_latency_s=scheduled_latency_s,
        network_successful_delivery_latency_s=successful_latency_s,
        network_energy_j=energy_j,
        hop_count=len(hop_records),
        is_full_graph_baseline=graph.is_full_selection(set(selected)),
    )


def _adjacency(graph: CandidateGraph, selected_edge_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in graph.node_ids}
    for edge_id in selected_edge_ids:
        edge = graph.get_edge(edge_id)
        neighbors[edge.node_u].add(edge.node_v)
        neighbors[edge.node_v].add(edge.node_u)
    return {node_id: tuple(sorted(values)) for node_id, values in neighbors.items()}


def _reachable_nodes(adjacency: Mapping[str, tuple[str, ...]], source_id: str) -> set[str]:
    seen = {source_id}
    queue: deque[str] = deque([source_id])
    while queue:
        node_id = queue.popleft()
        for neighbor in adjacency[node_id]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _primitive_trace(
    graph: CandidateGraph,
    adjacency: Mapping[str, tuple[str, ...]],
    selected_edge_ids: tuple[str, ...],
    source_id: str,
    target_ids: tuple[str, ...],
    primitive: str,
    max_hops: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if primitive == "route":
        if len(target_ids) != 1:
            raise ValueError("route primitive requires exactly one target_id")
        return _shortest_path_trace(adjacency, source_id, target_ids[0], max_hops)
    if not target_ids:
        raise ValueError("broadcast primitive requires at least one target_id")
    return _broadcast_trace(graph, selected_edge_ids, source_id)


def _shortest_path_trace(
    adjacency: Mapping[str, tuple[str, ...]],
    source_id: str,
    target_id: str,
    max_hops: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source_id, (source_id,))])
    seen = {source_id}
    while queue:
        node_id, path = queue.popleft()
        if node_id == target_id:
            if max_hops is not None and len(path) - 1 > max_hops:
                return (), ()
            return path, _edge_ids_for_node_path(path)
        for neighbor in adjacency[node_id]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + (neighbor,)))
    return (), ()


def _broadcast_trace(
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    source_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    adjacency = _adjacency(graph, selected_edge_ids)
    distances = _bfs_distances(adjacency, source_id)
    active_edges = tuple(
        edge_id
        for edge_id in selected_edge_ids
        if graph.get_edge(edge_id).node_u in distances
        and graph.get_edge(edge_id).node_v in distances
    )
    return tuple(sorted(distances, key=lambda node_id: (distances[node_id], node_id))), active_edges


def _bfs_distances(
    adjacency: Mapping[str, tuple[str, ...]],
    source_id: str,
) -> dict[str, int]:
    distances = {source_id: 0}
    queue: deque[str] = deque([source_id])
    while queue:
        node_id = queue.popleft()
        for neighbor in adjacency[node_id]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node_id] + 1
            queue.append(neighbor)
    return distances


def _edge_ids_for_node_path(path: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonical_edge_id(left, right) for left, right in zip(path, path[1:]))


def _transmission_specs_for_trace(
    route_nodes: tuple[str, ...],
    route_edge_ids: tuple[str, ...],
    resource_assignments: Mapping[str, str],
) -> tuple[NetworkTransmissionSpec, ...]:
    if not route_nodes or not route_edge_ids:
        return ()
    node_pairs = _node_pairs_for_edges(route_nodes, route_edge_ids)
    return tuple(
        NetworkTransmissionSpec(
            edge_id=edge_id,
            tx_id=tx_id,
            rx_id=rx_id,
            resource_id=resource_assignments.get(edge_id, "resource_0"),
        )
        for edge_id, (tx_id, rx_id) in zip(route_edge_ids, node_pairs)
    )


def _node_pairs_for_edges(
    route_nodes: tuple[str, ...],
    route_edge_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if len(route_edge_ids) == len(route_nodes) - 1:
        return tuple(zip(route_nodes, route_nodes[1:]))
    pairs: list[tuple[str, str]] = []
    order = {node_id: index for index, node_id in enumerate(route_nodes)}
    for edge_id in route_edge_ids:
        left, right = edge_id.split("--", 1)
        left_order = order.get(left, len(order))
        right_order = order.get(right, len(order))
        if left_order <= right_order:
            pairs.append((left, right))
        else:
            pairs.append((right, left))
    return tuple(pairs)


def _hop_record(
    hop_index: int,
    spec: NetworkTransmissionSpec,
    link_record: LinkTransmissionRecord,
    interference_tx_ids: tuple[str, ...],
) -> NetworkHopRecord:
    return NetworkHopRecord(
        hop_index=hop_index,
        edge_id=spec.edge_id,
        tx_id=spec.tx_id,
        rx_id=spec.rx_id,
        resource_id=spec.resource_id,
        transmission_id=spec.transmission_id,
        p2p_delivery_probability=link_record.deadline_delivery_probability,
        p2p_packet_success_probability=link_record.packet_success_probability,
        p2p_expected_attempts=link_record.expected_attempts,
        p2p_latency_s=link_record.p2p_latency_s,
        p2p_energy_j=link_record.p2p_energy_j,
        interference_tx_ids=tuple(sorted(interference_tx_ids)),
    )


def _delivery_probability(
    hop_records: tuple[NetworkHopRecord, ...],
    target_ids: tuple[str, ...],
    route_nodes: tuple[str, ...],
) -> float:
    if not target_ids:
        return 0.0
    if sorted(target_ids) != sorted(set(target_ids)):
        raise ValueError("target_ids must be unique")
    if not set(target_ids).issubset(set(route_nodes)):
        return 0.0
    if not hop_records:
        return 1.0 if set(target_ids) == {route_nodes[0]} else 0.0
    return prod(hop.p2p_delivery_probability for hop in hop_records)


def _network_scheduled_latency(
    primitive: str,
    hop_records: tuple[NetworkHopRecord, ...],
) -> float:
    if not hop_records:
        return 0.0
    if primitive == "broadcast":
        return max(hop.p2p_latency_s for hop in hop_records)
    return sum(hop.p2p_latency_s for hop in hop_records)


def _network_successful_delivery_latency(
    scheduled_latency_s: float,
    delivery_probability: float,
) -> float:
    if delivery_probability == 0.0:
        return 0.0
    return scheduled_latency_s


def _interference_group_ids(hop_records: tuple[NetworkHopRecord, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"{hop.resource_id}:active_tx={len(hop.interference_tx_ids) + 1}"
                for hop in hop_records
                if hop.interference_tx_ids
            }
        )
    )
