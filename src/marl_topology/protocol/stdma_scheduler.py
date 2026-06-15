"""STDMA conflict-graph + SINR-validated link scheduling (the MAC abstraction).

Background
----------
The Stage 21 evaluator can place every selected link on its own resource
(``orthogonal_resources=True`` -> zero mutual interference, an unrealistically
optimistic ceiling) or place every link on one shared resource
(``orthogonal_resources=False, use_background_interference=True`` -> every active
link interferes with every other, an unrealistically pessimistic worst case that
makes dense urban global PBFT infeasible). Neither is how real large-scale urban /
V2X wireless works.

Real systems use **spatial-reuse TDMA (STDMA)**: links are partitioned into a small
number of time slots so that links sharing a slot are mutually
*signal-to-interference-plus-noise-ratio (SINR)* feasible, while links that would
collide are separated in time. This is the classic minimum-slot link-scheduling
problem under the SINR (physical-interference) model -- NP-hard in general but well
approximated by greedy graph-coloring validated against the aggregate SINR
(Halldorsson & Mitra, ICALP 2011; Djukic & Valaee, IEEE/ACM ToN 2009; Gore et al.
2007). See ``docs/URBAN_V2X_RESEARCH_LOG.md`` (literature survey) for the sources.

What this module computes
-------------------------
Given the actor-selected link set, it builds a slot schedule:

1. **Pairwise conflict graph.** Two links conflict if they share a node
   (half-duplex primary conflict) or if co-activating them drops either link's
   SINR -- at either endpoint, worst interferer direction -- below the threshold.
2. **Greedy SINR-feasible packing.** Links are processed in descending
   conflict-degree order; each is placed in the first slot whose members remain
   *aggregate*-SINR-feasible (all co-slot interferers summed, not just pairwise)
   once it is added, otherwise a new slot is opened. Every emitted slot is therefore
   feasible against its own co-slot interference by construction -- the
   "conflict-graph + SINR-validated hybrid" the literature recommends over a pure
   conflict graph (which over-estimates feasibility for 3+ co-slot links). (A link's
   absolute link budget vs the noise floor is not a slot-packing concern; the physics
   layer evaluates that per hop and yields the actual delivery probability.)

The schedule is consumed by the Stage 21 evaluator in two ways:

* **Reliability:** co-slot links share a channel resource (they interfere, but are
  SINR-validated), cross-slot links are orthogonal -- so each link's reliability is
  computed against only its *co-slot* interferers, not the whole topology. This is
  what makes realistic urban global PBFT feasible instead of worst-case infeasible.
* **Latency:** a link in slot ``k`` (0-based) must wait for its slot to begin, so a
  route over it incurs ``k * slot_duration_s`` of scheduling latency on top of the
  link's own transmission time. Denser topologies need more slots, so routes over
  late slots exceed the PBFT phase deadline and are dropped -- the
  connectivity-vs-latency tradeoff the controller must navigate.

This module produces a *schedule* only. It does not define consensus reliability,
reward, topology optimality, or training behaviour, and it never mutates the
channel / link / PBFT models -- those evaluate the schedule the evaluator builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from marl_topology.channel import (
    ChannelModelConfig,
    dbm_to_mw,
    evaluate_channel,
    noise_power_dbm,
)
from marl_topology.topology import CandidateGraph

STDMA_SCHEDULER_ID = "stdma_conflict_graph_sinr_validated_v1"


@dataclass(frozen=True, slots=True)
class StdmaScheduleConfig:
    """Declared STDMA scheduling parameters."""

    sinr_threshold_db: float = 0.0
    slot_duration_s: float = 0.0006
    scheduler_id: str = STDMA_SCHEDULER_ID

    def __post_init__(self) -> None:
        if self.slot_duration_s <= 0.0:
            raise ValueError("slot_duration_s must be positive")
        if not self.scheduler_id:
            raise ValueError("scheduler_id must be non-empty")


@dataclass(frozen=True, slots=True)
class StdmaSchedule:
    """A computed slot assignment over the selected links."""

    scheduler_id: str
    slot_duration_s: float
    sinr_threshold_db: float
    slot_of_edge: Mapping[str, int]
    num_slots: int
    conflict_pair_count: int

    def __post_init__(self) -> None:
        if self.num_slots < 0:
            raise ValueError("num_slots must be nonnegative")
        if self.slot_of_edge and max(self.slot_of_edge.values()) + 1 != self.num_slots:
            raise ValueError("num_slots must match the maximum assigned slot")

    def resource_assignments(self) -> dict[str, str]:
        """Map each selected edge to the channel resource of its slot.

        Co-slot edges share a resource (they interfere; SINR-validated); edges in
        different slots get different resources (orthogonal)."""
        return {
            edge_id: f"resource_{slot}" for edge_id, slot in self.slot_of_edge.items()
        }

    def route_schedule_latency_s(self, route_edge_ids: tuple[str, ...]) -> float:
        """TDMA waiting latency a route incurs before its last hop can transmit:
        a hop in slot ``k`` (0-based) waits ``k * slot_duration`` for its slot to
        begin, then transmits (that transmission time is the link's own p2p latency,
        already accounted). The route's wait is set by its latest-slot hop."""
        slots = [self.slot_of_edge[edge_id] for edge_id in route_edge_ids if edge_id in self.slot_of_edge]
        if not slots:
            return 0.0
        return max(slots) * self.slot_duration_s


def build_received_power_table(
    scene,
    node_ids,
    channel_config: ChannelModelConfig,
) -> dict[tuple[str, str], float]:
    """Public wrapper: received signal power (mW) for every ordered node pair.

    Invariant per (scene, channel_config), so the Stage 21 evaluator computes it ONCE
    and feeds it to every ``build_stdma_schedule`` call (the SA teacher evaluates many
    topologies on one scene, and recomputing the ray-box visibility each time dominates
    runtime)."""
    return _received_power_table(scene, sorted(set(node_ids)), channel_config)


def build_stdma_schedule(
    *,
    scene,
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    channel_config: ChannelModelConfig,
    config: StdmaScheduleConfig,
    rx_power_mw: Mapping[tuple[str, str], float] | None = None,
) -> StdmaSchedule:
    """Partition ``selected_edge_ids`` into SINR-feasible TDMA slots.

    ``rx_power_mw`` is an optional precomputed received-power table (see
    ``build_received_power_table``); when omitted it is built from the involved nodes.
    Passing a precomputed table is purely a speedup and does not change the result."""

    selected = tuple(sorted(set(selected_edge_ids)))
    if not selected:
        return StdmaSchedule(
            scheduler_id=config.scheduler_id,
            slot_duration_s=config.slot_duration_s,
            sinr_threshold_db=config.sinr_threshold_db,
            slot_of_edge={},
            num_slots=0,
            conflict_pair_count=0,
        )

    endpoints: dict[str, tuple[str, str]] = {}
    involved: set[str] = set()
    for edge_id in selected:
        edge = graph.get_edge(edge_id)
        endpoints[edge_id] = (edge.node_u, edge.node_v)
        involved.add(edge.node_u)
        involved.add(edge.node_v)

    if rx_power_mw is None:
        rx_power_mw = _received_power_table(scene, sorted(involved), channel_config)
    noise_mw = dbm_to_mw(noise_power_dbm(channel_config.bandwidth_hz, channel_config.noise_figure_db))
    threshold_linear = 10.0 ** (config.sinr_threshold_db / 10.0)

    # Pairwise conflicts -> degree, only used to order packing (largest-degree first
    # is a strong greedy-coloring heuristic that tends to minimise slot count).
    conflict_degree: dict[str, int] = {edge_id: 0 for edge_id in selected}
    conflict_pair_count = 0
    for i, edge_a in enumerate(selected):
        for edge_b in selected[i + 1 :]:
            if _links_conflict(
                endpoints[edge_a],
                endpoints[edge_b],
                rx_power_mw,
                noise_mw,
                threshold_linear,
            ):
                conflict_degree[edge_a] += 1
                conflict_degree[edge_b] += 1
                conflict_pair_count += 1

    order = sorted(selected, key=lambda edge_id: (-conflict_degree[edge_id], edge_id))

    slots: list[list[str]] = []
    slot_of_edge: dict[str, int] = {}
    for edge_id in order:
        placed = False
        for slot_index, members in enumerate(slots):
            if _slot_feasible_with(
                members,
                edge_id,
                endpoints,
                rx_power_mw,
                noise_mw,
                threshold_linear,
            ):
                members.append(edge_id)
                slot_of_edge[edge_id] = slot_index
                placed = True
                break
        if not placed:
            slot_of_edge[edge_id] = len(slots)
            slots.append([edge_id])

    return StdmaSchedule(
        scheduler_id=config.scheduler_id,
        slot_duration_s=config.slot_duration_s,
        sinr_threshold_db=config.sinr_threshold_db,
        slot_of_edge=slot_of_edge,
        num_slots=len(slots),
        conflict_pair_count=conflict_pair_count,
    )


def _received_power_table(
    scene,
    nodes: list[str],
    channel_config: ChannelModelConfig,
) -> dict[tuple[str, str], float]:
    """Received signal power (mW) for every ordered node pair among ``nodes``.

    Computed once with the project's own channel model (FSPL + NLOS penalty), so
    the schedule's SINR is consistent with the SINR the evaluator later sees."""
    table: dict[tuple[str, str], float] = {}
    for tx_id in nodes:
        for rx_id in nodes:
            if tx_id == rx_id:
                continue
            record = evaluate_channel(scene, tx_id, rx_id, config=channel_config)
            table[(tx_id, rx_id)] = dbm_to_mw(record.rx_power_dbm)
    return table


def _interference_mw(
    interferer_endpoints: tuple[str, str],
    victim_rx: str,
    rx_power_mw: Mapping[tuple[str, str], float],
) -> float:
    """Worst-case interference an interferer link contributes at ``victim_rx``:
    the louder of its two endpoints transmitting (the realised slot direction is
    unknown, so we upper-bound -- a conservative, reliability-safe schedule)."""
    left, right = interferer_endpoints
    return max(
        rx_power_mw.get((left, victim_rx), 0.0),
        rx_power_mw.get((right, victim_rx), 0.0),
    )


def _link_min_sinr(
    victim_endpoints: tuple[str, str],
    other_endpoints: list[tuple[str, str]],
    rx_power_mw: Mapping[tuple[str, str], float],
    noise_mw: float,
    threshold_linear: float,
) -> bool:
    """Is ``victim`` SINR-feasible at both endpoints against the aggregate of the
    ``other`` co-slot links? PBFT uses both directions, so both must clear."""
    u, v = victim_endpoints
    for tx_id, rx_id in ((u, v), (v, u)):
        signal = rx_power_mw.get((tx_id, rx_id), 0.0)
        interference = sum(
            _interference_mw(other, rx_id, rx_power_mw) for other in other_endpoints
        )
        denominator = noise_mw + interference
        if denominator <= 0.0:
            continue
        if signal / denominator < threshold_linear:
            return False
    return True


def _links_conflict(
    endpoints_a: tuple[str, str],
    endpoints_b: tuple[str, str],
    rx_power_mw: Mapping[tuple[str, str], float],
    noise_mw: float,
    threshold_linear: float,
) -> bool:
    if set(endpoints_a) & set(endpoints_b):
        return True
    a_ok = _link_min_sinr(endpoints_a, [endpoints_b], rx_power_mw, noise_mw, threshold_linear)
    b_ok = _link_min_sinr(endpoints_b, [endpoints_a], rx_power_mw, noise_mw, threshold_linear)
    return not (a_ok and b_ok)


def _slot_feasible_with(
    members: list[str],
    candidate: str,
    endpoints: Mapping[str, tuple[str, str]],
    rx_power_mw: Mapping[tuple[str, str], float],
    noise_mw: float,
    threshold_linear: float,
) -> bool:
    """Would adding ``candidate`` to ``members`` keep the whole slot SINR-feasible?
    A shared node makes it half-duplex-infeasible immediately; otherwise every link
    (members + candidate) must clear the threshold against the others' aggregate."""
    candidate_nodes = set(endpoints[candidate])
    for member in members:
        if candidate_nodes & set(endpoints[member]):
            return False
    full = list(members) + [candidate]
    for index, edge_id in enumerate(full):
        others = [endpoints[other] for j, other in enumerate(full) if j != index]
        if not _link_min_sinr(
            endpoints[edge_id], others, rx_power_mw, noise_mw, threshold_linear
        ):
            return False
    return True
