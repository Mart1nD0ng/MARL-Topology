"""Stage 4.3 adapter from Stage 3 network records to PBFT message matrices."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from marl_topology.network import NetworkCommunicationRecord

from .pbft_reliability import (
    PBFT_PHASE_NAMES,
    PBFTThreePhaseConfig,
    PBFTThreePhaseReliabilityRecord,
    evaluate_pbft_three_phase_reliability,
)


MESSAGE_MATRIX_ADAPTER_ID = "stage4_stage3_network_to_pbft_matrix_v1"
PhaseRecordMap = Mapping[str, tuple[NetworkCommunicationRecord, ...]]
MessageMatrixDict = dict[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class PBFTPhaseBudgets:
    pre_prepare_budget_s: float
    prepare_budget_s: float
    commit_budget_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("pre_prepare_budget_s", self.pre_prepare_budget_s),
            ("prepare_budget_s", self.prepare_budget_s),
            ("commit_budget_s", self.commit_budget_s),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def budget_for_phase(self, phase_name: str) -> float:
        if phase_name == "pre_prepare":
            return self.pre_prepare_budget_s
        if phase_name == "prepare":
            return self.prepare_budget_s
        if phase_name == "commit":
            return self.commit_budget_s
        raise ValueError(f"unknown PBFT phase: {phase_name}")


@dataclass(frozen=True, slots=True)
class PBFTMessageMatrices:
    adapter_id: str
    node_ids: tuple[str, ...]
    phase_names: tuple[str, str, str]
    pre_prepare_matrix: Mapping[tuple[str, str], float]
    prepare_matrix: Mapping[tuple[str, str], float]
    commit_matrix: Mapping[tuple[str, str], float]
    phase_budgets_s: Mapping[str, float]
    record_count_by_phase: Mapping[str, int]
    deadline_filtered_count_by_phase: Mapping[str, int]
    zero_delivery_count_by_phase: Mapping[str, int]
    uses_stage3_network_records: bool = True
    exports_consensus_metric: bool = False

    def __post_init__(self) -> None:
        if self.adapter_id != MESSAGE_MATRIX_ADAPTER_ID:
            raise ValueError("unsupported adapter_id")
        if self.phase_names != PBFT_PHASE_NAMES:
            raise ValueError("phase_names must be pre_prepare, prepare, commit")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be non-empty and unique")
        for phase_name in self.phase_names:
            if phase_name not in self.phase_budgets_s:
                raise ValueError("phase_budgets_s must include every phase")
            if phase_name not in self.record_count_by_phase:
                raise ValueError("record_count_by_phase must include every phase")
            if phase_name not in self.deadline_filtered_count_by_phase:
                raise ValueError("deadline_filtered_count_by_phase must include every phase")
            if phase_name not in self.zero_delivery_count_by_phase:
                raise ValueError("zero_delivery_count_by_phase must include every phase")
        for matrix in (self.pre_prepare_matrix, self.prepare_matrix, self.commit_matrix):
            _validate_matrix(self.node_ids, matrix)
        if not self.uses_stage3_network_records:
            raise ValueError("Stage 4.3 adapter must state Stage 3 record use")
        if self.exports_consensus_metric:
            raise ValueError("Stage 4.3 adapter must not export consensus metrics")


def _multi_hop_reach(
    node_ids: tuple[str, ...],
    direct_matrix: MessageMatrixDict,
    relay_hops: int,
    *,
    latency_matrix: MessageMatrixDict | None = None,
    phase_budget_s: float | None = None,
) -> MessageMatrixDict:
    """End-to-end relayed reachability over the selected topology: reach[(i, j)] = the
    most-reliable path of at most ``relay_hops`` directed links from i to j (max-product of
    per-link delivery probabilities). With relay_hops=1 this is exactly the direct matrix.

    Relaying through RSU / intermediate nodes lets a node reach validators it has no direct
    (LOS) link to -- the missing ingredient for global PBFT under urban NLOS.

    R2b (Spec S4.2/S4.10) -- LATENCY-AWARE deadline propagation: when ``latency_matrix`` and
    ``phase_budget_s`` are given, a relayed path delivers ONLY if its CUMULATIVE link latency is
    within the phase budget (a relayed message that arrives after the deadline does not count).
    The reachability is then the max delivery product over paths whose latency sum is feasible --
    a constrained optimum, NOT max-delivery with a post-hoc latency check (a slow high-delivery
    path is dropped in favour of a fast feasible one). Computed by a Pareto-label DP over
    (delivery, latency). ``latency_matrix=None`` reproduces the legacy delivery-only relay."""
    if relay_hops <= 1:
        return dict(direct_matrix)
    if latency_matrix is None or phase_budget_s is None:
        return _multi_hop_reach_delivery_only(node_ids, direct_matrix, relay_hops)

    # Directed adjacency of latency-feasible direct links, with (target, delivery, latency).
    adjacency: dict[str, list[tuple[str, float, float]]] = {node: [] for node in node_ids}
    for (source, target), prob in direct_matrix.items():
        latency = latency_matrix.get((source, target), 0.0)
        if prob > 0.0 and latency <= phase_budget_s:
            adjacency[source].append((target, prob, latency))

    # labels[(i, j)] = Pareto front of (delivery, latency) over i->j paths (<= relay_hops links).
    labels: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for source in node_ids:
        for target, prob, latency in adjacency[source]:
            _add_pareto_label(labels.setdefault((source, target), []), prob, latency)

    # Relax up to relay_hops-1 more hops; extend every current label by one direct link.
    for _ in range(relay_hops - 1):
        extended = False
        for (i, k), klabels in list(labels.items()):
            for delivery_ik, latency_ik in list(klabels):
                for j, delivery_kj, latency_kj in adjacency[k]:
                    if j == i:
                        continue
                    cum_latency = latency_ik + latency_kj
                    if cum_latency > phase_budget_s:
                        continue
                    if _add_pareto_label(
                        labels.setdefault((i, j), []), delivery_ik * delivery_kj, cum_latency
                    ):
                        extended = True
        if not extended:
            break

    return {
        pair: max(delivery for delivery, _latency in front)
        for pair, front in labels.items()
        if front
    }


def _multi_hop_reach_delivery_only(
    node_ids: tuple[str, ...], direct_matrix: MessageMatrixDict, relay_hops: int
) -> MessageMatrixDict:
    """The legacy delivery-only relay (max-product over <= relay_hops links, no latency)."""
    nodes = list(node_ids)
    index = {node: k for k, node in enumerate(nodes)}
    size = len(nodes)
    direct = [[0.0] * size for _ in range(size)]
    for (source, target), prob in direct_matrix.items():
        direct[index[source]][index[target]] = prob
    best = [row[:] for row in direct]  # 1-hop
    for _ in range(relay_hops - 1):  # relax: extend best paths by one more hop
        nxt = [row[:] for row in best]
        for i in range(size):
            for k in range(size):
                via = best[i][k]
                if via <= 0.0:
                    continue
                for j in range(size):
                    if j == i or direct[k][j] <= 0.0:
                        continue
                    candidate = via * direct[k][j]
                    if candidate > nxt[i][j]:
                        nxt[i][j] = candidate
        best = nxt
    return {
        (nodes[i], nodes[j]): best[i][j]
        for i in range(size)
        for j in range(size)
        if i != j and best[i][j] > 0.0
    }


def _add_pareto_label(front: list[tuple[float, float]], delivery: float, latency: float) -> bool:
    """Insert (delivery, latency) into a Pareto front maximizing delivery, minimizing latency.

    Returns False (no insert) if an existing label dominates it (delivery >= and latency <=);
    otherwise drops the labels it dominates, appends it, and returns True.
    """
    for existing_d, existing_l in front:
        if existing_d >= delivery - 1e-15 and existing_l <= latency + 1e-15:
            return False
    front[:] = [
        (existing_d, existing_l)
        for existing_d, existing_l in front
        if not (delivery >= existing_d - 1e-15 and latency <= existing_l + 1e-15)
    ]
    front.append((delivery, latency))
    return True


def build_pbft_message_matrices_from_network_records(
    node_ids: tuple[str, ...],
    phase_records: PhaseRecordMap,
    phase_budgets: PBFTPhaseBudgets,
    relay_hops: int = 1,
    perfect_pairs: frozenset[tuple[str, str]] = frozenset(),
    one_hop_relay: bool = False,
) -> PBFTMessageMatrices:
    """Build PBFT phase matrices from Stage 3 network communication records.

    relay_hops > 1 enables multi-hop relaying: each phase's delivery matrix becomes the
    end-to-end most-reliable relayed path (<= relay_hops links) instead of direct-link only.
    Default 1 keeps the single-hop behaviour byte-identical.

    one_hop_relay (Phase 2, Spec S4.2): when True the per-phase matrix is built from ONLY the
    DIRECT-link records (route length 2), so the ``relay_hops`` DP is the SINGLE multi-hop
    layer. This is the correct route/relay semantics (A--B--C: relay_hops=1 -> P(A->C)=0;
    relay_hops=2 -> P(A->C)>0). Default False reproduces the legacy behaviour, where the
    matrix already holds the evaluator's end-to-end BFS-route probabilities and ``relay_hops``
    relays them AGAIN (double counting -- hard-constraint #10). Activation is deferred to the
    scenario recalibration (it shifts reliability); see docs/CURRENT_HEAD_STATUS.md.

    perfect_pairs marks directed (source, target) pairs with an out-of-band reliable channel
    (e.g. WIRED RSU-RSU BACKHAUL): their delivery is set to 1.0 BEFORE the multi-hop pass, so
    relayed paths can route THROUGH the backbone (vehicle -> RSU_a -> wire -> RSU_b -> vehicle).
    Default empty = byte-identical."""

    if relay_hops < 1:
        raise ValueError("relay_hops must be >= 1")
    checked_node_ids = _checked_node_ids(node_ids)
    node_set = set(checked_node_ids)
    for source_id, target_id in perfect_pairs:
        if source_id not in node_set or target_id not in node_set or source_id == target_id:
            raise ValueError("perfect_pairs must be distinct known node pairs")
    matrices: dict[str, MessageMatrixDict] = {}
    record_counts: dict[str, int] = {}
    deadline_counts: dict[str, int] = {}
    zero_counts: dict[str, int] = {}

    for phase_name in PBFT_PHASE_NAMES:
        records = tuple(phase_records.get(phase_name, ()))
        phase_budget = phase_budgets.budget_for_phase(phase_name)
        matrix, latency_matrix, deadline_filtered, zero_delivery = _matrix_for_phase(
            checked_node_ids,
            records,
            phase_budget,
            direct_only=one_hop_relay,
        )
        for pair in perfect_pairs:
            matrix[pair] = 1.0
            latency_matrix[pair] = 0.0  # wired RSU backhaul: out-of-band, negligible latency
        # R2b (Spec S4.2/S4.10): relay deadline propagation -- a relayed path delivers only if its
        # CUMULATIVE link latency is within the phase budget (a slow multi-hop path misses the
        # deadline). Direct links (relay_hops==1) already passed the per-link filter above.
        matrix = _multi_hop_reach(
            checked_node_ids, matrix, relay_hops,
            latency_matrix=latency_matrix, phase_budget_s=phase_budget,
        )
        matrices[phase_name] = matrix
        record_counts[phase_name] = len(records)
        deadline_counts[phase_name] = deadline_filtered
        zero_counts[phase_name] = zero_delivery

    return PBFTMessageMatrices(
        adapter_id=MESSAGE_MATRIX_ADAPTER_ID,
        node_ids=checked_node_ids,
        phase_names=PBFT_PHASE_NAMES,
        pre_prepare_matrix=matrices["pre_prepare"],
        prepare_matrix=matrices["prepare"],
        commit_matrix=matrices["commit"],
        phase_budgets_s={
            "pre_prepare": phase_budgets.pre_prepare_budget_s,
            "prepare": phase_budgets.prepare_budget_s,
            "commit": phase_budgets.commit_budget_s,
        },
        record_count_by_phase=record_counts,
        deadline_filtered_count_by_phase=deadline_counts,
        zero_delivery_count_by_phase=zero_counts,
    )


def evaluate_pbft_reliability_from_network_records(
    pbft_config: PBFTThreePhaseConfig,
    phase_records: PhaseRecordMap,
    phase_budgets: PBFTPhaseBudgets,
) -> PBFTThreePhaseReliabilityRecord:
    """Evaluate PBFT reliability after adapting Stage 3 network records."""

    matrices = build_pbft_message_matrices_from_network_records(
        pbft_config.node_ids,
        phase_records,
        phase_budgets,
    )
    return evaluate_pbft_three_phase_reliability(
        pbft_config,
        pre_prepare_matrix=matrices.pre_prepare_matrix,
        prepare_matrix=matrices.prepare_matrix,
        commit_matrix=matrices.commit_matrix,
    )


def _matrix_for_phase(
    node_ids: tuple[str, ...],
    records: tuple[NetworkCommunicationRecord, ...],
    phase_budget_s: float,
    direct_only: bool = False,
) -> tuple[MessageMatrixDict, MessageMatrixDict, int, int]:
    matrix: MessageMatrixDict = {}
    latency_matrix: MessageMatrixDict = {}  # latency of the kept (max-delivery) record per pair
    deadline_filtered = 0
    zero_delivery = 0
    for record in records:
        _validate_record_nodes(node_ids, record)
        if record.is_oracle:
            raise ValueError("Stage 3 oracle records cannot feed PBFT message matrices")
        # Phase 2 (Spec S4.2): a DIRECT link is a route of exactly two nodes (source, target).
        # Skip multi-hop route records so the relay DP is the single multi-hop layer.
        if direct_only and len(record.route_node_ids) != 2:
            continue
        delivery = 0.0
        if record.network_scheduled_latency_s <= phase_budget_s:
            delivery = record.network_delivery_probability
        else:
            deadline_filtered += len(record.target_ids)
        if delivery == 0.0:
            zero_delivery += len(record.target_ids)
        for target_id in record.target_ids:
            if target_id == record.source_id:
                raise ValueError("self-message targets are not allowed")
            key = (record.source_id, target_id)
            # Keep the max-delivery record per pair, and the LATENCY that goes with it (R2b: the
            # relay DP propagates this along the path and drops paths that miss the phase deadline).
            if key not in matrix or delivery > matrix[key]:
                matrix[key] = delivery
                latency_matrix[key] = record.network_scheduled_latency_s
    return matrix, latency_matrix, deadline_filtered, zero_delivery


def _checked_node_ids(node_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not node_ids:
        raise ValueError("node_ids must be non-empty")
    if any(not node_id for node_id in node_ids):
        raise ValueError("node_ids must contain non-empty ids")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must be unique")
    return node_ids


def _validate_record_nodes(
    node_ids: tuple[str, ...],
    record: NetworkCommunicationRecord,
) -> None:
    node_set = set(node_ids)
    if record.source_id not in node_set:
        raise ValueError("network record source_id must be in node_ids")
    unknown_targets = sorted(set(record.target_ids) - node_set)
    if unknown_targets:
        raise ValueError(f"network record target_ids must be in node_ids: {unknown_targets}")


def _validate_matrix(
    node_ids: tuple[str, ...],
    matrix: Mapping[tuple[str, str], float],
) -> None:
    node_set = set(node_ids)
    for (source_id, target_id), value in matrix.items():
        if source_id not in node_set or target_id not in node_set:
            raise ValueError("matrix endpoints must be in node_ids")
        if source_id == target_id:
            raise ValueError("matrix must not contain self messages")
        if not 0.0 <= value <= 1.0:
            raise ValueError("matrix values must be in [0, 1]")
