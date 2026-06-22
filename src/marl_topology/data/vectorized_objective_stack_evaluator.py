"""Vectorized, equivalence-verified Stage-21 objective-stack evaluator.

The canonical ``Stage21ObjectiveStackEvaluator`` is ~O(N^4): N^2 (src,tgt) routes, each hop
re-running ray-box visibility + a finite-blocklength required-time bisection. At N=48 a single
``evaluate()`` is seconds, which makes the SA teacher / density sweeps / large-N MARL rollouts
infeasible.

This evaluator keeps the physics IDENTICAL but removes the two redundancies:

  1. The per-ordered-pair desired rx_power + desired ``ChannelRecord`` are TOPOLOGY-INDEPENDENT
     (geometry + config + realization only). Precompute them ONCE per (scene, graph, config); a
     hop's SINR is then ``signal_mw / (noise_mw + sum of precomputed interferer powers)``.
  2. With a fixed transmission time, the per-hop required-time BISECTION only fills the record's
     ``required_*`` fields -- it does NOT affect delivery / latency / energy (all that consensus
     depends on). The fast hop-link skips it.

Everything downstream of the network records (PBFT matrices, multi-hop relay, coverage-gated
membership, expected-initiator reliability, accounting, diagnostics) is the SAME canonical src
code, so given identical records the consensus probability is identical. The equivalence is
pinned by ``tests/unit/test_vectorized_objective_stack_evaluator.py`` (|Δp0| < 1e-9 vs the
canonical evaluator across random topologies at several N).

It is a drop-in for ``Stage21ObjectiveStackEvaluator.evaluate`` and reuses a canonical evaluator
instance for ``link_records`` / validator membership / the MAC power table, so it never changes
the physics -- only its cost. Opt-in: the canonical evaluator remains the default and reference.
"""

from __future__ import annotations

from marl_topology.channel import dbm_to_mw, evaluate_channel, mw_to_dbm
from marl_topology.channel.model import noise_power_dbm
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_OBJECTIVE_CONTRACT_ID,
    STAGE21_PHYSICS_REGIME_ID,
    STAGE21_PROTOCOL_MODEL_ID,
    Stage21ObjectiveEvaluation,
    Stage21ObjectiveStackEvaluator,
    _apply_schedule_latency,
    _resource_assignments,
    _restrict_matrix,
    _stdma_schedule_for,
    _topology_diagnostics,
)
from marl_topology.link.transmission import (
    deadline_retransmission_probability,
    dbm_to_watt,
    finite_blocklength_packet_success_probability,
)
from marl_topology.network.communication import (
    NetworkCommunicationRecord,
    _adjacency,
    _delivery_probability,
    _hop_record,
    _interference_group_ids,
    _network_scheduled_latency,
    _network_successful_delivery_latency,
    _reachable_nodes,
    _shortest_path_trace,
    _transmission_specs_for_trace,
)
from marl_topology.protocol import (
    FAULT_FILTER_REMOVE_LARGEST,
    PBFTExpectedInitiatorConfig,
    PBFTPhaseBudgets,
    STRATEGY_AUTO,
    account_pbft_protocol_latency_energy,
    build_pbft_message_matrices_from_network_records,
    evaluate_expected_initiator_pbft_reliability,
    robust_consensus_reliability,
)
from marl_topology.topology.evaluator import topology_id_for_edges


class _FastLink:
    """The 5 fields ``_hop_record`` reads from a LinkTransmissionRecord."""

    __slots__ = (
        "deadline_delivery_probability",
        "packet_success_probability",
        "expected_attempts",
        "p2p_latency_s",
        "p2p_energy_j",
    )

    def __init__(self, delivery, psucc, attempts, latency, energy):
        self.deadline_delivery_probability = delivery
        self.packet_success_probability = psucc
        self.expected_attempts = attempts
        self.p2p_latency_s = latency
        self.p2p_energy_j = energy


class VectorizedStage21Evaluator:
    """Drop-in for ``Stage21ObjectiveStackEvaluator.evaluate``, equivalence-verified on p0."""

    def __init__(self, *, scene, graph, config, ref=None):
        self.scene = scene
        self.graph = graph
        self.config = config
        # Reuse a canonical evaluator for link_records, validator membership, MAC power table.
        # An existing canonical evaluator (same scene/graph/config) may be passed as ``ref`` to
        # avoid a second canonical __init__.
        self._ref = ref if ref is not None else Stage21ObjectiveStackEvaluator(
            scene=scene, graph=graph, config=config
        )
        self.link_records = self._ref.link_records
        self.validator_ids = self._ref.validator_ids
        self._mac_rx_power_mw = self._ref._mac_rx_power_mw
        self._cache: dict[tuple[str, ...], Stage21ObjectiveEvaluation] = {}

        chan = config.channel_config
        self._noise_mw = dbm_to_mw(noise_power_dbm(chan.bandwidth_hz, chan.noise_figure_db))
        self._floor_dbm = chan.no_interference_power_dbm
        # Per-ordered-pair desired channel: rx_power (mw) + a ChannelRecord template to clone.
        self._rxmw: dict[tuple[str, str], float] = {}
        self._desired: dict[tuple[str, str], object] = {}
        nodes = graph.node_ids
        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                rec = evaluate_channel(
                    scene, a, b, config=chan, active_transmissions=(),
                    resource_id="resource_0", tx_power_dbm=None, shadowing_seed=None,
                )
                self._rxmw[(a, b)] = dbm_to_mw(rec.rx_power_dbm)
                self._desired[(a, b)] = rec
        # link-config constants for the bisection-free hop link
        lc = config.link_config
        self._lc = lc
        self._bw = chan.bandwidth_hz if lc.bandwidth_hz is None else lc.bandwidth_hz
        # no-interference hop-link cache, keyed by directed pair (interference_mw == 0)
        self._noint_link: dict[tuple[str, str], _FastLink] = {}

    # ---- fast per-hop link (fixed transmission time -> no required-time bisection) -------
    def _hop_link(self, tx_id, rx_id, sinr_db, tx_power_dbm, distance_m):
        lc = self._lc
        propagation = distance_m / lc.propagation_speed_mps
        transmission = lc.fixed_transmission_time_s
        processing = lc.processing_delay_s
        queueing = lc.queueing_delay_s
        attempt_duration = propagation + transmission + processing + queueing
        psucc = finite_blocklength_packet_success_probability(
            sinr_db=sinr_db, bandwidth_hz=self._bw, transmission_time_s=transmission,
            payload_bits=lc.payload_bits,
            use_normal_approximation_correction=lc.use_normal_approximation_correction,
        )
        if lc.deadline_s is None:
            max_attempts, delivery, attempts = 1, psucc, 1.0
        else:
            max_attempts, delivery, attempts = deadline_retransmission_probability(
                packet_success_probability=psucc, attempt_duration_s=attempt_duration,
                deadline_s=lc.deadline_s,
            )
        tx_power_w = dbm_to_watt(tx_power_dbm)
        attempt_energy = (
            tx_power_w * transmission
            + lc.rx_circuit_power_w * transmission
            + lc.processing_power_w * processing
        )
        return _FastLink(delivery, psucc, attempts, attempts * attempt_duration,
                         attempts * attempt_energy)

    def _hop_channel_link(self, u, v, interferer_les):
        """``interferer_les``: ordered list of interferer tx ids (already filtered: not in
        {u,v}, edge != {src,tgt}, same resource). Returns (interference_tx_ids, _FastLink).
        ``_hop_record`` only consumes interference_tx_ids + the link, so no ChannelRecord is
        built."""
        signal_mw = self._rxmw[(u, v)]
        desired = self._desired[(u, v)]
        if not interferer_les:
            link = self._noint_link.get((u, v))
            if link is None:
                sinr_db = mw_to_dbm(signal_mw / self._noise_mw)
                link = self._hop_link(u, v, sinr_db, desired.tx_power_dbm, desired.distance_3d_m)
                self._noint_link[(u, v)] = link
            return (), link
        interference_mw = 0.0
        for le in interferer_les:
            interference_mw += self._rxmw[(le, v)]
        sinr_db = mw_to_dbm(signal_mw / (self._noise_mw + interference_mw))
        link = self._hop_link(u, v, sinr_db, desired.tx_power_dbm, desired.distance_3d_m)
        return tuple(sorted(set(interferer_les))), link

    # ---- fast _directed_network_records (mirrors canonical, fast hop channel/link) -------
    def _directed_records(self, selected, schedule):
        graph = self.graph
        cfg = self.config
        if schedule is not None:
            resource_assignments = schedule.resource_assignments()
            scheduled_interference = True
        else:
            resource_assignments = _resource_assignments(selected, cfg.orthogonal_resources)
            scheduled_interference = cfg.use_background_interference
        # background interferer specs (edge -> (left, right, resource)); canonical splits edge id.
        edge_lr = {}
        for eid in selected:
            left, right = eid.split("--", 1)
            edge_lr[eid] = (left, right, resource_assignments.get(eid, "resource_0"))

        adjacency = _adjacency(graph, selected)
        is_full = graph.is_full_selection(set(selected))  # hoist O(E) check out of the N^2 loop
        records = []
        nodes = graph.node_ids
        for source_id in nodes:
            reachable = _reachable_nodes(adjacency, source_id)
            for target_id in nodes:
                if source_id == target_id:
                    continue
                route_nodes, route_edges = _shortest_path_trace(
                    adjacency, source_id, target_id, _cfg_max_hops(cfg)
                )
                route_specs = _transmission_specs_for_trace(
                    route_nodes, route_edges, resource_assignments
                )
                current_pair = {source_id, target_id}
                hop_records = []
                for index, spec in enumerate(route_specs):
                    u, v, r = spec.tx_id, spec.rx_id, spec.resource_id
                    if scheduled_interference:
                        interferer_les = [
                            lr[0]
                            for eid, lr in edge_lr.items()
                            if lr[2] == r
                            and {lr[0], lr[1]} != current_pair
                            and lr[0] != u
                            and lr[0] != v
                        ]
                    else:
                        interferer_les = []
                    interference_tx_ids, link = self._hop_channel_link(u, v, interferer_les)
                    hop_records.append(_hop_record(index, spec, link, interference_tx_ids))
                hop_tuple = tuple(hop_records)
                delivery = _delivery_probability(hop_tuple, (target_id,), route_nodes)
                scheduled_latency = _network_scheduled_latency("route", hop_tuple)
                successful_latency = _network_successful_delivery_latency(scheduled_latency, delivery)
                energy = sum(h.p2p_energy_j for h in hop_records)
                record = NetworkCommunicationRecord(
                    scenario_id=graph.scenario_id,
                    network_model_id="stage3_network_communication_v1",
                    primitive="route",
                    source_id=source_id,
                    target_ids=(target_id,),
                    selected_edge_ids=selected,
                    active_transmission_ids=tuple(sorted(s.transmission_id for s in route_specs if s.active)),
                    interference_group_ids=_interference_group_ids(hop_tuple),
                    reachable_node_ids=tuple(sorted(reachable)),
                    route_node_ids=route_nodes,
                    route_edge_ids=route_edges,
                    hop_records=hop_tuple,
                    network_delivery_probability=delivery,
                    network_latency_s=successful_latency,
                    network_scheduled_latency_s=scheduled_latency,
                    network_successful_delivery_latency_s=successful_latency,
                    network_energy_j=energy,
                    hop_count=len(hop_records),
                    is_full_graph_baseline=is_full,
                )
                if schedule is not None:
                    record = _apply_schedule_latency(record, schedule)
                records.append(record)
        return tuple(records)

    # ---- evaluate (mirrors Stage21ObjectiveStackEvaluator.evaluate downstream) -----------
    def evaluate(self, selected_edge_ids, *, topology_id=None):
        selected = tuple(sorted(set(selected_edge_ids)))
        unknown = sorted(set(selected) - set(self.graph.edge_ids))
        if unknown:
            raise ValueError(f"selected unknown edge ids: {unknown}")
        if topology_id is None and selected in self._cache:
            return self._cache[selected]

        schedule = _stdma_schedule_for(
            self.scene, self.graph, selected, self.config, self._mac_rx_power_mw
        )
        records = self._directed_records(selected, schedule)
        phase_records = {"pre_prepare": records, "prepare": records, "commit": records}
        budgets = PBFTPhaseBudgets(
            pre_prepare_budget_s=self.config.phase_budget_s,
            prepare_budget_s=self.config.phase_budget_s,
            commit_budget_s=self.config.phase_budget_s,
        )
        if self.config.wired_rsu_backhaul:
            rsu_ids = [n for n in self.graph.node_ids if n.startswith("rsu_")]
            perfect_pairs = frozenset((a, b) for a in rsu_ids for b in rsu_ids if a != b)
        else:
            perfect_pairs = frozenset()
        matrices = build_pbft_message_matrices_from_network_records(
            self.graph.node_ids, phase_records, budgets,
            relay_hops=self.config.relay_hops, perfect_pairs=perfect_pairs,
            one_hop_relay=self.config.one_hop_relay,
        )
        validators = self.validator_ids
        if len(validators) >= 4:
            if validators == self.graph.node_ids:
                pre, pre2, com = matrices.pre_prepare_matrix, matrices.prepare_matrix, matrices.commit_matrix
            else:
                vs = set(validators)
                pre = _restrict_matrix(matrices.pre_prepare_matrix, vs)
                pre2 = _restrict_matrix(matrices.prepare_matrix, vs)
                com = _restrict_matrix(matrices.commit_matrix, vs)
            fault_tolerance = min(self.config.fault_tolerance, max(0, (len(validators) - 1) // 3))
            if self.config.fault_model == "fixed_set":
                reliability = robust_consensus_reliability(
                    validators, pre_prepare_matrix=pre, prepare_matrix=pre2, commit_matrix=com,
                    fault_tolerance=fault_tolerance, strategy=STRATEGY_AUTO,
                )
            else:
                reliability = evaluate_expected_initiator_pbft_reliability(
                    PBFTExpectedInitiatorConfig(
                        node_ids=validators,
                        fault_tolerance=fault_tolerance,
                        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
                    ),
                    pre_prepare_matrix=pre, prepare_matrix=pre2, commit_matrix=com,
                )
            probability = reliability.consensus_success_probability
            per_primary = dict(reliability.per_primary_reliability)
        else:
            probability = 0.0
            per_primary = {validator: 0.0 for validator in validators}
        accounting = account_pbft_protocol_latency_energy(
            node_ids=self.graph.node_ids, phase_records=phase_records, phase_budgets=budgets,
        )
        diagnostics = _topology_diagnostics(
            graph=self.graph, selected=selected, records=records, matrices=matrices,
            accounting=accounting, config=self.config, schedule=schedule,
        )
        metrics = {
            "consensus_success": int(probability >= self.config.tau_requirement_min),
            "consensus_success_probability": probability,
            "latency": accounting.protocol_latency_s,
            "energy": accounting.protocol_energy_j,
            "topology_diagnostics": diagnostics,
        }
        if self.config.coverage_gated_membership:
            metrics["membership_gated"] = True
            metrics["validator_count"] = len(self.validator_ids)
            metrics["coverage_rate"] = (
                len(self.validator_ids) / len(self.graph.node_ids) if self.graph.node_ids else 0.0
            )
        evaluation = Stage21ObjectiveEvaluation(
            scenario_id=self.graph.scenario_id,
            topology_id=topology_id or topology_id_for_edges(selected),
            selected_edge_ids=selected, metrics=metrics, per_primary_reliability=per_primary,
            records=records, evaluator_id=self.config.evaluator_id,
            physics_regime_id=STAGE21_PHYSICS_REGIME_ID,
            protocol_model_id=STAGE21_PROTOCOL_MODEL_ID,
            objective_contract_id=STAGE21_OBJECTIVE_CONTRACT_ID, diagnostics=diagnostics,
        )
        if topology_id is None:
            if len(self._cache) > 256:  # bound the cache: SA evaluates 1000s of topos, each
                self._cache.clear()      # holding N^2 records -> unbounded cache OOMs at N>=32.
            self._cache[selected] = evaluation
        return evaluation


def _cfg_max_hops(cfg):
    # canonical NetworkCommunicationConfig.max_hops default is None
    return None
