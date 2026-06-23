"""Stage 21 objective-stack-aligned learning evidence rebuild."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from math import isfinite
from statistics import fmean

from marl_topology.channel import ChannelModelConfig, evaluate_channel
from marl_topology.data.actor_feature_rebuild import (
    STAGE18_FEATURE_SCHEMA_ID,
    actor_safe_view_has_no_forbidden_fields,
    build_stage18_actor_safe_feature_rows,
)
from marl_topology.data.learning_evidence import (
    EdgeDeltaTarget,
    LearningEvidenceRow,
    build_learning_evidence_row,
)
from marl_topology.data.learning_evidence_stage16 import (
    STAGE16_SEQUENCE_ID,
    iter_stage16_evidence_scenario_fixtures,
)
from marl_topology.env import ActorObservation
from marl_topology.link import (
    LinkRecord,
    LinkTransmissionConfig,
    URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
    evaluate_link_transmission,
)
from marl_topology.network import (
    NetworkCommunicationConfig,
    NetworkCommunicationRecord,
    NetworkTransmissionSpec,
    evaluate_network_communication,
)
from marl_topology.network.communication import NETWORK_COMMUNICATION_REGIME_ID
from marl_topology.policies import PolicyBaselines, build_local_observations
from marl_topology.protocol import (
    FAULT_FILTER_REMOVE_LARGEST,
    MESSAGE_MATRIX_ADAPTER_ID,
    PBFT_PHASE_NAMES,
    PBFTExpectedInitiatorConfig,
    PBFTPhaseBudgets,
    PBFTQuorumSpec,
    STRATEGY_AUTO,
    account_pbft_protocol_latency_energy,
    build_pbft_message_matrices_from_network_records,
    evaluate_expected_initiator_pbft_reliability,
    robust_consensus_reliability,
)
from marl_topology.protocol.pbft_accounting import PBFT_PROTOCOL_ACCOUNTING_MODEL_ID
from marl_topology.protocol.stdma_scheduler import (
    StdmaSchedule,
    StdmaScheduleConfig,
    build_received_power_table,
    build_stdma_schedule,
)
from marl_topology.scenario import ScenarioFixture
from marl_topology.scenario.scene import Scene3D
from marl_topology.topology import CandidateGraph
from marl_topology.topology.evaluator import topology_id_for_edges

from .stage21_assembler_aware_targets import (
    aggregate_proposal_rejection_diagnostics,
    aggregate_target_distribution,
    actor_target_view_has_no_forbidden_global_fields,
    build_stage21_actor_target_view,
    build_stage21_critic_target_view,
    build_stage21_teacher_projection,
    critic_targets_are_critic_only,
)


STAGE21_STAGE_ID = "stage_21_objective_stack_aligned_assembler_aware_training_rerun"
STAGE21_DATASET_ID = "stage21_objective_stack_aligned_assembler_aware_evidence_v1"
STAGE21_EVALUATOR_ID = "stage21_stage3_urlcc_stage4_expected_initiator_pbft_objective_stack_v1"
STAGE21_PHYSICS_REGIME_ID = URLLC_FINITE_BLOCKLENGTH_REGIME_ID
STAGE21_PROTOCOL_MODEL_ID = "stage4_expected_initiator_pbft_over_stage3_network_v1"
STAGE21_OBJECTIVE_CONTRACT_ID = "stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1"
STAGE21_TAU_REQUIREMENT_MIN = 0.9
STAGE21_PHASE_BUDGET_S = 0.01
STAGE20_BEST_ACTOR_TAU_FEASIBLE_RATE = 0.5593220339
STAGE20_BEST_ACTOR_PROJECTION_REJECTION_RATE = 0.6647058824


class Stage21EvidenceViolation(ValueError):
    """Raised when Stage 21 evidence crosses a declared boundary."""


@dataclass(frozen=True, slots=True)
class Stage21ObjectiveStackConfig:
    """Declared Stage 21 objective-stack parameters."""

    channel_config: ChannelModelConfig
    link_config: LinkTransmissionConfig
    phase_budget_s: float = STAGE21_PHASE_BUDGET_S
    fault_tolerance: int = 1
    tau_requirement_min: float = STAGE21_TAU_REQUIREMENT_MIN
    use_background_interference: bool = False
    orthogonal_resources: bool = True
    # relay_hops > 1 enables multi-hop relaying in consensus: a validator's message reaches
    # peers it has no direct (LOS) link to, via relays through RSU / intermediate nodes. The
    # missing ingredient for global PBFT under urban NLOS. R2 (Spec S4.2): production default 2
    # (the corrected single-relay-layer pair with one_hop_relay=True).
    relay_hops: int = 2
    # scheduled_mac replaces the binary all-orthogonal / all-shared spectrum model with a
    # spatial-reuse TDMA (STDMA) schedule: the selected links are partitioned into a few
    # SINR-feasible time slots (conflict-graph + SINR-validated packing). Co-slot links share a
    # resource (interfere, but validated feasible); cross-slot links are orthogonal -- so each
    # link's reliability sees only its co-slot interferers (rescues urban global PBFT from the
    # worst-case all-interfere model), and a link in slot k incurs (k+1)*slot latency, so dense
    # topologies exceed the phase deadline (the connectivity-vs-latency tradeoff). When
    # scheduled_mac is on, the schedule supersedes orthogonal_resources / use_background_
    # interference. Default off keeps the existing resource model byte-identical.
    scheduled_mac: bool = False
    mac_sinr_threshold_db: float = 0.0
    mac_slot_duration_s: float = 0.0006
    # wired_rsu_backhaul: RSU-RSU pairs get an out-of-band reliable channel (fiber/ethernet
    # roadside backhaul, the standard deployment) -- delivery 1.0 injected before the
    # multi-hop relay pass, so vehicles reach across the city THROUGH the RSU backbone.
    # RSU nodes are identified by the project-wide "rsu_" id prefix. Default off.
    wired_rsu_backhaul: bool = False
    # coverage_gated_membership: validator membership is gated on SCENE-level coverage --
    # a node whose best incident CANDIDATE link delivery is below membership_min_link_delivery
    # is provably unable to act as PBFT primary under any topology (its primary rounds cap
    # expected-initiator consensus at (N - k_dead)/N), so it is demoted to a CLIENT: it does
    # not vote, consensus is evaluated over the covered validator set, and coverage_rate
    # reports the demoted fraction. Membership depends only on the candidate graph (never on
    # the selected topology), so the policy cannot game feasibility by isolating nodes. RSUs
    # stay validators whenever the wired backhaul is on (they reach peers out-of-band).
    # Fewer than 4 validators -> consensus 0.0 (no fault-tolerant quorum exists). Default off.
    coverage_gated_membership: bool = False
    membership_min_link_delivery: float = 0.5
    # --- Phase 0-4 corrected environment math (recalibration knobs; defaults reproduce the
    #     legacy behaviour byte-for-byte so they are inert until the recalibration flips them).
    # fault_model: "remove_largest" (legacy per-phase filter) | "fixed_set" (the principled
    #   single fixed Byzantine set C_robust = min_{|B|<=f} C(x;B), Spec S4.7).
    fault_model: str = "remove_largest"
    # one_hop_relay: when True the PBFT matrix is built from DIRECT links only, so relay_hops is
    #   the single multi-hop layer (Spec S4.2 -- no double-counting). R2: production DEFAULT True
    #   (paired with relay_hops>=2). Legacy double-count mode is opt-in: set False explicitly.
    one_hop_relay: bool = True
    # timeout_aware_latency: when True the "latency" metric is the quorum-completion timeout-aware
    #   consensus time E[min(T,B)] (Spec S4.10) summed over the three phases -- a FAILED topology
    #   pays the full phase budget -- instead of the degenerate min(max_all_pairs, budget).
    timeout_aware_latency: bool = False
    evaluator_id: str = STAGE21_EVALUATOR_ID

    def __post_init__(self) -> None:
        if self.link_config.link_transmission_model_id != URLLC_FINITE_BLOCKLENGTH_REGIME_ID:
            raise Stage21EvidenceViolation("Stage 21 requires finite-blocklength link records")
        if self.phase_budget_s < 0.0 or not isfinite(self.phase_budget_s):
            raise Stage21EvidenceViolation("phase_budget_s must be finite and nonnegative")
        if self.tau_requirement_min != STAGE21_TAU_REQUIREMENT_MIN:
            raise Stage21EvidenceViolation("Stage 21 must keep tau_requirement_min fixed at 0.9")
        if self.scheduled_mac and self.mac_slot_duration_s <= 0.0:
            raise Stage21EvidenceViolation("mac_slot_duration_s must be positive when scheduled_mac is on")
        if not 0.0 <= self.membership_min_link_delivery <= 1.0:
            raise Stage21EvidenceViolation("membership_min_link_delivery must be in [0, 1]")
        if self.fault_model not in ("remove_largest", "fixed_set"):
            raise Stage21EvidenceViolation("fault_model must be 'remove_largest' or 'fixed_set'")


@dataclass(frozen=True, slots=True)
class Stage21ObjectiveEvaluation:
    """Topology evaluation produced by the Stage 21 final objective stack."""

    scenario_id: str
    topology_id: str
    selected_edge_ids: tuple[str, ...]
    metrics: Mapping[str, object]
    per_primary_reliability: Mapping[str, float]
    records: tuple[NetworkCommunicationRecord, ...]
    evaluator_id: str
    physics_regime_id: str
    protocol_model_id: str
    objective_contract_id: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Stage21EvidenceRow:
    """One Stage 21 row with required separated views."""

    evidence_id: str
    scenario_id: str
    time_step: int
    topology_id: str
    topology_name: str
    topology_source: str
    evaluator_id: str
    physics_regime_id: str
    protocol_model_id: str
    objective_contract_id: str
    selected_edges: tuple[str, ...]
    selected_edge_count: int
    consensus_success_probability: float
    per_primary_reliability: Mapping[str, float]
    latency: float
    energy: float
    topology_diagnostics: Mapping[str, object]
    tau_requirement_min: float
    feasible_under_tau_requirement: bool
    actor_safe_view: tuple[Mapping[str, object], ...]
    actor_target_view: Mapping[str, object]
    critic_target_view: Mapping[str, object]
    diagnostics_view: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.evaluator_id != STAGE21_EVALUATOR_ID:
            raise Stage21EvidenceViolation("Stage 21 row used a non-Stage21 evaluator")
        if self.physics_regime_id != STAGE21_PHYSICS_REGIME_ID:
            raise Stage21EvidenceViolation("Stage 21 row used a non-finite-blocklength regime")
        if not actor_safe_view_has_no_forbidden_fields(self.actor_safe_view):
            raise Stage21EvidenceViolation("Stage 21 actor-safe view contains forbidden fields")
        if not actor_target_view_has_no_forbidden_global_fields(self.actor_target_view):
            raise Stage21EvidenceViolation("Stage 21 actor target view contains global fields")
        if not critic_targets_are_critic_only(self.critic_target_view):
            raise Stage21EvidenceViolation("Stage 21 critic targets must be critic-only")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "scenario_id": self.scenario_id,
            "time_step": self.time_step,
            "topology_id": self.topology_id,
            "topology_name": self.topology_name,
            "topology_source": self.topology_source,
            "evaluator_id": self.evaluator_id,
            "physics_regime_id": self.physics_regime_id,
            "protocol_model_id": self.protocol_model_id,
            "objective_contract_id": self.objective_contract_id,
            "selected_edges": list(self.selected_edges),
            "selected_edge_count": self.selected_edge_count,
            "consensus_success_probability": self.consensus_success_probability,
            "per_primary_reliability": dict(self.per_primary_reliability),
            "latency": self.latency,
            "energy": self.energy,
            "topology_diagnostics": _jsonable(self.topology_diagnostics),
            "tau_requirement_min": self.tau_requirement_min,
            "feasible_under_tau_requirement": self.feasible_under_tau_requirement,
            "actor_safe_view": _jsonable(self.actor_safe_view),
            "actor_target_view": _jsonable(self.actor_target_view),
            "critic_target_view": _jsonable(self.critic_target_view),
            "diagnostics_view": _jsonable(self.diagnostics_view),
        }


@dataclass(frozen=True, slots=True)
class Stage21EvidenceDataset:
    dataset_id: str
    rows: tuple[Stage21EvidenceRow, ...]
    source_audit: Mapping[str, object]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def actor_edge_sample_count(self) -> int:
        return sum(len(row.actor_safe_view) for row in self.rows)

    @property
    def ranking_pair_count(self) -> int:
        return sum(
            len(tuple(row.actor_target_view.get("pairwise_ranking_targets", ())))
            for row in self.rows
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "actor_edge_sample_count": self.actor_edge_sample_count,
            "ranking_pair_count": self.ranking_pair_count,
            "source_audit": _jsonable(self.source_audit),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class Stage21EvidenceBuild:
    dataset: Stage21EvidenceDataset
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Stage21EvaluationContext:
    fixture: ScenarioFixture
    time_step: int
    sequence_id: str | None
    graph: CandidateGraph
    evaluator: "Stage21ObjectiveStackEvaluator"
    link_records: Mapping[str, LinkRecord]
    topology_variants: Mapping[str, tuple[str, ...]]


class Stage21ObjectiveStackEvaluator:
    """Evaluate topologies through Stage 3 network and Stage 4 PBFT records."""

    def __init__(
        self,
        *,
        scene: Scene3D,
        graph: CandidateGraph,
        config: Stage21ObjectiveStackConfig,
    ) -> None:
        self.scene = scene
        self.graph = graph
        self.config = config
        self._cache: dict[tuple[str, ...], Stage21ObjectiveEvaluation] = {}
        self.link_records = _build_stage21_link_records(scene, graph, config)
        # Precompute the per-scene received-power table ONCE when scheduled MAC is on, so
        # the STDMA scheduler (called on every evaluate, incl. the SA teacher's thousands of
        # evaluations) does not redo n^2 ray-box visibility evals each time.
        self._mac_rx_power_mw: dict[tuple[str, str], float] | None = (
            build_received_power_table(scene, graph.node_ids, config.channel_config)
            if config.scheduled_mac
            else None
        )
        # Validator membership is a SCENE property (candidate graph + link physics), fixed
        # before any topology is chosen -- see Stage21ObjectiveStackConfig docs.
        self.validator_ids: tuple[str, ...] = (
            _coverage_gated_validators(graph, self.link_records, config)
            if config.coverage_gated_membership
            else graph.node_ids
        )

    def evaluate(
        self,
        selected_edge_ids: Iterable[str],
        *,
        topology_id: str | None = None,
    ) -> Stage21ObjectiveEvaluation:
        selected = tuple(sorted(set(selected_edge_ids)))
        unknown = sorted(set(selected) - set(self.graph.edge_ids))
        if unknown:
            raise Stage21EvidenceViolation(f"selected unknown edge ids: {unknown}")
        if topology_id is None and selected in self._cache:
            return self._cache[selected]

        schedule = _stdma_schedule_for(
            self.scene, self.graph, selected, self.config, self._mac_rx_power_mw
        )
        records = _directed_network_records(
            self.scene,
            self.graph,
            selected,
            self.config,
            schedule,
        )
        phase_records = {
            "pre_prepare": records,
            "prepare": records,
            "commit": records,
        }
        budgets = PBFTPhaseBudgets(
            pre_prepare_budget_s=self.config.phase_budget_s,
            prepare_budget_s=self.config.phase_budget_s,
            commit_budget_s=self.config.phase_budget_s,
        )
        if self.config.wired_rsu_backhaul:
            rsu_ids = [n for n in self.graph.node_ids if n.startswith("rsu_")]
            perfect_pairs = frozenset(
                (a, b) for a in rsu_ids for b in rsu_ids if a != b
            )
        else:
            perfect_pairs = frozenset()
        matrices = build_pbft_message_matrices_from_network_records(
            self.graph.node_ids,
            phase_records,
            budgets,
            relay_hops=self.config.relay_hops,
            perfect_pairs=perfect_pairs,
            one_hop_relay=self.config.one_hop_relay,
        )
        validators = self.validator_ids
        fault_tolerance = min(self.config.fault_tolerance, max(0, (len(validators) - 1) // 3))
        if len(validators) >= 4:
            if validators == self.graph.node_ids:
                pre_prepare = matrices.pre_prepare_matrix
                prepare = matrices.prepare_matrix
                commit = matrices.commit_matrix
            else:
                # Consensus runs among validators only; messages may still RELAY through
                # client nodes (the matrices were built over the full node set).
                validator_set = set(validators)
                pre_prepare = _restrict_matrix(matrices.pre_prepare_matrix, validator_set)
                prepare = _restrict_matrix(matrices.prepare_matrix, validator_set)
                commit = _restrict_matrix(matrices.commit_matrix, validator_set)
            if self.config.fault_model == "fixed_set":
                reliability = robust_consensus_reliability(
                    validators,
                    pre_prepare_matrix=pre_prepare,
                    prepare_matrix=prepare,
                    commit_matrix=commit,
                    fault_tolerance=fault_tolerance,
                    strategy=STRATEGY_AUTO,
                )
            else:
                reliability = evaluate_expected_initiator_pbft_reliability(
                    PBFTExpectedInitiatorConfig(
                        node_ids=validators,
                        fault_tolerance=fault_tolerance,
                        fault_filter_mode=FAULT_FILTER_REMOVE_LARGEST,
                    ),
                    pre_prepare_matrix=pre_prepare,
                    prepare_matrix=prepare,
                    commit_matrix=commit,
                )
            probability = reliability.consensus_success_probability
            per_primary = dict(reliability.per_primary_reliability)
            reliability_result = reliability
        else:
            # No fault-tolerant quorum exists below 4 validators (PBFT n >= 3f + 1, f >= 1).
            probability = 0.0
            per_primary = {validator: 0.0 for validator in validators}
            reliability_result = None
        fault_accounting = _build_fault_accounting(
            validators=tuple(validators),
            configured_fault_tolerance=int(self.config.fault_tolerance),
            effective_fault_tolerance=int(fault_tolerance),
            fault_model=self.config.fault_model,
            reliability=reliability_result,
        )
        accounting = account_pbft_protocol_latency_energy(
            node_ids=self.graph.node_ids,
            phase_records=phase_records,
            phase_budgets=budgets,
        )
        diagnostics = _topology_diagnostics(
            graph=self.graph,
            selected=selected,
            records=records,
            matrices=matrices,
            accounting=accounting,
            config=self.config,
            schedule=schedule,
        )
        if self.config.timeout_aware_latency:
            latency_value = _consensus_completion_latency(
                validators, fault_tolerance, phase_records, self.config.phase_budget_s
            )
        else:
            latency_value = accounting.protocol_latency_s
        metrics = {
            "consensus_success": int(probability >= self.config.tau_requirement_min),
            "consensus_success_probability": probability,
            "latency": latency_value,
            "energy": accounting.protocol_energy_j,
            "topology_diagnostics": diagnostics,
            "fault_accounting": fault_accounting,
        }
        if self.config.coverage_gated_membership:
            metrics["membership_gated"] = True
            metrics["validator_count"] = len(self.validator_ids)
            metrics["coverage_rate"] = (
                len(self.validator_ids) / len(self.graph.node_ids)
                if self.graph.node_ids
                else 0.0
            )
        evaluation = Stage21ObjectiveEvaluation(
            scenario_id=self.graph.scenario_id,
            topology_id=topology_id or topology_id_for_edges(selected),
            selected_edge_ids=selected,
            metrics=metrics,
            per_primary_reliability=per_primary,
            records=records,
            evaluator_id=self.config.evaluator_id,
            physics_regime_id=STAGE21_PHYSICS_REGIME_ID,
            protocol_model_id=STAGE21_PROTOCOL_MODEL_ID,
            objective_contract_id=STAGE21_OBJECTIVE_CONTRACT_ID,
            diagnostics=diagnostics,
        )
        if topology_id is None:
            # Bound the memo: an RL/SA caller evaluates thousands of DISTINCT topologies, each holding
            # O(N^2) link records -> an unbounded cache OOMs at large N. The cache is a pure performance
            # memo (no semantic effect), so a clear-all at the bound is behaviorally transparent.
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[selected] = evaluation
        return evaluation


def audit_stage16_to_stage20_evidence_stack() -> dict[str, object]:
    """Report whether prior learning/evaluation stages used the final stack."""

    return {
        "stage16_learning_evidence": {
            "uses_early_minimal_skeleton_evaluator": True,
            "uses_simple_link_model": True,
            "uses_topology_evaluator_min_link_abstraction": True,
            "uses_stage3_urlcc_finite_blocklength_stack": False,
            "uses_stage4_expected_initiator_pbft": False,
            "uses_stage5_objective_contract": "partial_tau_latency_energy_names_only",
            "evidence": "build_fixture_stack constructs SimpleLinkModel and TopologyEvaluator",
        },
        "stage18_evidence_rebuild": {
            "uses_early_minimal_skeleton_evaluator": True,
            "uses_simple_link_model": True,
            "uses_topology_evaluator_min_link_abstraction": True,
            "uses_stage3_urlcc_finite_blocklength_stack": False,
            "uses_stage4_expected_initiator_pbft": False,
            "uses_stage5_objective_contract": "inherits_stage16_partial_contract",
            "evidence": "Stage18 rebuild consumes Stage16 rows and targets",
        },
        "stage19_supervised_actor_rerun": {
            "uses_early_minimal_skeleton_evaluator": True,
            "uses_simple_link_model": True,
            "uses_topology_evaluator_min_link_abstraction": True,
            "uses_stage3_urlcc_finite_blocklength_stack": False,
            "uses_stage4_expected_initiator_pbft": False,
            "uses_stage5_objective_contract": "inherits_stage18_targets",
            "evidence": "Stage19 trains on Stage18 disambiguated evidence",
        },
        "stage20_assembler_evaluation": {
            "uses_early_minimal_skeleton_evaluator": True,
            "uses_simple_link_model": True,
            "uses_topology_evaluator_min_link_abstraction": True,
            "uses_stage3_urlcc_finite_blocklength_stack": False,
            "uses_stage4_expected_initiator_pbft": False,
            "uses_stage5_objective_contract": "inherits_stage16_tau_requirement",
            "evidence": "Stage20 fixture lookup uses build_fixture_stack for evaluator and baselines",
        },
        "stage21_required_repair": (
            "Main learning and evaluation evidence must use explicit Stage 3 "
            "network records and Stage 4 expected-initiator PBFT reliability."
        ),
    }


@lru_cache(maxsize=1)
def build_stage21_objective_stack_contexts() -> tuple[Stage21EvaluationContext, ...]:
    contexts: list[Stage21EvaluationContext] = []
    for fixture, time_step, sequence_id in iter_stage16_evidence_scenario_fixtures():
        graph = CandidateGraph.from_scene(
            fixture.scene,
            max_distance_m=fixture.max_candidate_distance_m,
        )
        config = _config_for_fixture(fixture)
        evaluator = Stage21ObjectiveStackEvaluator(
            scene=fixture.scene,
            graph=graph,
            config=config,
        )
        variants = _stage21_topology_variants(graph, evaluator.link_records, fixture.quorum_size)
        contexts.append(
            Stage21EvaluationContext(
                fixture=fixture,
                time_step=time_step,
                sequence_id=sequence_id,
                graph=graph,
                evaluator=evaluator,
                link_records=evaluator.link_records,
                topology_variants=variants,
            )
        )
    return tuple(contexts)


@lru_cache(maxsize=1)
def build_stage21_objective_stack_evidence_dataset() -> Stage21EvidenceDataset:
    """Build Stage 21 evidence rows using the final objective stack."""

    source_audit = audit_stage16_to_stage20_evidence_stack()
    previous_selected_by_key: dict[tuple[str, str, int], tuple[str, ...]] = {}
    rows: list[Stage21EvidenceRow] = []

    contexts = build_stage21_objective_stack_contexts()
    for context in contexts:
        observations = build_local_observations(
            scene=context.fixture.scene,
            graph=context.graph,
            link_records=context.link_records,
            time_step=context.time_step,
        )
        oracle_edges = _oracle_diagnostic_edges(context.evaluator)
        topology_variants = dict(context.topology_variants)
        if oracle_edges is not None:
            topology_variants["oracle_diagnostic"] = oracle_edges

        for topology_name, selected_edges in topology_variants.items():
            sequence_key = (
                str(context.sequence_id or ""),
                topology_name,
                context.time_step - 1,
            )
            previous_selected = previous_selected_by_key.get(sequence_key, ())
            evaluation = context.evaluator.evaluate(
                selected_edges,
                topology_id=(
                    f"stage21:{context.fixture.fixture_id}:{topology_name}:"
                    f"t{context.time_step}"
                ),
            )
            raw_targets = _build_stage21_edge_delta_targets(
                context.evaluator,
                evaluation.selected_edge_ids,
                topology_id=evaluation.topology_id,
                tau_requirement_min=STAGE21_TAU_REQUIREMENT_MIN,
            )
            source_row = _source_learning_evidence_row(
                evaluation=evaluation,
                topology_name=topology_name,
                observations=observations,
                context=context,
                learning_targets=(target.to_dict() for target in raw_targets),
            )
            actor_safe_view = build_stage18_actor_safe_feature_rows(
                source_row,
                previous_selected_edges=previous_selected,
            )
            teacher_projection = build_stage21_teacher_projection(
                actor_safe_view,
                proposed_edge_ids=context.topology_variants["greedy_reliability_raw"],
                teacher_source="projected_greedy_reliability_teacher",
                include_non_proposed_for_targets=False,
            )
            actor_target_view = build_stage21_actor_target_view(
                actor_safe_view=actor_safe_view,
                teacher_projection=teacher_projection,
            )
            critic_target_view = build_stage21_critic_target_view(
                raw_targets=(target.to_dict() for target in raw_targets),
                selected_edges=evaluation.selected_edge_ids,
                per_primary_reliability=evaluation.per_primary_reliability,
                objective_contract_id=STAGE21_OBJECTIVE_CONTRACT_ID,
            )
            diagnostics_view = _diagnostics_view(
                context=context,
                topology_name=topology_name,
                topology_source=_topology_source(topology_name),
                evaluation=evaluation,
                teacher_projection=teacher_projection,
            )
            row = Stage21EvidenceRow(
                evidence_id=(
                    f"stage21:{context.fixture.fixture_id}:t{context.time_step}:"
                    f"{topology_name}"
                ),
                scenario_id=evaluation.scenario_id,
                time_step=context.time_step,
                topology_id=evaluation.topology_id,
                topology_name=topology_name,
                topology_source=_topology_source(topology_name),
                evaluator_id=evaluation.evaluator_id,
                physics_regime_id=evaluation.physics_regime_id,
                protocol_model_id=evaluation.protocol_model_id,
                objective_contract_id=evaluation.objective_contract_id,
                selected_edges=evaluation.selected_edge_ids,
                selected_edge_count=len(evaluation.selected_edge_ids),
                consensus_success_probability=float(
                    evaluation.metrics["consensus_success_probability"]
                ),
                per_primary_reliability=evaluation.per_primary_reliability,
                latency=float(evaluation.metrics["latency"]),
                energy=float(evaluation.metrics["energy"]),
                topology_diagnostics=evaluation.metrics["topology_diagnostics"],
                tau_requirement_min=STAGE21_TAU_REQUIREMENT_MIN,
                feasible_under_tau_requirement=(
                    float(evaluation.metrics["consensus_success_probability"])
                    >= STAGE21_TAU_REQUIREMENT_MIN
                ),
                actor_safe_view=actor_safe_view,
                actor_target_view=actor_target_view,
                critic_target_view=critic_target_view,
                diagnostics_view=diagnostics_view,
            )
            rows.append(row)
            if context.sequence_id is not None:
                previous_selected_by_key[
                    (str(context.sequence_id), topology_name, context.time_step)
                ] = evaluation.selected_edge_ids

    return Stage21EvidenceDataset(
        dataset_id=STAGE21_DATASET_ID,
        rows=tuple(rows),
        source_audit=source_audit,
    )


def build_stage21_objective_stack_evidence_report() -> Stage21EvidenceBuild:
    dataset = build_stage21_objective_stack_evidence_dataset()
    readiness = _evidence_readiness(dataset)
    target_summary = aggregate_target_distribution(
        row.actor_target_view for row in dataset.rows
    )
    projection_summary = aggregate_proposal_rejection_diagnostics(
        row.actor_target_view["proposal_rejection_diagnostics"]
        for row in dataset.rows
    )
    report = {
        "stage": STAGE21_STAGE_ID,
        "dataset_id": dataset.dataset_id,
        "row_count": dataset.row_count,
        "actor_edge_sample_count": dataset.actor_edge_sample_count,
        "ranking_pair_count": dataset.ranking_pair_count,
        "source_audit": dataset.source_audit,
        "main_evaluator_id": STAGE21_EVALUATOR_ID,
        "physics_regime_id": STAGE21_PHYSICS_REGIME_ID,
        "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
        "objective_contract_id": STAGE21_OBJECTIVE_CONTRACT_ID,
        "tau_requirement_min": STAGE21_TAU_REQUIREMENT_MIN,
        "all_rows_declare_evaluator_id": all(bool(row.evaluator_id) for row in dataset.rows),
        "all_rows_use_final_objective_stack": all(
            row.evaluator_id == STAGE21_EVALUATOR_ID
            and row.physics_regime_id == STAGE21_PHYSICS_REGIME_ID
            and row.protocol_model_id == STAGE21_PROTOCOL_MODEL_ID
            for row in dataset.rows
        ),
        "silent_fallback_to_simple_link_or_min_link": False,
        "partial_readiness_flag": False,
        "evidence_readiness": readiness,
        "target_distribution": target_summary,
        "teacher_projection_rejection_diagnostics": projection_summary,
        "actor_safe_view_has_no_forbidden_fields": all(
            actor_safe_view_has_no_forbidden_fields(row.actor_safe_view)
            for row in dataset.rows
        ),
        "actor_target_view_has_no_global_delta_fields": all(
            actor_target_view_has_no_forbidden_global_fields(row.actor_target_view)
            for row in dataset.rows
        ),
        "critic_target_view_is_critic_only": all(
            critic_targets_are_critic_only(row.critic_target_view)
            for row in dataset.rows
        ),
        "stage21_evidence_ready_for_supervised_rerun": readiness["all_required_evidence_present"],
        "policy_gradient_allowed_from_evidence_only": False,
        "artifact_written": False,
        "checkpoint_written": False,
        "v5_modified": False,
    }
    return Stage21EvidenceBuild(dataset=dataset, report=report)


def stage21_oracle_diagnostic_edges(
    evaluator: Stage21ObjectiveStackEvaluator,
) -> tuple[str, ...] | None:
    """Return the small-fixture objective-stack oracle diagnostic topology."""

    return _oracle_diagnostic_edges(evaluator)


def _config_for_fixture(fixture: ScenarioFixture) -> Stage21ObjectiveStackConfig:
    fixture_id = fixture.fixture_id
    tx_power = 20.0
    bandwidth = 20e6
    deadline_s = 0.003
    use_background = False
    orthogonal = True
    if "weak_primary" in fixture_id:
        tx_power = -20.0
    if "interference_penalty_proxy" in fixture_id:
        tx_power = -5.0
        use_background = True
        orthogonal = False
    return Stage21ObjectiveStackConfig(
        channel_config=ChannelModelConfig(
            default_tx_power_dbm=tx_power,
            bandwidth_hz=bandwidth,
        ),
        link_config=LinkTransmissionConfig(
            payload_bits=12_000,
            bandwidth_hz=bandwidth,
            fixed_transmission_time_s=0.0005,
            target_reliability=0.99,
            deadline_s=deadline_s,
        ),
        use_background_interference=use_background,
        orthogonal_resources=orthogonal,
    )


def _build_stage21_link_records(
    scene: Scene3D,
    graph: CandidateGraph,
    config: Stage21ObjectiveStackConfig,
) -> dict[str, LinkRecord]:
    records: dict[str, LinkRecord] = {}
    for edge in graph.edges:
        channel = evaluate_channel(
            scene,
            edge.node_u,
            edge.node_v,
            config=config.channel_config,
            resource_id="resource_0",
        )
        link = evaluate_link_transmission(
            channel,
            config.link_config,
            selected=True,
            active=True,
        )
        records[edge.edge_id] = LinkRecord(
            edge_id=edge.edge_id,
            tx_id=edge.node_u,
            rx_id=edge.node_v,
            distance_3d_m=edge.distance_3d_m,
            link_success_probability=link.deadline_delivery_probability,
            latency_s=link.expected_latency_s,
            energy_j=link.expected_energy_j,
            physics_regime=URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
        )
    return records


def _coverage_gated_validators(
    graph: CandidateGraph,
    link_records: Mapping[str, LinkRecord],
    config: Stage21ObjectiveStackConfig,
) -> tuple[str, ...]:
    """Scene-level validator set: nodes whose best incident CANDIDATE link can deliver.

    A node below the delivery floor on every candidate link cannot complete a primary
    round under ANY topology (relay products through its incident links are no larger),
    so demoting it to client removes a certified-impossible obligation, not a hard case.
    RSUs keep membership whenever the wired backhaul provides out-of-band reach."""
    best_incident: dict[str, float] = {node_id: 0.0 for node_id in graph.node_ids}
    for edge in graph.edges:
        delivery = link_records[edge.edge_id].link_success_probability
        if delivery > best_incident[edge.node_u]:
            best_incident[edge.node_u] = delivery
        if delivery > best_incident[edge.node_v]:
            best_incident[edge.node_v] = delivery
    rsu_backhaul_ids = (
        {n for n in graph.node_ids if n.startswith("rsu_")}
        if config.wired_rsu_backhaul
        else set()
    )
    return tuple(
        node_id
        for node_id in graph.node_ids
        if best_incident[node_id] >= config.membership_min_link_delivery
        or node_id in rsu_backhaul_ids
    )


def _restrict_matrix(
    matrix: Mapping[tuple[str, str], float],
    node_set: set[str],
) -> dict[tuple[str, str], float]:
    return {
        key: value
        for key, value in matrix.items()
        if key[0] in node_set and key[1] in node_set
    }


def _build_fault_accounting(
    *,
    validators: tuple[str, ...],
    configured_fault_tolerance: int,
    effective_fault_tolerance: int,
    fault_model: str,
    reliability: object | None,
) -> dict[str, object]:
    """Auditable PBFT fault/quorum accounting for one production evaluation (Spec S4.7.2).

    Records what the evaluator ACTUALLY executed: validator count, the configured vs the
    effective (n-clamped) fault tolerance, the quorum and external quorum, the fault strategy,
    and -- for the principled ``fixed_set`` model -- the searched fault-set count, whether the
    enumeration was exact, whether the worst case is CERTIFIED (exact hard-min over all |B|<=f),
    and the worst-case fault set. ``remove_largest`` is a per-receiver/per-phase heuristic, NOT a
    single coherent fixed ``B``, so it is never certified.
    """

    accounting: dict[str, object] = {
        "validator_count": len(validators),
        "configured_fault_tolerance": int(configured_fault_tolerance),
        "effective_fault_tolerance": int(effective_fault_tolerance),
        "fault_strategy": fault_model,
        "quorum": None,
        "external_quorum": None,
        "fault_set_count": None,
        "enumeration_exact": False,
        "is_certified": False,
        "worst_case_fault_set": None,
    }
    if reliability is None:  # < 4 validators: no fault-tolerant quorum exists
        return accounting
    if fault_model == "fixed_set":
        accounting.update(
            quorum=int(reliability.quorum),
            external_quorum=int(reliability.external_quorum),
            fault_set_count=int(reliability.fault_set_count),
            enumeration_exact=bool(reliability.enumeration_exact),
            is_certified=bool(reliability.is_certified),
            worst_case_fault_set=tuple(reliability.worst_case_fault_set),
        )
    else:  # remove_largest: quorum is still well-defined; the worst case is not certified
        spec = PBFTQuorumSpec(node_count=len(validators), fault_tolerance=effective_fault_tolerance)
        accounting.update(quorum=int(spec.quorum), external_quorum=int(spec.external_quorum))
    return accounting


def _consensus_completion_latency(
    validators: tuple[str, ...],
    fault_tolerance: int,
    phase_records: Mapping[str, tuple],
    phase_budget_s: float,
) -> float:
    """Quorum-completion, timeout-aware consensus latency (Spec S4.10), summed over the
    three PBFT phases. A FAILED topology pays the full phase budget per phase (a timeout),
    instead of the degenerate max-all-pairs latency. Uses each validator-to-validator route
    record's scheduled latency + delivery as the per-message arrival CDF step."""

    from marl_topology.protocol.quorum_completion_latency import quorum_completion_latency

    if len(validators) < 4:
        # No fault-tolerant quorum exists -> consensus never completes -> full timeout/phase.
        return 3.0 * phase_budget_s
    spec = PBFTQuorumSpec(node_count=len(validators), fault_tolerance=fault_tolerance)
    validator_set = set(validators)
    total = 0.0
    for phase_name in PBFT_PHASE_NAMES:
        arrival: dict[tuple[str, str], float] = {}
        delivery: dict[tuple[str, str], float] = {}
        for record in phase_records.get(phase_name, ()):  # type: ignore[union-attr]
            if record.source_id not in validator_set:
                continue
            for target_id in record.target_ids:
                if target_id in validator_set and target_id != record.source_id:
                    arrival[(record.source_id, target_id)] = record.network_scheduled_latency_s
                    delivery[(record.source_id, target_id)] = record.network_delivery_probability
        total += quorum_completion_latency(
            validators,
            arrival_latencies=arrival,
            deliveries=delivery,
            external_quorum=spec.external_quorum,
            global_quorum=spec.quorum,
            phase_budget_s=phase_budget_s,
        ).expected_s
    return total


def _stdma_schedule_for(
    scene: Scene3D,
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    config: Stage21ObjectiveStackConfig,
    rx_power_mw: Mapping[tuple[str, str], float] | None = None,
) -> StdmaSchedule | None:
    """Build the STDMA slot schedule for the selected links, or ``None`` when
    scheduled MAC is disabled (preserving the legacy resource model exactly)."""
    if not config.scheduled_mac:
        return None
    return build_stdma_schedule(
        scene=scene,
        graph=graph,
        selected_edge_ids=selected_edge_ids,
        channel_config=config.channel_config,
        config=StdmaScheduleConfig(
            sinr_threshold_db=config.mac_sinr_threshold_db,
            slot_duration_s=config.mac_slot_duration_s,
        ),
        rx_power_mw=rx_power_mw,
    )


def _directed_network_records(
    scene: Scene3D,
    graph: CandidateGraph,
    selected_edge_ids: tuple[str, ...],
    config: Stage21ObjectiveStackConfig,
    schedule: StdmaSchedule | None = None,
) -> tuple[NetworkCommunicationRecord, ...]:
    records: list[NetworkCommunicationRecord] = []
    if schedule is not None:
        # Scheduled MAC: slots are the resources (co-slot links interfere and are
        # SINR-validated, cross-slot links are orthogonal); every other selected link
        # is a candidate interferer (the channel keeps only same-slot/same-resource
        # ones). The schedule supersedes orthogonal_resources / use_background.
        resource_assignments = schedule.resource_assignments()
        scheduled_interference = True
    else:
        resource_assignments = _resource_assignments(selected_edge_ids, config.orthogonal_resources)
        scheduled_interference = config.use_background_interference
    for source_id in graph.node_ids:
        for target_id in graph.node_ids:
            if source_id == target_id:
                continue
            record = evaluate_network_communication(
                scene=scene,
                graph=graph,
                selected_edge_ids=selected_edge_ids,
                source_id=source_id,
                target_ids=(target_id,),
                config=NetworkCommunicationConfig(
                    channel_config=config.channel_config,
                    link_config=config.link_config,
                ),
                resource_assignments=resource_assignments,
                background_transmissions=_background_transmissions(
                    selected_edge_ids,
                    source_id,
                    target_id,
                    resource_assignments,
                    scheduled_interference,
                ),
            )
            if schedule is not None:
                record = _apply_schedule_latency(record, schedule)
            records.append(record)
    return tuple(records)


def _apply_schedule_latency(
    record: NetworkCommunicationRecord,
    schedule: StdmaSchedule,
) -> NetworkCommunicationRecord:
    """Charge a route its TDMA scheduling delay: it cannot finish until the latest
    slot among its hops fires. This feeds the existing phase-deadline filter -- a
    route whose (transmission + scheduling) latency exceeds the phase budget is
    dropped -- so dense topologies (more slots) lose late-slot links."""
    delta = schedule.route_schedule_latency_s(record.route_edge_ids)
    if delta <= 0.0:
        return record
    scheduled = record.network_scheduled_latency_s + delta
    if record.network_delivery_probability > 0.0:
        successful = record.network_successful_delivery_latency_s + delta
    else:
        successful = record.network_successful_delivery_latency_s
    return replace(
        record,
        network_scheduled_latency_s=scheduled,
        network_successful_delivery_latency_s=successful,
        network_latency_s=successful,
    )


def _resource_assignments(
    selected_edge_ids: tuple[str, ...],
    orthogonal_resources: bool,
) -> dict[str, str]:
    if not orthogonal_resources:
        return {edge_id: "resource_0" for edge_id in selected_edge_ids}
    return {
        edge_id: f"resource_{index}"
        for index, edge_id in enumerate(selected_edge_ids)
    }


def _background_transmissions(
    selected_edge_ids: tuple[str, ...],
    source_id: str,
    target_id: str,
    resource_assignments: Mapping[str, str],
    use_background_interference: bool,
) -> tuple[NetworkTransmissionSpec, ...]:
    if not use_background_interference:
        return ()
    current_pair = {source_id, target_id}
    specs: list[NetworkTransmissionSpec] = []
    for edge_id in selected_edge_ids:
        left, right = edge_id.split("--", 1)
        if {left, right} == current_pair:
            continue
        specs.append(
            NetworkTransmissionSpec(
                edge_id=edge_id,
                tx_id=left,
                rx_id=right,
                resource_id=resource_assignments.get(edge_id, "resource_0"),
            )
        )
    return tuple(specs)


def _stage21_topology_variants(
    graph: CandidateGraph,
    link_records: Mapping[str, LinkRecord],
    quorum_size: int,
) -> dict[str, tuple[str, ...]]:
    greedy = PolicyBaselines.greedy_reliability(graph, link_records).edge_ids
    return {
        "empty_raw": (),
        "single_best_edge_raw": _single_best_edge(graph, link_records),
        "sparse_quorum_raw": _leader_quorum_edges(graph, link_records, quorum_size),
        "greedy_reliability_raw": greedy,
        "full_graph_raw": PolicyBaselines.full(graph).edge_ids,
        "random_raw": PolicyBaselines.random(graph, seed=21, edge_probability=0.5).edge_ids,
    }


def _single_best_edge(
    graph: CandidateGraph,
    link_records: Mapping[str, LinkRecord],
) -> tuple[str, ...]:
    if not graph.edge_ids:
        return ()
    return (
        max(
            graph.edge_ids,
            key=lambda edge_id: (
                link_records[edge_id].link_success_probability,
                -link_records[edge_id].energy_j,
                edge_id,
            ),
        ),
    )


def _leader_quorum_edges(
    graph: CandidateGraph,
    link_records: Mapping[str, LinkRecord],
    quorum_size: int,
) -> tuple[str, ...]:
    if not graph.node_ids:
        return ()
    leader = graph.node_ids[0]
    incident = [
        edge.edge_id
        for edge in graph.edges
        if edge.node_u == leader or edge.node_v == leader
    ]
    ranked = sorted(
        incident,
        key=lambda edge_id: (
            -link_records[edge_id].link_success_probability,
            link_records[edge_id].energy_j,
            edge_id,
        ),
    )
    target_edges = max(1, quorum_size - 1)
    return tuple(sorted(ranked[:target_edges]))


def _oracle_diagnostic_edges(
    evaluator: Stage21ObjectiveStackEvaluator,
) -> tuple[str, ...] | None:
    edge_ids = evaluator.graph.edge_ids
    if len(edge_ids) > 10:
        return None
    best: Stage21ObjectiveEvaluation | None = None
    for count in range(len(edge_ids) + 1):
        for combo in combinations(edge_ids, count):
            evaluation = evaluator.evaluate(combo)
            if best is None or _oracle_key(evaluation) < _oracle_key(best):
                best = evaluation
    return best.selected_edge_ids if best is not None else None


def _oracle_key(evaluation: Stage21ObjectiveEvaluation) -> tuple[float, float, float, int, str]:
    probability = float(evaluation.metrics["consensus_success_probability"])
    feasible_penalty = 0.0 if probability >= STAGE21_TAU_REQUIREMENT_MIN else 1.0
    return (
        feasible_penalty,
        float(evaluation.metrics["latency"]),
        float(evaluation.metrics["energy"]),
        len(evaluation.selected_edge_ids),
        "|".join(evaluation.selected_edge_ids),
    )


def _build_stage21_edge_delta_targets(
    evaluator: Stage21ObjectiveStackEvaluator,
    selected_edge_ids: Iterable[str],
    *,
    topology_id: str,
    tau_requirement_min: float,
) -> tuple[EdgeDeltaTarget, ...]:
    selected = set(selected_edge_ids)
    baseline = evaluator.evaluate(selected, topology_id=f"{topology_id}:baseline")
    baseline_probability = float(baseline.metrics["consensus_success_probability"])
    baseline_latency = float(baseline.metrics["latency"])
    baseline_energy = float(baseline.metrics["energy"])
    baseline_feasible = int(baseline_probability >= tau_requirement_min)
    targets: list[EdgeDeltaTarget] = []
    for edge_id in evaluator.graph.edge_ids:
        targets.append(
            EdgeDeltaTarget(
                topology_id=topology_id,
                edge_id=edge_id,
                action_type="keep_edge",
                delta_consensus_success_probability=0.0,
                delta_latency=0.0,
                delta_energy=0.0,
                delta_feasibility=0,
                delta_reward_surrogate_diagnostic={
                    "objective_stack_diagnostic_only": 0.0,
                },
            )
        )
        changed = set(selected)
        if edge_id in changed:
            changed.remove(edge_id)
            action_type = "remove_edge"
        else:
            changed.add(edge_id)
            action_type = "add_edge"
        changed_eval = evaluator.evaluate(
            changed,
            topology_id=f"{topology_id}:{action_type}:{edge_id}",
        )
        changed_probability = float(changed_eval.metrics["consensus_success_probability"])
        changed_latency = float(changed_eval.metrics["latency"])
        changed_energy = float(changed_eval.metrics["energy"])
        changed_feasible = int(changed_probability >= tau_requirement_min)
        targets.append(
            EdgeDeltaTarget(
                topology_id=topology_id,
                edge_id=edge_id,
                action_type=action_type,
                delta_consensus_success_probability=changed_probability - baseline_probability,
                delta_latency=changed_latency - baseline_latency,
                delta_energy=changed_energy - baseline_energy,
                delta_feasibility=changed_feasible - baseline_feasible,
                delta_reward_surrogate_diagnostic={
                    "delta_reliability_gap_penalty": _gap(changed_probability)
                    - _gap(baseline_probability),
                    "delta_latency": changed_latency - baseline_latency,
                    "delta_energy": changed_energy - baseline_energy,
                },
            )
        )
    return tuple(targets)


def _source_learning_evidence_row(
    *,
    evaluation: Stage21ObjectiveEvaluation,
    topology_name: str,
    observations: tuple[ActorObservation, ...],
    context: Stage21EvaluationContext,
    learning_targets: Iterable[Mapping[str, object]],
) -> LearningEvidenceRow:
    return build_learning_evidence_row(
        evaluation,  # type: ignore[arg-type]
        topology_name=topology_name,
        observations=observations,
        graph_node_ids=context.graph.node_ids,
        candidate_edge_ids=context.graph.edge_ids,
        tau_requirement_min=STAGE21_TAU_REQUIREMENT_MIN,
        learning_targets=learning_targets,
        diagnostics={
            "fixture_id": context.fixture.fixture_id,
            "sequence_id": context.sequence_id,
            "time_step": context.time_step,
            "topology_label_role": _topology_label_role(topology_name),
            "topology_source": _topology_source(topology_name),
            "evaluator_id": evaluation.evaluator_id,
            "physics_regime_id": evaluation.physics_regime_id,
            "protocol_model_id": evaluation.protocol_model_id,
            "objective_contract_id": evaluation.objective_contract_id,
            "is_deployment_actor_input": False,
        },
    )


def _diagnostics_view(
    *,
    context: Stage21EvaluationContext,
    topology_name: str,
    topology_source: str,
    evaluation: Stage21ObjectiveEvaluation,
    teacher_projection,
) -> dict[str, object]:
    return {
        "scenario_id": evaluation.scenario_id,
        "source_fixture": context.fixture.fixture_id,
        "sequence_id": context.sequence_id,
        "time_step": context.time_step,
        "topology_name": topology_name,
        "topology_source": topology_source,
        "target_source": teacher_projection.teacher_source,
        "feature_schema_id": STAGE18_FEATURE_SCHEMA_ID,
        "evaluator_id": evaluation.evaluator_id,
        "physics_regime_id": evaluation.physics_regime_id,
        "protocol_model_id": evaluation.protocol_model_id,
        "objective_contract_id": evaluation.objective_contract_id,
        "forbidden_field_scan_result": {
            "actor_safe_view_passed": True,
            "actor_target_view_has_global_delta_fields": False,
            "critic_target_role": TARGET_ROLE_CRITIC_ONLY,
        },
        "teacher_projection": {
            "selected_edges": list(teacher_projection.teacher_selected_physical_edges),
            "pre_projection_edge_count": teacher_projection.assembled.pre_projection_edge_count,
            "post_projection_edge_count": teacher_projection.assembled.post_projection_edge_count,
            "rejection_reason_counts": dict(
                teacher_projection.assembled.diagnostics["rejection_reason_counts"]
            ),
        },
    }


TARGET_ROLE_CRITIC_ONLY = "critic_only"


def _topology_diagnostics(
    *,
    graph: CandidateGraph,
    selected: tuple[str, ...],
    records: tuple[NetworkCommunicationRecord, ...],
    matrices,
    accounting,
    config: Stage21ObjectiveStackConfig,
    schedule: StdmaSchedule | None = None,
) -> dict[str, object]:
    probabilities = [record.network_delivery_probability for record in records]
    return {
        "edge_count": len(selected),
        "node_count": len(graph.node_ids),
        "is_full_graph_baseline": graph.is_full_selection(set(selected)),
        "physics_regime_id": STAGE21_PHYSICS_REGIME_ID,
        "network_regime_id": NETWORK_COMMUNICATION_REGIME_ID,
        "finite_blocklength_regime_id": URLLC_FINITE_BLOCKLENGTH_REGIME_ID,
        "matrix_adapter_id": MESSAGE_MATRIX_ADAPTER_ID,
        "protocol_accounting_model_id": PBFT_PROTOCOL_ACCOUNTING_MODEL_ID,
        "protocol_model_id": STAGE21_PROTOCOL_MODEL_ID,
        "objective_contract_id": STAGE21_OBJECTIVE_CONTRACT_ID,
        "phase_budgets_s": dict(accounting.phase_budgets_s),
        "stage3_network_record_count": len(records),
        "min_network_delivery_probability": min(probabilities) if probabilities else 0.0,
        "mean_network_delivery_probability": fmean(probabilities) if probabilities else 0.0,
        "max_network_scheduled_latency_s": max(
            (record.network_scheduled_latency_s for record in records),
            default=0.0,
        ),
        "max_network_successful_delivery_latency_s": max(
            (record.network_successful_delivery_latency_s for record in records),
            default=0.0,
        ),
        "total_network_energy_j": sum(record.network_energy_j for record in records),
        "deadline_filtered_count_by_phase": dict(matrices.deadline_filtered_count_by_phase),
        "zero_delivery_count_by_phase": dict(matrices.zero_delivery_count_by_phase),
        "interference_group_ids": sorted(
            {
                group_id
                for record in records
                for group_id in record.interference_group_ids
            }
        ),
        "background_interference_enabled": config.use_background_interference,
        "orthogonal_resources": config.orthogonal_resources,
        "scheduled_mac_enabled": schedule is not None,
        "mac_scheduler_id": schedule.scheduler_id if schedule is not None else None,
        "mac_num_slots": schedule.num_slots if schedule is not None else 0,
        "mac_conflict_pair_count": schedule.conflict_pair_count if schedule is not None else 0,
        "mac_slot_duration_s": schedule.slot_duration_s if schedule is not None else 0.0,
        "mac_frame_latency_s": (
            schedule.num_slots * schedule.slot_duration_s if schedule is not None else 0.0
        ),
        "mac_sinr_threshold_db": schedule.sinr_threshold_db if schedule is not None else 0.0,
        "fallback_used": False,
    }


def _topology_label_role(topology_name: str) -> str:
    if topology_name == "oracle_diagnostic":
        return "oracle_diagnostic_only"
    if topology_name.endswith("_raw"):
        return "raw_baseline_or_heuristic"
    return "projected_or_actor_evaluation"


def _topology_source(topology_name: str) -> str:
    if topology_name == "oracle_diagnostic":
        return "oracle_diagnostic_not_actor_input"
    if topology_name == "full_graph_raw":
        return "raw_full_graph_baseline"
    if topology_name == "greedy_reliability_raw":
        return "raw_heuristic_baseline"
    if topology_name == "random_raw":
        return "raw_random_baseline"
    if topology_name == "sparse_quorum_raw":
        return "raw_sparse_heuristic_baseline"
    return "diagnostic_topology_variant"


def _evidence_readiness(dataset: Stage21EvidenceDataset) -> dict[str, object]:
    rows = dataset.rows
    tag_counts = Counter()
    sequence_steps: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.consensus_success_probability >= STAGE21_TAU_REQUIREMENT_MIN:
            tag_counts["tau_ge_0_9_feasible_topology_row"] += 1
        if row.consensus_success_probability < STAGE21_TAU_REQUIREMENT_MIN and row.selected_edges:
            tag_counts["infeasible_hard_row"] += 1
        if abs(row.consensus_success_probability - STAGE21_TAU_REQUIREMENT_MIN) <= 0.1:
            tag_counts["near_threshold_row"] += 1
        if (
            row.topology_name in {"sparse_quorum_raw", "greedy_reliability_raw"}
            and row.feasible_under_tau_requirement
            and row.selected_edge_count < max(1, int(row.topology_diagnostics["edge_count"]) + 1)
        ):
            tag_counts["sparse_feasible_row"] += 1
        if row.topology_name == "full_graph_raw":
            tag_counts["full_graph_baseline_row"] += 1
        if row.topology_name == "greedy_reliability_raw":
            tag_counts["projected_greedy_baseline_teacher_row"] += 1
        if row.topology_name == "oracle_diagnostic":
            tag_counts["oracle_diagnostic_row"] += 1
        sequence_id = row.diagnostics_view.get("sequence_id")
        if sequence_id is not None:
            sequence_steps[str(sequence_id)].add(row.time_step)
    checks = {
        "tau_feasible_topology_present": tag_counts["tau_ge_0_9_feasible_topology_row"] > 0,
        "infeasible_hard_row_present": tag_counts["infeasible_hard_row"] > 0,
        "near_threshold_row_present": tag_counts["near_threshold_row"] > 0,
        "sparse_feasible_row_present": tag_counts["sparse_feasible_row"] > 0,
        "full_graph_baseline_row_present": tag_counts["full_graph_baseline_row"] > 0,
        "projected_greedy_baseline_row_present": tag_counts["projected_greedy_baseline_teacher_row"] > 0,
        "oracle_diagnostic_row_present": tag_counts["oracle_diagnostic_row"] > 0,
        "multi_step_actor_safe_sequence_present": any(
            len(steps) >= 2 for steps in sequence_steps.values()
        ),
        "all_rows_declare_evaluator_id": all(bool(row.evaluator_id) for row in rows),
    }
    return {
        "tag_counts": dict(tag_counts),
        "sequence_steps": {key: sorted(value) for key, value in sequence_steps.items()},
        **checks,
        "all_required_evidence_present": all(checks.values()),
    }


def _gap(probability: float) -> float:
    return max(0.0, STAGE21_TAU_REQUIREMENT_MIN - probability) ** 2


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
