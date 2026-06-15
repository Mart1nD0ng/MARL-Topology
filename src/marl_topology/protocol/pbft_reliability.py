"""Stage 4 PBFT reliability from declared message matrices."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .quorum_tail import heterogeneous_quorum_tail, remove_largest_probabilities


PBFT_RELIABILITY_VARIANT_ID = "stage4_pbft_three_phase_closed_form_v0"
PBFT_EXPECTED_INITIATOR_MODEL_ID = "pbft_expected_initiator_mean_field_v1"
PBFT_PHASE_NAMES = ("pre_prepare", "prepare", "commit")
FAULT_FILTER_NONE = "none"
FAULT_FILTER_REMOVE_LARGEST = "unknown_faults_remove_largest"
FAULT_FILTER_MODES = (FAULT_FILTER_NONE, FAULT_FILTER_REMOVE_LARGEST)
PRIMARY_DISTRIBUTION_UNIFORM = "uniform"
VIEW_CHANGE_DEFERRED = "deferred"
VIEW_CHANGE_NONE = "none"
MessageMatrix = Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class PBFTThreePhaseConfig:
    node_ids: tuple[str, ...]
    primary_id: str
    fault_tolerance: int
    fault_filter_mode: str = FAULT_FILTER_REMOVE_LARGEST
    protocol_variant: str = PBFT_RELIABILITY_VARIANT_ID

    def __post_init__(self) -> None:
        if not self.protocol_variant:
            raise ValueError("protocol_variant must be non-empty")
        if self.protocol_variant != PBFT_RELIABILITY_VARIANT_ID:
            raise ValueError("unsupported protocol_variant")
        if not self.node_ids:
            raise ValueError("node_ids must be non-empty")
        if any(not node_id for node_id in self.node_ids):
            raise ValueError("node_ids must contain non-empty ids")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if self.primary_id not in self.node_ids:
            raise ValueError("primary_id must be in node_ids")
        if isinstance(self.fault_tolerance, bool) or not isinstance(
            self.fault_tolerance,
            int,
        ):
            raise TypeError("fault_tolerance must be an integer")
        if self.fault_tolerance < 0:
            raise ValueError("fault_tolerance must be nonnegative")
        if len(self.node_ids) < 3 * self.fault_tolerance + 1:
            raise ValueError("PBFT requires n >= 3f + 1")
        if self.fault_filter_mode not in FAULT_FILTER_MODES:
            raise ValueError("fault_filter_mode is not supported")

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def total_quorum(self) -> int:
        return 2 * self.fault_tolerance + 1

    @property
    def external_quorum(self) -> int:
        return 2 * self.fault_tolerance


@dataclass(frozen=True, slots=True)
class PBFTExpectedInitiatorConfig:
    node_ids: tuple[str, ...]
    fault_tolerance: int
    fault_filter_mode: str = FAULT_FILTER_REMOVE_LARGEST
    primary_distribution: str = PRIMARY_DISTRIBUTION_UNIFORM
    self_vote_counted: bool = True
    model_id: str = PBFT_EXPECTED_INITIATOR_MODEL_ID
    mean_field_assumption: bool = True
    view_change_mode: str = VIEW_CHANGE_DEFERRED

    def __post_init__(self) -> None:
        if self.model_id != PBFT_EXPECTED_INITIATOR_MODEL_ID:
            raise ValueError("unsupported model_id")
        if not self.node_ids:
            raise ValueError("node_ids must be non-empty")
        if any(not node_id for node_id in self.node_ids):
            raise ValueError("node_ids must contain non-empty ids")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if isinstance(self.fault_tolerance, bool) or not isinstance(
            self.fault_tolerance,
            int,
        ):
            raise TypeError("fault_tolerance must be an integer")
        if self.fault_tolerance < 0:
            raise ValueError("fault_tolerance must be nonnegative")
        if len(self.node_ids) < 3 * self.fault_tolerance + 1:
            raise ValueError("PBFT requires n >= 3f + 1")
        if self.fault_filter_mode not in FAULT_FILTER_MODES:
            raise ValueError("fault_filter_mode is not supported")
        if self.primary_distribution != PRIMARY_DISTRIBUTION_UNIFORM:
            raise ValueError("only uniform primary_distribution is supported")
        if not self.self_vote_counted:
            raise ValueError("Stage 4.4 requires self_vote_counted=true")
        if not self.mean_field_assumption:
            raise ValueError("Stage 4.4 requires mean_field_assumption=true")
        if self.view_change_mode not in {VIEW_CHANGE_DEFERRED, VIEW_CHANGE_NONE}:
            raise ValueError("view_change_mode must be deferred or none")

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def distribution_weights(self) -> dict[str, float]:
        weight = 1.0 / float(self.node_count)
        return {node_id: weight for node_id in self.node_ids}


@dataclass(frozen=True, slots=True)
class PBFTThreePhaseReliabilityRecord:
    protocol_variant: str
    node_ids: tuple[str, ...]
    primary_id: str
    fault_tolerance: int
    fault_filter_mode: str
    total_quorum: int
    external_quorum: int
    phase_names: tuple[str, str, str]
    pre_prepare_readiness: Mapping[str, float]
    prepared_probability: Mapping[str, float]
    committed_probability: Mapping[str, float]
    consensus_success_probability: float
    mean_field_assumption: bool = True
    uses_declared_matrices_only: bool = True
    uses_stage3_adapter: bool = False

    def __post_init__(self) -> None:
        if self.protocol_variant != PBFT_RELIABILITY_VARIANT_ID:
            raise ValueError("unsupported protocol_variant")
        if self.phase_names != PBFT_PHASE_NAMES:
            raise ValueError("phase_names must be pre_prepare, prepare, commit")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if self.total_quorum <= 0:
            raise ValueError("total_quorum must be positive")
        if self.external_quorum < 0:
            raise ValueError("external_quorum must be nonnegative")
        for mapping_name, values in (
            ("pre_prepare_readiness", self.pre_prepare_readiness),
            ("prepared_probability", self.prepared_probability),
            ("committed_probability", self.committed_probability),
        ):
            if set(values) != set(self.node_ids):
                raise ValueError(f"{mapping_name} must contain exactly node_ids")
            for value in values.values():
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{mapping_name} values must be in [0, 1]")
        if not self.mean_field_assumption:
            raise ValueError("Stage 4.2 record must state the mean-field assumption")
        if not self.uses_declared_matrices_only:
            raise ValueError("Stage 4.2 record must use declared matrices only")
        if self.uses_stage3_adapter:
            raise ValueError("Stage 4.2 must not use the Stage 3 adapter")


@dataclass(frozen=True, slots=True)
class PBFTExpectedInitiatorReliabilityRecord:
    model_id: str
    node_ids: tuple[str, ...]
    fault_tolerance: int
    fault_filter_mode: str
    primary_distribution: Mapping[str, float]
    per_primary_reliability: Mapping[str, float]
    consensus_success_probability: float
    self_vote_counted: bool = True
    mean_field_assumption: bool = True
    view_change_mode: str = VIEW_CHANGE_DEFERRED
    uses_primary_specific_helper: bool = True
    uses_subset_enumeration: bool = False

    def __post_init__(self) -> None:
        if self.model_id != PBFT_EXPECTED_INITIATOR_MODEL_ID:
            raise ValueError("unsupported model_id")
        if set(self.primary_distribution) != set(self.node_ids):
            raise ValueError("primary_distribution must contain exactly node_ids")
        if set(self.per_primary_reliability) != set(self.node_ids):
            raise ValueError("per_primary_reliability must contain exactly node_ids")
        total_weight = 0.0
        for value in self.primary_distribution.values():
            if not 0.0 <= value <= 1.0:
                raise ValueError("primary_distribution values must be in [0, 1]")
            total_weight += value
        if abs(total_weight - 1.0) > 1e-12:
            raise ValueError("primary_distribution must sum to 1")
        for value in self.per_primary_reliability.values():
            if not 0.0 <= value <= 1.0:
                raise ValueError("per_primary_reliability values must be in [0, 1]")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if not self.self_vote_counted:
            raise ValueError("self_vote_counted must be true")
        if not self.mean_field_assumption:
            raise ValueError("mean_field_assumption must be true")
        if self.view_change_mode not in {VIEW_CHANGE_DEFERRED, VIEW_CHANGE_NONE}:
            raise ValueError("view_change_mode must be deferred or none")
        if not self.uses_primary_specific_helper:
            raise ValueError("Stage 4.4 must use the primary-specific helper")
        if self.uses_subset_enumeration:
            raise ValueError("Stage 4.4 must not use subset enumeration")


def evaluate_pbft_three_phase_reliability(
    config: PBFTThreePhaseConfig,
    *,
    pre_prepare_matrix: MessageMatrix,
    prepare_matrix: MessageMatrix,
    commit_matrix: MessageMatrix,
) -> PBFTThreePhaseReliabilityRecord:
    """Evaluate Stage 4.2 PBFT reliability from declared delivery matrices."""

    pre_prepare = _checked_message_matrix(config.node_ids, pre_prepare_matrix)
    prepare = _checked_message_matrix(config.node_ids, prepare_matrix)
    commit = _checked_message_matrix(config.node_ids, commit_matrix)

    alpha_1 = {
        node_id: (
            1.0
            if node_id == config.primary_id
            else _matrix_probability(pre_prepare, config.primary_id, node_id)
        )
        for node_id in config.node_ids
    }
    alpha_2 = _receiver_cascade(
        config=config,
        sender_readiness=alpha_1,
        receiver_readiness=alpha_1,
        message_matrix=prepare,
    )
    alpha_3 = _receiver_cascade(
        config=config,
        sender_readiness=alpha_2,
        receiver_readiness=alpha_2,
        message_matrix=commit,
    )
    global_inputs = _fault_filtered_values(tuple(alpha_3.values()), config)
    consensus_probability = heterogeneous_quorum_tail(
        global_inputs,
        config.total_quorum,
    )

    return PBFTThreePhaseReliabilityRecord(
        protocol_variant=config.protocol_variant,
        node_ids=config.node_ids,
        primary_id=config.primary_id,
        fault_tolerance=config.fault_tolerance,
        fault_filter_mode=config.fault_filter_mode,
        total_quorum=config.total_quorum,
        external_quorum=config.external_quorum,
        phase_names=PBFT_PHASE_NAMES,
        pre_prepare_readiness=alpha_1,
        prepared_probability=alpha_2,
        committed_probability=alpha_3,
        consensus_success_probability=consensus_probability,
    )


def evaluate_pbft_given_primary(
    primary_id: str,
    config: PBFTExpectedInitiatorConfig,
    *,
    pre_prepare_matrix: MessageMatrix,
    prepare_matrix: MessageMatrix,
    commit_matrix: MessageMatrix,
) -> PBFTThreePhaseReliabilityRecord:
    """Evaluate primary-specific PBFT reliability as an internal helper."""

    primary_config = PBFTThreePhaseConfig(
        node_ids=config.node_ids,
        primary_id=primary_id,
        fault_tolerance=config.fault_tolerance,
        fault_filter_mode=config.fault_filter_mode,
    )
    return evaluate_pbft_three_phase_reliability(
        primary_config,
        pre_prepare_matrix=pre_prepare_matrix,
        prepare_matrix=prepare_matrix,
        commit_matrix=commit_matrix,
    )


def evaluate_expected_initiator_pbft_reliability(
    config: PBFTExpectedInitiatorConfig,
    *,
    pre_prepare_matrix: MessageMatrix,
    prepare_matrix: MessageMatrix,
    commit_matrix: MessageMatrix,
) -> PBFTExpectedInitiatorReliabilityRecord:
    """Evaluate topology-level expected reliability over uniform initiators."""

    weights = config.distribution_weights
    per_primary: dict[str, float] = {}
    for primary_id in config.node_ids:
        primary_record = evaluate_pbft_given_primary(
            primary_id,
            config,
            pre_prepare_matrix=pre_prepare_matrix,
            prepare_matrix=prepare_matrix,
            commit_matrix=commit_matrix,
        )
        per_primary[primary_id] = primary_record.consensus_success_probability

    consensus_probability = sum(
        weights[primary_id] * per_primary[primary_id]
        for primary_id in config.node_ids
    )
    return PBFTExpectedInitiatorReliabilityRecord(
        model_id=config.model_id,
        node_ids=config.node_ids,
        fault_tolerance=config.fault_tolerance,
        fault_filter_mode=config.fault_filter_mode,
        primary_distribution=weights,
        per_primary_reliability=per_primary,
        consensus_success_probability=_clamp_probability(consensus_probability),
        self_vote_counted=config.self_vote_counted,
        mean_field_assumption=config.mean_field_assumption,
        view_change_mode=config.view_change_mode,
    )


def _receiver_cascade(
    *,
    config: PBFTThreePhaseConfig,
    sender_readiness: Mapping[str, float],
    receiver_readiness: Mapping[str, float],
    message_matrix: dict[tuple[str, str], float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for receiver_id in config.node_ids:
        incoming = tuple(
            sender_readiness[sender_id]
            * _matrix_probability(message_matrix, sender_id, receiver_id)
            for sender_id in config.node_ids
            if sender_id != receiver_id
        )
        filtered = _fault_filtered_values(incoming, config)
        quorum_probability = heterogeneous_quorum_tail(
            filtered,
            config.external_quorum,
        )
        result[receiver_id] = receiver_readiness[receiver_id] * quorum_probability
    return result


def _fault_filtered_values(
    probabilities: tuple[float, ...],
    config: PBFTThreePhaseConfig,
) -> tuple[float, ...]:
    if config.fault_filter_mode == FAULT_FILTER_NONE or config.fault_tolerance == 0:
        return probabilities
    return remove_largest_probabilities(probabilities, config.fault_tolerance)


def _checked_message_matrix(
    node_ids: tuple[str, ...],
    matrix: MessageMatrix,
) -> dict[tuple[str, str], float]:
    node_set = set(node_ids)
    checked: dict[tuple[str, str], float] = {}
    for key, raw_value in matrix.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("message matrix keys must be (sender_id, receiver_id)")
        sender_id, receiver_id = key
        if sender_id not in node_set or receiver_id not in node_set:
            raise ValueError("message matrix endpoints must be in node_ids")
        if sender_id == receiver_id:
            raise ValueError("message matrix must not include self messages")
        value = float(raw_value)
        if not isfinite(value):
            raise ValueError("message probabilities must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("message probabilities must be in [0, 1]")
        checked[(sender_id, receiver_id)] = value
    return checked


def _matrix_probability(
    matrix: Mapping[tuple[str, str], float],
    sender_id: str,
    receiver_id: str,
) -> float:
    if sender_id == receiver_id:
        return 0.0
    return matrix.get((sender_id, receiver_id), 0.0)


def _clamp_probability(value: float) -> float:
    if value < 0.0 and value > -1e-15:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-15:
        return 1.0
    return value
