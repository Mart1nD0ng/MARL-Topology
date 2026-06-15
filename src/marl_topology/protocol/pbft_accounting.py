"""Stage 4.6 PBFT protocol latency and energy accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from marl_topology.metrics import REGISTERED_METRICS, require_registered_metrics
from marl_topology.network import NetworkCommunicationRecord

from .message_matrix_adapter import PBFTPhaseBudgets
from .pbft_reliability import PBFT_PHASE_NAMES


PBFT_PROTOCOL_ACCOUNTING_MODEL_ID = "stage4_pbft_protocol_accounting_v1"
PhaseRecordMap = Mapping[str, tuple[NetworkCommunicationRecord, ...]]


@dataclass(frozen=True, slots=True)
class PBFTPhaseAccountingRecord:
    phase_name: str
    phase_budget_s: float
    scheduled_record_count: int
    scheduled_message_count: int
    deadline_eligible_message_count: int
    deadline_filtered_message_count: int
    zero_delivery_message_count: int
    max_scheduled_latency_s: float
    phase_latency_s: float
    phase_energy_j: float
    latency_accounting: str = "phase_max_clipped_to_budget"
    energy_accounting: str = "scheduled_attempt_energy_sum"

    def __post_init__(self) -> None:
        if self.phase_name not in PBFT_PHASE_NAMES:
            raise ValueError("phase_name must be a PBFT phase")
        numeric_values = (
            self.phase_budget_s,
            self.max_scheduled_latency_s,
            self.phase_latency_s,
            self.phase_energy_j,
        )
        if any(not isfinite(value) or value < 0.0 for value in numeric_values):
            raise ValueError("phase accounting numeric values must be finite and nonnegative")
        count_values = (
            self.scheduled_record_count,
            self.scheduled_message_count,
            self.deadline_eligible_message_count,
            self.deadline_filtered_message_count,
            self.zero_delivery_message_count,
        )
        if any(value < 0 for value in count_values):
            raise ValueError("phase accounting counts must be nonnegative")
        if self.phase_latency_s > self.phase_budget_s:
            raise ValueError("phase_latency_s must be clipped to phase_budget_s")
        if self.deadline_eligible_message_count + self.deadline_filtered_message_count != (
            self.scheduled_message_count
        ):
            raise ValueError("deadline message counts must partition scheduled messages")
        if self.zero_delivery_message_count > self.scheduled_message_count:
            raise ValueError("zero_delivery_message_count cannot exceed scheduled messages")

    def to_payload(self) -> dict[str, object]:
        return {
            "phase_name": self.phase_name,
            "phase_budget_s": self.phase_budget_s,
            "scheduled_record_count": self.scheduled_record_count,
            "scheduled_message_count": self.scheduled_message_count,
            "deadline_eligible_message_count": self.deadline_eligible_message_count,
            "deadline_filtered_message_count": self.deadline_filtered_message_count,
            "zero_delivery_message_count": self.zero_delivery_message_count,
            "max_scheduled_latency_s": self.max_scheduled_latency_s,
            "phase_latency_s": self.phase_latency_s,
            "phase_energy_j": self.phase_energy_j,
            "latency_accounting": self.latency_accounting,
            "energy_accounting": self.energy_accounting,
        }


@dataclass(frozen=True, slots=True)
class PBFTProtocolAccountingRecord:
    accounting_model_id: str
    node_ids: tuple[str, ...]
    phase_names: tuple[str, str, str]
    phase_budgets_s: Mapping[str, float]
    phase_accounting: Mapping[str, PBFTPhaseAccountingRecord]
    protocol_latency_s: float
    protocol_energy_j: float
    latency_accounting: str = "sum_phase_max_clipped_to_budget"
    energy_accounting: str = "sum_scheduled_attempt_energy"
    uses_stage3_network_records: bool = True
    exports_consensus_probability: bool = False
    reward_logic_implemented: bool = False

    def __post_init__(self) -> None:
        if self.accounting_model_id != PBFT_PROTOCOL_ACCOUNTING_MODEL_ID:
            raise ValueError("unsupported accounting_model_id")
        if self.phase_names != PBFT_PHASE_NAMES:
            raise ValueError("phase_names must be pre_prepare, prepare, commit")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be non-empty and unique")
        if set(self.phase_budgets_s) != set(PBFT_PHASE_NAMES):
            raise ValueError("phase_budgets_s must contain exactly PBFT phases")
        if set(self.phase_accounting) != set(PBFT_PHASE_NAMES):
            raise ValueError("phase_accounting must contain exactly PBFT phases")
        for phase_name in PBFT_PHASE_NAMES:
            if self.phase_accounting[phase_name].phase_name != phase_name:
                raise ValueError("phase_accounting keys must match records")
        if self.protocol_latency_s < 0.0 or self.protocol_energy_j < 0.0:
            raise ValueError("protocol latency and energy must be nonnegative")
        if not isfinite(self.protocol_latency_s) or not isfinite(self.protocol_energy_j):
            raise ValueError("protocol latency and energy must be finite")
        if not self.uses_stage3_network_records:
            raise ValueError("Stage 4.6 accounting must state Stage 3 record use")
        if self.exports_consensus_probability:
            raise ValueError("Stage 4.6 accounting must not export consensus probability")
        if self.reward_logic_implemented:
            raise ValueError("Stage 4.6 accounting must not implement reward")

    def diagnostics(self) -> dict[str, object]:
        return {
            "protocol_accounting_model_id": self.accounting_model_id,
            "phase_accounting": {
                phase_name: record.to_payload()
                for phase_name, record in self.phase_accounting.items()
            },
            "latency_accounting": self.latency_accounting,
            "energy_accounting": self.energy_accounting,
            "uses_stage3_network_records": self.uses_stage3_network_records,
            "exports_consensus_probability": self.exports_consensus_probability,
            "implements_reward": self.reward_logic_implemented,
        }

    def metric_rows(
        self,
        *,
        scenario_id: str,
        topology_id: str,
    ) -> tuple[Mapping[str, object], ...]:
        metrics: dict[str, object] = {
            "latency": self.protocol_latency_s,
            "energy": self.protocol_energy_j,
            "topology_diagnostics": self.diagnostics(),
        }
        require_registered_metrics(metrics.keys())
        rows: list[Mapping[str, object]] = []
        for name, value in metrics.items():
            definition = REGISTERED_METRICS[name]
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "topology_id": topology_id,
                    "metric_name": name,
                    "metric_level": definition.level,
                    "metric_value": value,
                    "used_for": definition.used_for,
                }
            )
        return tuple(rows)


def account_pbft_protocol_latency_energy(
    *,
    node_ids: tuple[str, ...],
    phase_records: PhaseRecordMap,
    phase_budgets: PBFTPhaseBudgets,
) -> PBFTProtocolAccountingRecord:
    """Account protocol latency and energy from scheduled Stage 3 records."""

    checked_node_ids = _checked_node_ids(node_ids)
    extra_phases = sorted(set(phase_records) - set(PBFT_PHASE_NAMES))
    if extra_phases:
        raise ValueError(f"unknown PBFT phase records: {extra_phases}")

    phase_accounting: dict[str, PBFTPhaseAccountingRecord] = {}
    for phase_name in PBFT_PHASE_NAMES:
        records = tuple(phase_records.get(phase_name, ()))
        phase_accounting[phase_name] = _account_phase(
            phase_name=phase_name,
            phase_budget_s=phase_budgets.budget_for_phase(phase_name),
            node_ids=checked_node_ids,
            records=records,
        )

    protocol_latency_s = sum(
        record.phase_latency_s for record in phase_accounting.values()
    )
    protocol_energy_j = sum(record.phase_energy_j for record in phase_accounting.values())
    return PBFTProtocolAccountingRecord(
        accounting_model_id=PBFT_PROTOCOL_ACCOUNTING_MODEL_ID,
        node_ids=checked_node_ids,
        phase_names=PBFT_PHASE_NAMES,
        phase_budgets_s={
            "pre_prepare": phase_budgets.pre_prepare_budget_s,
            "prepare": phase_budgets.prepare_budget_s,
            "commit": phase_budgets.commit_budget_s,
        },
        phase_accounting=phase_accounting,
        protocol_latency_s=protocol_latency_s,
        protocol_energy_j=protocol_energy_j,
    )


def _account_phase(
    *,
    phase_name: str,
    phase_budget_s: float,
    node_ids: tuple[str, ...],
    records: tuple[NetworkCommunicationRecord, ...],
) -> PBFTPhaseAccountingRecord:
    message_count = 0
    eligible_count = 0
    filtered_count = 0
    zero_delivery_count = 0
    max_latency_s = 0.0
    energy_j = 0.0

    for record in records:
        _validate_record_nodes(node_ids, record)
        if record.is_oracle:
            raise ValueError("oracle-labelled records cannot feed PBFT accounting")
        record_message_count = len(record.target_ids)
        message_count += record_message_count
        max_latency_s = max(max_latency_s, record.network_scheduled_latency_s)
        energy_j += record.network_energy_j
        if record.network_scheduled_latency_s <= phase_budget_s:
            eligible_count += record_message_count
        else:
            filtered_count += record_message_count
        if (
            record.network_delivery_probability == 0.0
            or record.network_scheduled_latency_s > phase_budget_s
        ):
            zero_delivery_count += record_message_count

    phase_latency_s = min(max_latency_s, phase_budget_s) if records else 0.0
    return PBFTPhaseAccountingRecord(
        phase_name=phase_name,
        phase_budget_s=phase_budget_s,
        scheduled_record_count=len(records),
        scheduled_message_count=message_count,
        deadline_eligible_message_count=eligible_count,
        deadline_filtered_message_count=filtered_count,
        zero_delivery_message_count=zero_delivery_count,
        max_scheduled_latency_s=max_latency_s,
        phase_latency_s=phase_latency_s,
        phase_energy_j=energy_j,
    )


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
    if any(target_id == record.source_id for target_id in record.target_ids):
        raise ValueError("self-message targets are not allowed")
