"""Diagnostics for non-learning topology assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class RejectionReason(str, Enum):
    INVALID_CANDIDATE = "invalid_candidate"
    ROLE_FORBIDDEN = "role_forbidden"
    LOW_SCORE = "low_score"
    TX_BUDGET_EXCEEDED = "tx_budget_exceeded"
    RX_CAPACITY_EXCEEDED = "rx_capacity_exceeded"
    CHANNEL_CONFLICT = "channel_conflict"
    INTERFERENCE_CONFLICT = "interference_conflict"
    DUPLICATE_EDGE = "duplicate_edge"
    HYSTERESIS_REJECTED = "hysteresis_rejected"
    PROJECTION_LIMIT = "projection_limit"


@dataclass(frozen=True, slots=True)
class AssemblerDiagnostics:
    assembler_id: str
    mode: str
    baseline_only: bool
    recommended_for_deployment: bool
    pre_projection_edge_count: int
    post_projection_edge_count: int
    rejected_count: int
    rejection_reason_counts: Mapping[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    oracle_used: bool = False
    objective_used: bool = False
    reward_signal_used: bool = False

    def __post_init__(self) -> None:
        if not self.assembler_id:
            raise ValueError("assembler_id must be declared")
        if not self.mode:
            raise ValueError("mode must be declared")
        for field_name in (
            "pre_projection_edge_count",
            "post_projection_edge_count",
            "rejected_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        object.__setattr__(
            self,
            "rejection_reason_counts",
            dict(sorted(self.rejection_reason_counts.items())),
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_payload(self) -> dict[str, object]:
        return {
            "assembler_id": self.assembler_id,
            "mode": self.mode,
            "baseline_only": self.baseline_only,
            "recommended_for_deployment": self.recommended_for_deployment,
            "pre_projection_edge_count": self.pre_projection_edge_count,
            "post_projection_edge_count": self.post_projection_edge_count,
            "rejected_count": self.rejected_count,
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "notes": list(self.notes),
            "oracle_used": self.oracle_used,
            "objective_used": self.objective_used,
            "reward_signal_used": self.reward_signal_used,
        }
