"""Minimal consensus abstraction for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    """Stage 2 quorum graph abstraction.

    This is not a full PBFT timing model. It names the variant explicitly so it
    cannot be confused with a migrated v5 effective-success metric.
    """

    quorum_size: int
    success_probability_threshold: float = 0.5
    deadline_s: float | None = None
    protocol_variant: str = "stage2_minimal_quorum_graph"

    def __post_init__(self) -> None:
        if self.quorum_size <= 0:
            raise ValueError("quorum_size must be positive")
        if not 0.0 <= self.success_probability_threshold <= 1.0:
            raise ValueError("success_probability_threshold must be in [0, 1]")
        if self.deadline_s is not None and self.deadline_s < 0:
            raise ValueError("deadline_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    consensus_success: bool
    consensus_success_probability: float
    reachable_node_count: int
    quorum_size: int
    deadline_s: float | None
    latency_s: float
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        if self.reachable_node_count < 0:
            raise ValueError("reachable_node_count must be nonnegative")
        if self.latency_s < 0:
            raise ValueError("latency_s must be nonnegative")
