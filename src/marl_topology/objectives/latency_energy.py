"""Minimal latency and energy aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from marl_topology.link import LinkRecord


@dataclass(frozen=True, slots=True)
class LatencyEnergy:
    latency_s: float
    energy_j: float

    def __post_init__(self) -> None:
        if self.latency_s < 0:
            raise ValueError("latency_s must be nonnegative")
        if self.energy_j < 0:
            raise ValueError("energy_j must be nonnegative")


def aggregate_latency_energy(records: Iterable[LinkRecord]) -> LatencyEnergy:
    selected = tuple(records)
    if not selected:
        return LatencyEnergy(latency_s=0.0, energy_j=0.0)
    return LatencyEnergy(
        latency_s=max(record.latency_s for record in selected),
        energy_j=sum(record.energy_j for record in selected),
    )
