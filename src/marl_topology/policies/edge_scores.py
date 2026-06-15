"""Actor edge-score records for Stage 8 topology assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite


EDGE_SCORE_RECORD_FIELDS = (
    "agent_id",
    "neighbor_id",
    "edge_id",
    "directed_edge_id",
    "score",
    "probability",
    "score_source",
    "time_step",
)


def make_directed_edge_id(agent_id: str, neighbor_id: str) -> str:
    if not agent_id or not neighbor_id:
        raise ValueError("agent_id and neighbor_id must be declared")
    if agent_id == neighbor_id:
        raise ValueError("self-directed edges are not valid policy outputs")
    return f"{agent_id}->{neighbor_id}"


@dataclass(frozen=True, slots=True)
class EdgeScoreRecord:
    agent_id: str
    neighbor_id: str
    edge_id: str
    directed_edge_id: str
    score: float
    probability: float | None = None
    score_source: str = "actor_edge_scorer"
    time_step: int = 0

    def __post_init__(self) -> None:
        for field_name in ("agent_id", "neighbor_id", "edge_id", "directed_edge_id"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be declared")
        expected_directed_id = make_directed_edge_id(self.agent_id, self.neighbor_id)
        if self.directed_edge_id != expected_directed_id:
            raise ValueError(f"directed_edge_id must be {expected_directed_id}")
        if not isfinite(float(self.score)):
            raise ValueError("score must be finite")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1] when provided")
        if self.time_step < 0:
            raise ValueError("time_step must be nonnegative")
        if not self.score_source:
            raise ValueError("score_source must be declared")

    def to_payload(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "neighbor_id": self.neighbor_id,
            "edge_id": self.edge_id,
            "directed_edge_id": self.directed_edge_id,
            "score": self.score,
            "probability": self.probability,
            "score_source": self.score_source,
            "time_step": self.time_step,
        }


@dataclass(frozen=True, slots=True)
class EdgeScoreBatch:
    edge_scores: tuple[EdgeScoreRecord, ...]
    batch_id: str = "edge_score_batch_stage8"
    source: str = "actor_policy_output"

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must be declared")
        if not self.source:
            raise ValueError("source must be declared")
        seen: set[tuple[str, int]] = set()
        for record in self.edge_scores:
            key = (record.directed_edge_id, record.time_step)
            if key in seen:
                raise ValueError(f"duplicate edge score record: {key}")
            seen.add(key)

    @classmethod
    def from_records(
        cls,
        records: Iterable[EdgeScoreRecord],
        *,
        batch_id: str = "edge_score_batch_stage8",
        source: str = "actor_policy_output",
    ) -> "EdgeScoreBatch":
        return cls(edge_scores=tuple(records), batch_id=batch_id, source=source)

    @property
    def field_names(self) -> tuple[str, ...]:
        return EDGE_SCORE_RECORD_FIELDS

    @property
    def directed_edge_ids(self) -> tuple[str, ...]:
        return tuple(record.directed_edge_id for record in self.edge_scores)

    def to_payload(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "source": self.source,
            "field_names": list(self.field_names),
            "edge_scores": [record.to_payload() for record in self.edge_scores],
            "contains_final_topology": False,
        }
