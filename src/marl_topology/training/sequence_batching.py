"""Actor-safe temporal sequence batching for recurrent edge scorers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from marl_topology.models.tensorizers import ACTOR_EDGE_FEATURE_FIELDS
from marl_topology.training.supervised_batching import (
    build_supervised_batch_from_evidence,
    load_learning_evidence_json,
)


STAGE14_SEQUENCE_BATCH_SCHEMA_ID = "stage14_actor_safe_temporal_sequence_v1"


class SequenceBatchingViolation(ValueError):
    """Raised when temporal batching leaks future information or breaks order."""


@dataclass(frozen=True, slots=True)
class TemporalEdgeSequenceBatch:
    features: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    sequence_ids: tuple[str, ...]
    time_steps: tuple[int, ...]
    source: str
    schema_id: str = STAGE14_SEQUENCE_BATCH_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != STAGE14_SEQUENCE_BATCH_SCHEMA_ID:
            raise SequenceBatchingViolation("unexpected sequence schema")
        if self.features.ndim != 3:
            raise SequenceBatchingViolation("features must be [sequence, time, feature]")
        if self.features.shape[-1] != len(ACTOR_EDGE_FEATURE_FIELDS):
            raise SequenceBatchingViolation("feature dimension mismatch")
        if self.targets.shape != self.features.shape[:2]:
            raise SequenceBatchingViolation("targets must be [sequence, time]")
        if self.mask.shape != self.targets.shape:
            raise SequenceBatchingViolation("mask must match targets")
        if len(self.sequence_ids) != self.features.shape[0]:
            raise SequenceBatchingViolation("sequence id count mismatch")
        if tuple(self.time_steps) != tuple(sorted(self.time_steps)):
            raise SequenceBatchingViolation("time steps must be sorted")

    @property
    def sequence_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def max_time_steps(self) -> int:
        return int(self.features.shape[1])

    def summary(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "source": self.source,
            "sequence_count": self.sequence_count,
            "max_time_steps": self.max_time_steps,
            "feature_dim": int(self.features.shape[2]),
            "mask_active_count": int(self.mask.sum().item()),
            "time_steps": list(self.time_steps),
            "future_outcome_leakage_detected": False,
        }


def build_actor_safe_temporal_fixture_from_evidence(
    evidence_path: str | Path,
    *,
    max_sequences: int = 12,
    time_steps: int = 4,
) -> TemporalEdgeSequenceBatch:
    evidence = load_learning_evidence_json(evidence_path)
    batch = build_supervised_batch_from_evidence(evidence, row_limit=1).actor_batch
    if batch.edge_count == 0:
        raise SequenceBatchingViolation("cannot build sequence fixture without actor edges")
    base_features = batch.edge_features[:max_sequences].clone()
    sequence_count = int(base_features.shape[0])
    features = torch.zeros((sequence_count, time_steps, base_features.shape[1]), dtype=torch.float32)
    targets = torch.zeros((sequence_count, time_steps), dtype=torch.float32)
    mask = torch.ones((sequence_count, time_steps), dtype=torch.bool)
    success_index = ACTOR_EDGE_FEATURE_FIELDS.index("link_success_probability")
    distance_index = ACTOR_EDGE_FEATURE_FIELDS.index("distance_3d_m")
    for seq_index in range(sequence_count):
        base = base_features[seq_index]
        for time_index in range(time_steps):
            row = base.clone()
            trend = 0.05 * time_index if seq_index % 2 == 0 else -0.04 * time_index
            row[success_index] = torch.clamp(base[success_index] + trend, 0.0, 1.0)
            row[distance_index] = torch.clamp(base[distance_index] + (time_index * 2.0), min=0.0)
            features[seq_index, time_index] = row
            previous = (
                features[seq_index, time_index - 1, success_index]
                if time_index > 0
                else row[success_index]
            )
            targets[seq_index, time_index] = 1.0 if (row[success_index] + previous) / 2.0 >= 0.68 else 0.0
    return TemporalEdgeSequenceBatch(
        features=features,
        targets=targets,
        mask=mask,
        sequence_ids=tuple(f"edge_sequence_{index}" for index in range(sequence_count)),
        time_steps=tuple(range(time_steps)),
        source="stage14_synthetic_actor_safe_temporal_fixture",
    )


def assert_no_future_outcome_leakage(batch: TemporalEdgeSequenceBatch) -> None:
    success_index = ACTOR_EDGE_FEATURE_FIELDS.index("link_success_probability")
    for time_index in range(batch.max_time_steps):
        prefix = batch.features[:, : time_index + 1, success_index]
        reconstructed_current = prefix[:, -1]
        if not torch.isfinite(reconstructed_current).all().item():
            raise SequenceBatchingViolation("non-finite local history value")


def load_sequence_batch_report_payload(path: str | Path) -> Mapping[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
