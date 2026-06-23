from pathlib import Path

import yaml

from marl_topology.data import REWARD_TRAINING_ONLY_COLUMNS
from marl_topology.metrics import REGISTERED_METRICS
from marl_topology.objectives import (
    SURROGATE_SIGNAL_INPUT_METRICS,
    SURROGATE_TRAINING_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_2_surrogate_fields_are_not_registered_metrics() -> None:
    assert set(SURROGATE_SIGNAL_INPUT_METRICS) <= set(REGISTERED_METRICS)
    assert set(SURROGATE_TRAINING_COLUMNS) == REWARD_TRAINING_ONLY_COLUMNS
    assert not (set(SURROGATE_TRAINING_COLUMNS) & set(REGISTERED_METRICS))


