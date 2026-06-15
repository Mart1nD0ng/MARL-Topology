"""Objective abstractions for latency, energy, and training-side signals."""

from .latency_energy import LatencyEnergy, aggregate_latency_energy
from .normalization import (
    NORMALIZATION_REFERENCE_MODEL_ID,
    NORMALIZATION_REFERENCE_SOURCE_STAGE_ID,
    NORMALIZATION_SELECTION_POLICY,
    NormalizationReferenceConfig,
    NormalizationReferenceRecord,
    select_normalization_references,
)
from .surrogate_signal import (
    SURROGATE_SIGNAL_INPUT_METRICS,
    SURROGATE_SIGNAL_MODEL_ID,
    SURROGATE_TRAINING_COLUMNS,
    SurrogateSignalConfig,
    SurrogateSignalInput,
    SurrogateSignalRecord,
    evaluate_reward_surrogate,
)

__all__ = [
    "LatencyEnergy",
    "NORMALIZATION_REFERENCE_MODEL_ID",
    "NORMALIZATION_REFERENCE_SOURCE_STAGE_ID",
    "NORMALIZATION_SELECTION_POLICY",
    "SURROGATE_SIGNAL_INPUT_METRICS",
    "SURROGATE_SIGNAL_MODEL_ID",
    "SURROGATE_TRAINING_COLUMNS",
    "NormalizationReferenceConfig",
    "NormalizationReferenceRecord",
    "SurrogateSignalConfig",
    "SurrogateSignalInput",
    "SurrogateSignalRecord",
    "aggregate_latency_energy",
    "evaluate_reward_surrogate",
    "select_normalization_references",
]
