"""Normalization reference records for the Stage 5 reward-surrogate path."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from .surrogate_signal import SurrogateSignalConfig


NORMALIZATION_REFERENCE_MODEL_ID = "stage5_3_feasible_positive_max_v1"
NORMALIZATION_REFERENCE_SOURCE_STAGE_ID = (
    "stage_5_0l_stage3_backed_sweep_range_expansion_and_realism_review"
)
NORMALIZATION_SELECTION_POLICY = "feasible_positive_max_v1"


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class NormalizationReferenceConfig:
    """Selection policy for fixed latency and energy references."""

    tau_requirement_min: float = 0.9
    source_stage_id: str = NORMALIZATION_REFERENCE_SOURCE_STAGE_ID
    selection_policy: str = NORMALIZATION_SELECTION_POLICY
    minimum_eligible_rows: int = 1
    config_id: str = "stage5_3_normalization_reference_config"

    def __post_init__(self) -> None:
        tau_requirement_min = _finite_float(
            "tau_requirement_min", self.tau_requirement_min
        )
        if not 0.0 <= tau_requirement_min <= 1.0:
            raise ValueError("tau_requirement_min must be in [0, 1]")
        object.__setattr__(self, "tau_requirement_min", tau_requirement_min)
        if self.selection_policy != NORMALIZATION_SELECTION_POLICY:
            raise ValueError("unsupported normalization selection_policy")
        if self.minimum_eligible_rows < 1:
            raise ValueError("minimum_eligible_rows must be positive")
        if not self.source_stage_id.strip():
            raise ValueError("source_stage_id must be nonempty")
        if not self.config_id.strip():
            raise ValueError("config_id must be nonempty")


@dataclass(frozen=True, slots=True)
class NormalizationReferenceRecord:
    """Fixed references selected from approved evaluation evidence."""

    latency_reference_s: float
    energy_reference_j: float
    selection_policy: str
    source_stage_id: str
    source_row_count: int
    eligible_row_count: int
    excluded_row_count: int
    tau_requirement_min: float
    latency_source_values_s: tuple[float, ...]
    energy_source_values_j: tuple[float, ...]
    config_id: str
    model_id: str = NORMALIZATION_REFERENCE_MODEL_ID
    reference_quality: str = "alpha_stage3_backed_reference_not_deployment_calibration"
    reward_weight_calibration_performed: bool = False
    training_ready: bool = False

    def __post_init__(self) -> None:
        latency_reference_s = _finite_float(
            "latency_reference_s", self.latency_reference_s
        )
        if latency_reference_s <= 0.0:
            raise ValueError("latency_reference_s must be positive")
        object.__setattr__(self, "latency_reference_s", latency_reference_s)

        energy_reference_j = _finite_float("energy_reference_j", self.energy_reference_j)
        if energy_reference_j <= 0.0:
            raise ValueError("energy_reference_j must be positive")
        object.__setattr__(self, "energy_reference_j", energy_reference_j)

        tau_requirement_min = _finite_float(
            "tau_requirement_min", self.tau_requirement_min
        )
        if not 0.0 <= tau_requirement_min <= 1.0:
            raise ValueError("tau_requirement_min must be in [0, 1]")
        object.__setattr__(self, "tau_requirement_min", tau_requirement_min)

        if self.selection_policy != NORMALIZATION_SELECTION_POLICY:
            raise ValueError("unsupported selection_policy")
        if not self.source_stage_id.strip():
            raise ValueError("source_stage_id must be nonempty")
        if self.source_row_count < 1:
            raise ValueError("source_row_count must be positive")
        if self.eligible_row_count < 1:
            raise ValueError("eligible_row_count must be positive")
        if self.excluded_row_count < 0:
            raise ValueError("excluded_row_count must be nonnegative")
        if self.source_row_count != self.eligible_row_count + self.excluded_row_count:
            raise ValueError("source rows must equal eligible plus excluded rows")
        if not self.config_id.strip():
            raise ValueError("config_id must be nonempty")
        if not self.model_id.strip():
            raise ValueError("model_id must be nonempty")
        if not self.reference_quality.strip():
            raise ValueError("reference_quality must be nonempty")
        if self.reward_weight_calibration_performed:
            raise ValueError("normalization reference selection must not calibrate weights")
        if self.training_ready:
            raise ValueError("normalization reference selection must not declare training ready")

        latency_values = tuple(
            _positive_value("latency_source_values_s", value)
            for value in self.latency_source_values_s
        )
        energy_values = tuple(
            _positive_value("energy_source_values_j", value)
            for value in self.energy_source_values_j
        )
        if len(latency_values) != self.eligible_row_count:
            raise ValueError("latency source values must match eligible row count")
        if len(energy_values) != self.eligible_row_count:
            raise ValueError("energy source values must match eligible row count")
        object.__setattr__(self, "latency_source_values_s", latency_values)
        object.__setattr__(self, "energy_source_values_j", energy_values)

    def to_payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "config_id": self.config_id,
            "selection_policy": self.selection_policy,
            "source_stage_id": self.source_stage_id,
            "source_row_count": self.source_row_count,
            "eligible_row_count": self.eligible_row_count,
            "excluded_row_count": self.excluded_row_count,
            "tau_requirement_min": self.tau_requirement_min,
            "latency_reference_s": self.latency_reference_s,
            "energy_reference_j": self.energy_reference_j,
            "latency_source_values_s": list(self.latency_source_values_s),
            "energy_source_values_j": list(self.energy_source_values_j),
            "reference_quality": self.reference_quality,
            "reward_weight_calibration_performed": self.reward_weight_calibration_performed,
            "training_ready": self.training_ready,
        }

    def build_surrogate_config(
        self,
        *,
        tau: float,
        reliability_weight: float,
        latency_weight: float,
        energy_weight: float,
        clip_min: float = 0.0,
        clip_max: float | None = None,
        reliability_penalty_power: float = 2.0,
        config_id: str = "stage5_3_surrogate_config_with_selected_references",
    ) -> SurrogateSignalConfig:
        """Build a Stage 5.2 surrogate config using these fixed references."""

        return SurrogateSignalConfig(
            tau=tau,
            reliability_weight=reliability_weight,
            latency_weight=latency_weight,
            energy_weight=energy_weight,
            latency_reference_s=self.latency_reference_s,
            energy_reference_j=self.energy_reference_j,
            clip_min=clip_min,
            clip_max=clip_max,
            reliability_penalty_power=reliability_penalty_power,
            config_id=config_id,
        )


def select_normalization_references(
    rows: Iterable[Mapping[str, object]],
    config: NormalizationReferenceConfig | None = None,
) -> NormalizationReferenceRecord:
    """Select fixed references from approved evaluation rows."""

    cfg = config or NormalizationReferenceConfig()
    source_rows = tuple(rows)
    if not source_rows:
        raise ValueError("normalization reference source rows must be nonempty")

    eligible_latency: list[float] = []
    eligible_energy: list[float] = []
    for row in source_rows:
        probability = _finite_float(
            "consensus_success_probability",
            row["consensus_success_probability"],
        )
        latency = _finite_float("latency", row["latency"])
        energy = _finite_float("energy", row["energy"])
        if probability >= cfg.tau_requirement_min and latency > 0.0 and energy > 0.0:
            eligible_latency.append(latency)
            eligible_energy.append(energy)

    if len(eligible_latency) < cfg.minimum_eligible_rows:
        raise ValueError("not enough eligible rows for normalization reference selection")

    latency_values = tuple(sorted(eligible_latency))
    energy_values = tuple(sorted(eligible_energy))
    return NormalizationReferenceRecord(
        latency_reference_s=max(latency_values),
        energy_reference_j=max(energy_values),
        selection_policy=cfg.selection_policy,
        source_stage_id=cfg.source_stage_id,
        source_row_count=len(source_rows),
        eligible_row_count=len(latency_values),
        excluded_row_count=len(source_rows) - len(latency_values),
        tau_requirement_min=cfg.tau_requirement_min,
        latency_source_values_s=latency_values,
        energy_source_values_j=energy_values,
        config_id=cfg.config_id,
    )


def _positive_value(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} values must be positive")
    return result
