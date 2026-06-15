"""Link-model regime metadata and validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID = "stage2_deterministic_distance"


class LinkRecordLike(Protocol):
    edge_id: str
    distance_3d_m: float
    link_success_probability: float
    latency_s: float
    energy_j: float
    physics_regime: str


@dataclass(frozen=True, slots=True)
class LinkModelRegime:
    regime_id: str
    description: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    units: dict[str, str]
    assumptions: tuple[str, ...]
    omitted_components: tuple[str, ...]
    nonclaims: tuple[str, ...]
    required_tests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.regime_id:
            raise ValueError("regime_id must be non-empty")
        if not self.inputs:
            raise ValueError("link regime must declare inputs")
        if not self.outputs:
            raise ValueError("link regime must declare outputs")
        if not self.omitted_components:
            raise ValueError("link regime must declare omitted components")
        missing_units = sorted(set(self.inputs + self.outputs) - set(self.units))
        if missing_units:
            raise ValueError(f"missing units for regime fields: {missing_units}")


STAGE2_DETERMINISTIC_DISTANCE_REGIME = LinkModelRegime(
    regime_id=STAGE2_DETERMINISTIC_DISTANCE_REGIME_ID,
    description=(
        "Deterministic Stage 2 distance-only link abstraction for clean-core "
        "skeleton tests and baseline smoke reports."
    ),
    inputs=("distance_3d_m",),
    outputs=("link_success_probability", "latency_s", "energy_j"),
    units={
        "distance_3d_m": "m",
        "link_success_probability": "[0, 1]",
        "latency_s": "s",
        "energy_j": "J",
    },
    assumptions=(
        "link_success_probability = exp(-distance_3d_m / reference_distance_m)",
        "latency_s = base_latency_s + distance_3d_m / speed_of_light_mps",
        "energy_j = fixed_tx_energy_j + energy_per_m_j * distance_3d_m",
        "all links are deterministic and independent for Stage 2 evaluation",
        "candidate edges already encode the selected communication range",
    ),
    omitted_components=(
        "building LoS/NLoS",
        "path loss",
        "shadowing",
        "SINR",
        "interference",
        "bandwidth and packet size",
        "queueing delay",
        "processing delay",
        "receive idle and processing energy",
        "mobility",
    ),
    nonclaims=(
        "not a full 3D V2X physical simulator",
        "not evidence of cross-regime policy quality",
        "not a reward definition",
        "not a PBFT timing model",
        "not calibrated against measurements",
    ),
    required_tests=(
        "distance_3d_m is nonnegative",
        "link_success_probability is in [0, 1]",
        "link_success_probability is nonincreasing with distance",
        "latency_s is nondecreasing and nonnegative with distance",
        "energy_j is nondecreasing and nonnegative with distance",
        "records carry the declared physics_regime",
    ),
)


def get_stage2_link_regime() -> LinkModelRegime:
    return STAGE2_DETERMINISTIC_DISTANCE_REGIME


def validate_link_record_against_regime(
    record: LinkRecordLike,
    regime: LinkModelRegime = STAGE2_DETERMINISTIC_DISTANCE_REGIME,
) -> None:
    if record.physics_regime != regime.regime_id:
        raise ValueError(
            f"link record physics_regime {record.physics_regime!r} does not match "
            f"{regime.regime_id!r}"
        )
    if record.distance_3d_m < 0:
        raise ValueError("distance_3d_m must be nonnegative")
    if not 0.0 <= record.link_success_probability <= 1.0:
        raise ValueError("link_success_probability must be in [0, 1]")
    if record.latency_s < 0:
        raise ValueError("latency_s must be nonnegative")
    if record.energy_j < 0:
        raise ValueError("energy_j must be nonnegative")
