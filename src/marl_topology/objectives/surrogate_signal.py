"""Stage 5.2 training-only surrogate signal interface.

The interface is intentionally pure: it consumes registered evaluation
quantities and returns a decomposed training-side signal record. It does not
write replay files, run training, calibrate weights, or inspect actor inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from marl_topology.metrics import require_registered_metrics


SURROGATE_SIGNAL_MODEL_ID = "stage5_2_reward_surrogate_interface_v0"

# Surrogate scalarization structures.
# - flat_weighted_sum_v1: the original Stage 5.2 additive penalty sum. A flat
#   sum cannot represent the lexicographic objective when most samples are
#   infeasible, so reward can worsen while latency/energy improve (Stage 26-30
#   B1 blocker).
# - feasibility_first_barrier_v2 (Stage 31, owner-approved): every feasible
#   point strictly beats every infeasible point via a feasibility barrier; the
#   gradient is reliability-only while infeasible and latency/energy-only while
#   feasible. Reliability plateaus above tau. Keeps the decomposition invariant
#   training_signal_value == -(reliability + latency + energy penalties).
SURROGATE_STRUCTURE_FLAT_WEIGHTED_SUM_V1 = "flat_weighted_sum_v1"
SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2 = "feasibility_first_barrier_v2"
SURROGATE_STRUCTURES = (
    SURROGATE_STRUCTURE_FLAT_WEIGHTED_SUM_V1,
    SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2,
)

SURROGATE_SIGNAL_INPUT_METRICS = (
    "consensus_success_probability",
    "latency",
    "energy",
    "topology_diagnostics",
)

SURROGATE_TRAINING_COLUMNS = (
    "reward_surrogate",
    "reward_reliability_penalty",
    "reward_latency_penalty",
    "reward_energy_penalty",
    "reward_config_id",
)


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SurrogateSignalConfig:
    """Explicit Stage 5.2 config for the training-only surrogate interface."""

    tau: float
    reliability_weight: float
    latency_weight: float
    energy_weight: float
    latency_reference_s: float
    energy_reference_j: float
    clip_min: float = 0.0
    clip_max: float | None = None
    reliability_penalty_power: float = 2.0
    structure: str = SURROGATE_STRUCTURE_FLAT_WEIGHTED_SUM_V1
    feasibility_margin: float = 1.0
    config_id: str = "stage5_2_explicit_untrained_config"
    model_id: str = SURROGATE_SIGNAL_MODEL_ID

    def __post_init__(self) -> None:
        tau = _finite_float("tau", self.tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        object.__setattr__(self, "tau", tau)

        for name in (
            "reliability_weight",
            "latency_weight",
            "energy_weight",
        ):
            value = _finite_float(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)

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

        clip_min = _finite_float("clip_min", self.clip_min)
        if clip_min < 0.0:
            raise ValueError("clip_min must be nonnegative")
        object.__setattr__(self, "clip_min", clip_min)

        if self.clip_max is not None:
            clip_max = _finite_float("clip_max", self.clip_max)
            if clip_max < clip_min:
                raise ValueError("clip_max must be greater than or equal to clip_min")
            object.__setattr__(self, "clip_max", clip_max)

        reliability_penalty_power = _finite_float(
            "reliability_penalty_power", self.reliability_penalty_power
        )
        if reliability_penalty_power < 1.0:
            raise ValueError("reliability_penalty_power must be at least 1")
        object.__setattr__(
            self, "reliability_penalty_power", reliability_penalty_power
        )

        feasibility_margin = _finite_float("feasibility_margin", self.feasibility_margin)
        if feasibility_margin < 0.0:
            raise ValueError("feasibility_margin must be nonnegative")
        object.__setattr__(self, "feasibility_margin", feasibility_margin)

        if self.structure not in SURROGATE_STRUCTURES:
            raise ValueError(f"unsupported surrogate structure: {self.structure}")
        if self.structure == SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2:
            if self.clip_max is None:
                raise ValueError(
                    "feasibility_first_barrier_v2 requires a finite clip_max so the "
                    "feasible objective cost is bounded and the barrier guarantees "
                    "strict feasibility-first ordering"
                )
            if self.feasibility_margin <= 0.0:
                raise ValueError(
                    "feasibility_first_barrier_v2 requires a positive feasibility_margin"
                )

        if not str(self.config_id).strip():
            raise ValueError("config_id must be nonempty")
        if not str(self.model_id).strip():
            raise ValueError("model_id must be nonempty")

    @property
    def max_feasible_objective_cost(self) -> float:
        """Upper bound on the feasible-region cost (latency+energy penalties)."""

        clip_max = self.clip_max if self.clip_max is not None else 0.0
        return (self.latency_weight + self.energy_weight) * clip_max

    @property
    def feasibility_barrier(self) -> float:
        """Offset that makes every infeasible signal below every feasible one."""

        return self.feasibility_margin + self.max_feasible_objective_cost


@dataclass(frozen=True, slots=True)
class SurrogateSignalInput:
    """Registered evaluation quantities consumed by the Stage 5.2 interface."""

    consensus_success_probability: float
    latency: float
    energy: float
    topology_diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        consensus_success_probability = _finite_float(
            "consensus_success_probability", self.consensus_success_probability
        )
        if not 0.0 <= consensus_success_probability <= 1.0:
            raise ValueError("consensus_success_probability must be in [0, 1]")
        object.__setattr__(
            self, "consensus_success_probability", consensus_success_probability
        )

        latency = _finite_float("latency", self.latency)
        if latency < 0.0:
            raise ValueError("latency must be nonnegative")
        object.__setattr__(self, "latency", latency)

        energy = _finite_float("energy", self.energy)
        if energy < 0.0:
            raise ValueError("energy must be nonnegative")
        object.__setattr__(self, "energy", energy)


@dataclass(frozen=True, slots=True)
class SurrogateSignalRecord:
    """Decomposed training-side signal returned by the Stage 5.2 interface."""

    training_signal_value: float
    reliability_violation: float
    reliability_penalty: float
    normalized_latency: float
    normalized_energy: float
    latency_penalty: float
    energy_penalty: float
    constraint_satisfied: bool
    tau: float
    config_id: str
    model_id: str
    structure: str = SURROGATE_STRUCTURE_FLAT_WEIGHTED_SUM_V1
    input_metric_names: tuple[str, ...] = SURROGATE_SIGNAL_INPUT_METRICS
    training_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "training_signal_value",
            "reliability_violation",
            "reliability_penalty",
            "normalized_latency",
            "normalized_energy",
            "latency_penalty",
            "energy_penalty",
            "tau",
        ):
            value = _finite_float(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if not str(self.config_id).strip():
            raise ValueError("config_id must be nonempty")
        if not str(self.model_id).strip():
            raise ValueError("model_id must be nonempty")
        require_registered_metrics(self.input_metric_names)

    def to_training_payload(self) -> dict[str, object]:
        """Return the training-only columns admitted by the replay contract."""

        return {
            "reward_surrogate": self.training_signal_value,
            "reward_reliability_penalty": self.reliability_penalty,
            "reward_latency_penalty": self.latency_penalty,
            "reward_energy_penalty": self.energy_penalty,
            "reward_config_id": self.config_id,
        }


def _clip_nonnegative(value: float, config: SurrogateSignalConfig) -> float:
    clipped = max(config.clip_min, value)
    if config.clip_max is not None:
        clipped = min(config.clip_max, clipped)
    return clipped


def evaluate_reward_surrogate(
    record: SurrogateSignalInput,
    config: SurrogateSignalConfig,
) -> SurrogateSignalRecord:
    """Evaluate the Stage 5.2 training-only surrogate interface."""

    require_registered_metrics(SURROGATE_SIGNAL_INPUT_METRICS)

    reliability_violation = max(0.0, config.tau - record.consensus_success_probability)
    normalized_latency = _clip_nonnegative(
        record.latency / config.latency_reference_s,
        config,
    )
    normalized_energy = _clip_nonnegative(
        record.energy / config.energy_reference_j,
        config,
    )
    feasible = reliability_violation == 0.0

    if config.structure == SURROGATE_STRUCTURE_FEASIBILITY_FIRST_BARRIER_V2:
        # Feasibility-first lexicographic barrier. Above tau (feasible) the
        # reliability term plateaus and the cost is latency+energy only; below
        # tau the cost is the barrier plus a reliability-only gradient, so the
        # only way to improve reward while infeasible is to raise psucc toward
        # tau. The barrier makes every infeasible signal strictly below every
        # feasible one, matching the objective contract's hard feasibility gate.
        # Penalties are weight-folded here so that
        #   signal = -(reliability + latency + energy penalties).
        reliability_raw = reliability_violation**config.reliability_penalty_power
        if feasible:
            reliability_penalty = 0.0
            latency_penalty = config.latency_weight * normalized_latency
            energy_penalty = config.energy_weight * normalized_energy
        else:
            reliability_penalty = (
                config.feasibility_barrier
                + config.reliability_weight * reliability_raw
            )
            latency_penalty = 0.0
            energy_penalty = 0.0
        signal_value = -(reliability_penalty + latency_penalty + energy_penalty)
    else:
        # flat_weighted_sum_v1 (original Stage 5.2 behavior, unchanged). The
        # record stores unweighted penalties; weights are applied in the sum.
        reliability_penalty = reliability_violation**config.reliability_penalty_power
        latency_penalty = normalized_latency
        energy_penalty = normalized_energy
        signal_value = -(
            config.reliability_weight * reliability_penalty
            + config.latency_weight * latency_penalty
            + config.energy_weight * energy_penalty
        )

    return SurrogateSignalRecord(
        training_signal_value=signal_value,
        reliability_violation=reliability_violation,
        reliability_penalty=reliability_penalty,
        normalized_latency=normalized_latency,
        normalized_energy=normalized_energy,
        latency_penalty=latency_penalty,
        energy_penalty=energy_penalty,
        constraint_satisfied=feasible,
        tau=config.tau,
        config_id=config.config_id,
        model_id=config.model_id,
        structure=config.structure,
    )
