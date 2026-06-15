"""Rollout batch records for the Stage 24 critic-integrated micro-loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    actor_safe_observation: Mapping[str, object]
    centralized_critic_input: Mapping[str, object]
    actor_logits: tuple[float, ...]
    sampler_id: str
    proposal_action: Mapping[str, object]
    proposal_logprob: torch.Tensor
    proposal_entropy: torch.Tensor
    pre_projection_proposals: tuple[str, ...]
    post_projection_selected_physical_edges: tuple[str, ...]
    projection_diagnostics: Mapping[str, object]
    consensus_success_probability: float
    latency: float
    energy: float
    reward_surrogate: float
    value_prediction: torch.Tensor
    done: bool
    mask: float
    scenario_id: str
    time_step: int
    seed: int
    row_index: int
    step_index: int
    raw_sample_data: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.sampler_id:
            raise ValueError("sampler_id must be declared")
        if not torch.isfinite(self.proposal_logprob).all().item():
            raise ValueError("proposal_logprob must be finite")
        if not torch.isfinite(self.proposal_entropy).all().item():
            raise ValueError("proposal_entropy must be finite")
        if not torch.isfinite(self.value_prediction).all().item():
            raise ValueError("value_prediction must be finite")
        if self.mask not in {0.0, 1.0}:
            raise ValueError("mask must be 0.0 or 1.0")
        object.__setattr__(
            self,
            "pre_projection_proposals",
            tuple(sorted(str(edge_id) for edge_id in self.pre_projection_proposals)),
        )
        object.__setattr__(
            self,
            "post_projection_selected_physical_edges",
            tuple(sorted(str(edge_id) for edge_id in self.post_projection_selected_physical_edges)),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "actor_safe_observation": _jsonable(dict(self.actor_safe_observation)),
            "centralized_critic_input": _jsonable(dict(self.centralized_critic_input)),
            "actor_logits": list(self.actor_logits),
            "sampler_id": self.sampler_id,
            "proposal_action": dict(self.proposal_action),
            "proposal_logprob": _scalar(self.proposal_logprob),
            "proposal_entropy": _scalar(self.proposal_entropy),
            "pre_projection_proposals": list(self.pre_projection_proposals),
            "post_projection_selected_physical_edges": list(
                self.post_projection_selected_physical_edges
            ),
            "projection_diagnostics": dict(self.projection_diagnostics),
            "consensus_success_probability": self.consensus_success_probability,
            "latency": self.latency,
            "energy": self.energy,
            "reward_surrogate": self.reward_surrogate,
            "value_prediction": _scalar(self.value_prediction),
            "done": self.done,
            "mask": self.mask,
            "scenario_id": self.scenario_id,
            "time_step": self.time_step,
            "seed": self.seed,
            "row_index": self.row_index,
            "step_index": self.step_index,
            "raw_sample_data": _jsonable(self.raw_sample_data),
        }


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    transitions: tuple[RolloutTransition, ...]
    observations: tuple[Mapping[str, object], ...]
    centralized_inputs: tuple[Mapping[str, object], ...]
    actions: tuple[Mapping[str, object], ...]
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    masks: torch.Tensor
    entropies: torch.Tensor
    projection_diagnostics: tuple[Mapping[str, object], ...]
    scenario_time_metadata: tuple[Mapping[str, object], ...]
    num_scenarios: int
    rollout_steps: int

    def __post_init__(self) -> None:
        expected_count = self.num_scenarios * self.rollout_steps
        if len(self.transitions) != expected_count:
            raise ValueError("transition count must match rollout dimensions")
        for name in ("old_logprobs", "rewards", "values", "dones", "masks", "entropies"):
            tensor = getattr(self, name)
            if tensor.reshape(-1).shape != (expected_count,):
                raise ValueError(f"{name} shape must match transition count")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must be finite")
        if len(self.observations) != expected_count:
            raise ValueError("observations length mismatch")
        if len(self.centralized_inputs) != expected_count:
            raise ValueError("centralized_inputs length mismatch")

    @property
    def total_transitions(self) -> int:
        return self.num_scenarios * self.rollout_steps

    def to_payload(self, *, include_records: bool = False) -> dict[str, object]:
        payload = {
            "total_transitions": self.total_transitions,
            "num_scenarios": self.num_scenarios,
            "rollout_steps": self.rollout_steps,
            "old_logprob_mean": _scalar(self.old_logprobs.mean()),
            "reward_surrogate_mean": _scalar(self.rewards.mean()),
            "value_prediction_mean": _scalar(self.values.mean()),
            "entropy_mean": _scalar(self.entropies.mean()),
            "done_count": int(self.dones.sum().detach().cpu().item()),
            "mask_sum": _scalar(self.masks.sum()),
        }
        if include_records:
            payload["transitions"] = [transition.to_payload() for transition in self.transitions]
        return payload


def build_rollout_batch(
    transitions: tuple[RolloutTransition, ...],
    *,
    num_scenarios: int,
    rollout_steps: int,
) -> RolloutBatch:
    if not transitions:
        raise ValueError("rollout batch requires transitions")
    return RolloutBatch(
        transitions=transitions,
        observations=tuple(transition.actor_safe_observation for transition in transitions),
        centralized_inputs=tuple(
            transition.centralized_critic_input for transition in transitions
        ),
        actions=tuple(transition.proposal_action for transition in transitions),
        old_logprobs=torch.stack(
            [transition.proposal_logprob.detach().reshape(()) for transition in transitions]
        ),
        rewards=torch.tensor(
            [float(transition.reward_surrogate) for transition in transitions],
            dtype=torch.float32,
        ),
        values=torch.stack(
            [transition.value_prediction.detach().reshape(()) for transition in transitions]
        ),
        dones=torch.tensor(
            [1.0 if transition.done else 0.0 for transition in transitions],
            dtype=torch.float32,
        ),
        masks=torch.tensor([float(transition.mask) for transition in transitions], dtype=torch.float32),
        entropies=torch.stack(
            [transition.proposal_entropy.detach().reshape(()) for transition in transitions]
        ),
        projection_diagnostics=tuple(
            transition.projection_diagnostics for transition in transitions
        ),
        scenario_time_metadata=tuple(
            {
                "scenario_id": transition.scenario_id,
                "time_step": transition.time_step,
                "seed": transition.seed,
                "row_index": transition.row_index,
                "step_index": transition.step_index,
            }
            for transition in transitions
        ),
        num_scenarios=num_scenarios,
        rollout_steps=rollout_steps,
    )


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(()).item())


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
