"""Stage 23 stochastic proposal samplers for physical-link policy gradients.

The policy action is a stochastic proposal over undirected physical links.
The environment owns projection from proposals to the selected physical
topology, so the log-probability here is the proposal log-probability only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

import torch
import torch.nn.functional as F


PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID = "physical_bernoulli_proposal_sampler"
PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID = "physical_plackett_luce_top_k_sampler"
ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID = (
    "endpoint_budgeted_physical_proposal_sampler"
)
# Stage 31 (B3 projection-friction repair): a sequential sampler that only ever
# proposes endpoint-budget-feasible edge sets, so the deployment assembler never
# rejects a proposal for TX_BUDGET_EXCEEDED.
BUDGET_AWARE_SEQUENTIAL_PROPOSAL_SAMPLER_ID = (
    "physical_budget_aware_sequential_sampler"
)

ACTIVE_POLICY_GRADIENT_SAMPLER_ID = PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID
STAGE31_ACTIVE_POLICY_GRADIENT_SAMPLER_ID = BUDGET_AWARE_SEQUENTIAL_PROPOSAL_SAMPLER_ID
ARCHIVED_TRIAL_SAMPLER_IDS = (
    PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID,
    ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID,
)

FORBIDDEN_SAMPLER_INPUT_FIELDS = frozenset(
    {
        "consensus_success_probability",
        "latency",
        "energy",
        "reward_surrogate",
        "global_objective_value",
        "objective_value",
        "future_outcome",
        "selected_topology",
        "global_topology",
        "stage4_pbft_result",
    }
)


class PhysicalProposalSampler(Protocol):
    sampler_id: str

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: "ProposalSamplerConfig",
        rng: torch.Generator,
    ) -> "ProposalSample":
        ...

    def logprob_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: "ProposalSamplerConfig",
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        ...

    def entropy_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: "ProposalSamplerConfig",
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True, slots=True)
class ProposalSamplerConfig:
    """Sampler-only config; no objective or reward fields are admitted."""

    physical_edge_ids: tuple[str, ...]
    top_k: int = 3
    endpoint_budget: int = 1
    # Optional per-endpoint (per-node) budgets. When provided they override the
    # scalar endpoint_budget for those nodes, so the sampler can match a
    # kind-aware deployment assembler (e.g. RSU=4, vehicle=2).
    endpoint_budgets: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    sampler_config_id: str = "stage23_selected_physical_sampler_config_v1"
    forbidden_input_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.physical_edge_ids:
            raise ValueError("physical_edge_ids must be non-empty")
        if len(set(self.physical_edge_ids)) != len(self.physical_edge_ids):
            raise ValueError("physical_edge_ids must be unique")
        if any("--" not in edge_id for edge_id in self.physical_edge_ids):
            raise ValueError("physical_edge_ids must use canonical a--b form")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if self.endpoint_budget < 0:
            raise ValueError("endpoint_budget must be nonnegative")
        if any(budget < 0 for _node, budget in self.endpoint_budgets):
            raise ValueError("per-endpoint budgets must be nonnegative")
        forbidden = sorted(set(self.forbidden_input_fields) & FORBIDDEN_SAMPLER_INPUT_FIELDS)
        if forbidden:
            raise ValueError(f"forbidden sampler input fields: {forbidden}")

    def budget_for(self, endpoint: str) -> int:
        for node, budget in self.endpoint_budgets:
            if node == endpoint:
                return budget
        return self.endpoint_budget

    def endpoints_for(self, edge_index: int) -> tuple[str, str]:
        edge_id = self.physical_edge_ids[edge_index]
        left, right = edge_id.split("--", 1)
        if not left or not right or left == right:
            raise ValueError("physical edge endpoints must be distinct")
        return left, right

    def endpoint_to_indices(self, mask: torch.Tensor) -> dict[str, tuple[int, ...]]:
        active = set(_active_indices(mask))
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, _edge_id in enumerate(self.physical_edge_ids):
            if index not in active:
                continue
            for endpoint in self.endpoints_for(index):
                grouped[endpoint].append(index)
        return {endpoint: tuple(indices) for endpoint, indices in sorted(grouped.items())}


@dataclass(frozen=True, slots=True)
class ProposalSample:
    proposed_physical_edges: tuple[str, ...]
    logprob: torch.Tensor
    entropy: torch.Tensor
    sampler_id: str
    raw_sample_data: Mapping[str, object]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.sampler_id:
            raise ValueError("sampler_id must be declared")
        if not torch.isfinite(self.logprob).all().item():
            raise ValueError("proposal logprob must be finite")
        if not torch.isfinite(self.entropy).all().item():
            raise ValueError("proposal entropy must be finite")
        if len(set(self.proposed_physical_edges)) != len(self.proposed_physical_edges):
            raise ValueError("proposed_physical_edges must be unique")
        object.__setattr__(
            self,
            "proposed_physical_edges",
            tuple(sorted(str(edge_id) for edge_id in self.proposed_physical_edges)),
        )
        object.__setattr__(self, "raw_sample_data", dict(self.raw_sample_data))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_payload(self) -> dict[str, object]:
        return {
            "proposed_physical_edges": list(self.proposed_physical_edges),
            "logprob": float(self.logprob.detach().cpu().reshape(()).item()),
            "entropy": float(self.entropy.detach().cpu().reshape(()).item()),
            "sampler_id": self.sampler_id,
            "raw_sample_data": _jsonable_raw_sample(self.raw_sample_data),
            "diagnostics": dict(self.diagnostics),
        }


class _SamplerBase:
    sampler_id: str

    def sample_batch(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        rng: torch.Generator,
    ) -> tuple[ProposalSample, ...]:
        if logits.ndim == 1:
            return (self.sample(logits, mask, config, rng),)
        if logits.ndim != 2:
            raise ValueError("logits batch must be [batch, physical_edge_count]")
        if mask.shape != logits.shape:
            raise ValueError("mask batch shape must match logits batch")
        return tuple(
            self.sample(logits[index], mask[index], config, rng)
            for index in range(logits.shape[0])
        )


class PhysicalBernoulliProposalSampler(_SamplerBase):
    """Sampler A: independent Bernoulli proposal per valid physical link."""

    sampler_id = PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        rng: torch.Generator,
    ) -> ProposalSample:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        action_vector = logits.new_zeros(logits.shape)
        if bool(mask.any().item()):
            valid_logits = logits[mask]
            draws = torch.rand(
                valid_logits.shape,
                dtype=valid_logits.dtype,
                device=valid_logits.device,
                generator=rng,
            )
            action_vector[mask] = (draws < torch.sigmoid(valid_logits)).to(logits.dtype)
        logprob = self.logprob_of(logits, mask, config, {"action_vector": action_vector})
        entropy = _bernoulli_entropy(logits[mask]).sum() if bool(mask.any().item()) else logits.new_tensor(0.0)
        proposed = _edges_from_action_vector(action_vector, config)
        return ProposalSample(
            proposed_physical_edges=proposed,
            logprob=logprob,
            entropy=entropy,
            sampler_id=self.sampler_id,
            raw_sample_data={
                "action_vector": action_vector.detach(),
                "logprob_semantics": "sum_independent_bernoulli_over_valid_physical_links",
            },
            diagnostics={
                "valid_candidate_count": int(mask.sum().item()),
                "proposal_count": len(proposed),
                "proposal_distribution": "bernoulli_sigmoid_logits",
                "logprob_exact_for_policy_action": True,
            },
        )

    def logprob_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        actions = _tensor_from_raw(raw_sample_data, "action_vector", logits)
        if actions.shape != logits.shape:
            raise ValueError("action_vector shape must match logits")
        if bool((actions[~mask] != 0.0).any().item()):
            raise ValueError("invalid masked candidates cannot be proposed")
        if not bool(mask.any().item()):
            return logits.new_tensor(0.0)
        return -F.binary_cross_entropy_with_logits(
            logits[mask],
            actions[mask],
            reduction="none",
        ).sum()

    def entropy_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        if not bool(mask.any().item()):
            return logits.new_tensor(0.0)
        return _bernoulli_entropy(logits[mask]).sum()


class PhysicalPlackettLuceTopKSampler(_SamplerBase):
    """Sampler B: sequential without-replacement top-k proposal sampler."""

    sampler_id = PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        rng: torch.Generator,
        deterministic: bool = False,
    ) -> ProposalSample:
        # deterministic=True selects the policy MODE (greedy argmax at each Plackett-Luce
        # step) instead of sampling -- the action a deployed controller would take. It is
        # for EVALUATION only (training rollouts stay stochastic for exploration), so it
        # never affects the PPO ratio. logprob/entropy are still computed for the picked
        # action. Default False is byte-identical to the original stochastic sampler.
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        remaining = list(_active_indices(mask))
        chosen: list[int] = []
        logprob = logits.new_tensor(0.0)
        entropy = logits.new_tensor(0.0)
        steps = min(config.top_k, len(remaining))
        for _step in range(steps):
            remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=logits.device)
            step_logits = logits[remaining_tensor]
            probs = torch.softmax(step_logits, dim=0)
            if deterministic:
                local_index = int(torch.argmax(step_logits).item())
            else:
                local_index = int(torch.multinomial(probs, 1, generator=rng).item())
            logprob = logprob + torch.log_softmax(step_logits, dim=0)[local_index]
            entropy = entropy + _categorical_entropy(step_logits)
            chosen_index = remaining.pop(local_index)
            chosen.append(chosen_index)
        proposed = tuple(config.physical_edge_ids[index] for index in sorted(chosen))
        return ProposalSample(
            proposed_physical_edges=proposed,
            logprob=logprob,
            entropy=entropy,
            sampler_id=self.sampler_id,
            raw_sample_data={
                "ordered_indices": tuple(chosen),
                "top_k": config.top_k,
                "logprob_semantics": "sequential_plackett_luce_without_replacement",
            },
            diagnostics={
                "valid_candidate_count": int(mask.sum().item()),
                "proposal_count": len(proposed),
                "top_k": config.top_k,
                "proposal_distribution": "plackett_luce_top_k_without_replacement",
                "entropy_is_sequential_categorical_sum": True,
                "logprob_exact_for_policy_action": True,
            },
        )

    def logprob_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        ordered = tuple(int(index) for index in raw_sample_data.get("ordered_indices", ()))
        remaining = list(_active_indices(mask))
        logprob = logits.new_tensor(0.0)
        for chosen_index in ordered:
            if chosen_index not in remaining:
                raise ValueError("ordered_indices include masked, duplicate, or invalid candidate")
            remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=logits.device)
            step_logits = logits[remaining_tensor]
            local_index = remaining.index(chosen_index)
            logprob = logprob + torch.log_softmax(step_logits, dim=0)[local_index]
            remaining.pop(local_index)
        return logprob

    def entropy_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        ordered = tuple(int(index) for index in raw_sample_data.get("ordered_indices", ()))
        remaining = list(_active_indices(mask))
        entropy = logits.new_tensor(0.0)
        for chosen_index in ordered:
            if chosen_index not in remaining:
                raise ValueError("ordered_indices include masked, duplicate, or invalid candidate")
            remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=logits.device)
            step_logits = logits[remaining_tensor]
            entropy = entropy + _categorical_entropy(step_logits)
            remaining.pop(remaining.index(chosen_index))
        return entropy


class EndpointBudgetedPhysicalProposalSampler(_SamplerBase):
    """Sampler C: exact endpoint-action proposal sampler."""

    sampler_id = ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        rng: torch.Generator,
    ) -> ProposalSample:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        endpoint_choices: list[tuple[str, int]] = []
        endpoint_proposal_indices: dict[str, list[int]] = {}
        endpoint_logprob_terms: dict[str, torch.Tensor] = {}
        logprob = logits.new_tensor(0.0)
        entropy = logits.new_tensor(0.0)
        proposed_indices: set[int] = set()
        for endpoint, incident in config.endpoint_to_indices(mask).items():
            remaining = list(incident)
            steps = min(config.endpoint_budget, len(remaining))
            endpoint_term = logits.new_tensor(0.0)
            endpoint_proposal_indices[endpoint] = []
            for _step in range(steps):
                remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=logits.device)
                step_logits = logits[remaining_tensor]
                probs = torch.softmax(step_logits, dim=0)
                local_index = int(torch.multinomial(probs, 1, generator=rng).item())
                chosen_index = remaining.pop(local_index)
                step_logprob = torch.log_softmax(step_logits, dim=0)[local_index]
                endpoint_term = endpoint_term + step_logprob
                logprob = logprob + step_logprob
                entropy = entropy + _categorical_entropy(step_logits)
                endpoint_choices.append((endpoint, chosen_index))
                endpoint_proposal_indices[endpoint].append(chosen_index)
                proposed_indices.add(chosen_index)
            endpoint_logprob_terms[endpoint] = endpoint_term
        proposed = tuple(config.physical_edge_ids[index] for index in sorted(proposed_indices))
        endpoint_proposal_sets = {
            endpoint: tuple(config.physical_edge_ids[index] for index in indices)
            for endpoint, indices in sorted(endpoint_proposal_indices.items())
        }
        endpoint_logprobs = {
            endpoint: float(term.detach().cpu().reshape(()).item())
            for endpoint, term in sorted(endpoint_logprob_terms.items())
        }
        return ProposalSample(
            proposed_physical_edges=proposed,
            logprob=logprob,
            entropy=entropy,
            sampler_id=self.sampler_id,
            raw_sample_data={
                "endpoint_choices": tuple(endpoint_choices),
                "endpoint_budget_config": {
                    "default_endpoint_budget": config.endpoint_budget,
                    "local_without_replacement": True,
                },
                "endpoint_proposal_sets": endpoint_proposal_sets,
                "endpoint_logprobs": endpoint_logprobs,
                "joint_proposal_logprob": logprob.detach(),
                "proposal_entropy": entropy.detach(),
                "aggregated_physical_proposals": proposed,
                "logprob_semantics": "exact_endpoint_proposal_logprob",
                "projected_topology_logprob_exact": False,
            },
            diagnostics={
                "valid_candidate_count": int(mask.sum().item()),
                "proposal_count": len(proposed),
                "endpoint_budget_config": {
                    "default_endpoint_budget": config.endpoint_budget,
                    "local_without_replacement": True,
                },
                "endpoint_proposal_sets": endpoint_proposal_sets,
                "endpoint_logprobs": endpoint_logprobs,
                "joint_proposal_logprob": float(logprob.detach().cpu().reshape(()).item()),
                "proposal_entropy": float(entropy.detach().cpu().reshape(()).item()),
                "aggregated_physical_proposals": proposed,
                "proposal_distribution": "endpoint_local_categorical_without_replacement",
                "aggregated_physical_logprob_claimed_exact": False,
                "logprob_semantics": "exact_endpoint_proposal_logprob",
                "projected_topology_logprob_exact": False,
                "endpoint_choice_logprob_exact": True,
                "objective_or_reward_inputs_used": False,
            },
        )

    def logprob_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        choices = tuple(raw_sample_data.get("endpoint_choices", ()))
        choices_by_endpoint: dict[str, list[int]] = defaultdict(list)
        for item in choices:
            endpoint, chosen_index = item  # type: ignore[misc]
            choices_by_endpoint[str(endpoint)].append(int(chosen_index))
        logprob = logits.new_tensor(0.0)
        endpoint_indices = config.endpoint_to_indices(mask)
        for endpoint in sorted(endpoint_indices):
            remaining = list(endpoint_indices[endpoint])
            for chosen_index in choices_by_endpoint.get(endpoint, ()):
                if chosen_index not in remaining:
                    raise ValueError("endpoint_choices include masked, duplicate, or invalid candidate")
                remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=logits.device)
                step_logits = logits[remaining_tensor]
                local_index = remaining.index(chosen_index)
                logprob = logprob + torch.log_softmax(step_logits, dim=0)[local_index]
                remaining.pop(local_index)
        return logprob

    def entropy_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        choices = tuple(raw_sample_data.get("endpoint_choices", ()))
        choices_by_endpoint: dict[str, list[int]] = defaultdict(list)
        for item in choices:
            endpoint, chosen_index = item  # type: ignore[misc]
            choices_by_endpoint[str(endpoint)].append(int(chosen_index))
        entropy = logits.new_tensor(0.0)
        endpoint_indices = config.endpoint_to_indices(mask)
        for endpoint in sorted(endpoint_indices):
            remaining = list(endpoint_indices[endpoint])
            for chosen_index in choices_by_endpoint.get(endpoint, ()):
                if chosen_index not in remaining:
                    raise ValueError("endpoint_choices include masked, duplicate, or invalid candidate")
                remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=logits.device)
                step_logits = logits[remaining_tensor]
                entropy = entropy + _categorical_entropy(step_logits)
                remaining.pop(remaining.index(chosen_index))
        return entropy


class BudgetAwareSequentialProposalSampler(_SamplerBase):
    """Stage 31 Sampler D: sequential proposal that respects the endpoint budget.

    At each step only edges whose both endpoints still have remaining budget are
    eligible. The proposal is therefore guaranteed endpoint-budget-feasible, so
    the deployment assembler accepts every proposed edge (no TX_BUDGET_EXCEEDED).
    The log-probability is the exact sequential categorical log-probability over
    the *eligible* set at each step, which makes the gradient consistent with the
    proposal the environment actually projects.
    """

    sampler_id = BUDGET_AWARE_SEQUENTIAL_PROPOSAL_SAMPLER_ID

    def _eligible(
        self,
        remaining: list[int],
        endpoint_usage: Mapping[str, int],
        config: ProposalSamplerConfig,
    ) -> list[int]:
        eligible: list[int] = []
        for index in remaining:
            if all(
                endpoint_usage.get(endpoint, 0) < config.budget_for(endpoint)
                for endpoint in config.endpoints_for(index)
            ):
                eligible.append(index)
        return eligible

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        rng: torch.Generator,
    ) -> ProposalSample:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        remaining = list(_active_indices(mask))
        endpoint_usage: dict[str, int] = defaultdict(int)
        chosen: list[int] = []
        logprob = logits.new_tensor(0.0)
        entropy = logits.new_tensor(0.0)
        steps_cap = config.top_k if config.top_k > 0 else len(remaining)
        while remaining and len(chosen) < steps_cap:
            eligible = self._eligible(remaining, endpoint_usage, config)
            if not eligible:
                break
            eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=logits.device)
            step_logits = logits[eligible_tensor]
            probs = torch.softmax(step_logits, dim=0)
            local_index = int(torch.multinomial(probs, 1, generator=rng).item())
            chosen_index = eligible[local_index]
            logprob = logprob + torch.log_softmax(step_logits, dim=0)[local_index]
            entropy = entropy + _categorical_entropy(step_logits)
            chosen.append(chosen_index)
            for endpoint in config.endpoints_for(chosen_index):
                endpoint_usage[endpoint] += 1
            remaining.remove(chosen_index)
        proposed = tuple(config.physical_edge_ids[index] for index in sorted(chosen))
        return ProposalSample(
            proposed_physical_edges=proposed,
            logprob=logprob,
            entropy=entropy,
            sampler_id=self.sampler_id,
            raw_sample_data={
                "ordered_indices": tuple(chosen),
                "endpoint_budget": config.endpoint_budget,
                "top_k": config.top_k,
                "logprob_semantics": "budget_feasible_sequential_categorical",
                "proposal_is_endpoint_budget_feasible": True,
            },
            diagnostics={
                "valid_candidate_count": int(mask.sum().item()),
                "proposal_count": len(proposed),
                "endpoint_budget": config.endpoint_budget,
                "proposal_distribution": "budget_feasible_sequential_categorical",
                "proposal_is_endpoint_budget_feasible": True,
                "logprob_exact_for_policy_action": True,
            },
        )

    def _replay(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
        *,
        accumulate_entropy: bool,
    ) -> torch.Tensor:
        logits, mask = _validate_logits_mask_config(logits, mask, config)
        ordered = tuple(int(index) for index in raw_sample_data.get("ordered_indices", ()))
        remaining = list(_active_indices(mask))
        endpoint_usage: dict[str, int] = defaultdict(int)
        total = logits.new_tensor(0.0)
        for chosen_index in ordered:
            eligible = self._eligible(remaining, endpoint_usage, config)
            if chosen_index not in eligible:
                raise ValueError("ordered_indices include an ineligible or invalid candidate")
            eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=logits.device)
            step_logits = logits[eligible_tensor]
            if accumulate_entropy:
                total = total + _categorical_entropy(step_logits)
            else:
                local_index = eligible.index(chosen_index)
                total = total + torch.log_softmax(step_logits, dim=0)[local_index]
            for endpoint in config.endpoints_for(chosen_index):
                endpoint_usage[endpoint] += 1
            remaining.remove(chosen_index)
        return total

    def logprob_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        return self._replay(logits, mask, config, raw_sample_data, accumulate_entropy=False)

    def entropy_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: ProposalSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        return self._replay(logits, mask, config, raw_sample_data, accumulate_entropy=True)


def build_stage31_active_sampler_registry() -> dict[str, PhysicalProposalSampler]:
    """Return the Stage 31 active budget-aware sampler registry."""

    return {
        STAGE31_ACTIVE_POLICY_GRADIENT_SAMPLER_ID: BudgetAwareSequentialProposalSampler(),
    }


def get_stage31_active_sampler() -> PhysicalProposalSampler:
    return build_stage31_active_sampler_registry()[
        STAGE31_ACTIVE_POLICY_GRADIENT_SAMPLER_ID
    ]


def build_stage23_trial_sampler_registry() -> dict[str, PhysicalProposalSampler]:
    """Return the A/B/C trial samplers used only during Stage 23 comparison."""

    return {
        PHYSICAL_BERNOULLI_PROPOSAL_SAMPLER_ID: PhysicalBernoulliProposalSampler(),
        PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID: PhysicalPlackettLuceTopKSampler(),
        ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID: EndpointBudgetedPhysicalProposalSampler(),
    }


def build_stage24_candidate_sampler_registry() -> dict[str, PhysicalProposalSampler]:
    """Return the active sampler and the repaired endpoint comparison candidate."""

    return {
        PHYSICAL_PLACKETT_LUCE_TOP_K_SAMPLER_ID: PhysicalPlackettLuceTopKSampler(),
        ENDPOINT_BUDGETED_PHYSICAL_PROPOSAL_SAMPLER_ID: EndpointBudgetedPhysicalProposalSampler(),
    }


def build_active_policy_gradient_sampler_registry() -> dict[str, PhysicalProposalSampler]:
    """Return the post-Stage-23 low-entropy active sampler registry."""

    return {
        ACTIVE_POLICY_GRADIENT_SAMPLER_ID: PhysicalPlackettLuceTopKSampler(),
    }


def get_active_policy_gradient_sampler() -> PhysicalProposalSampler:
    return build_active_policy_gradient_sampler_registry()[ACTIVE_POLICY_GRADIENT_SAMPLER_ID]


def sampler_cleanup_report() -> dict[str, object]:
    active = build_active_policy_gradient_sampler_registry()
    trial = build_stage23_trial_sampler_registry()
    return {
        "active_policy_gradient_sampler_id": ACTIVE_POLICY_GRADIENT_SAMPLER_ID,
        "active_sampler_count": len(active),
        "active_sampler_ids": sorted(active),
        "archived_trial_sampler_ids": list(ARCHIVED_TRIAL_SAMPLER_IDS),
        "trial_sampler_ids": sorted(trial),
        "losing_samplers_removed_from_active_registry": (
            sorted(set(trial) - set(active)) == sorted(ARCHIVED_TRIAL_SAMPLER_IDS)
        ),
    }


def _validate_logits_mask_config(
    logits: torch.Tensor,
    mask: torch.Tensor,
    config: ProposalSamplerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional physical-link logits")
    if mask.shape != logits.shape:
        raise ValueError("mask shape must match logits")
    if logits.shape[0] != len(config.physical_edge_ids):
        raise ValueError("logit count must match physical_edge_ids")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    return logits, mask.to(dtype=torch.bool, device=logits.device)


def _active_indices(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(index) for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist())


def _edges_from_action_vector(
    action_vector: torch.Tensor,
    config: ProposalSamplerConfig,
) -> tuple[str, ...]:
    return tuple(
        config.physical_edge_ids[index]
        for index, value in enumerate(action_vector.detach().cpu().tolist())
        if float(value) >= 0.5
    )


def _tensor_from_raw(
    raw_sample_data: Mapping[str, object],
    field: str,
    like: torch.Tensor,
) -> torch.Tensor:
    value = raw_sample_data.get(field)
    if isinstance(value, torch.Tensor):
        return value.to(dtype=like.dtype, device=like.device)
    return torch.tensor(value, dtype=like.dtype, device=like.device)


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).clamp(1e-8, 1.0 - 1e-8)
    return -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log(1.0 - probabilities)
    )


def _categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    probabilities = torch.softmax(logits, dim=0)
    log_probabilities = torch.log_softmax(logits, dim=0)
    return -(probabilities * log_probabilities).sum()


def _jsonable_raw_sample(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable_raw_sample(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable_raw_sample(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            return str(value)
    return value
