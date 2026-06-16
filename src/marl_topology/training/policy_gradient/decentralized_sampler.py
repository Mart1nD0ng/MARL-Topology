"""Decentralized per-node mutual-acceptance proposal sampler (Task 2 decoder).

This replaces the scene-GLOBAL Plackett-Luce top-k on the LEARNED actor path with a
genuinely decentralized decoder. Each node ``u`` decides ONLY over its own incident
directed edges; an undirected physical link ``(u, v)`` activates iff BOTH endpoints pick
each other (mutual acceptance). The decode is therefore per-node computable, uses zero
global state, and respects each node's radio budget by construction.

Per-node action: budget-truncated Plackett-Luce WITH a STOP token
-----------------------------------------------------------------
Node ``u`` selects sequentially, without replacement, from its incident directed edges
plus a STOP token whose logit is fixed at ``stop_logit`` (default 0). It takes at most
``b_u`` (its radio budget) picks and halts as soon as it samples STOP. Consequences:

* variable proposal size 0..b_u (a node can propose nothing),
* budget respected by construction,
* the "logit >= 0 => propose" intuition realised softly (an incident edge is taken before
  STOP iff its logit beats 0),
* an EXACT sequential-categorical log-probability per node.

Joint log-prob factorisation (the crux for PPO / per-agent MARL)
----------------------------------------------------------------
The joint policy factorises across nodes:

    log pi(joint | obs) = sum_u  log pi_u(a_u | obs_u)

Every *directed* accept variable ``u->v`` is sampled inside node ``u``'s sequence and
contributes to exactly ONE node's factor. The mutual edge ``(u, v)`` is active iff both
``u->v`` and ``v->u`` were picked, with probability ``P_u(u->v) * P_v(v->u)`` -- each
factor living in its own owner's term. So there is NO log-prob double counting even though
an undirected edge appears in two nodes' incident sets. ``per_owner_logprobs`` is exposed
so the trainer can form per-agent advantages (CTDE), and their sum equals the scalar
``logprob`` (asserted by the unit tests).

This sampler reads only directed edge logits + the candidate-graph incidence + per-node
radio budgets. It never reads consensus / latency / energy / reward / global topology
(the same FORBIDDEN_SAMPLER_INPUT_FIELDS contract the other samplers enforce).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from .samplers import (
    FORBIDDEN_SAMPLER_INPUT_FIELDS,
    ProposalSample,
    _categorical_entropy,
)


DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID = "decentralized_per_node_mutual_acceptance_sampler"


def physical_edge_for_directed(directed_edge_id: str) -> str:
    """Canonical undirected ``a--b`` physical id for a directed ``u->v`` edge."""

    if "->" not in directed_edge_id:
        raise ValueError("directed_edge_id must use canonical u->v form")
    left, right = directed_edge_id.split("->", 1)
    if not left or not right or left == right:
        raise ValueError("directed edge endpoints must be distinct")
    a, b = sorted((left, right))
    return f"{a}--{b}"


def owner_of_directed(directed_edge_id: str) -> str:
    return directed_edge_id.split("->", 1)[0]


@dataclass(frozen=True, slots=True)
class DecentralizedMutualSamplerConfig:
    """Per-node sampler config; carries directed-edge incidence + radio budgets only."""

    directed_edge_ids: tuple[str, ...]
    node_budgets: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    default_budget: int = 2
    stop_logit: float = 0.0
    sampler_config_id: str = "decentralized_per_node_mutual_v1"
    forbidden_input_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.directed_edge_ids:
            raise ValueError("directed_edge_ids must be non-empty")
        if len(set(self.directed_edge_ids)) != len(self.directed_edge_ids):
            raise ValueError("directed_edge_ids must be unique")
        for directed_edge_id in self.directed_edge_ids:
            if "->" not in directed_edge_id:
                raise ValueError("directed_edge_ids must use canonical u->v form")
        if self.default_budget < 0:
            raise ValueError("default_budget must be nonnegative")
        if any(budget < 0 for _node, budget in self.node_budgets):
            raise ValueError("per-node budgets must be nonnegative")
        forbidden = sorted(set(self.forbidden_input_fields) & FORBIDDEN_SAMPLER_INPUT_FIELDS)
        if forbidden:
            raise ValueError(f"forbidden sampler input fields: {forbidden}")

    def budget_for(self, node: str) -> int:
        for candidate, budget in self.node_budgets:
            if candidate == node:
                return budget
        return self.default_budget

    def owner_for(self, index: int) -> str:
        return owner_of_directed(self.directed_edge_ids[index])

    def physical_for(self, index: int) -> str:
        return physical_edge_for_directed(self.directed_edge_ids[index])

    def owner_to_indices(self, mask: torch.Tensor) -> dict[str, tuple[int, ...]]:
        active = set(_active(mask))
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, directed_edge_id in enumerate(self.directed_edge_ids):
            if index not in active:
                continue
            grouped[owner_of_directed(directed_edge_id)].append(index)
        return {owner: tuple(indices) for owner, indices in sorted(grouped.items())}

    def physical_to_indices(self, mask: torch.Tensor) -> dict[str, tuple[int, ...]]:
        active = set(_active(mask))
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in range(len(self.directed_edge_ids)):
            if index not in active:
                continue
            grouped[self.physical_for(index)].append(index)
        return {physical: tuple(indices) for physical, indices in sorted(grouped.items())}


class DecentralizedPerNodeMutualSampler:
    """Budget-truncated per-node Plackett-Luce with STOP + mutual-acceptance assembly."""

    sampler_id = DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID

    def sample(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: DecentralizedMutualSamplerConfig,
        rng: torch.Generator,
        deterministic: bool = False,
    ) -> ProposalSample:
        logits, mask = _validate(logits, mask, config)
        owner_indices = config.owner_to_indices(mask)
        accepted: set[int] = set()
        per_owner_logprob: dict[str, float] = {}
        per_owner_entropy: dict[str, float] = {}
        ordered_by_owner: dict[str, tuple[int, ...]] = {}
        total_logprob = logits.new_tensor(0.0)
        total_entropy = logits.new_tensor(0.0)
        for owner in sorted(owner_indices):
            picks, owner_logprob, owner_entropy = self._sample_owner(
                logits, list(owner_indices[owner]), config.budget_for(owner), config.stop_logit, rng, deterministic
            )
            ordered_by_owner[owner] = tuple(picks)
            per_owner_logprob[owner] = float(owner_logprob.detach().cpu().item())
            per_owner_entropy[owner] = float(owner_entropy.detach().cpu().item())
            total_logprob = total_logprob + owner_logprob
            total_entropy = total_entropy + owner_entropy
            accepted.update(picks)
        proposed = _mutual_physical_edges(config.physical_to_indices(mask), accepted)
        return ProposalSample(
            proposed_physical_edges=proposed,
            logprob=total_logprob,
            entropy=total_entropy,
            sampler_id=self.sampler_id,
            raw_sample_data={
                "ordered_indices_by_owner": {owner: list(indices) for owner, indices in ordered_by_owner.items()},
                "per_owner_logprobs": per_owner_logprob,
                "per_owner_entropies": per_owner_entropy,
                "logprob_semantics": "per_node_budget_truncated_plackett_luce_with_stop",
            },
            diagnostics={
                "valid_candidate_count": int(mask.sum().item()),
                "proposal_count": len(proposed),
                "active_owner_count": len(owner_indices),
                "proposal_distribution": "decentralized_per_node_mutual_acceptance",
                "logprob_exact_for_policy_action": True,
                "logprob_factorizes_per_agent": True,
            },
        )

    def sample_batch(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: DecentralizedMutualSamplerConfig,
        rng: torch.Generator,
    ) -> tuple[ProposalSample, ...]:
        if logits.ndim == 1:
            return (self.sample(logits, mask, config, rng),)
        if logits.ndim != 2:
            raise ValueError("logits batch must be [batch, directed_edge_count]")
        if mask.shape != logits.shape:
            raise ValueError("mask batch shape must match logits batch")
        return tuple(self.sample(logits[i], mask[i], config, rng) for i in range(logits.shape[0]))

    def _sample_owner(
        self,
        logits: torch.Tensor,
        incident: list[int],
        budget: int,
        stop_logit: float,
        rng: torch.Generator,
        deterministic: bool,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        remaining = list(incident)
        picks: list[int] = []
        logprob = logits.new_tensor(0.0)
        entropy = logits.new_tensor(0.0)
        steps = min(budget, len(incident))
        for _step in range(steps):
            candidate_logits = _candidate_logits(logits, remaining, stop_logit)
            log_probs = torch.log_softmax(candidate_logits, dim=0)
            entropy = entropy + _categorical_entropy(candidate_logits)
            if deterministic:
                choice = int(torch.argmax(candidate_logits).item())
            else:
                probs = torch.softmax(candidate_logits, dim=0)
                choice = int(torch.multinomial(probs, 1, generator=rng).item())
            logprob = logprob + log_probs[choice]
            if choice == len(remaining):  # STOP token
                break
            picks.append(remaining.pop(choice))
        return picks, logprob, entropy

    def logprob_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: DecentralizedMutualSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        return self._replay(logits, mask, config, raw_sample_data, accumulate_entropy=False)

    def entropy_of(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: DecentralizedMutualSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> torch.Tensor:
        return self._replay(logits, mask, config, raw_sample_data, accumulate_entropy=True)

    def per_owner_logprobs(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: DecentralizedMutualSamplerConfig,
        raw_sample_data: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        """Per-agent (per-owner-node) log-probabilities, for CTDE per-agent advantages."""

        logits, mask = _validate(logits, mask, config)
        owner_indices = config.owner_to_indices(mask)
        ordered = _ordered_by_owner(raw_sample_data)
        result: dict[str, torch.Tensor] = {}
        for owner in sorted(owner_indices):
            result[owner] = self._replay_owner(
                logits, list(owner_indices[owner]), config.budget_for(owner), config.stop_logit,
                ordered.get(owner, ()), accumulate_entropy=False,
            )
        return result

    def _replay(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        config: DecentralizedMutualSamplerConfig,
        raw_sample_data: Mapping[str, object],
        *,
        accumulate_entropy: bool,
    ) -> torch.Tensor:
        logits, mask = _validate(logits, mask, config)
        owner_indices = config.owner_to_indices(mask)
        ordered = _ordered_by_owner(raw_sample_data)
        total = logits.new_tensor(0.0)
        for owner in sorted(owner_indices):
            total = total + self._replay_owner(
                logits, list(owner_indices[owner]), config.budget_for(owner), config.stop_logit,
                ordered.get(owner, ()), accumulate_entropy=accumulate_entropy,
            )
        return total

    def _replay_owner(
        self,
        logits: torch.Tensor,
        incident: list[int],
        budget: int,
        stop_logit: float,
        picks: tuple[int, ...],
        *,
        accumulate_entropy: bool,
    ) -> torch.Tensor:
        remaining = list(incident)
        total = logits.new_tensor(0.0)
        for chosen_index in picks:
            if chosen_index not in remaining:
                raise ValueError("ordered picks include an inactive, duplicate, or non-incident edge")
            candidate_logits = _candidate_logits(logits, remaining, stop_logit)
            if accumulate_entropy:
                total = total + _categorical_entropy(candidate_logits)
            else:
                local_index = remaining.index(chosen_index)
                total = total + torch.log_softmax(candidate_logits, dim=0)[local_index]
            remaining.pop(remaining.index(chosen_index))
        # Trailing STOP: the node halted early (sampled STOP) iff it took fewer than its
        # budget AND incident edges still remained. Otherwise it stopped by budget/exhaustion.
        if len(picks) < budget and remaining:
            candidate_logits = _candidate_logits(logits, remaining, stop_logit)
            if accumulate_entropy:
                total = total + _categorical_entropy(candidate_logits)
            else:
                total = total + torch.log_softmax(candidate_logits, dim=0)[len(remaining)]
        return total


def build_decentralized_sampler_registry() -> dict[str, DecentralizedPerNodeMutualSampler]:
    return {DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID: DecentralizedPerNodeMutualSampler()}


def get_decentralized_mutual_sampler() -> DecentralizedPerNodeMutualSampler:
    return build_decentralized_sampler_registry()[DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID]


def _candidate_logits(logits: torch.Tensor, remaining: list[int], stop_logit: float) -> torch.Tensor:
    if remaining:
        edge_logits = logits[torch.tensor(remaining, dtype=torch.long, device=logits.device)]
    else:
        edge_logits = logits.new_empty((0,))
    stop = logits.new_tensor([stop_logit])
    return torch.cat((edge_logits, stop), dim=0)


def _mutual_physical_edges(
    physical_indices: Mapping[str, tuple[int, ...]],
    accepted: set[int],
) -> tuple[str, ...]:
    proposed = []
    for physical, indices in physical_indices.items():
        if len(indices) == 2 and all(index in accepted for index in indices):
            proposed.append(physical)
    return tuple(sorted(set(proposed)))


def _ordered_by_owner(raw_sample_data: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    raw = raw_sample_data.get("ordered_indices_by_owner", {})
    if not isinstance(raw, Mapping):
        raise ValueError("raw_sample_data must include ordered_indices_by_owner mapping")
    return {str(owner): tuple(int(index) for index in indices) for owner, indices in raw.items()}


def _active(mask: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(index) for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist())


def _validate(
    logits: torch.Tensor,
    mask: torch.Tensor,
    config: DecentralizedMutualSamplerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional directed-edge logits")
    if mask.shape != logits.shape:
        raise ValueError("mask shape must match logits")
    if logits.shape[0] != len(config.directed_edge_ids):
        raise ValueError("logit count must match directed_edge_ids")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    return logits, mask.to(dtype=torch.bool, device=logits.device)
