"""Controlled clipped policy-gradient pilot for Stage 15."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
import torch.nn.functional as F

from marl_topology.evaluation.fixture_stack import build_fixture_stack
from marl_topology.models import LocalGNNEdgeScorer
from marl_topology.objectives import (
    SurrogateSignalConfig,
    SurrogateSignalInput,
    evaluate_reward_surrogate,
)
from marl_topology.policies import ActorPolicyInput, ConflictAwareGreedyAssembler
from marl_topology.policies.topology_assembler import CandidateEdgeConstraint
from marl_topology.scenario import get_scenario_fixture
from marl_topology.training.supervised_actor_trainer import (
    SupervisedActorTrainingConfig,
    run_supervised_actor_training,
)
from marl_topology.training.supervised_batching import load_learning_evidence_json


STAGE15_POLICY_PILOT_STAGE_ID = "stage_15_controlled_policy_gradient_pilot"


class PolicyPilotViolation(ValueError):
    """Raised when the controlled policy pilot violates preconditions."""


@dataclass(frozen=True, slots=True)
class PolicyPilotConfig:
    seed: int = 15
    clip_epsilon: float = 0.2
    learning_rate: float = 0.001
    update_epochs: int = 2
    tau_requirement_min: float = 0.9
    max_projection_rejection_rate: float = 0.95
    surrogate_config_id: str = "stage15_pilot_equal_component_weights_not_calibrated"


@dataclass(frozen=True, slots=True)
class PolicyPilotResult:
    stage_id: str
    completed: bool
    stop_reason: str | None
    before: dict[str, object]
    after: dict[str, object]
    policy_loss: float
    value_loss: float
    kl: float
    entropy: float
    clip_fraction: float
    projection_rejection_rate: float
    actor_score_distribution: dict[str, float]
    violation_rate_worsened: bool
    actor_collapsed: bool
    checkpoint_written: bool = False
    artifact_written: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "completed": self.completed,
            "stop_reason": self.stop_reason,
            "before": dict(self.before),
            "after": dict(self.after),
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "kl": self.kl,
            "entropy": self.entropy,
            "clip_fraction": self.clip_fraction,
            "projection_rejection_rate": self.projection_rejection_rate,
            "actor_score_distribution": dict(self.actor_score_distribution),
            "violation_rate_worsened": self.violation_rate_worsened,
            "actor_collapsed": self.actor_collapsed,
            "checkpoint_written": self.checkpoint_written,
            "artifact_written": self.artifact_written,
        }


def run_controlled_policy_pilot(
    evidence_path,
    *,
    config: PolicyPilotConfig | None = None,
) -> PolicyPilotResult:
    cfg = config or PolicyPilotConfig()
    torch.manual_seed(cfg.seed)
    evidence = load_learning_evidence_json(evidence_path)
    rows = tuple(dict(row) for row in evidence["rows"])  # type: ignore[index]
    actor, _warm_start = run_supervised_actor_training(
        rows,
        config=SupervisedActorTrainingConfig(seed=cfg.seed, epochs=40, tiny_batch_epochs=120),
        model_factory=LocalGNNEdgeScorer,
    )
    fixture = get_scenario_fixture("demo_stage2")
    stack = build_fixture_stack(fixture)
    observations = tuple(
        ActorPolicyInput.from_actor_safe_row(observation.to_payload())
        for observation in _reset_observations(stack)
    )
    constraints = _constraints_from_observations(observations)
    before = _evaluate_sampled_policy(actor, observations, constraints, stack, cfg, sample=True)
    old_log_prob = torch.tensor(float(before["proposal_log_prob"]), dtype=torch.float32)
    signal_value = float(before["training_signal_value"])
    update_rule = torch.optim.AdamW(actor.parameters(), lr=cfg.learning_rate)
    last_loss = torch.tensor(0.0)
    last_ratio = torch.tensor(1.0)
    for _ in range(cfg.update_epochs):
        update_rule.zero_grad()
        proposal = _proposal_log_prob_for_actions(actor, observations, before["sampled_actions"])
        ratio = torch.exp(proposal - old_log_prob)
        advantage = torch.tensor(signal_value, dtype=torch.float32)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantage
        last_loss = -torch.minimum(unclipped, clipped)
        last_loss.backward()
        update_rule.step()
        last_ratio = ratio.detach()
    after = _evaluate_sampled_policy(actor, observations, constraints, stack, cfg, sample=True)
    violation_rate_worsened = (
        float(after["violation_rate"]) > float(before["violation_rate"]) + 0.05
    )
    actor_collapsed = bool(after["edge_count"] in {0, int(after["candidate_edge_count"])})
    stop_reason = None
    completed = True
    if violation_rate_worsened:
        stop_reason = "violation_rate_worsened_against_supervised_baseline"
        completed = False
    elif actor_collapsed:
        stop_reason = "actor_collapsed_to_empty_or_full_graph"
        completed = False
    elif float(after["projection_rejection_rate"]) > cfg.max_projection_rejection_rate:
        stop_reason = "assembler_rejected_most_actions"
        completed = False
    return PolicyPilotResult(
        stage_id=STAGE15_POLICY_PILOT_STAGE_ID,
        completed=completed,
        stop_reason=stop_reason,
        before=before,
        after=after,
        policy_loss=float(last_loss.detach().item()),
        value_loss=float(signal_value**2),
        kl=float((old_log_prob - torch.tensor(float(after["proposal_log_prob"]))).abs().item()),
        entropy=float(after["proposal_entropy"]),
        clip_fraction=float((torch.abs(last_ratio - 1.0) > cfg.clip_epsilon).float().item()),
        projection_rejection_rate=float(after["projection_rejection_rate"]),
        actor_score_distribution=dict(after["actor_score_distribution"]),
        violation_rate_worsened=violation_rate_worsened,
        actor_collapsed=actor_collapsed,
    )


def _reset_observations(stack):
    from marl_topology.env import MinimalDecPOMDPEnv

    env = MinimalDecPOMDPEnv(
        scene=stack.scene,
        graph=stack.graph,
        evaluator=stack.evaluator,
        horizon=1,
    )
    return env.reset().observations


def _evaluate_sampled_policy(
    actor,
    observations,
    constraints,
    stack,
    cfg: PolicyPilotConfig,
    *,
    sample: bool,
) -> dict[str, object]:
    from marl_topology.models import tensorize_actor_policy_inputs

    batch = tensorize_actor_policy_inputs(observations)
    logits = actor.score_tensor_batch(batch)
    probabilities = torch.sigmoid(logits)
    distribution = torch.distributions.Bernoulli(probs=probabilities)
    actions = distribution.sample() if sample else (probabilities >= 0.5).float()
    active_indices = [index for index, action in enumerate(actions.tolist()) if action >= 0.5]
    score_batch = batch.to_edge_score_batch(
        logits,
        batch_id="stage15_policy_scores",
        source="stage15_policy_pilot",
    )
    active_scores = tuple(score_batch.edge_scores[index] for index in active_indices)
    assembled = ConflictAwareGreedyAssembler().assemble(active_scores, constraints)
    selected_edges = _selected_undirected_edges(assembled.selected_directed_edges, constraints)
    evaluation = stack.evaluator.evaluate(selected_edges, topology_id="stage15_policy_eval")
    signal = evaluate_reward_surrogate(
        SurrogateSignalInput(
            consensus_success_probability=float(
                evaluation.metrics["consensus_success_probability"]
            ),
            latency=float(evaluation.metrics["latency"]),
            energy=float(evaluation.metrics["energy"]),
            topology_diagnostics=evaluation.metrics["topology_diagnostics"],
        ),
        _surrogate_config(cfg),
    )
    log_prob = distribution.log_prob(actions).sum()
    entropy = distribution.entropy().mean()
    candidate_edge_count = len(stack.graph.edge_ids)
    edge_count = len(selected_edges)
    rejection_rate = (
        len(assembled.rejected_edges) / assembled.pre_projection_edge_count
        if assembled.pre_projection_edge_count
        else 0.0
    )
    return {
        "consensus_success_probability": float(
            evaluation.metrics["consensus_success_probability"]
        ),
        "violation_rate": 0.0
        if float(evaluation.metrics["consensus_success_probability"]) >= cfg.tau_requirement_min
        else 1.0,
        "latency": float(evaluation.metrics["latency"]),
        "energy": float(evaluation.metrics["energy"]),
        "topology_diagnostics": dict(evaluation.metrics["topology_diagnostics"]),
        "edge_count": edge_count,
        "candidate_edge_count": candidate_edge_count,
        "sampled_actions": tuple(float(item) for item in actions.detach().tolist()),
        "proposal_log_prob": float(log_prob.detach().item()),
        "proposal_entropy": float(entropy.detach().item()),
        "projection_rejection_rate": rejection_rate,
        "training_signal_value": float(signal.training_signal_value),
        "actor_score_distribution": {
            "mean": float(logits.detach().mean().item()),
            "std": float(logits.detach().std(unbiased=False).item()),
            "min": float(logits.detach().min().item()),
            "max": float(logits.detach().max().item()),
        },
        "reward_surrogate_config_id": cfg.surrogate_config_id,
    }


def _proposal_log_prob_for_actions(actor, observations, sampled_actions) -> torch.Tensor:
    from marl_topology.models import tensorize_actor_policy_inputs

    batch = tensorize_actor_policy_inputs(observations)
    logits = actor.score_tensor_batch(batch)
    actions = torch.tensor(sampled_actions, dtype=torch.float32)
    return torch.distributions.Bernoulli(logits=logits).log_prob(actions).sum()


def _constraints_from_observations(observations) -> tuple[CandidateEdgeConstraint, ...]:
    constraints = []
    for observation in observations:
        for neighbor in observation.local_neighbor_observations:
            constraints.append(
                CandidateEdgeConstraint(
                    edge_id=str(_neighbor_value(neighbor, "edge_id")),
                    tx_id=observation.agent_id,
                    rx_id=str(_neighbor_value(neighbor, "neighbor_id")),
                    edge_type="actor_local_candidate",
                    role_allowed=True,
                    channel_slot=None,
                    conflict_group=None,
                )
            )
    return tuple(constraints)


def _selected_undirected_edges(
    selected_directed_edges,
    constraints: tuple[CandidateEdgeConstraint, ...],
) -> set[str]:
    by_directed = {constraint.directed_edge_id: constraint.edge_id for constraint in constraints}
    return {by_directed[edge_id] for edge_id in selected_directed_edges if edge_id in by_directed}


def _surrogate_config(cfg: PolicyPilotConfig) -> SurrogateSignalConfig:
    return SurrogateSignalConfig(
        tau=cfg.tau_requirement_min,
        reliability_weight=1.0,
        latency_weight=1.0,
        energy_weight=1.0,
        latency_reference_s=0.01,
        energy_reference_j=1.0,
        config_id=cfg.surrogate_config_id,
    )


def _neighbor_value(neighbor: object, name: str) -> object:
    if hasattr(neighbor, name):
        return getattr(neighbor, name)
    return neighbor[name]  # type: ignore[index]
