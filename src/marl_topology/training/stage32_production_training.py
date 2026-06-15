"""Stage 32 archived training loop: GNN actor + repaired graph critic baseline.

Stage 33 retired this custom loop from the active production path. The code
remains for historical reproducibility of Stage 32/32a reports only; production
training must route through :mod:`marl_topology.training.production_mappo_adapter`
and the official Stage24/25/28 mappo infrastructure.

Originally implemented the owner-approved Stage 32 design contract
(`docs/STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md`): the message-passing GNN
edge scorer replaces the Stage 31 MLP, the Stage 27/28 repaired centralized graph
value critic provides the policy-gradient baseline, and the deployment policy uses
a variable proposal size (a learned per-edge inclusion threshold) instead of a
fixed N-1. The feasibility-first surrogate, budget-aware sampler, kind-aware
budgets, and tau=0.9 are inherited unchanged from Stage 31.

Reuses the Stage 31 readiness building blocks for scenario tensors, the surrogate,
and evaluation; swaps the actor/critic/selection per the contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Sequence

import torch
from torch import nn

from marl_topology.models.centralized_message_passing_graph_critic import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    CentralizedMessagePassingGraphCritic,
    CentralizedMessagePassingGraphCriticConfig,
    GraphCriticBatch,
)
from marl_topology.models.local_gnn_edge_scorer import (
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalGNNEdgeScorer,
    LocalGNNEdgeScorerConfig,
)
from marl_topology.data.stage31_production_dataset import Stage31ProductionDataset
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioSpec,
    enumerate_candidate_topologies,
)
from marl_topology.data.stage31_surrogate_recalibration import (
    build_objective_records_from_specs,
    build_stage31_surrogate_config,
    recalibrate_references,
)
from marl_topology.training.mappo.losses import (
    ClippedPolicyValueLossInputs,
    clipped_policy_value_loss,
)
from marl_topology.training.policy_gradient.samplers import (
    BudgetAwareSequentialProposalSampler,
    ProposalSamplerConfig,
)
from marl_topology.training.stage31_readiness import (
    EDGE_FEATURE_FIELDS,
    ScenarioTensors,
    _surrogate_value,
    build_scenario_tensors,
)

TAU_REQUIREMENT_MIN = 0.9
NODE_FEATURE_DIM = 8
EDGE_FEATURE_DIM = 8
STAGE32_CUSTOM_LOOP_ACTIVE_PRODUCTION_PATH = False
STAGE32_CUSTOM_LOOP_STATUS = "archived_inactive_reproducibility_only_stage33"


@dataclass(frozen=True, slots=True)
class Stage32Config:
    seed: int = 3201
    warm_start_epochs: int = 30
    warm_start_lr: float = 0.01
    epochs: int = 12
    actor_lr: float = 0.01
    critic_lr: float = 0.02
    entropy_coefficient: float = 0.01
    hidden_dim: int = 32
    critic_hidden_dim: int = 64
    message_passing_layers: int = 2
    critic_pretrain_scenarios: int = 200
    critic_pretrain_epochs: int = 40
    critic_pretrain_lr: float = 0.005
    pg_minibatch: int = 256
    # Real clipped-PPO update settings (mappo/losses.clipped_policy_value_loss).
    ppo_update_epochs: int = 4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    include_threshold: float = 0.0  # logit threshold for variable-size selection
    tau_requirement_min: float = TAU_REQUIREMENT_MIN


# --------------------------------------------------------------------------- #
# GNN actor (full message passing over each scenario's candidate graph).
# --------------------------------------------------------------------------- #
def build_gnn_actor(config: Stage32Config) -> LocalGNNEdgeScorer:
    return LocalGNNEdgeScorer(
        LocalGNNEdgeScorerConfig(
            input_dim=len(EDGE_FEATURE_FIELDS),
            hidden_dim=config.hidden_dim,
            message_passing_layers=config.message_passing_layers,
        )
    )


def _build_directed(tensors: ScenarioTensors) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed ego-graph rows: two per physical edge, grouped by ego node.

    This is the designed use of the message-passing scorer (Stage 13/22): each
    group is one ego node's local star. Row 2k and 2k+1 are the two directed
    views of physical edge k (ego=u and ego=v); their scores are max-aggregated.
    """

    node_to_idx = _node_index(tensors)
    rows: list[list[float]] = []
    group_ids: list[int] = []
    feats = tensors.features
    for e_idx, edge_id in enumerate(tensors.edge_ids):
        u, v = edge_id.split("--", 1)
        f = feats[e_idx]
        d, lp, lat, en, cu, cv = (float(f[0]), float(f[1]), float(f[2]),
                                  float(f[3]), float(f[4]), float(f[5]))
        rows.append([d, lp, lat, en, cu, cv]); group_ids.append(node_to_idx[u])
        rows.append([d, lp, lat, en, cv, cu]); group_ids.append(node_to_idx[v])
    return (
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(group_ids, dtype=torch.long),
    )


def score_scenario(
    actor: LocalGNNEdgeScorer, tensors: ScenarioTensors, directed_cache: dict | None = None
) -> torch.Tensor:
    if directed_cache is None:
        df, gids = _build_directed(tensors)
    else:
        key = tensors.spec.scenario_id
        if key not in directed_cache:
            directed_cache[key] = _build_directed(tensors)
        df, gids = directed_cache[key]
    directed_logits = actor.forward_with_groups(df, gids)  # [2E]
    return directed_logits.reshape(-1, 2).max(dim=1).values  # [E] physical edges


# --------------------------------------------------------------------------- #
# Centralized graph critic batch construction (training-only baseline).
# --------------------------------------------------------------------------- #
def _node_index(tensors: ScenarioTensors) -> dict[str, int]:
    node_ids = sorted({n for edge in tensors.edge_ids for n in edge.split("--")})
    return {node_id: index for index, node_id in enumerate(node_ids)}


def build_critic_batch(
    tensors: ScenarioTensors, selected_edges: Sequence[str]
) -> GraphCriticBatch:
    """One-scenario centralized critic view of the selected topology."""

    node_to_idx = _node_index(tensors)
    budgets = tensors.budget_map()
    selected = set(selected_edges)
    # Critic features are normalized to comparable scales so the value head's
    # regression stays well-conditioned (raw distance is tens-hundreds of metres
    # while link prob/latency/energy are ~0-1 / ~1e-3).
    node_rows = []
    for node_id, _idx in sorted(node_to_idx.items(), key=lambda kv: kv[1]):
        budget = float(budgets.get(node_id, 2)) / 4.0
        is_rsu = 1.0 if node_id.startswith("rsu") else 0.0
        sel_degree = float(sum(1 for e in selected if node_id in e.split("--"))) / 4.0
        node_rows.append((budget, is_rsu, 1.0 - is_rsu, sel_degree, 0.0, 0.0, 0.0, 0.0))
    # Feed only the SELECTED edges so the critic sees the actual topology (not a
    # near-identical candidate set for every topology of a scenario).
    edge_rows = []
    edge_index_rows = []
    edge_valid = []
    for row_index, edge_id in enumerate(tensors.edge_ids):
        if edge_id not in selected:
            continue
        u, v = edge_id.split("--", 1)
        feats = tensors.features[row_index]
        edge_rows.append(
            (
                float(feats[0]) / 100.0,   # distance (normalized)
                float(feats[1]),           # link_success_probability (0-1)
                float(feats[2]) * 100.0,   # est link latency (scaled)
                float(feats[3]) * 100.0,   # est link energy (scaled)
                float(feats[4]),           # contention u (0-1)
                float(feats[5]),           # contention v (0-1)
                1.0,                       # selected
                0.0,
            )
        )
        edge_index_rows.append((node_to_idx[u], node_to_idx[v]))
        edge_valid.append(True)
    if not edge_rows:
        # Empty topology: one masked-out placeholder edge so shapes are valid.
        edge_rows.append((0.0,) * EDGE_FEATURE_DIM)
        edge_index_rows.append((0, 0))
        edge_valid.append(False)
    node_features = torch.tensor([node_rows], dtype=torch.float32)
    edge_features = torch.tensor([edge_rows], dtype=torch.float32)
    edge_index = torch.tensor([edge_index_rows], dtype=torch.long)
    node_mask = torch.ones((1, len(node_rows)), dtype=torch.bool)
    edge_mask = torch.tensor([edge_valid], dtype=torch.bool)
    return GraphCriticBatch(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )


def build_critic(config: Stage32Config) -> CentralizedMessagePassingGraphCritic:
    return CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(
            node_feature_dim=NODE_FEATURE_DIM,
            edge_feature_dim=EDGE_FEATURE_DIM,
            hidden_dim=config.critic_hidden_dim,
            message_layers=config.message_passing_layers,
        )
    )


def _build_critic_examples(
    specs, tensor_cache, surrogate_config, *, sample: int
) -> list[tuple[str, tuple[str, ...], float]]:
    """Diverse (scenario, topology, signal) examples across candidate variants.

    Mirrors the Stage 27 critic dataset: a spread of feasible and infeasible
    topologies so the value head has real structure to learn.
    """

    examples: list[tuple[str, tuple[str, ...], float]] = []
    for spec in list(specs)[:sample]:
        tensors = tensor_cache[spec.scenario_id]
        if not tensors.edge_ids:
            continue
        graph = tensors.evaluator.graph
        variants = enumerate_candidate_topologies(
            graph, tensors.evaluator.link_records, spec.quorum_size
        )
        for _name, edges in variants.items():
            evaluation = tensors.evaluator.evaluate(set(edges))
            signal = _surrogate_value(evaluation, surrogate_config)
            examples.append((spec.scenario_id, tuple(edges), signal))
    return examples


def _pretrain_critic(
    critic, train_examples, eval_examples, tensor_cache, mean, std,
    *, epochs, lr, minibatch, generator
) -> float:
    """Pretrain the critic on diverse topologies; keep-best on held-out EV.

    Returns the best held-out explained variance (an honest repaired-critic metric).
    """

    update_rule = torch.optim.Adam(critic.parameters(), lr=lr)
    count = len(train_examples)
    if count < 2 or len(eval_examples) < 2:
        return 0.0
    best_ev = -1e9
    best_state = copy.deepcopy(critic.state_dict())
    minibatch = max(8, min(minibatch, count))
    for _epoch in range(epochs):
        critic.train()
        order = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count, minibatch):
            chunk = order[start : start + minibatch]
            values, targets = [], []
            for idx in chunk:
                scenario_id, edges, signal = train_examples[idx]
                tensors = tensor_cache[scenario_id]
                value = critic(build_critic_batch(tensors, edges)).normalized_value.reshape(())
                values.append(value)
                targets.append((signal - mean) / std)
            loss = nn.functional.mse_loss(
                torch.stack(values), torch.tensor(targets, dtype=torch.float32)
            )
            update_rule.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            update_rule.step()
        ev = _critic_ev_on_examples(critic, eval_examples, tensor_cache, mean, std)
        if ev > best_ev:
            best_ev = ev
            best_state = copy.deepcopy(critic.state_dict())
    critic.load_state_dict(best_state)
    return best_ev


def _critic_ev_on_examples(critic, examples, tensor_cache, mean, std) -> float:
    critic.eval()
    preds, targets = [], []
    with torch.no_grad():
        for scenario_id, edges, signal in examples:
            tensors = tensor_cache[scenario_id]
            value = critic(build_critic_batch(tensors, edges)).normalized_value.reshape(())
            preds.append(float(value.item()))
            targets.append((signal - mean) / std)
    return _explained_variance(preds, targets)


# --------------------------------------------------------------------------- #
# Variable-size, budget-feasible deployment selection (score-only, no leakage).
# --------------------------------------------------------------------------- #
def threshold_budget_feasible(
    logits: torch.Tensor,
    tensors: ScenarioTensors,
    threshold: float,
) -> tuple[str, ...]:
    budgets = tensors.budget_map()
    order = torch.argsort(logits, descending=True).tolist()
    usage: dict[str, int] = {}
    chosen: list[str] = []
    quorum_floor = tensors.proposal_size()  # ensure at least a spanning attempt
    for rank, index in enumerate(order):
        edge_id = tensors.edge_ids[index]
        u, v = edge_id.split("--", 1)
        above = float(logits[index].item()) > threshold or len(chosen) < quorum_floor
        if above and usage.get(u, 0) < budgets.get(u, 2) and usage.get(v, 0) < budgets.get(v, 2):
            chosen.append(edge_id)
            usage[u] = usage.get(u, 0) + 1
            usage[v] = usage.get(v, 0) + 1
    return tuple(sorted(chosen))


@dataclass
class SeedResult:
    seed: int
    before_eval: dict[str, float]
    warm_start_eval: dict[str, float]
    after_eval: dict[str, float]
    test_eval: dict[str, float]
    critic_explained_variance: float
    actor_state: dict


def _evaluate(
    actor: LocalGNNEdgeScorer,
    specs: Sequence[ProductionScenarioSpec],
    tensor_cache: dict,
    config: Stage32Config,
    surrogate_config,
    directed_cache: dict,
) -> dict[str, float]:
    tau = config.tau_requirement_min
    feasible = violations = 0
    signal_sum = latency_sum = energy_sum = 0.0
    feasible_cost = proposed = 0
    actor.eval()
    with torch.no_grad():
        for spec in specs:
            tensors = tensor_cache[spec.scenario_id]
            if not tensors.edge_ids:
                continue
            logits = score_scenario(actor, tensors, directed_cache)
            selected = threshold_budget_feasible(logits, tensors, config.include_threshold)
            proposed += len(selected)
            evaluation = tensors.evaluator.evaluate(set(selected))
            psucc = float(evaluation.metrics["consensus_success_probability"])
            signal_sum += _surrogate_value(evaluation, surrogate_config)
            if psucc >= tau:
                feasible += 1
                latency_sum += float(evaluation.metrics["latency"])
                energy_sum += float(evaluation.metrics["energy"])
                feasible_cost += 1
            else:
                violations += 1
    count = max(1, len(specs))
    return {
        "scenario_count": len(specs),
        "tau_feasible_rate": feasible / count,
        "violation_rate": violations / count,
        "mean_surrogate_signal": signal_sum / count,
        "mean_feasible_latency": latency_sum / max(1, feasible_cost),
        "mean_feasible_energy": energy_sum / max(1, feasible_cost),
        "projection_rejection_rate": 0.0,
        "mean_selected_edges": proposed / count,
    }


def _warm_start(actor, train_specs, teacher_labels, tensor_cache, config, directed_cache) -> None:
    update_rule = torch.optim.Adam(actor.parameters(), lr=config.warm_start_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    targets = {}
    for spec in train_specs:
        label = teacher_labels[spec.scenario_id]
        if not label["feasible_exists"]:
            continue
        tensors = tensor_cache[spec.scenario_id]
        selected = set(label["selected_physical_edges"])
        targets[spec.scenario_id] = torch.tensor(
            [1.0 if e in selected else 0.0 for e in tensors.edge_ids], dtype=torch.float32
        )
    actor.train()
    for _epoch in range(config.warm_start_epochs):
        for spec_id, target in targets.items():
            tensors = tensor_cache[spec_id]
            logits = score_scenario(actor, tensors, directed_cache)
            loss = loss_fn(logits, target)
            update_rule.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            update_rule.step()


def train_one_seed(
    dataset: Stage31ProductionDataset,
    config: Stage32Config,
    tensor_cache: dict | None = None,
    surrogate_config=None,
) -> SeedResult:
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_specs = dataset.specs_for_split("train")
    eval_specs = dataset.specs_for_split("eval")
    test_specs = dataset.specs_for_split("test")

    if surrogate_config is None:
        references = recalibrate_references(build_objective_records_from_specs(train_specs))
        surrogate_config = build_stage31_surrogate_config(references)

    if tensor_cache is None:
        tensor_cache = {s.scenario_id: build_scenario_tensors(s) for s in dataset.specs}
    directed_cache: dict = {}

    actor = build_gnn_actor(config)
    critic = build_critic(config)
    actor_update = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_update = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    sampler = BudgetAwareSequentialProposalSampler()

    before_eval = _evaluate(actor, eval_specs, tensor_cache, config, surrogate_config, directed_cache)
    _warm_start(actor, train_specs, dataset.teacher_labels, tensor_cache, config, directed_cache)
    warm_start_eval = _evaluate(actor, eval_specs, tensor_cache, config, surrogate_config, directed_cache)

    # Pretrain the repaired graph critic on a diverse spread of topologies
    # (Stage 27 style), standardizing the bimodal surrogate signal so the value
    # head stays well-conditioned.
    critic_examples = _build_critic_examples(
        train_specs, tensor_cache, surrogate_config, sample=config.critic_pretrain_scenarios
    )
    example_signals = torch.tensor([s for _id, _e, s in critic_examples])
    signal_mean = float(example_signals.mean().item()) if critic_examples else 0.0
    signal_std = (
        float(example_signals.std(unbiased=False).clamp_min(1e-3).item())
        if critic_examples
        else 1.0
    )
    # Held-out split of critic examples for an honest generalization EV.
    split_at = int(0.8 * len(critic_examples))
    critic_train_ex = critic_examples[:split_at]
    critic_eval_ex = critic_examples[split_at:]
    critic_ev = _pretrain_critic(
        critic, critic_train_ex, critic_eval_ex, tensor_cache, signal_mean, signal_std,
        epochs=config.critic_pretrain_epochs, lr=config.critic_pretrain_lr,
        minibatch=config.pg_minibatch, generator=generator,
    )

    best_state = copy.deepcopy(actor.state_dict())
    best_signal = warm_start_eval["mean_surrogate_signal"]

    for epoch in range(config.epochs):
        # --- Rollout phase: sample budget-aware actions, record old logprobs,
        # standardized returns (feasibility-first surrogate), and proposals. ---
        order = torch.randperm(len(train_specs), generator=generator).tolist()
        batch = order[: config.pg_minibatch]
        rollout: list[dict] = []
        actor.eval()
        with torch.no_grad():
            for index in batch:
                tensors = tensor_cache[train_specs[index].scenario_id]
                if not tensors.edge_ids:
                    continue
                logits = score_scenario(actor, tensors, directed_cache)
                sampler_config = ProposalSamplerConfig(
                    physical_edge_ids=tensors.edge_ids,
                    top_k=min(len(tensors.edge_ids), 2 * tensors.proposal_size()),
                    endpoint_budget=1,
                    endpoint_budgets=tensors.node_budgets,
                )
                mask = torch.ones(len(tensors.edge_ids), dtype=torch.bool)
                sample = sampler.sample(logits, mask, sampler_config, generator)
                evaluation = tensors.evaluator.evaluate(set(sample.proposed_physical_edges))
                std_return = (
                    _surrogate_value(evaluation, surrogate_config) - signal_mean
                ) / signal_std
                rollout.append(
                    {
                        "tensors": tensors,
                        "mask": mask,
                        "sampler_config": sampler_config,
                        "raw_sample_data": sample.raw_sample_data,
                        "old_logprob": float(sample.logprob.item()),
                        "std_return": std_return,
                        "proposed": sample.proposed_physical_edges,
                    }
                )
        if not rollout:
            continue
        returns = torch.tensor([r["std_return"] for r in rollout], dtype=torch.float32)
        old_logprobs = torch.tensor([r["old_logprob"] for r in rollout], dtype=torch.float32)

        # --- Clipped-PPO update epochs over the rollout (mappo/losses). ---
        actor.train()
        critic.train()
        for _update in range(config.ppo_update_epochs):
            new_logprobs, new_entropies, new_values = [], [], []
            for item in rollout:
                logits = score_scenario(actor, item["tensors"], directed_cache)
                new_logprobs.append(
                    sampler.logprob_of(logits, item["mask"], item["sampler_config"], item["raw_sample_data"])
                )
                new_entropies.append(
                    sampler.entropy_of(logits, item["mask"], item["sampler_config"], item["raw_sample_data"])
                )
                critic_batch = build_critic_batch(item["tensors"], item["proposed"])
                new_values.append(critic(critic_batch).normalized_value.reshape(()))
            value_predictions = torch.stack(new_values)
            advantages = returns - value_predictions.detach()
            std = float(advantages.std().item())
            if std > 1e-6:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)
            loss_result = clipped_policy_value_loss(
                ClippedPolicyValueLossInputs(
                    new_logprobs=torch.stack(new_logprobs),
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    value_predictions=value_predictions,
                    returns=returns,
                    entropies=torch.stack(new_entropies),
                    clip_eps=config.clip_eps,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coefficient,
                )
            )
            actor_update.zero_grad()
            critic_update.zero_grad()
            loss_result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            actor_update.step()
            critic_update.step()
            if float(loss_result.approx_kl.item()) > 0.03:
                break  # early-stop the update like the Stage 25 protocol

        current = _evaluate(actor, eval_specs, tensor_cache, config, surrogate_config, directed_cache)
        if current["mean_surrogate_signal"] > best_signal:
            best_signal = current["mean_surrogate_signal"]
            best_state = copy.deepcopy(actor.state_dict())

    actor.load_state_dict(best_state)
    after_eval = _evaluate(actor, eval_specs, tensor_cache, config, surrogate_config, directed_cache)
    test_eval = _evaluate(actor, test_specs, tensor_cache, config, surrogate_config, directed_cache)
    return SeedResult(
        seed=config.seed,
        before_eval=before_eval,
        warm_start_eval=warm_start_eval,
        after_eval=after_eval,
        test_eval=test_eval,
        critic_explained_variance=critic_ev,
        actor_state=best_state,
    )


def _explained_variance(preds: list[float], targets: list[float]) -> float:
    if len(targets) < 2:
        return 0.0
    t = torch.tensor(targets)
    p = torch.tensor(preds)
    var = float(t.var(unbiased=False).item())
    if var < 1e-12:
        return 0.0
    return float(1.0 - ((t - p).var(unbiased=False) / var).item())


@dataclass
class Stage32Result:
    actor_model_id: str
    critic_model_id: str
    seeds: list[SeedResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, object]:
        def mean(key_path):
            vals = [getattr(s, key_path[0])[key_path[1]] for s in self.seeds]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "seed_count": len(self.seeds),
            "actor_model_id": self.actor_model_id,
            "critic_model_id": self.critic_model_id,
            "before_tau_feasible_mean": mean(("before_eval", "tau_feasible_rate")),
            "after_tau_feasible_mean": mean(("after_eval", "tau_feasible_rate")),
            "test_tau_feasible_mean": mean(("test_eval", "tau_feasible_rate")),
            "after_violation_mean": mean(("after_eval", "violation_rate")),
            "after_surrogate_signal_mean": mean(("after_eval", "mean_surrogate_signal")),
            "projection_rejection_mean": mean(("after_eval", "projection_rejection_rate")),
            "critic_explained_variance_mean": (
                sum(s.critic_explained_variance for s in self.seeds) / len(self.seeds)
                if self.seeds
                else 0.0
            ),
            "mean_selected_edges": mean(("after_eval", "mean_selected_edges")),
        }


def build_stage32_manifest(scenario_count: int, seeds: Sequence[int]) -> dict[str, object]:
    """Stage 5.9-compliant run manifest for the Stage 32 production training run."""

    seeds = tuple(seeds)
    return {
        "run_id": "stage32_gnn_critic_production_run_v1",
        "created_at_utc": "2026-06-02T00:00:00Z",
        "stage_id": "stage32_production_training_execution_with_gnn_actor_repaired_critic_and_scaled_data",
        "owner_approval_id": "owner_approved_stage32_production_training_execution",
        "config_id": "stage32_gnn_critic_scaled_config_v1",
        "scenario_set_id": f"stage32_procedural_scenarios_{scenario_count}",
        "split_id": "stage32_context_keyed_leakage_checked_split",
        "seed": seeds[0],
        "seed_group_id": "stage32_seed_group_" + "_".join(str(s) for s in seeds),
        "code_version_marker": "stage32_production_training_workspace_marker",
        "contract_ids": [
            "stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1",
            "feasibility_first_barrier_v2",
            "policy_architecture_contract_stage5_7",
            "run_manifest_artifact_contract_stage5_9",
            "stage32_production_training_design_contract",
        ],
        "metric_registry_version": "metric_governance_stage5",
        "physics_regime_id": "urlcc_finite_blocklength_v1",
        "protocol_model_id": "stage4_expected_initiator_pbft_over_stage3_network_v1",
        "objective_contract_id": "stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1",
        "surrogate_config_id": "stage31_feasibility_first_recalibrated_v2",
        "normalization_reference_id": "stage31_feasible_positive_max_recalibrated",
        "architecture_contract_id": "stage32_gnn_actor_graph_critic_v1",
        "replay_schema_version": "learning_target_replay_contract_stage5_8",
        "artifact_policy_id": "run_manifest_artifact_contract_stage5_9",
        "artifact_root": "result_save",
    }


def train_production(
    dataset: Stage31ProductionDataset,
    *,
    seeds: Sequence[int] = (3201, 3202, 3203, 3204, 3205),
    base_config: Stage32Config | None = None,
) -> Stage32Result:
    base = base_config or Stage32Config()
    result = Stage32Result(
        actor_model_id=LOCAL_GNN_EDGE_SCORER_MODEL_ID,
        critic_model_id=CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    )
    # Build the scenario-tensor cache and the recalibrated surrogate once and
    # reuse across seeds (deterministic per scenario).
    tensor_cache = {s.scenario_id: build_scenario_tensors(s) for s in dataset.specs}
    references = recalibrate_references(
        build_objective_records_from_specs(dataset.specs_for_split("train"))
    )
    surrogate_config = build_stage31_surrogate_config(references)
    for seed in seeds:
        cfg = replace(base, seed=seed)
        result.seeds.append(
            train_one_seed(dataset, cfg, tensor_cache=tensor_cache, surrogate_config=surrogate_config)
        )
    return result
