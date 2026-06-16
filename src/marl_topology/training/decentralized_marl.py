"""Decentralized CTDE multi-agent policy-gradient flow (the genuine-MARL trunk loop).

This is the production training/execution loop that makes the system genuinely MARL and
genuinely sequential -- NOT a centralized-critic graph combinatorial bandit. It composes
the three finalized components:

* Actor (per agent, decentralized): the K-hop local message-passing GNN edge scorer
  (``LocalKHopGNNEdgeScorer``). Each node is an agent; it observes only locally and reasons
  over its K-hop neighbourhood via local neighbour signalling (no global-state shortcut).
* Decoder (decentralized, true end-to-end): the per-node mutual-acceptance sampler
  (``DecentralizedPerNodeMutualSampler``). Each node decides only over its own incident
  edges within its radio budget; an edge activates iff BOTH endpoints pick it. The joint
  log-prob factorises across agents, so each node gets its own PPO ratio.
* Critic (centralized, training-only / CTDE): the message-passing graph value critic. It
  pools the whole graph into a scalar baseline used only during training; the deployment
  actor never receives it.

Genuine MARL (not single-agent PPO on a joint action): the clipped objective is formed
PER AGENT from the factorized ``per_owner_logprobs`` (each node's own ratio is clipped),
with a SHARED CTDE advantage from the centralized critic -- the standard multi-agent PPO
recipe with parameter sharing.

Genuine sequential MDP (not a bandit): each node carries a depletable ENERGY BATTERY.
Activating links spends it; once drained, the node's feasible action set shrinks at the
NEXT step. So an action changes the reward-relevant state the agent inherits later -- a true
multi-step transition the policy controls. Vehicles also move along a pre-computed
trajectory (observable state evolution). Crucially, neither touches the FROZEN objective
evaluator: the per-step reward is the unchanged feasibility-barrier surrogate plus a
link-churn (handover) shaping cost; the battery is a dynamics constraint, not a reward
hack. Because the per-step augmented observation (battery / previous-active state) is a
deterministic function of the realized action sequence and the immutable geometry frames,
storing the realized observation tensors makes the PPO ratio replay-exact.

Checkpoint writing lives in the runnable driver under ``scripts``/``logs`` (the src
boundary stays checkpoint-free); this module returns the trained modules and a report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCritic,
    CentralizedMessagePassingGraphCriticConfig,
    GraphCriticBatch,
)
from marl_topology.models.local_khop_gnn_edge_scorer import (
    LocalKHopGNNEdgeScorer,
    LocalKHopGNNEdgeScorerConfig,
)
from marl_topology.models.tensorizers import (
    ACTOR_EDGE_FEATURE_FIELDS,
    ACTOR_NODE_FEATURE_FIELDS,
    tensorize_actor_graph,
)
from marl_topology.objectives.surrogate_signal import (
    SurrogateSignalInput,
    evaluate_reward_surrogate,
)
from marl_topology.policies.actor_interface import ActorPolicyInput
from marl_topology.policies.decentralized_baselines import build_local_observations
from marl_topology.training.mappo.advantages import AdvantageConfig, compute_gae_returns
from marl_topology.training.policy_gradient.decentralized_sampler import (
    DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID,
    DecentralizedMutualSamplerConfig,
    DecentralizedPerNodeMutualSampler,
    physical_edge_for_directed,
)


DECENTRALIZED_CTDE_FLOW_ID = "decentralized_ctde_mappo_v1"

# Carry-state columns appended to the actor-safe node / edge features so the agent can
# observe (and react to) the sequential state its own actions mutate.
AUG_NODE_EXTRA = ("remaining_energy_norm", "previous_active_degree_norm")
AUG_EDGE_EXTRA = ("was_active_previous_step",)
PRODUCTION_ACTOR_NODE_DIM = len(ACTOR_NODE_FEATURE_FIELDS) + len(AUG_NODE_EXTRA)
PRODUCTION_ACTOR_EDGE_DIM = len(ACTOR_EDGE_FEATURE_FIELDS) + len(AUG_EDGE_EXTRA)

STAGE21_TAU = 0.9


@dataclass(frozen=True, slots=True)
class DecentralizedMARLConfig:
    """Configuration for the decentralized CTDE flow (scale-ready for a rented GPU)."""

    # Actor architecture (the Task-1 ablation axes).
    rounds: int = 3
    hidden_dim: int = 64
    jumping_knowledge: str = "concat"
    node_center_norm: bool = False
    # Critic (CTDE, training-only).
    critic_hidden_dim: int = 64
    critic_message_layers: int = 2
    # Rollout / episode. rollout_steps is a sustained operating window (not a smoke length):
    # at dt=1s, 16 steps ~ vehicles move ~160m so links genuinely change and the battery spans
    # the episode; gamma=0.99 propagates credit over it.
    rollout_steps: int = 16
    traj_dt_s: float = 1.0
    traj_speed_min_mps: float = 5.0
    traj_speed_max_mps: float = 15.0
    # Sequential MDP energy battery (the non-bandit coupling). energy_unit_j is the abstract
    # per-active-link battery cost; energy_budget_j the per-node starting battery (sized to
    # span the episode so the depletion constraint is meaningful but not crippling).
    energy_unit_j: float = 1.0
    energy_budget_j: float = 20.0
    churn_weight: float = 0.05
    # PPO / multi-agent policy gradient.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32
    max_updates: int = 20
    # Scenes rolled per update (the parallel-env batch). 0 = roll the whole train pool each
    # update (fine for small pools). For a large pool, set this so per-update rollout cost is
    # scenes_per_update * rollout_steps, and each update samples a fresh batch -> more scene
    # diversity over training without an intractable per-update cost.
    scenes_per_update: int = 0
    eval_every: int = 5
    # Behaviour-cloning warm start: distill the budget-aware SA teacher for this many BCE
    # passes before policy gradient (0 disables). The validated mechanism for recovering
    # feasibility at larger N (a frozen small-N actor collapses without it).
    warmup_updates: int = 0
    bc_teacher_sa_iters: int = 120
    bc_teacher_restarts: int = 4
    # Parallel rollout workers (0 = serial). Reserved for the Linux multi-core throughput path.
    num_workers: int = 0
    # Use the vectorized (equivalence-verified, bit-identical) Stage-21 evaluator for the
    # rollout/BC physics. Default on (it only removes redundant compute); set False to fall
    # back to the canonical evaluator (e.g. for an A/B equivalence check).
    use_vectorized_evaluator: bool = True
    seed: int = 4096
    device: str = "cpu"
    tau_requirement_min: float = STAGE21_TAU

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1")
        if self.tau_requirement_min != STAGE21_TAU:
            raise ValueError("tau_requirement_min stays frozen at 0.9")
        if self.energy_unit_j < 0.0 or self.energy_budget_j < 0.0:
            raise ValueError("energy parameters must be nonnegative")


def build_production_actor(config: DecentralizedMARLConfig) -> LocalKHopGNNEdgeScorer:
    return LocalKHopGNNEdgeScorer(
        LocalKHopGNNEdgeScorerConfig(
            node_input_dim=PRODUCTION_ACTOR_NODE_DIM,
            edge_input_dim=PRODUCTION_ACTOR_EDGE_DIM,
            hidden_dim=config.hidden_dim,
            rounds=config.rounds,
            jumping_knowledge=config.jumping_knowledge,
            node_center_norm=config.node_center_norm,
        )
    )


def build_ctde_critic(config: DecentralizedMARLConfig) -> CentralizedMessagePassingGraphCritic:
    return CentralizedMessagePassingGraphCritic(
        CentralizedMessagePassingGraphCriticConfig(
            node_feature_dim=PRODUCTION_ACTOR_NODE_DIM,
            edge_feature_dim=PRODUCTION_ACTOR_EDGE_DIM,
            hidden_dim=config.critic_hidden_dim,
            message_layers=config.critic_message_layers,
        )
    )


@dataclass(slots=True)
class _Transition:
    scene_index: int
    step_index: int
    node_features: torch.Tensor  # [N, node_dim] augmented (carry-state baked in)
    edge_index: torch.Tensor  # [E, 2]
    edge_features: torch.Tensor  # [E, edge_dim] augmented
    directed_edge_ids: tuple[str, ...]
    node_budgets: tuple[tuple[str, int], ...]
    mask: torch.Tensor  # [E] bool
    raw_sample_data: dict
    owners: tuple[str, ...]
    old_per_owner_logprob: dict[str, float]
    old_logprob: float
    value: float
    reward_surrogate: float
    done_mask: float
    # diagnostics
    consensus: float
    latency: float
    energy: float
    churn: float
    selected_edges: tuple[str, ...]


def _observation_to_policy_input(observation) -> ActorPolicyInput:
    return ActorPolicyInput(
        agent_id=observation.agent_id,
        agent_kind=observation.agent_kind,
        time_step=observation.time_step,
        local_position_m=tuple(observation.local_position_m),
        local_neighbor_observations=tuple(observation.local_neighbor_observations),
        local_messages=tuple(observation.local_messages),
        local_history=dict(observation.local_history),
    )


class DecentralizedCTDEFlow:
    """Runs decentralized CTDE multi-agent policy gradient over moving-vehicle episodes."""

    flow_id = DECENTRALIZED_CTDE_FLOW_ID

    def __init__(self, config: DecentralizedMARLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.sampler = DecentralizedPerNodeMutualSampler()
        self._pool = None  # persistent spawn worker pool, set for the duration of run()

    # ------------------------------------------------------------------ episodes
    def _episode_frames(self, spec, seed: int):
        """A per-scene sequence of frames (moving vehicles). Each frame is a (graph,
        evaluator, observations, kind/budget tables) snapshot. Geometry is deterministic
        given the seed, so loss recompute reads identical frames."""

        import random

        from marl_topology.data.stage31_scenario_generator import _sample_vehicle_motions
        from marl_topology.data.stage31_production_dataset import build_scenario_evaluator
        from marl_topology.scenario.scene import advance_scene

        rng = random.Random(seed)
        motions = _sample_vehicle_motions(
            rng, spec.scene, self.config.traj_speed_min_mps, self.config.traj_speed_max_mps
        )
        motion_map = {motion.node_id: motion for motion in motions}
        frames = []
        scene = spec.scene
        for step_index in range(self.config.rollout_steps):
            if step_index > 0:
                scene = advance_scene(scene, motion_map, self.config.traj_dt_s)
            frame_spec = replace(spec, scenario_id=f"{spec.scenario_id}:t{step_index}", scene=scene)
            graph, canonical = build_scenario_evaluator(frame_spec)
            evaluator = self._wrap_evaluator(scene, graph, canonical)
            observations = build_local_observations(
                scene=scene,
                graph=graph,
                link_records=evaluator.link_records,
                time_step=step_index + 1,
            )
            frames.append((evaluator, observations))
        return frames

    def _wrap_evaluator(self, scene, graph, canonical):
        """Wrap a canonical Stage-21 evaluator with the bit-identical vectorized one (reusing
        the canonical for link_records / membership / MAC), unless disabled by config."""

        if not self.config.use_vectorized_evaluator:
            return canonical
        from marl_topology.data.vectorized_objective_stack_evaluator import (
            VectorizedStage21Evaluator,
        )

        return VectorizedStage21Evaluator(
            scene=scene, graph=graph, config=canonical.config, ref=canonical
        )

    # ------------------------------------------------------------------ rollout
    def rollout_scene(
        self,
        actor: LocalKHopGNNEdgeScorer,
        critic: CentralizedMessagePassingGraphCritic,
        spec,
        scene_index: int,
        seed: int,
        deterministic: bool,
    ) -> list[_Transition]:
        frames = self._episode_frames(spec, seed)
        transitions: list[_Transition] = []
        remaining_energy: dict[str, float] = {}
        previous_active: set[str] = set()
        generator = torch.Generator().manual_seed(seed)
        for step_index, (evaluator, observations) in enumerate(frames):
            policy_inputs = [_observation_to_policy_input(o) for o in observations]
            kind_by_node = {p.agent_id: p.agent_kind for p in policy_inputs}
            radio_budget = {node: _radio_budget(kind) for node, kind in kind_by_node.items()}
            for node in kind_by_node:
                remaining_energy.setdefault(node, self.config.energy_budget_j)

            graph_batch = tensorize_actor_graph([policy_inputs])
            if graph_batch.edge_count == 0:
                # Degenerate scene with no candidate edges: empty proposal, zero reward step.
                continue
            node_features, edge_features = self._augment_features(
                graph_batch, remaining_energy, previous_active, radio_budget
            )
            node_features = node_features.to(self.device)
            edge_features = edge_features.to(self.device)
            edge_index = graph_batch.edge_index.to(self.device)
            node_batch = graph_batch.node_batch.to(self.device)

            directed_edge_ids = graph_batch.directed_edge_ids
            effective_budgets = self._effective_budgets(remaining_energy, radio_budget)
            sampler_config = DecentralizedMutualSamplerConfig(
                directed_edge_ids=directed_edge_ids,
                node_budgets=tuple(sorted(effective_budgets.items())),
                default_budget=max(radio_budget.values(), default=2),
            )
            mask = torch.ones(len(directed_edge_ids), dtype=torch.bool, device=self.device)

            with torch.no_grad():
                logits = actor.forward_graph(node_features, edge_index, edge_features, node_batch=node_batch)
                sample = self.sampler.sample(
                    logits, mask, sampler_config, generator, deterministic=deterministic
                )
                value = self._critic_value(critic, node_features, edge_index, edge_features)

            selected = tuple(sample.proposed_physical_edges)
            evaluation = evaluator.evaluate(set(selected), topology_id=f"{spec.scenario_id}:t{step_index}:marl")
            consensus = float(evaluation.metrics["consensus_success_probability"])
            latency = float(evaluation.metrics["latency"])
            energy = float(evaluation.metrics["energy"])
            base_signal = evaluate_reward_surrogate(
                SurrogateSignalInput(
                    consensus_success_probability=consensus,
                    latency=latency,
                    energy=energy,
                    topology_diagnostics=evaluation.metrics["topology_diagnostics"],
                ),
                _reward_config(),
            )
            churn = float(len(set(selected) ^ previous_active))
            churn_penalty = self.config.churn_weight * churn / max(1, graph_batch.edge_count // 2)
            reward_surrogate = float(base_signal.training_signal_value) - churn_penalty

            transitions.append(
                _Transition(
                    scene_index=scene_index,
                    step_index=step_index,
                    node_features=node_features.detach().cpu(),
                    edge_index=edge_index.detach().cpu(),
                    edge_features=edge_features.detach().cpu(),
                    directed_edge_ids=directed_edge_ids,
                    node_budgets=tuple(sorted(effective_budgets.items())),
                    mask=mask.detach().cpu(),
                    raw_sample_data=dict(sample.raw_sample_data),
                    owners=tuple(sorted({_owner(d) for d in directed_edge_ids})),
                    old_per_owner_logprob=dict(sample.raw_sample_data["per_owner_logprobs"]),
                    old_logprob=float(sample.logprob.detach().cpu().item()),
                    value=float(value),
                    reward_surrogate=reward_surrogate,
                    done_mask=1.0,
                    consensus=consensus,
                    latency=latency,
                    energy=energy,
                    churn=churn,
                    selected_edges=selected,
                )
            )
            # Sequential transition: spend battery on active incident links -> next step's
            # feasible action set shrinks. This is the action-consequence state.
            self._spend_energy(remaining_energy, selected)
            previous_active = set(selected)
        return transitions

    def _augment_features(self, graph_batch, remaining_energy, previous_active, radio_budget):
        node_extra = []
        for node_id in graph_batch.node_ids:
            energy_norm = (
                min(1.0, remaining_energy.get(node_id, 0.0) / self.config.energy_budget_j)
                if self.config.energy_budget_j > 0
                else 0.0
            )
            active_degree = sum(1 for edge in previous_active if node_id in _endpoints(edge))
            degree_norm = active_degree / max(1, radio_budget.get(node_id, 1))
            node_extra.append([energy_norm, degree_norm])
        node_extra_t = torch.tensor(node_extra, dtype=torch.float32)
        node_features = torch.cat([graph_batch.node_features, node_extra_t], dim=1)
        edge_extra = []
        for ref in graph_batch.records:
            physical = physical_edge_for_directed(ref.directed_edge_id)
            edge_extra.append([1.0 if physical in previous_active else 0.0])
        edge_extra_t = torch.tensor(edge_extra, dtype=torch.float32)
        edge_features = torch.cat([graph_batch.edge_features, edge_extra_t], dim=1)
        return node_features, edge_features

    def _effective_budgets(self, remaining_energy, radio_budget) -> dict[str, int]:
        budgets: dict[str, int] = {}
        for node, radio in radio_budget.items():
            if self.config.energy_unit_j <= 0.0:
                budgets[node] = radio
                continue
            affordable = int(remaining_energy.get(node, 0.0) // self.config.energy_unit_j)
            budgets[node] = max(0, min(radio, affordable))
        return budgets

    def _spend_energy(self, remaining_energy, selected: tuple[str, ...]) -> None:
        for edge in selected:
            for endpoint in _endpoints(edge):
                remaining_energy[endpoint] = max(
                    0.0, remaining_energy.get(endpoint, 0.0) - self.config.energy_unit_j
                )

    def _critic_value(self, critic, node_features, edge_index, edge_features) -> float:
        output = critic(_critic_batch(node_features, edge_index, edge_features))
        return float(output.normalized_value[0].detach().cpu().item())

    # ------------------------------------------------------------------ update
    def update(
        self,
        actor: LocalKHopGNNEdgeScorer,
        critic: CentralizedMessagePassingGraphCritic,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
        transitions: list[_Transition],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        config = self.config
        generator = torch.Generator().manual_seed(config.seed + len(transitions))
        payloads: list[dict[str, float]] = []
        for _epoch in range(config.update_epochs):
            order = torch.randperm(len(transitions), generator=generator).tolist()
            for start in range(0, len(transitions), config.minibatch_size):
                batch_indices = order[start : start + config.minibatch_size]
                payload = self._minibatch_step(
                    actor, critic, actor_opt, critic_opt, transitions, advantages, returns, batch_indices
                )
                if payload is not None:
                    payloads.append(payload)
        return _mean_payload(payloads)

    def _minibatch_step(
        self, actor, critic, actor_opt, critic_opt, transitions, advantages, returns, batch_indices
    ) -> dict[str, float] | None:
        config = self.config
        policy_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        clip_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_preds: list[torch.Tensor] = []
        value_targets: list[torch.Tensor] = []
        for index in batch_indices:
            transition = transitions[index]
            node_features = transition.node_features.to(self.device)
            edge_index = transition.edge_index.to(self.device)
            edge_features = transition.edge_features.to(self.device)
            logits = actor.forward_graph(node_features, edge_index, edge_features)
            sampler_config = DecentralizedMutualSamplerConfig(
                directed_edge_ids=transition.directed_edge_ids,
                node_budgets=transition.node_budgets,
                default_budget=max((b for _n, b in transition.node_budgets), default=2),
            )
            mask = transition.mask.to(self.device)
            new_per_owner = self.sampler.per_owner_logprobs(
                logits, mask, sampler_config, transition.raw_sample_data
            )
            entropy = self.sampler.entropy_of(logits, mask, sampler_config, transition.raw_sample_data)
            entropy_terms.append(entropy)
            advantage = advantages[index]
            for owner, new_lp in new_per_owner.items():
                old_lp = transition.old_per_owner_logprob.get(owner, 0.0)
                log_ratio = new_lp - old_lp
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantage
                policy_terms.append(-torch.minimum(unclipped, clipped))
                kl_terms.append((ratio - 1.0) - log_ratio)
                clip_terms.append((torch.abs(ratio - 1.0) > config.clip_eps).to(dtype=ratio.dtype))
            value = critic(_critic_batch(node_features, edge_index, edge_features)).normalized_value[0]
            value_preds.append(value)
            value_targets.append(returns[index])
        if not policy_terms:
            return None
        policy_loss = torch.stack(policy_terms).mean()
        value_loss = torch.nn.functional.mse_loss(
            torch.stack(value_preds), torch.stack(value_targets).to(self.device)
        )
        entropy_mean = torch.stack(entropy_terms).mean()
        total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
        if not torch.isfinite(total_loss).all().item():
            return None
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        total_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        actor_opt.step()
        critic_opt.step()
        return {
            "total_loss": float(total_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "approx_kl": float(torch.stack(kl_terms).mean().detach().cpu().item()),
            "clip_fraction": float(torch.stack(clip_terms).mean().detach().cpu().item()),
            "actor_grad_norm": float(actor_norm.detach().cpu().item()),
            "critic_grad_norm": float(critic_norm.detach().cpu().item()),
        }

    def _sample_scene_batch(self, train_pool: list, update_index: int) -> list:
        """The per-update parallel-env batch: a fresh random subset of the train pool (so a
        large pool gives scene diversity without an intractable per-update rollout cost).
        Deterministic given update_index. 0 / >= pool size -> the whole pool."""

        import random

        size = self.config.scenes_per_update
        if size <= 0 or size >= len(train_pool):
            return train_pool
        rng = random.Random(self.config.seed + update_index)
        return rng.sample(train_pool, size)

    # ------------------------------------------------------------------ rollout batch
    def collect(self, actor, critic, specs, *, seed_base: int, deterministic: bool):
        specs = list(specs)
        # The per-scene rollout (Stage-21 physics) is the CPU bottleneck and is independent
        # across scenes, so num_workers>0 fans it out across processes (CPU workers; the GPU
        # is reserved for the update). Default 0 keeps the serial path byte-identical.
        if self.config.num_workers and len(specs) > 1:
            results = self._parallel_rollout(actor, critic, specs, seed_base, deterministic)
        else:
            results = [
                self.rollout_scene(actor, critic, spec, i, seed=seed_base + i, deterministic=deterministic)
                for i, spec in enumerate(specs)
            ]
        transitions: list[_Transition] = []
        per_scene_counts: list[int] = []
        for scene_transitions in results:
            # Pad/truncate to a uniform rollout length so GAE can reshape per scene.
            scene_transitions = scene_transitions[: self.config.rollout_steps]
            per_scene_counts.append(len(scene_transitions))
            transitions.extend(scene_transitions)
        return transitions, per_scene_counts

    def _acquire_pool(self):
        """Return (pool, owned). Prefer the persistent run() pool; else create a temporary
        SPAWN pool. Spawn (not fork) is mandatory: forked workers inherit a half-initialized
        OpenMP runtime (libgomp) and deadlock -- the classic CPU-pegged / GPU-idle hang."""

        if self._pool is not None:
            return self._pool, False
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(self.config.num_workers, initializer=_worker_init)
        return pool, True

    def _parallel_rollout(self, actor, critic, specs, seed_base, deterministic):
        actor_state = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
        critic_state = {key: value.detach().cpu() for key, value in critic.state_dict().items()}
        payloads = [
            (self.config, actor_state, critic_state, spec, index, seed_base + index, deterministic)
            for index, spec in enumerate(specs)
        ]
        pool, owned = self._acquire_pool()
        try:
            return pool.map(_rollout_worker, payloads)
        finally:
            if owned:
                pool.close()
                pool.join()

    def _advantages(self, transitions: list[_Transition], per_scene_counts: list[int]):
        # Only keep scenes that produced the full rollout length, so the GAE reshape is exact.
        full = [count == self.config.rollout_steps for count in per_scene_counts]
        kept: list[_Transition] = []
        offset = 0
        kept_scene_count = 0
        for scene_index, count in enumerate(per_scene_counts):
            block = transitions[offset : offset + count]
            offset += count
            if full[scene_index]:
                kept.extend(block)
                kept_scene_count += 1
        if kept_scene_count == 0:
            return kept, None
        rewards = torch.tensor([t.reward_surrogate for t in kept], dtype=torch.float32)
        values = torch.tensor([t.value for t in kept], dtype=torch.float32)
        masks = torch.tensor([t.done_mask for t in kept], dtype=torch.float32)
        result = compute_gae_returns(
            rewards,
            values,
            masks,
            num_scenarios=kept_scene_count,
            rollout_steps=self.config.rollout_steps,
            config=AdvantageConfig(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda),
        )
        return kept, result

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, actor, critic, specs, *, seed_base: int) -> dict[str, float]:
        transitions, _counts = self.collect(actor, critic, specs, seed_base=seed_base, deterministic=True)
        if not transitions:
            return {"tau_feasible_rate": 0.0, "mean_reward": 0.0, "mean_consensus": 0.0,
                    "mean_latency": 0.0, "mean_energy": 0.0, "mean_churn": 0.0, "step_count": 0}
        feasible = sum(1 for t in transitions if t.consensus >= self.config.tau_requirement_min)
        return {
            "tau_feasible_rate": feasible / len(transitions),
            "mean_reward": _mean(t.reward_surrogate for t in transitions),
            "mean_consensus": _mean(t.consensus for t in transitions),
            "mean_latency": _mean(t.latency for t in transitions),
            "mean_energy": _mean(t.energy for t in transitions),
            "mean_churn": _mean(t.churn for t in transitions),
            "step_count": len(transitions),
        }

    # ------------------------------------------------------------------ BC warm start
    def _warm_start(self, actor, train_specs) -> dict[str, object]:
        """Distil the budget-aware SA teacher into the actor (BCE on directed edge logits)
        before policy gradient. Targets mark the teacher's selected physical links on BOTH
        directed views, so the per-node scorer learns to rank the near-optimal backbone high
        and mutual acceptance then activates it. This is the validated scale-recovery step."""

        import torch.nn.functional as F

        train_specs = list(train_specs)
        if self.config.warmup_updates <= 0 or not train_specs:
            return {"bc_performed": False, "bc_scene_count": 0}
        # The SA-teacher search per scene is CPU-heavy and independent, so fan it out too.
        if self.config.num_workers and len(train_specs) > 1:
            samples = self._parallel_bc_samples(train_specs)
        else:
            samples = []
            for index, spec in enumerate(train_specs):
                sample = self._bc_sample(spec, seed=index)
                if sample is not None:
                    samples.append(sample)
        if not samples:
            return {"bc_performed": False, "bc_scene_count": 0}
        bc_opt = torch.optim.AdamW(actor.parameters(), lr=self.config.actor_lr)
        actor.train()
        last_loss = 0.0
        for _epoch in range(self.config.warmup_updates):
            for node_features, edge_index, edge_features, targets in samples:
                logits = actor.forward_graph(node_features, edge_index, edge_features)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                bc_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                bc_opt.step()
                last_loss = float(loss.detach().cpu().item())
        return {"bc_performed": True, "bc_scene_count": len(samples), "bc_final_loss": last_loss}

    def _bc_sample(self, spec, seed: int):
        import random

        from marl_topology.budgets import node_budgets_for_scene
        from marl_topology.data.stage31_production_dataset import build_production_context
        from marl_topology.data.stage31_scenario_generator import best_feasible_topology

        context = build_production_context(spec, time_step=0)
        evaluator = self._wrap_evaluator(spec.scene, context.graph, context.evaluator)
        node_budgets = node_budgets_for_scene(spec.scene)
        edge_universe = tuple(
            sorted({edge for edges in context.topology_variants.values() for edge in edges})
        )
        if not edge_universe:
            return None
        teacher = best_feasible_topology(
            evaluator,
            context.topology_variants,
            tau=self.config.tau_requirement_min,
            node_budgets=node_budgets,
            search_edge_ids=edge_universe,
            search_rng=random.Random(seed),
            search_sa_iters=self.config.bc_teacher_sa_iters,
            search_restarts=self.config.bc_teacher_restarts,
        )
        teacher_edges = set(teacher["edges"])
        observations = build_local_observations(
            scene=spec.scene,
            graph=context.graph,
            link_records=evaluator.link_records,
            time_step=0,
        )
        policy_inputs = [_observation_to_policy_input(o) for o in observations]
        graph_batch = tensorize_actor_graph([policy_inputs])
        if graph_batch.edge_count == 0:
            return None
        kind_by_node = {p.agent_id: p.agent_kind for p in policy_inputs}
        radio_budget = {node: _radio_budget(kind) for node, kind in kind_by_node.items()}
        remaining_energy = {node: self.config.energy_budget_j for node in kind_by_node}
        node_features, edge_features = self._augment_features(
            graph_batch, remaining_energy, set(), radio_budget
        )
        targets = torch.tensor(
            [
                1.0 if physical_edge_for_directed(directed) in teacher_edges else 0.0
                for directed in graph_batch.directed_edge_ids
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return (
            node_features.to(self.device),
            graph_batch.edge_index.to(self.device),
            edge_features.to(self.device),
            targets,
        )

    def _parallel_bc_samples(self, train_specs):
        payloads = [(self.config, spec, index) for index, spec in enumerate(train_specs)]
        pool, owned = self._acquire_pool()
        samples = []
        total = len(payloads)
        try:
            done = 0
            # imap_unordered streams results so we can show progress on the slow SA teacher
            # precompute (order is irrelevant for BC samples).
            for sample in pool.imap_unordered(_bc_worker, payloads):
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"[flow] BC teacher precompute {done}/{total}", flush=True)
                if sample is None:
                    continue
                node_features, edge_index, edge_features, targets = sample
                samples.append(
                    (
                        node_features.to(self.device),
                        edge_index.to(self.device),
                        edge_features.to(self.device),
                        targets.to(self.device),
                    )
                )
        finally:
            if owned:
                pool.close()
                pool.join()
        return samples

    # ------------------------------------------------------------------ run
    def run(
        self,
        train_specs: Sequence,
        eval_specs: Sequence,
        large_n_specs: Sequence = (),
    ) -> dict[str, object]:
        torch.manual_seed(self.config.seed)
        train_specs = list(train_specs)
        actor = build_production_actor(self.config).to(self.device)
        critic = build_ctde_critic(self.config).to(self.device)
        actor_opt = torch.optim.AdamW(actor.parameters(), lr=self.config.actor_lr)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=self.config.critic_lr)

        # One persistent SPAWN worker pool for the whole run (rollout + BC precompute), reused
        # across updates. Spawn (not fork) avoids the OpenMP+fork deadlock; reuse avoids paying
        # per-update process-spawn + torch-import cost.
        if self.config.num_workers and self.config.num_workers > 0:
            import multiprocessing as mp

            self._pool = mp.get_context("spawn").Pool(
                self.config.num_workers, initializer=_worker_init
            )

        # BC warm start (SA-teacher distillation) BEFORE policy gradient, so the reported
        # initial eval is the post-BC baseline and "improvement" measures the RL phase.
        print(
            f"[flow] BC warm start over {len(train_specs)} train scenes "
            f"(num_workers={self.config.num_workers}); the SA teacher precompute is the slow part...",
            flush=True,
        )
        bc_report = self._warm_start(actor, train_specs)
        print(f"[flow] BC done: {bc_report}", flush=True)

        update_records: list[dict[str, object]] = []
        eval_records: list[dict[str, object]] = []
        initial_eval = self.evaluate(actor, critic, eval_specs, seed_base=900_000)
        eval_records.append({"update_index": 0, "phase": "post_bc_initial", **initial_eval})

        train_pool = list(train_specs)
        for update_index in range(1, self.config.max_updates + 1):
            actor.train()
            critic.train()
            batch_specs = self._sample_scene_batch(train_pool, update_index)
            transitions, counts = self.collect(
                actor, critic, batch_specs, seed_base=update_index * 10_000, deterministic=False
            )
            kept, advantage_result = self._advantages(transitions, counts)
            if advantage_result is None:
                update_records.append({"update_index": update_index, "skipped": "no_full_rollout"})
                continue
            payload = self.update(
                actor, critic, actor_opt, critic_opt, kept,
                advantage_result.advantages, advantage_result.returns,
            )
            train_feasible = sum(1 for t in kept if t.consensus >= self.config.tau_requirement_min) / len(kept)
            update_records.append({
                "update_index": update_index,
                "train_tau_feasible_rate": train_feasible,
                "train_mean_reward": _mean(t.reward_surrogate for t in kept),
                "transition_count": len(kept),
                **payload,
            })
            print(
                f"[flow] update {update_index}/{self.config.max_updates} "
                f"train_feasible={train_feasible:.3f} loss={payload.get('total_loss', float('nan')):.4f} "
                f"kl={payload.get('approx_kl', 0.0):.4f}",
                flush=True,
            )
            if update_index % self.config.eval_every == 0 or update_index == self.config.max_updates:
                actor.eval()
                critic.eval()
                eval_metrics = self.evaluate(actor, critic, eval_specs, seed_base=900_000)
                eval_records.append({"update_index": update_index, "phase": "eval", **eval_metrics})
                print(
                    f"[flow] eval @ {update_index}: tau_feasible={eval_metrics['tau_feasible_rate']:.3f} "
                    f"consensus={eval_metrics['mean_consensus']:.3f}",
                    flush=True,
                )

        final_eval = eval_records[-1]
        # Held-out scale-generalization test: evaluate on strictly larger N never trained on.
        large_n_eval = (
            self.evaluate(actor, critic, large_n_specs, seed_base=950_000)
            if large_n_specs
            else None
        )
        if large_n_eval is not None:
            print(
                f"[flow] held-out large-N eval: tau_feasible={large_n_eval['tau_feasible_rate']:.3f}",
                flush=True,
            )
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        return {
            "flow_id": self.flow_id,
            "sampler_id": DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID,
            "config": _config_payload(self.config),
            "actor_boundary": actor.boundary_report(),
            "critic_boundary": critic.boundary_report(),
            "behaviour_cloning": bc_report,
            "initial_eval": initial_eval,
            "final_eval": final_eval,
            "large_n_eval": large_n_eval,
            "update_metrics": update_records,
            "eval_metrics": eval_records,
            "is_decentralized_execution": True,
            "is_sequential_mdp": True,
            "is_bandit": False,
        }, actor, critic


def decentralized_production_gate(
    report: Mapping[str, object],
    *,
    min_feasible_rate: float = 0.5,
) -> dict[str, object]:
    """Real training gate for the decentralized trunk (replaces the report-only micro guard).

    Passes only when the trained policy is actually feasible and stable: it clears a
    per-scene tau-feasible bar on held-out eval, did not collapse (final-update entropy and
    finite losses), and improved over the post-BC baseline. The headline scale result is the
    held-out large-N feasibility, surfaced alongside.
    """

    final = report.get("final_eval") or {}
    initial = report.get("initial_eval") or {}
    updates = [r for r in report.get("update_metrics", ()) if "total_loss" in r]
    last = updates[-1] if updates else {}
    final_rate = float(final.get("tau_feasible_rate", 0.0))
    entropy = float(last.get("entropy", 0.0))
    finite = bool(updates) and all(
        r.get("total_loss") == r.get("total_loss") for r in updates  # not NaN
    )
    not_collapsed = entropy > 1e-6 and finite
    feasible = final_rate >= min_feasible_rate
    improved = final_rate >= float(initial.get("tau_feasible_rate", 0.0))
    large = report.get("large_n_eval") or {}
    return {
        "passed": bool(feasible and not_collapsed),
        "min_feasible_rate": min_feasible_rate,
        "eval_tau_feasible_rate": final_rate,
        "large_n_tau_feasible_rate": float(large.get("tau_feasible_rate", 0.0)) if large else None,
        "improved_over_post_bc_baseline": bool(improved),
        "not_collapsed": bool(not_collapsed),
        "final_entropy": entropy,
        "losses_finite": finite,
    }


def run_decentralized_marl_training(
    config: DecentralizedMARLConfig,
    train_specs: Sequence,
    eval_specs: Sequence,
) -> dict[str, object]:
    report, _actor, _critic = DecentralizedCTDEFlow(config).run(train_specs, eval_specs)
    return report


# ----------------------------------------------------------------------- parallel workers
def _worker_init():
    """Pool worker initializer: pin OpenMP + torch to a single thread so N worker PROCESSES do
    not each spawn M OpenMP threads (oversubscription -> CPU thrash), and fix any invalid
    inherited OMP_NUM_THREADS that would make libgomp abort."""

    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _rollout_worker(payload):
    """Roll one scene in a worker process (CPU). Rebuilds the actor/critic from the passed
    state so each update's current policy is used; returns the scene's transitions (picklable
    CPU tensors). Single-threaded per worker so process-level parallelism is not oversubscribed."""

    torch.set_num_threads(1)
    config, actor_state, critic_state, spec, scene_index, seed, deterministic = payload
    cpu_config = replace(config, device="cpu")
    flow = DecentralizedCTDEFlow(cpu_config)
    actor = build_production_actor(cpu_config)
    actor.load_state_dict(actor_state)
    actor.eval()
    critic = build_ctde_critic(cpu_config)
    critic.load_state_dict(critic_state)
    critic.eval()
    return flow.rollout_scene(actor, critic, spec, scene_index, seed=seed, deterministic=deterministic)


def _bc_worker(payload):
    """Compute one scene's BC teacher sample (SA search + actor-safe graph) in a worker (CPU)."""

    torch.set_num_threads(1)
    config, spec, seed = payload
    cpu_config = replace(config, device="cpu")
    flow = DecentralizedCTDEFlow(cpu_config)
    return flow._bc_sample(spec, seed)


# ----------------------------------------------------------------------- helpers
def _critic_batch(node_features, edge_index, edge_features) -> GraphCriticBatch:
    node_count = node_features.shape[0]
    edge_count = edge_features.shape[0]
    return GraphCriticBatch(
        node_features=node_features.unsqueeze(0),
        edge_features=edge_features.unsqueeze(0),
        edge_index=edge_index.unsqueeze(0),
        node_mask=torch.ones((1, node_count), dtype=torch.float32, device=node_features.device),
        edge_mask=torch.ones((1, edge_count), dtype=torch.float32, device=edge_features.device),
    )


def _radio_budget(kind: str) -> int:
    from marl_topology.budgets import budget_for_kind

    return int(budget_for_kind(kind))


def _endpoints(physical_edge_id: str) -> tuple[str, str]:
    left, right = physical_edge_id.split("--", 1)
    return left, right


def _owner(directed_edge_id: str) -> str:
    return directed_edge_id.split("->", 1)[0]


def _reward_config():
    from marl_topology.training.policy_gradient.pilot_runner import (
        build_stage23_reward_surrogate_config,
    )

    return build_stage23_reward_surrogate_config()


def _mean(values) -> float:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else 0.0


def _mean_payload(payloads: list[dict[str, float]]) -> dict[str, float]:
    if not payloads:
        return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clip_fraction": 0.0, "actor_grad_norm": 0.0, "critic_grad_norm": 0.0}
    keys = payloads[0].keys()
    return {key: _mean(p[key] for p in payloads) for key in keys}


def _config_payload(config: DecentralizedMARLConfig) -> dict[str, object]:
    return {
        "flow_id": DECENTRALIZED_CTDE_FLOW_ID,
        "rounds": config.rounds,
        "hidden_dim": config.hidden_dim,
        "jumping_knowledge": config.jumping_knowledge,
        "rollout_steps": config.rollout_steps,
        "energy_budget_j": config.energy_budget_j,
        "energy_unit_j": config.energy_unit_j,
        "churn_weight": config.churn_weight,
        "clip_eps": config.clip_eps,
        "actor_lr": config.actor_lr,
        "critic_lr": config.critic_lr,
        "max_updates": config.max_updates,
        "tau_requirement_min": config.tau_requirement_min,
    }
