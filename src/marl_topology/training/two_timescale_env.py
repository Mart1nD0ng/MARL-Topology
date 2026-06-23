"""Two-timescale dynamic topology environment + Temporal Value Test (Phase 5, Spec S3.3-3.6).

The production trunk is a ``T=1`` contextual bandit. This OPT-IN module composes the static
per-frame objective into a multi-step episode to answer ONE architectural question (the Temporal
Value Test): does the task actually need temporal modeling, or does the static bandit suffice?

Two timescales (Spec S3.3): a MACRO topology decision is held for ``hold_interval`` (= H_PBFT)
consensus micro-rounds; the trajectory has ``T`` macro frames. The per-step reward is the static
per-frame objective held over the interval MINUS the reconfiguration cost of changing the topology
from the previous frame (Spec S3.5: ``E_reconfig = e_edge * |E_t triangle E_{t-1}|``, likewise
``L_reconfig``) -- a REAL physical cost, not a hand-tuned stability bonus.

The Temporal Value Test (Spec S3.6): ``Delta_H = J_myopic - J_horizon``. Myopic picks the per-frame
cheapest topology (ignoring reconfiguration); horizon minimizes the discounted total INCLUDING
reconfiguration (a DP over the candidate topologies). ``Delta_H > 0`` iff horizon-awareness (not
switching needlessly) genuinely lowers cost -> the task needs temporal return; ``Delta_H ~ 0`` ->
the static bandit is the main task and dynamic is an extension.

This is a TRAINING/eval harness (gate-exempt ``training/``); it touches no deployed path and the
``T=1`` limit reproduces the static single-step objective exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import product

TWO_TIMESCALE_ENV_MODEL_ID = "two_timescale_dynamic_topology_env_v1"

# cost_fn(frame, topology) -> scalar per-frame objective J (lower is better cost).
CostFn = Callable[[object, frozenset], float]


@dataclass(frozen=True)
class ReconfigCost:
    """Topology-switch physical cost (Spec S3.5): proportional to the toggled-edge count."""

    e_edge: float = 0.0  # energy per toggled edge
    l_edge: float = 0.0  # latency per toggled edge

    def __post_init__(self) -> None:
        if self.e_edge < 0.0 or self.l_edge < 0.0:
            raise ValueError("reconfiguration cost coefficients must be nonnegative")

    def cost(self, previous: frozenset, current: frozenset) -> float:
        """``(e_edge + l_edge) * |E_t triangle E_{t-1}|`` -- the symmetric-difference edge count."""
        n_changed = len(frozenset(previous) ^ frozenset(current))
        return (self.e_edge + self.l_edge) * n_changed


@dataclass(frozen=True)
class DynamicEnvState:
    """Immutable env state -> forking is copy-by-construction (no mutation leakage)."""

    frame_index: int
    previous_topology: frozenset


@dataclass(frozen=True)
class StepResult:
    reward: float
    base_objective: float
    reconfig_cost: float
    next_state: DynamicEnvState
    done: bool


class TwoTimescaleTopologyEnv:
    """An opt-in two-timescale dynamic topology env (Spec S3.3-3.5)."""

    def __init__(
        self,
        frames: Sequence[object],
        cost_fn: CostFn,
        reconfig: ReconfigCost = ReconfigCost(),
        *,
        hold_interval: int = 1,
        gamma: float = 1.0,
    ) -> None:
        if not frames:
            raise ValueError("at least one frame is required")
        if hold_interval < 1:
            raise ValueError("hold_interval (H_PBFT) must be >= 1")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.frames = list(frames)
        self.cost_fn = cost_fn
        self.reconfig = reconfig
        self.hold_interval = hold_interval
        self.gamma = gamma
        self.model_id = TWO_TIMESCALE_ENV_MODEL_ID

    def reset(self) -> DynamicEnvState:
        return DynamicEnvState(frame_index=0, previous_topology=frozenset())

    def step(self, state: DynamicEnvState, topology) -> StepResult:
        """Apply a macro topology decision: hold it for ``hold_interval`` micro-rounds, pay the
        per-frame objective each round plus a one-time reconfiguration cost vs the previous frame.
        Returns ``reward = -(base*hold + reconfig)`` and the next (immutable) state.
        """
        topology = frozenset(topology)
        frame = self.frames[state.frame_index]
        base_objective = float(self.cost_fn(frame, topology)) * self.hold_interval
        # No reconfiguration is charged for the very first decision (no prior topology to switch from).
        reconfig_cost = (
            self.reconfig.cost(state.previous_topology, topology) if state.frame_index > 0 else 0.0
        )
        reward = -(base_objective + reconfig_cost)
        next_state = DynamicEnvState(
            frame_index=state.frame_index + 1, previous_topology=topology
        )
        done = next_state.frame_index >= len(self.frames)
        return StepResult(reward, base_objective, reconfig_cost, next_state, done)

    def fork(self, state: DynamicEnvState) -> DynamicEnvState:
        """Fork the env state for counterfactual rollouts (Spec S10.3). The state is immutable,
        so the fork is independent by construction (stepping one never mutates the other)."""
        return replace(state)

    def trajectory_cost(self, topologies: Sequence[frozenset]) -> float:
        """Discounted total cost of holding ``topologies[t]`` at each frame (lower is better)."""
        if len(topologies) != len(self.frames):
            raise ValueError("one topology per frame is required")
        state = self.reset()
        total = 0.0
        discount = 1.0
        for topology in topologies:
            result = self.step(state, topology)
            total += discount * (-result.reward)  # reward is negative cost
            discount *= self.gamma
            state = result.next_state
        return total


def temporal_value_test(
    env: TwoTimescaleTopologyEnv,
    candidates_per_frame: Sequence[Sequence[frozenset]],
) -> dict[str, float]:
    """``Delta_H = J_myopic - J_horizon`` over the trajectory (Spec S3.6).

    Myopic picks the per-frame cheapest topology (ignoring reconfiguration); horizon minimizes the
    discounted total INCLUDING reconfiguration via an exact DP over the candidate topologies.
    Returns ``{j_myopic, j_horizon, delta_h}``; ``delta_h >= 0`` always (horizon optimizes a
    superset objective), and ``delta_h > 0`` iff horizon-awareness strictly helps.
    """
    if len(candidates_per_frame) != len(env.frames):
        raise ValueError("one candidate set per frame is required")
    candidates = [list(c) for c in candidates_per_frame]
    if any(not c for c in candidates):
        raise ValueError("each frame needs at least one candidate topology")

    # Myopic: per-frame argmin of the base objective (reconfiguration-blind).
    myopic = [min(c, key=lambda x: env.cost_fn(frame, x)) for frame, c in zip(env.frames, candidates)]
    j_myopic = env.trajectory_cost([frozenset(x) for x in myopic])

    # Horizon: exact DP. value[x] = min discounted cost-to-here ending frame t with topology x.
    # transition cost at frame t with prev p, current x = gamma^t * (base(t,x)*hold + reconfig(p,x)).
    j_horizon = _horizon_optimal_cost(env, candidates)

    return {"j_myopic": j_myopic, "j_horizon": j_horizon, "delta_h": j_myopic - j_horizon}


def _horizon_optimal_cost(env: TwoTimescaleTopologyEnv, candidates: list[list[frozenset]]) -> float:
    """Exact minimum discounted total cost over the candidate topologies (DP over frames)."""
    discount = 1.0
    # frame 0: no reconfiguration (no prior topology).
    best = {frozenset(x): discount * env.cost_fn(env.frames[0], x) * env.hold_interval
            for x in candidates[0]}
    for t in range(1, len(env.frames)):
        discount *= env.gamma
        frame = env.frames[t]
        nxt: dict[frozenset, float] = {}
        for x in candidates[t]:
            xf = frozenset(x)
            base = discount * env.cost_fn(frame, xf) * env.hold_interval
            best_prev = min(
                prev_cost + discount * env.reconfig.cost(p, xf) for p, prev_cost in best.items()
            )
            nxt[xf] = base + best_prev
        best = nxt
    return min(best.values())
