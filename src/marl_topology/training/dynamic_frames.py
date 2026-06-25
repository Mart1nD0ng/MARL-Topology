"""Multi-frame (two-timescale) dynamic topology data: per-frame channel observations.

The production trunk is a ``T=1`` contextual bandit. This OPT-IN training/eval substrate turns a
moving-vehicle trajectory into a genuine ``T>1`` MDP: a MACRO topology decision held for
``hold_interval`` PBFT micro-rounds at each of ``T`` frames, with the per-frame objective minus the
reconfiguration cost of changing the topology from the previous frame (the two_timescale_env
accounting). Vehicles move at constant velocity (``advance_scene``), so links evolve PREDICTABLY
across the window -- the temporal signal a recurrent actor can learn from (unlike i.i.d. fading).

Invariants the dynamic rollout relies on (pinned by tests/unit/test_dynamic_frames.py):

  * the node SET and the all-pairs candidate edge-id SET are INVARIANT across frames (vehicles move
    but never appear/disappear; candidate edges are all node pairs) -> the action space, and hence
    the reconfiguration cost ``|E_t triangle E_{t-1}|``, is well defined frame-to-frame;
  * each frame's CHANNEL is the real Stage-21 measurement on the moved geometry (link success /
    latency / energy from ``context.link_records``), which the actor observes via the edge features;
  * the previous topology and the step index enter the observation through ``graph_payload``'s
    ``previous_selected_edges`` / ``step_index`` hooks (already present in the shared data path);
  * frame 0 is the static scene exactly (the ``T=1`` limit reproduces the static objective).

Gate-exempt ``training/``: this touches no deployed path. The per-frame rollout uses the SAME
torch-free ``local_mutual_assemble`` decoder as deploy (train == deploy); the central critic stays
training-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from marl_topology.data.graph_payload import graph_payload
from marl_topology.data.stage31_production_dataset import build_production_context
from marl_topology.data.stage31_scenario_generator import (
    ProductionScenarioSpec,
    ProductionTrajectorySpec,
    PhysicsRegime,
    _quorum_for_node_count,
    _sample_scene,
    _sample_vehicle_motions,
    measure_reliable_range_m,
    solvability_from_finite_search,
)
from marl_topology.data.row_context_builder import build_stage33_training_row
from marl_topology.scenario.scene import NodeKind, Scene3D, advance_scene
from marl_topology.training.two_timescale_env import ReconfigCost

DYNAMIC_FRAMES_MODEL_ID = "two_timescale_mobility_frames_v1"

# Placeholder spec metadata for a frame whose feasibility was not pre-measured (the training rollout
# reads feasibility from the LIVE evaluator, never from the spec metadata -- these only populate the
# fixture's recorded fields). Valid ranges so ProductionScenarioSpec/ScenarioFixture validation passes.
_PLACEHOLDER_MEASUREMENT = {
    "full_graph_psucc": 1.0, "full_graph_feasible": 1.0, "best_feasible_psucc": 1.0,
    "best_feasible_energy_j": 0.1, "best_feasible_edge_count": 1.0, "full_graph_energy_j": 0.1,
    "feasible_exists": 1.0, "sparse_beats_full_energy": 0.0, "candidate_edge_count": 1.0,
}


def _frame_spec(scene: Scene3D, regime: PhysicsRegime, quorum_size: int, reliable_range_m: float,
                measurement: Mapping[str, float] | None) -> ProductionScenarioSpec:
    m = dict(measurement) if measurement is not None else dict(_PLACEHOLDER_MEASUREMENT)
    return ProductionScenarioSpec(
        scenario_id=scene.scenario_id, scene=scene, quorum_size=quorum_size, regime=regime,
        family=("feasible_sparse" if float(m["feasible_exists"]) >= 0.5 else "infeasible"),
        reliable_range_m=float(reliable_range_m),
        full_graph_psucc=float(m["full_graph_psucc"]), full_graph_feasible=bool(m["full_graph_feasible"]),
        best_feasible_psucc=float(m["best_feasible_psucc"]),
        best_feasible_energy_j=float(m["best_feasible_energy_j"]),
        best_feasible_edge_count=int(m["best_feasible_edge_count"]),
        full_graph_energy_j=float(m["full_graph_energy_j"]),
        feasible_exists=bool(m["feasible_exists"]),
        sparse_beats_full_energy=bool(m["sparse_beats_full_energy"]),
        candidate_edge_count=int(m["candidate_edge_count"]),
        solvability_status=solvability_from_finite_search(
            float(m["best_feasible_psucc"]), 0.9).status,
    )


@dataclass
class DynamicScene:
    """A scenario evolved over ``n_frames`` macro-steps: per-frame contexts on a shared action space.

    Built lazily -- the per-frame ``(context, row)`` are constructed on first use and cached, so the
    expensive evaluator build happens once per (scene, frame) for a whole run. The ``ReconfigCost``
    and ``hold_interval`` / ``gamma`` carry the two_timescale_env accounting used by the rollout.
    """

    sequence_id: str
    regime: PhysicsRegime
    quorum_size: int
    reliable_range_m: float
    scenes: tuple[Scene3D, ...]
    measurements: tuple[Mapping[str, float] | None, ...]
    reconfig: ReconfigCost = ReconfigCost()
    hold_interval: int = 1
    gamma: float = 0.95
    _ctx_cache: dict = field(default_factory=dict, repr=False)
    _row_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.scenes:
            raise ValueError("a dynamic scene needs at least one frame")
        if len(self.measurements) != len(self.scenes):
            raise ValueError("one measurement slot per frame is required")

    @property
    def n_frames(self) -> int:
        return len(self.scenes)

    def context(self, t: int):
        ctx = self._ctx_cache.get(t)
        if ctx is None:
            spec = _frame_spec(self.scenes[t], self.regime, self.quorum_size,
                               self.reliable_range_m, self.measurements[t])
            ctx = build_production_context(spec, time_step=t, sequence_id=self.sequence_id)
            self._ctx_cache[t] = ctx
        return ctx

    def _row(self, t: int):
        row = self._row_cache.get(t)
        if row is None:
            ctx = self.context(t)
            all_edges = tuple(e.edge_id for e in ctx.graph.edges)
            # Reference topology for the row's radio-bookkeeping features = the full candidate graph
            # (a fixed, non-leaky choice; the actor never sees teacher edges, only nf/ef/ei).
            teacher = {"selected_physical_edges": all_edges, "feasible_exists": True}
            row = build_stage33_training_row(
                context=ctx, teacher_label=teacher,
                structural_family=_frame_spec(self.scenes[t], self.regime, self.quorum_size,
                                              self.reliable_range_m, self.measurements[t]).family)
            self._row_cache[t] = row
        return row

    @property
    def edge_ids(self) -> list[str]:
        return list(self.edge_ids_at(0))

    def edge_ids_at(self, t: int) -> list[str]:
        return [e.edge_id for e in self.context(t).graph.edges]

    def observation(self, t: int, previous_selected_edges: Sequence[str]) -> dict:
        """The per-frame actor observation (and a sample dict compatible with the trunk helpers).

        nf / ef encode the CURRENT channel (moved geometry) + the previous topology + the step index;
        ei / edge_ids are the frame-invariant action space; context / label feed reward_of / eval.
        """
        ctx = self.context(t)
        row = self._row(t)
        payload = graph_payload(row=row, context=ctx,
                                previous_selected_edges=list(previous_selected_edges), step_index=t)
        nf = torch.tensor([list(map(float, n)) for n in payload["node_features"]])
        ef = torch.tensor([list(map(float, e)) for e in payload["edge_features"]])
        ei = torch.tensor([list(map(int, p)) for p in payload["edge_index"]], dtype=torch.long)
        edge_ids = [e.edge_id for e in ctx.graph.edges]
        m = self.measurements[t]
        feasible_exists = bool(m["feasible_exists"]) if m is not None else True
        label = {
            "selected_physical_edges": (),
            "feasible_exists": feasible_exists,
            "solvability_status": solvability_from_finite_search(
                float(m["best_feasible_psucc"]) if m is not None else 1.0, 0.9).status,
        }
        return {"nf": nf, "ef": ef, "ei": ei, "edge_ids": edge_ids,
                "context": ctx, "label": label, "time_index": t}


def dynamic_scene_from_trajectory(
    traj: ProductionTrajectorySpec,
    *,
    reconfig: ReconfigCost = ReconfigCost(),
    hold_interval: int = 1,
    gamma: float = 0.95,
) -> DynamicScene:
    """Build a DynamicScene from a measured mobility trajectory (frames carry the real measurement)."""
    return DynamicScene(
        sequence_id=str(traj.sequence_id), regime=traj.regime, quorum_size=int(traj.quorum_size),
        reliable_range_m=float(measure_reliable_range_m(traj.regime)),
        scenes=tuple(f.scene for f in traj.frames),
        measurements=tuple(dict(f.measurement) for f in traj.frames),
        reconfig=reconfig, hold_interval=hold_interval, gamma=gamma,
    )


def dynamic_scene_from_motion(
    initial_scene: Scene3D,
    motions,
    regime: PhysicsRegime,
    quorum_size: int,
    *,
    num_frames: int,
    dt_s: float,
    reliable_range_m: float,
    reconfig: ReconfigCost = ReconfigCost(),
    hold_interval: int = 1,
    gamma: float = 0.95,
) -> DynamicScene:
    """Roll geometry forward (advance_scene; NO per-frame SA measurement) -> a cheap DynamicScene.

    This is the dataset hot path: building the per-frame EVALUATOR (link records) is cheap; the
    expensive best-feasible SA search is NOT run (feasibility is read live from the evaluator during
    rollout). Frame metadata is a placeholder (unused by the reward).
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    motion_map = {m.node_id: m for m in motions}
    scenes = []
    scene = initial_scene
    for t in range(num_frames):
        if t > 0:
            scene = advance_scene(scene, motion_map, dt_s)
        scenes.append(scene)
    return DynamicScene(
        sequence_id=str(initial_scene.scenario_id), regime=regime, quorum_size=int(quorum_size),
        reliable_range_m=float(reliable_range_m), scenes=tuple(scenes),
        measurements=tuple([None] * num_frames),
        reconfig=reconfig, hold_interval=hold_interval, gamma=gamma,
    )


def sample_dynamic_scenes(
    *,
    seed: int,
    count: int,
    node_count_choices: Sequence[int],
    regime: PhysicsRegime,
    num_frames: int,
    dt_s: float,
    speed_min_mps: float,
    speed_max_mps: float,
    reconfig: ReconfigCost,
    hold_interval: int,
    gamma: float,
) -> list[DynamicScene]:
    """Deterministic set of moving-vehicle DynamicScenes (cheap path; no per-frame SA).

    Mirrors generate_production_trajectories' geometry sampling, but skips the per-frame measurement
    so building a multi-frame dataset is fast. Family cycle keeps a feasible/near/infeasible mix.
    """
    import random

    rng = random.Random(seed)
    reliable_range_m = measure_reliable_range_m(regime)
    cycle = ("feasible_sparse", "infeasible", "near_threshold")
    out = []
    for index in range(count):
        family = cycle[index % len(cycle)]
        node_count = rng.choice(tuple(node_count_choices))
        quorum = _quorum_for_node_count(node_count)
        scene = _sample_scene(rng, f"dyn_{index:05d}_{family}", node_count, family, reliable_range_m)
        motions = _sample_vehicle_motions(rng, scene, speed_min_mps, speed_max_mps)
        out.append(dynamic_scene_from_motion(
            scene, motions, regime, quorum, num_frames=num_frames, dt_s=dt_s,
            reliable_range_m=reliable_range_m, reconfig=reconfig, hold_interval=hold_interval, gamma=gamma))
    return out
