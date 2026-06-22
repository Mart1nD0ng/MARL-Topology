"""Stage 31 procedural scenario generator with a measured tau-feasibility gradient.

This module produces production-scale 3D V2X scenarios that flow through the
*existing* Stage 21 objective stack (finite-blocklength links + Stage 3 network
records + Stage 4 expected-initiator PBFT reliability). It does not change the
physics or the objective; it generates diverse node geometries and measures the
resulting feasibility so that the dataset contains a genuine gradient:

- *feasible* scenarios where a sparse, reliable-link, quorum-spanning topology
  reaches ``consensus_success_probability >= tau`` (tau = 0.9);
- *near-threshold* scenarios that sit close to the reliable-range boundary;
- *infeasible* scenarios where at least one node is beyond reliable range so no
  topology reaches tau.

The generator is *self-calibrating*: it measures the effective reliable
communication range ``R`` for the chosen power/bandwidth/deadline regime and
sizes all geometry relative to ``R``. This keeps the gradient valid regardless
of the exact physical regime and avoids faking feasibility by saturating links.

tau is fixed at 0.9 (owner decision, Stage 31). No link probability is inflated;
feasibility is always *measured* on the real stack.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from marl_topology.channel import ChannelModelConfig, evaluate_channel
from marl_topology.geometry3d import Point3D
from marl_topology.link import LinkTransmissionConfig, evaluate_link_transmission
from marl_topology.policies import PolicyBaselines
from marl_topology.scenario.scene import (
    Node3D,
    NodeKind,
    NodeMotion,
    Scene3D,
    advance_scene,
)
from marl_topology.scenario.urban_grid import UrbanGridConfig, build_urban_grid_scene
from marl_topology.topology import CandidateGraph

TAU_REQUIREMENT_MIN = 0.9
RSU_HEIGHT_M = 5.0
VEHICLE_HEIGHT_M = 1.5
RELIABLE_LINK_PROBABILITY = 0.99


class ProceduralGeneratorViolation(ValueError):
    """Raised when generation parameters or measured output cross a boundary."""


@dataclass(frozen=True, slots=True)
class PhysicsRegime:
    """A declared, realistic low-power NLOS sidelink regime.

    The defaults give a finite, measurable reliable range so node geometry
    (tens to ~150 m, an urban V2X scale) actually controls feasibility.
    """

    tx_power_dbm: float = -3.0
    bandwidth_hz: float = 20e6
    deadline_s: float = 0.003
    payload_bits: int = 12_000
    fixed_transmission_time_s: float = 0.0005
    target_reliability: float = 0.99
    # Shared spectrum with interference is what makes topology control
    # non-trivial: a dense (full-graph) topology self-interferes and fails, so
    # the controller must find a sparse, reliable, quorum-connected topology.
    use_background_interference: bool = True
    orthogonal_resources: bool = False
    # relay_hops > 1: consensus messages may be relayed through up to this many links (via
    # RSU / intermediate nodes) instead of needing a direct link -- enables global PBFT under
    # urban NLOS where direct connectivity is fragmented. Default 1 = single-hop (unchanged).
    relay_hops: int = 1
    # scheduled_mac: partition the selected links into SINR-feasible spatial-reuse TDMA slots
    # (the realistic MAC abstraction) instead of the worst-case all-shared spectrum. Co-slot
    # links interfere (validated), cross-slot links are orthogonal, and slot count adds latency.
    # When on, it supersedes use_background_interference / orthogonal_resources. Default off.
    scheduled_mac: bool = False
    mac_sinr_threshold_db: float = 0.0
    mac_slot_duration_s: float = 0.0006
    # path_loss_model: propagation formula for the channel (calibration knob). Default
    # "fspl_nlos_penalty" = byte-identical FSPL + flat NLOS penalty; "umi_street_canyon_38901"
    # = 3GPP TR 38.901 UMi-Street-Canyon LOS/NLOS (distance-dependent NLOS exponent);
    # "v2x_37885" = TR 37.885 urban V2V for vehicle-vehicle links + UMi for V2I (the
    # standards-calibrated V2X pair).
    path_loss_model: str = "fspl_nlos_penalty"
    # wired_rsu_backhaul: RSU-RSU pairs communicate over reliable roadside backhaul (the
    # standard deployment); injected as delivery-1.0 before the relay pass so vehicles can
    # reach across the city through the RSU backbone. Default off.
    wired_rsu_backhaul: bool = False
    # coverage_gated_membership: PBFT validators are the scene-level COVERED nodes (best
    # incident candidate link delivery >= membership_min_link_delivery); uncovered nodes
    # are demoted to clients (temporarily out of the consensus group, resynced on rejoin),
    # consensus is evaluated over the validator set, and coverage_rate is reported beside
    # it. Membership never depends on the selected topology. Default off.
    coverage_gated_membership: bool = False
    membership_min_link_delivery: float = 0.5
    # shadowing_37885 / nlosv_37885: standardized stochastic large-scale fading (per-state
    # sigma spatially-correlated shadowing; stochastic NLOSv vehicle-blockage state + loss
    # + UE-type-RSU link mapping). Require path_loss_model "v2x_37885". Default off.
    # shadowing_realization indexes independent robustness draws for the same scene.
    shadowing_37885: bool = False
    nlosv_37885: bool = False
    shadowing_realization: int = 0
    # --- Phase 0-4 corrected environment math (recalibration; defaults reproduce legacy
    #     byte-for-byte). Appended LAST so positional PhysicsRegime construction is unaffected.
    # fault_model: "remove_largest" (legacy per-phase filter) | "fixed_set" (the principled
    #   single fixed Byzantine set C_robust = min_{|B|<=f} C(x;B), Spec S4.7).
    fault_model: str = "remove_largest"
    # one_hop_relay: build the PBFT matrix from DIRECT links only so relay_hops is the single
    #   multi-hop layer (Spec S4.2). MUST be paired with relay_hops >= 2 to keep reachability.
    one_hop_relay: bool = False

    def channel_config(self) -> ChannelModelConfig:
        return ChannelModelConfig(
            default_tx_power_dbm=self.tx_power_dbm,
            bandwidth_hz=self.bandwidth_hz,
            # getattr: PhysicsRegime instances unpickled from datasets serialized BEFORE this
            # field existed (slots dataclass -> missing attribute) default to FSPL.
            path_loss_model=getattr(self, "path_loss_model", "fspl_nlos_penalty"),
            shadowing_37885=getattr(self, "shadowing_37885", False),
            nlosv_37885=getattr(self, "nlosv_37885", False),
            shadowing_realization=getattr(self, "shadowing_realization", 0),
        )

    def link_config(self) -> LinkTransmissionConfig:
        return LinkTransmissionConfig(
            payload_bits=self.payload_bits,
            bandwidth_hz=self.bandwidth_hz,
            fixed_transmission_time_s=self.fixed_transmission_time_s,
            target_reliability=self.target_reliability,
            deadline_s=self.deadline_s,
        )


# Realistic urban regime: 20 dBm (typical C-V2X) so NLOS (20 dB building penalty) is the
# binding constraint at a fixed ~450 m city-grid scale -- LOS links along a street deliver,
# cross-block NLOS links mostly fail, so topology must route along streets / relay.
URBAN_PHYSICS_REGIME = PhysicsRegime(tx_power_dbm=20.0)


@dataclass(frozen=True, slots=True)
class ProductionScenarioConfig:
    """Generation knobs for a production scenario family."""

    seed: int = 31
    scenario_count: int = 120
    # N >= 4 so PBFT fault tolerance f >= 1 (quorum > 1). A spread keeps small,
    # learnable graphs so the GNN-vs-MLP comparison stays meaningful. The endpoint
    # budget is now SCALE-READY (budgets.py rsu=64) and logs/probe_large_n_feasibility
    # confirms N=10/12 are feasible with a healthy ~13/18 mix under it -- the OLD rsu=8
    # rejected the feasible degree-(N-1) star at N>=10 (0-1/18 feasible). BUT lifting
    # this cap to 10/12 made the readiness pilot stop improving feasibility: the
    # current memoryless actor cannot learn the larger graphs in a short pilot, so the
    # readiness gate correctly reports not-ready. Large-N production training is
    # therefore deferred to the temporal-actor work that can learn it; keep small-N
    # here until then (the budget no longer blocks scale, the actor does).
    node_count_choices: tuple[int, ...] = (4, 5, 6, 7, 8)
    regime: PhysicsRegime = field(default_factory=PhysicsRegime)
    # Urban mode (opt-in): place nodes on a city street grid with real building NLOS
    # blockage instead of free-space (the FSPL path is always-LOS). 1 RSU + (N-1) vehicles
    # per scene; blockage -- not just distance -- drives feasibility. Uses URBAN_PHYSICS_REGIME
    # (realistic 20 dBm) so NLOS is binding at a fixed ~450 m urban scale. Default off keeps
    # the validated free-space generator byte-identical.
    urban_mode: bool = False
    urban_block_size_m: float = 150.0
    urban_street_width_m: float = 24.0
    # urban_blocks_per_side controls grid DENSITY: with few vehicles a large grid leaves
    # nodes NLOS-isolated (their per-initiator PBFT reliability ~0 caps the expected-over-
    # initiator consensus below tau), so a denser grid (smaller blocks_per_side) is needed
    # for a trainable feasibility mix at modest vehicle counts. Default 3 = the ~450 m grid.
    urban_blocks_per_side: int = 3
    # urban_rsu_count: RSUs per scene (placed at distinct intersections). >1 enables the
    # multi-RSU regimes (pairs with PhysicsRegime.wired_rsu_backhaul). Default 1 (unchanged).
    urban_rsu_count: int = 1
    # Target fractions of the measured feasibility families. They are guidance
    # for binning; the generator records the realized distribution.
    target_feasible_fraction: float = 0.5
    target_near_threshold_fraction: float = 0.2
    target_infeasible_fraction: float = 0.3
    max_attempts_per_scenario: int = 40
    # Hard GLOBAL attempt budget. 0 = auto (scenario_count * max_attempts_per_scenario). At low-yield
    # regimes (e.g. urban N>=24 where the feasible bin is hard to fill) this terminates the rejection-
    # sampling loop with a SMALLER dataset rather than thrashing for hours. Default 0 keeps the high-yield
    # paths byte-identical (they finish far below the cap).
    max_total_attempts: int = 0
    tau_requirement_min: float = TAU_REQUIREMENT_MIN

    def __post_init__(self) -> None:
        if self.scenario_count <= 0:
            raise ProceduralGeneratorViolation("scenario_count must be positive")
        if not self.node_count_choices or any(n < 3 for n in self.node_count_choices):
            raise ProceduralGeneratorViolation("node_count_choices must be >= 3")
        if self.tau_requirement_min != TAU_REQUIREMENT_MIN:
            raise ProceduralGeneratorViolation("Stage 31 keeps tau_requirement_min fixed at 0.9")
        total = (
            self.target_feasible_fraction
            + self.target_near_threshold_fraction
            + self.target_infeasible_fraction
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ProceduralGeneratorViolation("target family fractions must sum to 1.0")


@dataclass(frozen=True, slots=True)
class ProductionScenarioSpec:
    """One generated scenario plus its measured feasibility labels."""

    scenario_id: str
    scene: Scene3D
    quorum_size: int
    regime: PhysicsRegime
    family: str
    reliable_range_m: float
    full_graph_psucc: float
    full_graph_feasible: bool
    best_feasible_psucc: float
    best_feasible_energy_j: float
    best_feasible_edge_count: int
    full_graph_energy_j: float
    feasible_exists: bool
    sparse_beats_full_energy: bool
    candidate_edge_count: int

    def __post_init__(self) -> None:
        if self.family not in {"feasible_sparse", "near_threshold", "infeasible"}:
            raise ProceduralGeneratorViolation(f"unknown family: {self.family}")
        if not 0.0 <= self.full_graph_psucc <= 1.0:
            raise ProceduralGeneratorViolation("full_graph_psucc out of range")


@dataclass(frozen=True, slots=True)
class TrajectoryFrame:
    """One time slice of a trajectory: the (moved) scene plus its measured feasibility.

    ``measurement`` is the same dict ``_measure_scene`` produces for a static scene,
    so each frame carries its OWN feasibility -- the trajectory captures links
    crossing the tau boundary as vehicles move.
    """

    time_index: int
    time_s: float
    scene: Scene3D
    measurement: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ProductionTrajectorySpec:
    """A scenario evolved over time: an ordered sequence of frames sharing one id.

    The same scenario, sampled once, then advanced ``num_frames`` time steps with
    vehicles at constant velocity (the RSU stays fixed). Every frame is measured on
    the real Stage 21 stack, so the sequence is a genuine time series of geometry +
    feasibility -- the predictable link evolution a temporal actor learns from. This
    is additive substrate for the rollout-evolution / temporal-actor work; it does
    not change the existing static-scene generator.
    """

    sequence_id: str
    regime: PhysicsRegime
    quorum_size: int
    dt_s: float
    motions: tuple[NodeMotion, ...]
    frames: tuple[TrajectoryFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ProceduralGeneratorViolation("a trajectory needs at least one frame")
        if self.dt_s <= 0.0:
            raise ProceduralGeneratorViolation("dt_s must be positive")
        indices = [frame.time_index for frame in self.frames]
        if indices != list(range(len(self.frames))):
            raise ProceduralGeneratorViolation("trajectory frames must be contiguous from 0")

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def feasible_frame_count(self) -> int:
        return sum(
            1 for frame in self.frames if frame.measurement.get("feasible_exists", 0.0) >= 0.5
        )


# --------------------------------------------------------------------------- #
# Self-calibration: measure the reliable range for a regime.
# --------------------------------------------------------------------------- #
def _single_link_probability(distance_m: float, regime: PhysicsRegime) -> float:
    scene = Scene3D(
        scenario_id="calib",
        nodes=(
            Node3D("a", NodeKind.RSU, Point3D(0.0, 0.0, RSU_HEIGHT_M)),
            Node3D("b", NodeKind.VEHICLE, Point3D(distance_m, 0.0, VEHICLE_HEIGHT_M)),
        ),
    )
    channel = evaluate_channel(
        scene, "a", "b", config=regime.channel_config(), resource_id="resource_0"
    )
    link = evaluate_link_transmission(
        channel, regime.link_config(), selected=True, active=True
    )
    return link.deadline_delivery_probability


def measure_reliable_range_m(
    regime: PhysicsRegime, *, probability: float = RELIABLE_LINK_PROBABILITY
) -> float:
    """Largest 2D separation whose link reliability stays >= ``probability``.

    Binary search over distance. Returns a finite positive range.
    """

    lo, hi = 1.0, 4.0
    # Expand hi until the link fails (or a sane cap).
    while _single_link_probability(hi, regime) >= probability and hi < 5000.0:
        hi *= 2.0
    if _single_link_probability(lo, regime) < probability:
        raise ProceduralGeneratorViolation(
            "regime cannot deliver a reliable link even at 1 m; pick a stronger regime"
        )
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _single_link_probability(mid, regime) >= probability:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Geometry samplers (positions in the x-y plane; z is role height).
# --------------------------------------------------------------------------- #
def _node(node_index: int, x: float, y: float) -> Node3D:
    if node_index == 0:
        return Node3D("rsu_0", NodeKind.RSU, Point3D(x, y, RSU_HEIGHT_M))
    return Node3D(f"veh_{node_index - 1}", NodeKind.VEHICLE, Point3D(x, y, VEHICLE_HEIGHT_M))


def _connected_core_positions(
    rng: random.Random, node_count: int, step: float
) -> list[tuple[float, float]]:
    """A random connected chain/cluster with consecutive spacing ~ ``step``."""

    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    for _ in range(node_count - 1):
        anchor = rng.choice(positions)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = step * rng.uniform(0.55, 0.95)
        positions.append(
            (anchor[0] + radius * math.cos(angle), anchor[1] + radius * math.sin(angle))
        )
    return positions


def _sample_scene(
    rng: random.Random,
    scenario_id: str,
    node_count: int,
    family: str,
    reliable_range_m: float,
) -> Scene3D:
    # consecutive spacing inside a reliable core
    step = reliable_range_m * rng.uniform(0.45, 0.7)
    positions = _connected_core_positions(rng, node_count, step)

    if family == "near_threshold":
        # move the last node out to just inside/around the boundary from its anchor
        anchor = positions[rng.randrange(node_count - 1)]
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = reliable_range_m * rng.uniform(0.9, 1.05)
        positions[-1] = (
            anchor[0] + radius * math.cos(angle),
            anchor[1] + radius * math.sin(angle),
        )
    elif family == "infeasible":
        # push the last node well beyond reliable range from the whole core
        cx = sum(p[0] for p in positions[:-1]) / (node_count - 1)
        cy = sum(p[1] for p in positions[:-1]) / (node_count - 1)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = reliable_range_m * rng.uniform(1.6, 2.4)
        positions[-1] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    nodes = tuple(_node(i, x, y) for i, (x, y) in enumerate(positions))
    return Scene3D(scenario_id=scenario_id, nodes=nodes)


# --------------------------------------------------------------------------- #
# Measurement: evaluate a scene on the real Stage 21 objective stack.
# --------------------------------------------------------------------------- #
def build_stack_config(regime: PhysicsRegime):
    """Construct the Stage 21 objective-stack config for a regime (lazy import)."""

    from marl_topology.data.stage21_objective_stack_evidence import (
        Stage21ObjectiveStackConfig,
    )

    return Stage21ObjectiveStackConfig(
        channel_config=regime.channel_config(),
        link_config=regime.link_config(),
        use_background_interference=regime.use_background_interference,
        orthogonal_resources=regime.orthogonal_resources,
        relay_hops=regime.relay_hops,
        scheduled_mac=regime.scheduled_mac,
        mac_sinr_threshold_db=regime.mac_sinr_threshold_db,
        mac_slot_duration_s=regime.mac_slot_duration_s,
        # getattr: regimes unpickled from datasets serialized before this field existed
        wired_rsu_backhaul=getattr(regime, "wired_rsu_backhaul", False),
        coverage_gated_membership=getattr(regime, "coverage_gated_membership", False),
        membership_min_link_delivery=getattr(regime, "membership_min_link_delivery", 0.5),
        # corrected env-math (recalibration); getattr legacy default keeps old regimes byte-identical
        fault_model=getattr(regime, "fault_model", "remove_largest"),
        one_hop_relay=getattr(regime, "one_hop_relay", False),
    )


def enumerate_candidate_topologies(
    graph: CandidateGraph,
    link_records: Mapping[str, object],
    quorum_size: int,
) -> dict[str, tuple[str, ...]]:
    """Named candidate topologies, reusing the project's Stage 21 variants.

    This is also the heuristic-teacher candidate pool used by Phase E for
    graphs that exceed the exhaustive-oracle edge cap.
    """

    from marl_topology.data.stage21_objective_stack_evidence import (
        _stage21_topology_variants,
    )

    return _stage21_topology_variants(graph, link_records, quorum_size)


def search_relay_topology(
    evaluator,
    edge_ids,
    seed_edge_sets,
    rng: random.Random,
    *,
    sa_iters: int = 200,
    restarts: int = 6,
    polish: bool = True,
    node_budgets: tuple[tuple[str, int], ...] | None = None,
) -> tuple[str, ...]:
    """Near-optimal sparse relay topology via simulated annealing (single-toggle + 2-swap
    neighborhood) over the consensus objective, polished by a final hill-climb, from many
    restarts. Maximizes (consensus_success, then sparsity). This is the RELAY-AWARE teacher:
    it finds the sparse multi-hop relay backbones the fixed heuristics miss under urban NLOS,
    so behaviour cloning gets a near-optimal target (validated gap=0 vs exhaustive at small N)
    -- removing the heuristic-teacher soft ceiling.

    When ``node_budgets`` is given the search is BUDGET-AWARE: budget-violating topologies are
    penalised below any budget-feasible one (a constant -2.0 keeps them strictly worse since
    consensus is in [0,1]), so the SA annealing dynamics on consensus are preserved but the
    optimum is the best BUDGET-FEASIBLE backbone. This stops the teacher from emitting targets
    the deployment assembler (degree-capped) must drop, and from under-counting feasibility.
    Default None = budget-blind (byte-identical)."""
    from marl_topology.budgets import is_budget_feasible

    edge_list = list(edge_ids)

    def score(edges):
        metrics = evaluator.evaluate(set(edges)).metrics
        consensus = float(metrics["consensus_success_probability"])
        if node_budgets is not None and not is_budget_feasible(tuple(edges), node_budgets):
            consensus -= 2.0
        return (consensus, -len(edges))

    def hill_climb(start):
        current, current_score = set(start), score(set(start))
        improved = True
        while improved:
            improved = False
            for edge in edge_list:
                trial = current ^ {edge}
                trial_score = score(trial)
                if trial_score > current_score:
                    current, current_score, improved = trial, trial_score, True
        return current, current_score

    best_edges, best_score = set(), (-1.0, 0)
    starts = list(seed_edge_sets) + [()] + [
        tuple(edge for edge in edge_list if rng.random() < p)
        for p in (0.15, 0.3, 0.5)
        for _ in range(restarts)
    ]
    for seed in starts:
        current = set(seed)
        current_score = score(current)
        run_best, run_best_score = set(current), current_score
        temperature = 0.25
        for _ in range(sa_iters):
            trial = set(current)
            if rng.random() < 0.5 and edge_list:
                trial ^= {rng.choice(edge_list)}
            elif len(edge_list) >= 2:
                first, second = rng.sample(edge_list, 2)
                trial ^= {first, second}
            trial_score = score(trial)
            delta = trial_score[0] - current_score[0]
            if trial_score > current_score or rng.random() < math.exp(delta / max(temperature, 1e-6)):
                current, current_score = trial, trial_score
                if current_score > run_best_score:
                    run_best, run_best_score = set(current), current_score
            temperature *= 0.985
        # The final hill-climb polish dominates cost on larger graphs (it re-evaluates every
        # edge toggle to convergence per restart). polish=False skips it -- a lighter,
        # good-enough search for fast family-binning / teacher labels under expensive regimes.
        if polish:
            polished, polished_score = hill_climb(run_best)
        else:
            polished, polished_score = run_best, run_best_score
        if polished_score > best_score:
            best_score, best_edges = polished_score, polished
    return tuple(sorted(best_edges))


def best_feasible_topology(
    evaluator,
    candidates: Mapping[str, tuple[str, ...]],
    tau: float,
    node_budgets: tuple[tuple[str, int], ...] | None = None,
    *,
    search_edge_ids=None,
    search_rng: random.Random | None = None,
    search_sa_iters: int = 200,
    search_restarts: int = 6,
    search_polish: bool = True,
) -> dict[str, object]:
    """Pick the best topology by (feasible, latency, energy, sparsity).

    Route B: when ``node_budgets`` is provided, a candidate counts as feasible
    only if it is ALSO endpoint-budget-feasible (no node exceeds its tx/rx
    budget), so the teacher never targets a topology the actor's deployment
    pipeline cannot realize. ``node_budgets=None`` preserves the legacy
    budget-blind behaviour. ``search_sa_iters`` / ``search_restarts`` bound the
    relay-aware SA cost (lighter for fast family-binning, heavier for the BC teacher).
    """

    from marl_topology.budgets import is_budget_feasible

    candidates = dict(candidates)
    if search_edge_ids is not None and search_rng is not None:
        # relay-aware teacher: add the near-optimal search topology as a candidate, so it
        # competes with (and, on relay scenes, beats) the fixed heuristics. Opt-in: when not
        # provided, behaviour is byte-identical to the heuristic-only teacher.
        candidates["search_relay"] = search_relay_topology(
            evaluator, search_edge_ids, list(candidates.values()), search_rng,
            sa_iters=search_sa_iters, restarts=search_restarts, polish=search_polish,
            node_budgets=node_budgets,
        )

    scored: list[tuple[bool, float, float, int, str, tuple[str, ...]]] = []
    for name, edges in candidates.items():
        ev = evaluator.evaluate(set(edges))
        psucc = float(ev.metrics["consensus_success_probability"])
        latency = float(ev.metrics["latency"])
        energy = float(ev.metrics["energy"])
        budget_ok = node_budgets is None or is_budget_feasible(edges, node_budgets)
        scored.append(
            (psucc >= tau and budget_ok, latency, energy, len(edges), name, tuple(edges))
        )

    feasible = [s for s in scored if s[0]]
    if feasible:
        winner = min(feasible, key=lambda s: (s[1], s[2], s[3]))
        feasible_exists = True
    else:
        # No feasible candidate. Prefer a BUDGET-FEASIBLE target (so BC learns deployable,
        # degree-respecting proposals even on infeasible scenes -- a budget-violating target
        # is one the assembler must drop); only if none is budget-feasible (impossible when
        # node_budgets is given, since the empty topology is always budget-feasible) fall back
        # to the highest-consensus topology. node_budgets=None -> byte-identical to before.
        def _winner_consensus(s):
            return float(evaluator.evaluate(set(s[5])).metrics["consensus_success_probability"])

        budget_feasible = [
            s for s in scored if node_budgets is None or is_budget_feasible(s[5], node_budgets)
        ]
        winner = max(budget_feasible or scored, key=_winner_consensus)
        feasible_exists = False
    win_eval = evaluator.evaluate(set(winner[5]))
    return {
        "feasible_exists": feasible_exists,
        "name": winner[4],
        "edges": winner[5],
        "edge_count": winner[3],
        "psucc": float(win_eval.metrics["consensus_success_probability"]),
        "latency": float(win_eval.metrics["latency"]),
        "energy": float(win_eval.metrics["energy"]),
    }


def relay_aware_search_kwargs(
    regime: PhysicsRegime,
    graph: CandidateGraph,
    scenario_id: str,
    *,
    sa_iters: int = 200,
    restarts: int = 6,
    polish: bool = True,
) -> dict[str, object]:
    """Hand ``best_feasible_topology`` the near-optimal relay-aware search teacher when
    the regime needs multi-hop relay or a scheduled MAC -- the fixed heuristic candidate
    pool misses the sparse relay/scheduled backbones those regimes make feasible, so the
    label/family would be wrongly capped without it. Deterministic per scenario. The
    default regime (single-hop, worst-case spectrum) returns ``{}`` -> byte-identical.

    ``sa_iters`` / ``restarts`` bound the SA cost: the family-binning measurement uses a
    LIGHT budget (it only needs to detect feasibility), while the behaviour-cloning teacher
    uses a heavier budget (its topology is the actual training target)."""
    if regime.relay_hops > 1 or regime.scheduled_mac:
        return {
            "search_edge_ids": graph.edge_ids,
            "search_rng": random.Random(f"search:{scenario_id}"),
            "search_sa_iters": sa_iters,
            "search_restarts": restarts,
            "search_polish": polish,
        }
    return {}


def _measure_scene(
    scene: Scene3D, quorum_size: int, regime: PhysicsRegime, tau: float, *, vectorized: bool = False
) -> dict[str, float]:
    from marl_topology.data.stage21_objective_stack_evidence import (
        Stage21ObjectiveStackEvaluator,
    )

    graph = CandidateGraph.from_scene(scene, max_distance_m=None)
    config = build_stack_config(regime)
    evaluator = Stage21ObjectiveStackEvaluator(scene=scene, graph=graph, config=config)
    if vectorized:
        # Bounded-cache, ~6x-faster, float-identical evaluator -- this is the dominant cost on the
        # generation path (the SA family-binning search runs it thousands of times per scene), so the
        # vectorized one is what makes N>=24 builds tractable (canonical OOMs/thrashes).
        from marl_topology.data.vectorized_objective_stack_evaluator import (
            VectorizedStage21Evaluator,
        )
        evaluator = VectorizedStage21Evaluator(scene=scene, graph=graph, config=config, ref=evaluator)
    link_records = evaluator.link_records

    from marl_topology.budgets import node_budgets_for_scene

    candidates = enumerate_candidate_topologies(graph, link_records, quorum_size)
    best = best_feasible_topology(
        evaluator, candidates, tau, node_budgets=node_budgets_for_scene(scene),
        # LIGHT SA budget: family-binning only needs to detect that a feasible relay
        # topology exists, and this runs on every generation attempt (incl. rejections), so
        # skip the expensive hill-climb polish and use few restarts.
        **relay_aware_search_kwargs(
            regime, graph, scene.scenario_id, sa_iters=40, restarts=1, polish=False
        ),
    )

    full = PolicyBaselines.full(graph)
    full_eval = evaluator.evaluate(set(full.edge_ids))
    full_p = float(full_eval.metrics["consensus_success_probability"])
    full_e = float(full_eval.metrics["energy"])

    return {
        "candidate_edge_count": float(len(graph.edge_ids)),
        "full_graph_psucc": full_p,
        "full_graph_energy_j": full_e,
        "full_graph_feasible": float(full_p >= tau),
        "best_feasible_psucc": float(best["psucc"]),
        "best_feasible_energy_j": float(best["energy"]),
        "best_feasible_edge_count": float(best["edge_count"]),
        "feasible_exists": float(best["feasible_exists"]),
        # The sparsification signal: the best feasible topology is strictly
        # sparser than the (interference-prone) full graph.
        "sparse_beats_full_energy": float(
            best["feasible_exists"] and best["edge_count"] < len(full.edge_ids)
        ),
    }


def _family_of_measurement(measure: Mapping[str, float], requested_family: str) -> str:
    """Re-label a scene by its *measured* feasibility, not just intent."""

    if measure["feasible_exists"] < 0.5:
        return "infeasible"
    # feasible: distinguish a comfortable margin from a near-threshold one
    if requested_family == "near_threshold":
        return "near_threshold"
    return "feasible_sparse"


def _sample_urban_scene(
    rng: random.Random,
    scenario_id: str,
    node_count: int,
    config: "ProductionScenarioConfig",
) -> Scene3D:
    """One city-grid scene: 1 RSU + (node_count-1) vehicles placed on the street grid, with
    real building NLOS. The realized feasibility family emerges from placement (a vehicle on
    a street with no RSU line-of-sight is hard to reach), then _measure_scene bins it."""
    rsu_count = max(1, getattr(config, "urban_rsu_count", 1))
    grid = UrbanGridConfig(
        blocks_per_side=config.urban_blocks_per_side,
        block_size_m=config.urban_block_size_m,
        street_width_m=config.urban_street_width_m,
        rsu_count=rsu_count,
        vehicle_count=max(1, node_count - rsu_count),
    )
    return build_urban_grid_scene(scenario_id, grid, rng)


# --------------------------------------------------------------------------- #
# Public generation entry point.
# --------------------------------------------------------------------------- #
def generate_production_scenarios(
    config: ProductionScenarioConfig | None = None,
    *,
    vectorized: bool = False,
) -> tuple[ProductionScenarioSpec, ...]:
    """Generate a deterministic production scenario set with a measured gradient."""

    config = config or ProductionScenarioConfig()
    rng = random.Random(config.seed)
    reliable_range_m = measure_reliable_range_m(config.regime)

    targets = {
        "feasible_sparse": round(config.target_feasible_fraction * config.scenario_count),
        "near_threshold": round(config.target_near_threshold_fraction * config.scenario_count),
        "infeasible": round(config.target_infeasible_fraction * config.scenario_count),
    }
    # Fix rounding so the bins sum exactly to scenario_count.
    drift = config.scenario_count - sum(targets.values())
    targets["feasible_sparse"] += drift

    counts = {family: 0 for family in targets}
    specs: list[ProductionScenarioSpec] = []
    attempt = 0
    # Hard global attempt cap (terminator for low-yield regimes -- see max_total_attempts). On the
    # default high-yield paths the loop finishes in ~scenario_count attempts, far below the cap, so
    # behavior is byte-identical; the cap only fires on the pathological urban N>=24 thrash path.
    attempt_cap = config.max_total_attempts or (
        config.scenario_count * config.max_attempts_per_scenario
    )
    requested_cycle = ("feasible_sparse", "infeasible", "near_threshold")

    while len(specs) < config.scenario_count and attempt < attempt_cap:
        # Choose which family still needs scenarios (round-robin over deficits).
        wanted = [f for f in requested_cycle if counts[f] < targets[f]]
        if not wanted:
            wanted = [f for f in requested_cycle if counts[f] < targets[f] + 1]
        requested_family = wanted[attempt % len(wanted)]
        node_count = rng.choice(config.node_count_choices)
        quorum_size = _quorum_for_node_count(node_count)
        scenario_id = f"stage31_proc_{len(specs):05d}_{requested_family}"
        if config.urban_mode:
            scene = _sample_urban_scene(rng, scenario_id, node_count, config)
        else:
            scene = _sample_scene(
                rng, scenario_id, node_count, requested_family, reliable_range_m
            )
        measure = _measure_scene(
            scene, quorum_size, config.regime, config.tau_requirement_min, vectorized=vectorized
        )
        realized_family = _family_of_measurement(measure, requested_family)

        attempt += 1
        # Accept if this realized family still has deficit, else keep trying a
        # bounded number of times before accepting anyway (avoid infinite loops).
        accept = counts[realized_family] < targets[realized_family]
        force = attempt % config.max_attempts_per_scenario == 0
        if not (accept or force):
            continue

        spec = ProductionScenarioSpec(
            scenario_id=scenario_id,
            scene=scene,
            quorum_size=quorum_size,
            regime=config.regime,
            family=realized_family,
            reliable_range_m=reliable_range_m,
            full_graph_psucc=measure["full_graph_psucc"],
            full_graph_feasible=bool(measure["full_graph_feasible"]),
            best_feasible_psucc=measure["best_feasible_psucc"],
            best_feasible_energy_j=measure["best_feasible_energy_j"],
            best_feasible_edge_count=int(measure["best_feasible_edge_count"]),
            full_graph_energy_j=measure["full_graph_energy_j"],
            feasible_exists=bool(measure["feasible_exists"]),
            sparse_beats_full_energy=bool(measure["sparse_beats_full_energy"]),
            candidate_edge_count=int(measure["candidate_edge_count"]),
        )
        specs.append(spec)
        counts[realized_family] += 1

    return tuple(specs)


def _quorum_for_node_count(node_count: int) -> int:
    """PBFT quorum 2f+1, with f matching the Stage 21 evaluator (f capped at 1)."""

    fault_tolerance = min(1, (node_count - 1) // 3)
    return max(2, 2 * fault_tolerance + 1)


# --------------------------------------------------------------------------- #
# Trajectory generation (Part A2): the same scene evolved over time, so a rollout
# step becomes a real time step and a temporal actor has a link-evolution signal.
# Additive: does not touch the static-scene generator above.
# --------------------------------------------------------------------------- #
def _sample_vehicle_motions(
    rng: random.Random,
    scene: Scene3D,
    speed_min_mps: float,
    speed_max_mps: float,
) -> tuple[NodeMotion, ...]:
    """Constant-velocity motion for each VEHICLE (random heading in the x-y plane).

    RSUs / base stations get no entry, so ``advance_scene`` keeps them fixed.
    """

    if not 0.0 <= speed_min_mps <= speed_max_mps:
        raise ProceduralGeneratorViolation("require 0 <= speed_min_mps <= speed_max_mps")
    motions: list[NodeMotion] = []
    for node in scene.nodes:
        if node.kind is not NodeKind.VEHICLE:
            continue
        speed = rng.uniform(speed_min_mps, speed_max_mps)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        motions.append(
            NodeMotion(
                node.node_id,
                (speed * math.cos(heading), speed * math.sin(heading), 0.0),
            )
        )
    return tuple(motions)


def roll_trajectory(
    initial_scene: Scene3D,
    motions: Sequence[NodeMotion],
    regime: PhysicsRegime,
    quorum_size: int,
    tau: float,
    *,
    num_frames: int,
    dt_s: float,
) -> ProductionTrajectorySpec:
    """Advance ``initial_scene`` ``num_frames`` steps under ``motions`` and measure
    each frame on the real Stage 21 objective stack. Frame 0 is the initial scene.
    """

    if num_frames < 1:
        raise ProceduralGeneratorViolation("num_frames must be >= 1")
    motion_map = {motion.node_id: motion for motion in motions}
    frames: list[TrajectoryFrame] = []
    scene = initial_scene
    for time_index in range(num_frames):
        if time_index > 0:
            scene = advance_scene(scene, motion_map, dt_s)
        measurement = _measure_scene(scene, quorum_size, regime, tau)
        frames.append(
            TrajectoryFrame(
                time_index=time_index,
                time_s=round(time_index * dt_s, 9),
                scene=scene,
                measurement=dict(measurement),
            )
        )
    return ProductionTrajectorySpec(
        sequence_id=str(initial_scene.scenario_id),
        regime=regime,
        quorum_size=quorum_size,
        dt_s=dt_s,
        motions=tuple(motions),
        frames=tuple(frames),
    )


def generate_production_trajectories(
    config: ProductionScenarioConfig | None = None,
    *,
    num_frames: int = 8,
    dt_s: float = 1.0,
    speed_min_mps: float = 5.0,
    speed_max_mps: float = 15.0,
) -> tuple[ProductionTrajectorySpec, ...]:
    """Generate a deterministic set of moving-vehicle trajectories.

    Each trajectory is a sampled scenario (reusing the static generator's geometry)
    advanced ``num_frames`` time steps; vehicles move at constant velocity so links
    evolve predictably across the window. Defaults (speed 5-15 m/s, dt 1 s, 8 frames)
    move a vehicle ~40-120 m over the window -- a meaningful fraction of the ~125 m
    reliable range, so some links cross the tau boundary within a trajectory.
    """

    config = config or ProductionScenarioConfig()
    if num_frames < 1:
        raise ProceduralGeneratorViolation("num_frames must be >= 1")
    rng = random.Random(config.seed)
    reliable_range_m = measure_reliable_range_m(config.regime)
    tau = config.tau_requirement_min
    requested_cycle = ("feasible_sparse", "infeasible", "near_threshold")

    trajectories: list[ProductionTrajectorySpec] = []
    for index in range(config.scenario_count):
        family = requested_cycle[index % len(requested_cycle)]
        node_count = rng.choice(config.node_count_choices)
        quorum_size = _quorum_for_node_count(node_count)
        scene = _sample_scene(
            rng, f"stage31_traj_{index:05d}_{family}", node_count, family, reliable_range_m
        )
        motions = _sample_vehicle_motions(rng, scene, speed_min_mps, speed_max_mps)
        trajectories.append(
            roll_trajectory(
                scene,
                motions,
                config.regime,
                quorum_size,
                tau,
                num_frames=num_frames,
                dt_s=dt_s,
            )
        )
    return tuple(trajectories)


def scenario_fixture_from_spec(spec: ProductionScenarioSpec):
    """Build a Stage 2.5 ``ScenarioFixture`` from a generated spec.

    The fixture carries the scene/quorum/measured-feasibility for the existing
    context pipeline. ``link_reference_distance_m`` is a nominal positive value
    only; the production stack derives reliability from the finite-blocklength
    channel/link config, not from this field.
    """

    from marl_topology.scenario.fixtures import ScenarioFixture

    expected_status = "feasible" if spec.feasible_exists else "infeasible"
    return ScenarioFixture(
        fixture_id=spec.scenario_id,
        description=f"Stage 31 procedural scenario ({spec.family}).",
        scene=spec.scene,
        max_candidate_distance_m=max(1.0, spec.reliable_range_m * 3.0),
        quorum_size=spec.quorum_size,
        # The fixture's success_probability_threshold is a legacy-evaluator field unused by
        # the Stage 21 tau path; default it when the regime drops target_reliability (a perf
        # option: with a fixed transmission time the URLLC solver result is unused).
        success_probability_threshold=(
            spec.regime.target_reliability if spec.regime.target_reliability is not None else 0.9
        ),
        deadline_s=spec.regime.deadline_s,
        link_reference_distance_m=max(1.0, spec.reliable_range_m),
        expected_oracle_status=expected_status,
        expected_full_graph_success=spec.full_graph_feasible,
    )


def summarize_feasibility_distribution(
    specs: Sequence[ProductionScenarioSpec],
) -> dict[str, object]:
    """Aggregate the realized feasibility gradient for acceptance checks."""

    total = len(specs)
    by_family: dict[str, int] = {}
    feasible = 0
    sparse_better = 0
    full_feasible = 0
    for spec in specs:
        by_family[spec.family] = by_family.get(spec.family, 0) + 1
        feasible += int(spec.feasible_exists)
        sparse_better += int(spec.sparse_beats_full_energy)
        full_feasible += int(spec.full_graph_feasible)
    node_counts = sorted({len(spec.scene.nodes) for spec in specs})
    return {
        "scenario_count": total,
        "by_family": by_family,
        "feasible_fraction": feasible / total if total else 0.0,
        "infeasible_fraction": (total - feasible) / total if total else 0.0,
        "sparse_beats_full_fraction": sparse_better / total if total else 0.0,
        "full_graph_feasible_fraction": full_feasible / total if total else 0.0,
        "reliable_range_m": specs[0].reliable_range_m if specs else 0.0,
        "node_counts_present": node_counts,
        "unique_scenarios": len({spec.scenario_id for spec in specs}),
    }
