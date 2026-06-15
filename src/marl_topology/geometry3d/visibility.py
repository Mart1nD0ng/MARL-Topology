"""Deterministic 3D visibility checks for Stage 3.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .primitives import BuildingBox, Point3D, distance_3d

if TYPE_CHECKING:  # pragma: no cover
    from marl_topology.scenario import Scene3D


class LoSState(str, Enum):
    LOS = "los"
    NLOS = "nlos"


@dataclass(frozen=True, slots=True)
class RayBoxIntersection:
    building_id: str
    entry_fraction: float
    exit_fraction: float
    entry_distance_m: float
    exit_distance_m: float

    def __post_init__(self) -> None:
        if not self.building_id:
            raise ValueError("building_id must be non-empty")
        if not 0.0 <= self.entry_fraction <= self.exit_fraction <= 1.0:
            raise ValueError("intersection fractions must satisfy 0 <= entry <= exit <= 1")
        if self.entry_distance_m < 0 or self.exit_distance_m < 0:
            raise ValueError("intersection distances must be nonnegative")
        if self.entry_distance_m > self.exit_distance_m:
            raise ValueError("entry_distance_m must be <= exit_distance_m")


@dataclass(frozen=True, slots=True)
class BlockerRecord:
    scenario_id: str
    building_id: str
    entry_distance_m: float
    exit_distance_m: float
    material_class: str

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not self.building_id:
            raise ValueError("building_id must be non-empty")
        if self.entry_distance_m < 0 or self.exit_distance_m < 0:
            raise ValueError("blocker distances must be nonnegative")
        if self.entry_distance_m > self.exit_distance_m:
            raise ValueError("entry_distance_m must be <= exit_distance_m")
        if not self.material_class:
            raise ValueError("material_class must be non-empty")


@dataclass(frozen=True, slots=True)
class VisibilityRecord:
    scenario_id: str
    tx_id: str
    rx_id: str
    distance_3d_m: float
    los_state: LoSState
    blocker_records: tuple[BlockerRecord, ...]
    ray_start_m: Point3D
    ray_end_m: Point3D
    visibility_regime: str = "stage3_axis_aligned_boxes"

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if not self.tx_id or not self.rx_id:
            raise ValueError("tx_id and rx_id must be non-empty")
        if self.tx_id == self.rx_id:
            raise ValueError("tx_id and rx_id must differ")
        if self.distance_3d_m < 0:
            raise ValueError("distance_3d_m must be nonnegative")
        if self.los_state == LoSState.LOS and self.blocker_records:
            raise ValueError("LOS record cannot contain blockers")
        if self.los_state == LoSState.NLOS and not self.blocker_records:
            raise ValueError("NLOS record must contain at least one blocker")


def ray_box_intersection(
    start: Point3D,
    end: Point3D,
    box: BuildingBox,
    epsilon: float = 1e-12,
) -> RayBoxIntersection | None:
    """Return the segment intersection with an axis-aligned box, if any."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    dz = end.z_m - start.z_m
    total_distance_m = distance_3d(start, end)
    if total_distance_m == 0.0:
        raise ValueError("ray segment endpoints must differ")

    t_min = 0.0
    t_max = 1.0
    for start_value, delta, box_min, box_max in (
        (start.x_m, dx, box.min_corner.x_m, box.max_corner.x_m),
        (start.y_m, dy, box.min_corner.y_m, box.max_corner.y_m),
        (start.z_m, dz, box.min_corner.z_m, box.max_corner.z_m),
    ):
        if abs(delta) < epsilon:
            if start_value < box_min or start_value > box_max:
                return None
            continue

        t1 = (box_min - start_value) / delta
        t2 = (box_max - start_value) / delta
        entry = min(t1, t2)
        exit_ = max(t1, t2)
        t_min = max(t_min, entry)
        t_max = min(t_max, exit_)
        if t_min > t_max:
            return None

    entry_fraction = max(t_min, 0.0)
    exit_fraction = min(t_max, 1.0)
    if exit_fraction <= epsilon or entry_fraction >= 1.0 - epsilon:
        return None
    return RayBoxIntersection(
        building_id=box.building_id,
        entry_fraction=entry_fraction,
        exit_fraction=exit_fraction,
        entry_distance_m=entry_fraction * total_distance_m,
        exit_distance_m=exit_fraction * total_distance_m,
    )


def evaluate_visibility(scene: "Scene3D", tx_id: str, rx_id: str) -> VisibilityRecord:
    """Compute Stage 3.1 LoS/NLoS between two scene nodes."""

    tx = scene.get_node(tx_id)
    rx = scene.get_node(rx_id)
    distance_m = distance_3d(tx.position, rx.position)
    blockers: list[BlockerRecord] = []
    for building in scene.buildings:
        hit = ray_box_intersection(tx.position, rx.position, building)
        if hit is None:
            continue
        blockers.append(
            BlockerRecord(
                scenario_id=scene.scenario_id,
                building_id=building.building_id,
                entry_distance_m=hit.entry_distance_m,
                exit_distance_m=hit.exit_distance_m,
                material_class=building.material_class,
            )
        )

    blockers_tuple = tuple(
        sorted(blockers, key=lambda blocker: (blocker.entry_distance_m, blocker.building_id))
    )
    return VisibilityRecord(
        scenario_id=scene.scenario_id,
        tx_id=tx_id,
        rx_id=rx_id,
        distance_3d_m=distance_m,
        los_state=LoSState.NLOS if blockers_tuple else LoSState.LOS,
        blocker_records=blockers_tuple,
        ray_start_m=tx.position,
        ray_end_m=rx.position,
    )
