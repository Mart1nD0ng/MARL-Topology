"""Small deterministic 3D geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True, slots=True)
class Point3D:
    """A point in meters in the scenario coordinate system."""

    x_m: float
    y_m: float
    z_m: float = 0.0

    def distance_to(self, other: "Point3D") -> float:
        return distance_3d(self, other)


@dataclass(frozen=True, slots=True)
class BuildingBox:
    """Axis-aligned building volume in meters."""

    building_id: str
    min_corner: Point3D
    max_corner: Point3D
    material_class: str = "generic"

    def __post_init__(self) -> None:
        if not self.building_id:
            raise ValueError("building_id must be non-empty")
        if not self.material_class:
            raise ValueError("material_class must be non-empty")
        if self.min_corner.x_m > self.max_corner.x_m:
            raise ValueError("building min x must be <= max x")
        if self.min_corner.y_m > self.max_corner.y_m:
            raise ValueError("building min y must be <= max y")
        if self.min_corner.z_m > self.max_corner.z_m:
            raise ValueError("building min z must be <= max z")

    @classmethod
    def from_footprint_height(
        cls,
        building_id: str,
        min_x_m: float,
        max_x_m: float,
        min_y_m: float,
        max_y_m: float,
        height_m: float,
        base_z_m: float = 0.0,
        material_class: str = "generic",
    ) -> "BuildingBox":
        if height_m < 0:
            raise ValueError("height_m must be nonnegative")
        return cls(
            building_id=building_id,
            min_corner=Point3D(min_x_m, min_y_m, base_z_m),
            max_corner=Point3D(max_x_m, max_y_m, base_z_m + height_m),
            material_class=material_class,
        )

    @property
    def height_m(self) -> float:
        return self.max_corner.z_m - self.min_corner.z_m


def distance_3d(a: Point3D, b: Point3D) -> float:
    """Return Euclidean 3D distance in meters."""

    dx = a.x_m - b.x_m
    dy = a.y_m - b.y_m
    dz = a.z_m - b.z_m
    return sqrt(dx * dx + dy * dy + dz * dz)
