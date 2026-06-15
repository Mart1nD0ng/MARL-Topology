"""Minimal 3D geometry primitives for the clean skeleton."""

from .primitives import BuildingBox, Point3D, distance_3d
from .visibility import (
    BlockerRecord,
    LoSState,
    RayBoxIntersection,
    VisibilityRecord,
    evaluate_visibility,
    ray_box_intersection,
)

__all__ = [
    "BlockerRecord",
    "BuildingBox",
    "LoSState",
    "Point3D",
    "RayBoxIntersection",
    "VisibilityRecord",
    "distance_3d",
    "evaluate_visibility",
    "ray_box_intersection",
]
