"""Split-isolated witness memory (Phase 3, Technical-Spec S5.3).

A witness is a concrete topology achieving some consensus reliability on a scene. The memory
keeps, per scene, the BEST witness found so far (monotone -- it only ever improves), which is
the scene's solvability lower bound.

Isolation rule (S5.3): train / validation / held-test memories must be kept separate, and
held/test discoveries must NEVER feed training (that would leak test feasibility into the
learner). ``merge_from`` enforces this: only a ``train`` memory may be merged into ``train``.
"""

from __future__ import annotations

from dataclasses import dataclass


WITNESS_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class Witness:
    """A concrete topology and its measured objectives on a scene."""

    topology: tuple[str, ...]
    reliability: float
    energy: float
    latency: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")
        if self.energy < 0.0:
            raise ValueError("energy must be nonnegative")
        if self.latency < 0.0:
            raise ValueError("latency must be nonnegative")


class WitnessMemory:
    """Per-split best-witness store, keyed by scene id."""

    def __init__(self, split: str) -> None:
        if split not in WITNESS_SPLITS:
            raise ValueError(f"split must be one of {WITNESS_SPLITS}")
        self._split = split
        self._best: dict[str, Witness] = {}

    @property
    def split(self) -> str:
        return self._split

    def update(self, scene_id: str, witness: Witness) -> bool:
        """Record ``witness`` if it strictly improves the scene's best reliability.

        Returns True iff it replaced the previous best (monotone, never regresses).
        """

        current = self._best.get(scene_id)
        if current is None or witness.reliability > current.reliability:
            self._best[scene_id] = witness
            return True
        return False

    def witness(self, scene_id: str) -> Witness | None:
        return self._best.get(scene_id)

    def lower_bound(self, scene_id: str) -> float:
        """The best witnessed consensus reliability for the scene (0.0 if none)."""

        witness = self._best.get(scene_id)
        return witness.reliability if witness is not None else 0.0

    def scene_ids(self) -> tuple[str, ...]:
        return tuple(self._best)

    def merge_from(self, other: "WitnessMemory") -> int:
        """Fold another memory's witnesses in (monotone). Returns the number improved.

        Enforces S5.3 isolation: a ``train`` memory may only absorb witnesses from another
        ``train`` memory -- never from ``val``/``test`` (that would leak held feasibility into
        training).
        """

        if self._split == "train" and other._split != "train":
            raise ValueError(
                "isolation: a train witness memory must not absorb held/test discoveries "
                f"(attempted merge from split={other._split!r})"
            )
        improved = 0
        for scene_id, witness in other._best.items():
            if self.update(scene_id, witness):
                improved += 1
        return improved

    def __len__(self) -> int:
        return len(self._best)
