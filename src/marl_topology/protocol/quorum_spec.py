"""Safe, configurable PBFT quorum specification (Phase 1, Technical-Spec S4.3).

The committee runs ``n`` validators tolerating ``f`` Byzantine faults. A *quorum*
of size ``q`` must satisfy two conditions for a BFT agreement protocol to be both
safe and live:

* **Quorum intersection (safety).** Any two quorums must share at least one honest
  node, i.e. their intersection ``2q - n`` must exceed ``f``::

      2 * q - n > f

* **Liveness.** The ``n - f`` honest nodes must by themselves be able to form a
  quorum::

      q <= n - f

Both are simultaneously satisfiable iff ``n >= 3f + 1``.

Two modes are supported:

* ``classic_exact`` — the textbook PBFT point ``n = 3f + 1``, ``q = 2f + 1``. It is
  an error to request this mode when ``n != 3f + 1``.
* ``safe_generalized`` — for an arbitrary committee ``n >= 3f + 1`` the smallest safe
  quorum ``q = floor((n + f) / 2) + 1``. This equals ``2f + 1`` exactly at
  ``n = 3f + 1`` (so classic fixtures are numerically unchanged) and is strictly
  larger when ``n > 3f + 1`` (where the naive ``2f + 1`` would be unsafe).

The cascade uses ``external_quorum = q - 1`` (matching messages required from the
*other* ``n - 1`` nodes; a node's own readiness is counted separately).
"""

from __future__ import annotations

from dataclasses import dataclass


QUORUM_MODE_CLASSIC = "classic_exact"
QUORUM_MODE_SAFE = "safe_generalized"
QUORUM_MODES = (QUORUM_MODE_CLASSIC, QUORUM_MODE_SAFE)


def safe_quorum_size(node_count: int, fault_tolerance: int) -> int:
    """Smallest quorum satisfying intersection + liveness for ``n >= 3f + 1``."""

    return (node_count + fault_tolerance) // 2 + 1


@dataclass(frozen=True, slots=True)
class PBFTQuorumSpec:
    """A validated PBFT quorum size for a given committee size and fault budget."""

    node_count: int
    fault_tolerance: int
    mode: str = QUORUM_MODE_SAFE

    def __post_init__(self) -> None:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int):
            raise TypeError("node_count must be an integer")
        if isinstance(self.fault_tolerance, bool) or not isinstance(self.fault_tolerance, int):
            raise TypeError("fault_tolerance must be an integer")
        if self.fault_tolerance < 0:
            raise ValueError("fault_tolerance must be nonnegative")
        if self.mode not in QUORUM_MODES:
            raise ValueError(f"mode must be one of {QUORUM_MODES}")
        if self.node_count < 3 * self.fault_tolerance + 1:
            raise ValueError("PBFT requires n >= 3f + 1")
        if self.mode == QUORUM_MODE_CLASSIC and self.node_count != 3 * self.fault_tolerance + 1:
            raise ValueError(
                "classic_exact requires n == 3f + 1; use safe_generalized for n > 3f + 1"
            )
        # Validate the chosen quorum against the safety + liveness conditions.
        quorum = self.quorum
        if not (2 * quorum - self.node_count > self.fault_tolerance):
            raise ValueError(
                "quorum violates intersection safety (2q - n > f): "
                f"n={self.node_count} f={self.fault_tolerance} q={quorum}"
            )
        if not (quorum <= self.node_count - self.fault_tolerance):
            raise ValueError(
                "quorum violates liveness (q <= n - f): "
                f"n={self.node_count} f={self.fault_tolerance} q={quorum}"
            )

    @property
    def quorum(self) -> int:
        """The agreement quorum size ``q`` (a.k.a. ``total_quorum``)."""

        if self.mode == QUORUM_MODE_CLASSIC:
            return 2 * self.fault_tolerance + 1
        return safe_quorum_size(self.node_count, self.fault_tolerance)

    @property
    def external_quorum(self) -> int:
        """Matching messages required from the other ``n - 1`` nodes (self separate)."""

        return self.quorum - 1

    @property
    def intersection_margin(self) -> int:
        """``2q - n - f`` — strictly positive honest overlap between any two quorums."""

        return 2 * self.quorum - self.node_count - self.fault_tolerance
