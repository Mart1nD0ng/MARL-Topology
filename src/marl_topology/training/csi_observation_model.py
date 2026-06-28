"""Q1 (POMDP-QP-FAR): stale / partial CSI observation model.

The dynamic actor decides on an OBSERVED channel ``g_hat_t`` that is stale and/or partially probed,
while the reward / evaluator / critic keep using the TRUE current channel ``g_t`` (Spec S4, S4.6). This
turns the ~Markov dynamic task into a genuine POMDP: history is needed to estimate the current true
channel from the lagged/masked observation (Spec S5). This module is a PURE, deterministic plan: given
``(edge_id, t)`` it returns which past frame the actor is allowed to observe, the resulting age, and
whether the edge was probed this frame. The DynamicScene does the actual value lookup from the cached
true per-frame contexts and never lets the observation reach the reward path.

Modes (Spec S4.2-S4.3):
    current        -> g_hat = g_t                 (inactive; byte-identical to HEAD)
    delay          -> g_hat = g_{t-delta}         (every frame probed, fixed lag)
    partial        -> probe ~ Bernoulli(prob); unprobed frames HOLD the last observed value, age grows
    delay_partial  -> both: a probe returns g_{t'-delta}, held until the next probe

The probe mask is a STABLE, seeded, deterministic function of ``(edge_id, frame, seed)`` (hashlib, not
the process-randomized ``hash``), so the whole plan is reproducible and stateless. Frame 0 is always
probed (the initial measurement, Spec S4.2: ``t-delta < 0`` falls back to frame 0).

NOTE (hard invariant, Spec S4.6 / Contract D1): this changes ONLY the actor observation. It does not
touch ``context.evaluator`` (the closed-form whole-network quorum-tail on the true current channel), so
reliability / energy / latency / reward all stay on ``g_t``. The centralized critic's regression TARGET
is the true-channel reward (so it too is on ``g_t``); its INPUT features are the same observation tensor
as the actor -- i.e. the Q1 critic reads the STALE CSI. Spec S4.6 PERMITS (but does not require) the
critic to read true current CSI as a training-only signal; wiring a true-CSI critic is a deferred CTDE
option (Q2+), not done here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

# Channel-CSI columns of the graph_payload edge features (data/graph_payload.py L61-71):
# 0,1 = link_success_probability, 2 = latency_s, 3 = energy_j. Cols 4 (prev-topo), 5 (distance),
# 6,7 (roles) are NOT channel CSI and are never staleified.
DEFAULT_CSI_COLUMNS: tuple[int, ...] = (0, 1, 2, 3)

_VALID_MODES = ("current", "delay", "partial", "delay_partial")


@dataclass(frozen=True, slots=True)
class CsiObservationModel:
    """A deterministic stale/partial CSI observation plan (Spec S4)."""

    mode: str = "current"
    delay_frames: int = 0
    probe_probability: float = 1.0
    noise_std: float = 0.0
    seed: int = 0
    csi_columns: tuple[int, ...] = DEFAULT_CSI_COLUMNS

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")
        if not 0.0 <= self.probe_probability <= 1.0:
            raise ValueError("probe_probability must be in [0, 1]")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        # Normalise mode-implied parameters so is_active() and the plan are unambiguous.
        if self.mode == "current":
            object.__setattr__(self, "delay_frames", 0)
            object.__setattr__(self, "probe_probability", 1.0)
        elif self.mode == "delay":
            object.__setattr__(self, "probe_probability", 1.0)
        elif self.mode == "partial":
            object.__setattr__(self, "delay_frames", 0)

    # -- activation ---------------------------------------------------------------------------------
    def is_active(self) -> bool:
        """True iff the model actually alters the observation. ``current`` (or a degenerate delay=0,
        prob>=1, noise=0 config) is INACTIVE -> the observation is byte-identical to HEAD."""
        if self.mode == "current":
            return False
        return self.delay_frames > 0 or self.probe_probability < 1.0 or self.noise_std > 0.0

    # -- probe mask (deterministic) -----------------------------------------------------------------
    def probe(self, edge_id: str, frame: int) -> bool:
        """Whether edge ``edge_id`` is freshly probed at ``frame``. Frame 0 is always probed; for
        ``current``/``delay`` every frame is probed; for partial modes it is a stable Bernoulli draw."""
        if frame <= 0:
            return True
        if self.probe_probability >= 1.0:
            return True
        if self.probe_probability <= 0.0:
            return False
        return self._uniform(edge_id, frame) < self.probe_probability

    def last_probe_frame(self, edge_id: str, t: int) -> int:
        """The most recent frame ``t' <= t`` at which ``edge_id`` was probed (>= 0; frame 0 is a floor)."""
        for frame in range(t, 0, -1):
            if self.probe(edge_id, frame):
                return frame
        return 0

    def edge_plan(self, edge_id: str, t: int) -> tuple[int, int, bool]:
        """Return ``(source_frame, age, observed_now)`` for ``edge_id`` at frame ``t``.

        ``source_frame`` is the (cached, true) past frame whose channel the actor may observe;
        ``age = t - source_frame`` is the real staleness in frames; ``observed_now`` is the probe flag.
        Unified across modes: ``src = max(0, last_probe(t) - delay)``.
        """
        if t < 0:
            raise ValueError("t must be >= 0")
        probed = self.probe(edge_id, t)
        last = self.last_probe_frame(edge_id, t)
        src = max(0, last - self.delay_frames)
        src = min(src, t)
        return src, t - src, probed

    # -- optional logit-domain noise (Spec S4.4; default off in Q1) ----------------------------------
    def apply_noise(self, probability: float, edge_id: str, t: int) -> float:
        """Add seeded Gaussian noise in the logit domain to a probability-valued CSI (Spec S4.4).
        A no-op when ``noise_std == 0`` (the Q1 default). Deterministic given (edge_id, t, seed)."""
        if self.noise_std <= 0.0:
            return probability
        p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        z = self._gaussian(edge_id, t)
        logit = math.log(p / (1.0 - p)) + self.noise_std * z
        return 1.0 / (1.0 + math.exp(-logit))

    # -- deterministic hashing ----------------------------------------------------------------------
    def _uniform(self, edge_id: str, frame: int) -> float:
        """A stable U[0,1) keyed by (seed, edge_id, frame) -- reproducible across processes."""
        digest = hashlib.sha256(f"probe|{self.seed}|{edge_id}|{frame}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def _gaussian(self, edge_id: str, frame: int) -> float:
        """A stable standard-normal keyed by (seed, edge_id, frame) via Box-Muller on two stable U."""
        d = hashlib.sha256(f"noise|{self.seed}|{edge_id}|{frame}".encode("utf-8")).digest()
        u1 = max(int.from_bytes(d[:8], "big") / float(1 << 64), 1e-12)
        u2 = int.from_bytes(d[8:16], "big") / float(1 << 64)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # -- diagnostics record -------------------------------------------------------------------------
    def manifest(self) -> dict:
        return {
            "mode": self.mode, "delay_frames": int(self.delay_frames),
            "probe_probability": float(self.probe_probability), "noise_std": float(self.noise_std),
            "seed": int(self.seed), "csi_columns": list(self.csi_columns), "is_active": self.is_active(),
        }


def build_csi_observation_model(args) -> CsiObservationModel | None:
    """Build a model from CLI args (``--csi-mode/--csi-delay/--csi-probe-prob/--csi-seed``). Returns
    None when no stale-CSI experiment is requested (default ``current``) -> byte-identical HEAD path."""
    mode = str(getattr(args, "csi_mode", "current"))
    if mode == "current":
        return None
    return CsiObservationModel(
        mode=mode,
        delay_frames=int(getattr(args, "csi_delay", 1)),
        probe_probability=float(getattr(args, "csi_probe_prob", 1.0)),
        noise_std=float(getattr(args, "csi_noise_std", 0.0)),
        seed=int(getattr(args, "csi_seed", getattr(args, "seed", 0))),
    )
