"""Q9 (POMDP-QP-FAR): potential-based reward shaping (PBRS) with the quorum-deficit potential.

Spec §9. The shaping term ``F_t = lam_phi * (gamma * Phi_{t+1} - Phi_t)`` with ``Phi = -eta * D_quorum``
and a TERMINAL ``Phi_T = 0`` is a Ng-Harada-Russell potential-based shaping: over any episode the
discounted shaping sum TELESCOPES to ``-lam_phi * Phi_0 + lam_phi * gamma^T * Phi_T = -lam_phi * Phi_0``
(constant in the initial state), so the optimal policy is PROVABLY UNCHANGED -- the shaping can only help
learning, never alter the true objective. ``D_quorum`` is the Q4-authorized AUXILIARY; it enters the
TRAINING reward ONLY as this optimum-preserving shaping. The final EVALUATION uses NO shaping (true C/E/L).

PBRS is only valid in the T>1 dynamic task (it needs the state transition Phi(s_{t+1})-Phi(s_t)); it is
meaningless in the single-step T=1 bandit and must not be claimed there (Spec §16.4).
"""

from __future__ import annotations

from marl_topology.training.quorum_deficit_bridge import topology_quorum_deficit


def dquorum_potential(evaluator, topology, *, eta: float = 1.0) -> float:
    """The state potential ``Phi = -eta * D_quorum(topology)`` (training-only, via the bridge). Higher
    Phi = lower quorum deficit = closer to feasibility. An empty/degenerate topology (D undefined) -> 0."""
    d = topology_quorum_deficit(evaluator, topology)
    return -float(eta) * (float(d["d_quorum_mean"]) if d is not None else 0.0)


def pbrs_term(phi_t: float, phi_next: float, gamma: float, lam_phi: float) -> float:
    """The Ng-Harada potential-based shaping term ``F_t = lam_phi * (gamma * Phi_{t+1} - Phi_t)``."""
    return float(lam_phi) * (float(gamma) * float(phi_next) - float(phi_t))


def episode_pbrs(potentials, gamma: float, lam_phi: float) -> list[float]:
    """Per-frame shaping ``[F_0, ..., F_{T-1}]`` over an episode whose state potentials are
    ``potentials = [Phi_0, ..., Phi_{T-1}]``. The TERMINAL potential ``Phi_T = 0`` (Spec §9.1, eta_T=0),
    so ``F_{T-1} = lam_phi * (gamma * 0 - Phi_{T-1})``. The shaped reward is ``r'_t = r_t + F_t``."""
    n = len(potentials)
    out: list[float] = []
    for t in range(n):
        phi_t = float(potentials[t])
        phi_next = float(potentials[t + 1]) if t + 1 < n else 0.0      # terminal Phi_T = 0
        out.append(pbrs_term(phi_t, phi_next, gamma, lam_phi))
    return out


def discounted_shaping_sum(shaping, gamma: float) -> float:
    """``sum_t gamma^t F_t`` -- which for PBRS telescopes to ``-lam_phi * Phi_0`` (terminal Phi=0)."""
    total = 0.0
    disc = 1.0
    for f in shaping:
        total += disc * float(f)
        disc *= float(gamma)
    return total
