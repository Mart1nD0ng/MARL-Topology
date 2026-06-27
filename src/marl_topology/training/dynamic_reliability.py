"""D11: episode-level reliability constraints / risk objectives for the dynamic (T>1) task.

Thin EPISODE-level wrappers over the verified static Phase-10 primitives
(``training/reliability_constraints.py``), applied to per-FRAME consensus across an episode batch:

  - CHANCE (Spec S6.2): the frame-average failure rate ``Pr(C_t < tau) <= delta``. The sign-flexible
    dual ``lam_chance`` ASCENDS when the failure rate exceeds ``delta`` and FALLS when it is met; the
    reward gains a per-frame penalty ``-lam_chance * 1[C_t < tau]``. No extra evaluator call (the frame
    feasibility flag is already produced by the reward evaluator).
  - CVaR (Spec S6.3): the tail mean of the per-frame reliability shortfalls ``D_t = (tau - C_t)_+`` --
    the mean of the worst ``(1-alpha)`` fraction. A reported risk METRIC (no extra evaluator call: the
    shortfalls are the recorded consensus margins).
  - PARETO (Spec S6.4): select the deployed checkpoint from the VALIDATION archive by reliability-risk
    -> min violation -> energy-latency non-dominated, NEVER by raw feasibility alone.
  - PREFERENCE: an energy/latency-weighted objective ``base - (omega_E * E~ + omega_L * L~)`` so a
    preference over the (energy, latency) trade-off changes which topology is reward-best.

Training-only (these shape the training reward / checkpoint, never the deployed actor). Pure functions.
"""

from __future__ import annotations

from marl_topology.training.reliability_constraints import (
    chance_dual_update,
    chance_residual,
    pareto_archive_select,
)


def episode_chance_dual_step(lam, feasible_flags, *, delta, lr, lam_max=float("inf")):
    """One sign-flexible chance-dual step over an episode batch's per-frame feasibility flags
    (``feasible_flags[t]`` = ``C_t >= tau``). Returns ``(new_lam, residual)`` with
    ``residual = Pr(C_t < tau) - delta`` (signed): ``lam`` RISES when the failure rate exceeds
    ``delta``, FALLS (toward 0) when it is met. Empty batch -> no change, residual 0."""
    if not feasible_flags:
        return float(lam), 0.0
    # encode each frame as a pseudo-consensus c=1.0 (>= tau) if feasible, c=0.0 (< tau) if not, with
    # tau=0.5 -> chance_residual counts exactly the below-tau fraction (reuses the verified primitive).
    pseudo = [1.0 if f else 0.0 for f in feasible_flags]
    residual = chance_residual(pseudo, tau=0.5, delta=delta)
    return chance_dual_update(float(lam), residual, lr, lam_max), residual


def episode_cvar_shortfall(margins, *, alpha):
    """``CVaR_alpha`` of the per-frame reliability shortfalls ``D_t = (tau - C_t)_+ >= 0`` (Spec S6.3),
    via the exact empirical Rockafellar-Uryasev minimum ``min_nu [nu + 1/(1-alpha) * mean((D-nu)_+)]``
    over the observed shortfalls. ``alpha`` in [0,1); larger = more extreme tail. Empty -> 0.0."""
    d = [max(0.0, float(m)) for m in margins]
    if not d:
        return 0.0
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    n = len(d)
    inv = 1.0 / (1.0 - alpha)
    return min(nu + inv * (sum(max(0.0, di - nu) for di in d) / n) for nu in d)


def cvar_bruteforce(margins, *, alpha):
    """Independent oracle for :func:`episode_cvar_shortfall`: the mean of the worst
    ``ceil((1-alpha)*n)`` shortfalls. Equals CVaR exactly when ``(1-alpha)*n`` is integral (test oracle)."""
    import math
    d = sorted(max(0.0, float(m)) for m in margins)
    if not d:
        return 0.0
    k = max(1, math.ceil((1.0 - alpha) * len(d)))
    return sum(d[-k:]) / k


def dynamic_pareto_select(val_archive, *, risk_budget=0.0):
    """Select the deployed checkpoint from the VALIDATION archive (Spec S6.4): reliability-risk
    satisfied -> min reliability_violation -> energy-latency non-dominated -> hypervolume -> stability.
    NEVER by raw feasibility. Each entry must carry ``split == "val"`` (held/train must NOT seed the
    archive -- Contract v3 §3.4). Returns the chosen entry (or None for an empty archive)."""
    val_only = [e for e in val_archive if e.get("split") == "val"]
    if val_archive and not val_only:
        raise ValueError("the Pareto archive must be seeded ONLY by validation checkpoints (Contract §3.4)")
    return pareto_archive_select(val_only, risk_budget=risk_budget)


def preference_weighted_objective(base_r, energy_norm, latency_norm, *, omega_e, omega_l):
    """An energy/latency-weighted per-frame objective ``base_r - (omega_E * E~ + omega_L * L~)`` (the
    preference over the energy-latency trade-off). With ``omega_E = omega_L = 0`` it is the plain
    feasibility objective (byte-identical); a nonzero preference penalizes energy/latency, so the
    reward-best topology shifts toward cheaper ones."""
    return float(base_r) - (float(omega_e) * float(energy_norm) + float(omega_l) * float(latency_norm))
