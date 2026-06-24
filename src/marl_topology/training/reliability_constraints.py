"""Phase 10 (Technical-Spec S5.4/S6.2/S6.3/S6.4): distribution-level reliability constraints.

The trunk's existing dual ascends ``lam_c`` on the MEAN consensus margin ``g_c = mean_s max(0, tau - c_s)``.
That bounds the average shortfall but not the FREQUENCY or the TAIL of reliability failures. Phase 10
adds the two Spec constraints (training-only, opt-in) plus the Pareto checkpoint selection:

  - CHANCE (S5.4/S6.2): ``J_C = E[1(C < tau)] <= delta`` -- bound the FRACTION of scenes below tau.
    Residual ``g_C = Pr(C<tau) - delta`` (signed); dual ``lam <- [lam + lr*g_C]_+`` (rises when the
    failure rate exceeds delta, FALLS when it is met -- residual must be sign-flexible).
  - CVaR (S6.3): with shortfall ``D = tau - C``, ``CVaR_alpha(D) = min_nu [nu + 1/(1-alpha) E[(D-nu)_+]]``
    (Rockafellar-Uryasev) -- the mean of the worst ``(1-alpha)`` fraction of shortfalls (the TAIL).
  - PARETO ARCHIVE (S6.4): pick the deployed checkpoint by reliability-risk -> min violation ->
    energy-latency non-dominated -> constrained hypervolume -> stability; NEVER by raw feasibility alone.

All pure, evaluator-free functions over a batch's measured consensus probabilities ``c_s``.
"""

from __future__ import annotations


def chance_residual(c_values, tau: float, delta: float) -> float:
    """``g_C = Pr(C < tau) - delta`` (Spec S6.2): the empirical fraction of scenes below the reliability
    threshold ``tau`` minus the allowed failure rate ``delta``. SIGNED -- positive when the failure rate
    is violated, negative when it is met (so the dual can fall). Empty batch -> 0.0 (no signal)."""
    if not c_values:
        return 0.0
    frac_below = sum(1 for c in c_values if c < tau) / len(c_values)
    return frac_below - float(delta)


def chance_dual_update(lam: float, residual: float, lr: float, lam_max: float = float("inf")) -> float:
    """Projected dual ascent ``lam <- clip([lam + lr*residual]_+, 0, lam_max)`` (Spec S6.2). The
    residual is sign-flexible so ``lam`` RISES when the chance constraint is violated and FALLS (toward
    0) when it is satisfied -- never negative."""
    return min(max(0.0, lam + lr * residual), lam_max)


def cvar_shortfall(c_values, tau: float, alpha: float) -> float:
    """``CVaR_alpha(D)`` of the reliability shortfall ``D = tau - C`` (Spec S6.3), via the exact
    empirical Rockafellar-Uryasev minimum ``min_nu [nu + 1/(1-alpha) * mean((D-nu)_+)]``. For the
    empirical distribution the minimizer is an order statistic, so the min over the observed ``D`` values
    is exact; it equals the mean of the worst ``(1-alpha)`` fraction of shortfalls when ``(1-alpha)*n``
    is integral. ``alpha`` in [0,1); larger ``alpha`` = more extreme tail. Empty batch -> 0.0."""
    if not c_values:
        return 0.0
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    d = [tau - c for c in c_values]
    n = len(d)
    inv = 1.0 / (1.0 - alpha)
    # the Rockafellar minimum over nu is attained at one of the order statistics of D
    return min(nu + inv * (sum(max(0.0, di - nu) for di in d) / n) for nu in d)


def tail_mean_shortfall(c_values, tau: float, alpha: float) -> float:
    """Independent ground truth for :func:`cvar_shortfall`: the mean of the worst ``ceil((1-alpha)*n)``
    shortfalls ``D = tau - C`` (the largest deficits). Equals CVaR exactly when ``(1-alpha)*n`` is an
    integer. (Test oracle, not used in training.)"""
    if not c_values:
        return 0.0
    d = sorted(tau - c for c in c_values)          # ascending
    n = len(d)
    import math
    k = max(1, math.ceil((1.0 - alpha) * n))
    return sum(d[-k:]) / k


def _dominates(a, b) -> bool:
    """``a`` Pareto-dominates ``b`` on (energy, latency) -- both minimized -- iff a is <= on both and < on
    at least one."""
    ae, al = a["energy"], a["latency"]
    be, bl = b["energy"], b["latency"]
    return ae <= be and al <= bl and (ae < be or al < bl)


def pareto_archive_select(archive, *, risk_budget: float = 0.0):
    """Spec S6.4 checkpoint selection order over a validation archive of dicts with keys
    ``reliability_violation, energy, latency`` (+ optional ``hypervolume, stability, id``):

    1. reliability-risk SATISFIED (``reliability_violation <= risk_budget``);
    2. among those, MIN reliability_violation;
    3. among those, energy-latency NON-DOMINATED;
    4. max constrained hypervolume;
    5. stability tie-break.

    Falls back to the whole archive if none satisfy the risk budget (then steps 2-5). NEVER selects by
    raw feasibility alone. Returns the chosen entry (or None for an empty archive)."""
    if not archive:
        return None
    feasible = [e for e in archive if e["reliability_violation"] <= risk_budget]
    pool = feasible if feasible else list(archive)                 # (1) reliability-risk satisfied
    min_v = min(e["reliability_violation"] for e in pool)          # (2) min violation
    pool = [e for e in pool if e["reliability_violation"] <= min_v + 1e-12]
    nd = [e for e in pool if not any(_dominates(o, e) for o in pool if o is not e)]  # (3) non-dominated
    return max(nd, key=lambda e: (e.get("hypervolume", 0.0), e.get("stability", 0.0)))  # (4),(5)
