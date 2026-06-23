"""Budget-Conditioned Unordered Subset Policy -- BCSP (Phase 6, Technical-Spec S7.2-7.10).

The production node action is an UNORDERED budget-capped subset ``S subseteq E_i, |S| <= b_i``:

    pi_i(S) = 1[|S| <= b_i] exp(sum_{e in S} theta_e) / Z_i,    theta_e = z_e / T.

This REPLACES the ordered Plackett-Luce action of ``decentralized_action.py``, which (1) wrongly
treated the ``k!`` permutations of one subset as distinct actions (a wrong entropy / exploration
signal: at ``k = m`` ordered entropy approaches ``log(m!)`` while the true subset entropy is 0), and
(2) enumerated ``perm(m, k)`` orderings -- factorial, the source of the ``m=15, b=64`` trunk hang.

Everything here is exact in ``O(m b)`` (and ``O(m)`` when ``b >= m``, the regime of the real
blocker), differentiable, and device-preserving:

  - log-partition  ``log Z`` via the cardinality DP ``A[j,r] = logaddexp(A[j-1,r], A[j-1,r-1]+theta_j)``;
  - subset log-prob ``log pi(S) = sum_{e in S} theta_e - log Z`` (order-IRRELEVANT, never enters PPO
    as an ordering);
  - exact sampling: draw the cardinality ``P(K=r) = exp(A[m,r] - log Z)`` then backward-sample edges;
  - exact entropy ``H = log Z - sum_e theta_e mu_e`` with ``mu_e = d log Z / d theta_e = P(e in S)``;
  - the ``b >= m`` fast path: ``log Z = sum_e softplus(theta_e)``, an independent Bernoulli ``sigma(theta_e)``;
  - the MAP action ``argmax_{|S|<=b} sum_{e in S} theta_e`` = the positive-``theta`` edges, top ``b`` --
    EXACTLY the deployed ``local_mutual_assemble`` decoder (train == deploy, D1).

Lives under ``training/`` (it carries autograd for ``logp``/``entropy``); the DEPLOYED path uses the
torch-free decoder. ``budget`` is the node's radio budget ``b_i``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

BCSP_MODEL_ID = "budget_conditioned_unordered_subset_policy_v1"
_NEG_INF = float("-inf")


def num_subset_actions(m: int, budget: int) -> int:
    """``|A_i| = sum_{r=0}^{min(b,m)} C(m, r)`` -- the count of legal budget-capped subsets."""
    b = min(int(budget), int(m))
    return sum(math.comb(int(m), r) for r in range(b + 1))


def log_partition(theta: Tensor, budget: int) -> Tensor:
    """``log Z = log sum_{|S| <= b} exp(sum_{e in S} theta_e)``. ``O(m)`` when ``b >= m`` else ``O(m b)``.

    The ``b < m`` path uses a GROWING-row DP: row ``r`` is materialized only once size-``r`` subsets
    are reachable, so every ``logaddexp`` combines finite (reachable) cells -- no ``logaddexp(-inf,
    -inf)`` ever occurs, which keeps the SECOND derivative (the entropy bonus needs ``d/dtheta`` of
    ``mu = d log Z / d theta``) finite. Autograd-safe and device-preserving.
    """
    m = int(theta.shape[-1])
    b = int(budget)
    if b < 0:
        raise ValueError("budget must be nonnegative")
    if b >= m:
        # all subsets legal -> Z = prod_e (1 + e^{theta_e}) -> log Z = sum_e softplus(theta_e). O(m).
        return torch.nn.functional.softplus(theta).sum(dim=-1)
    cap = b + 1  # the row never grows past r = b
    row = theta.new_zeros(1)  # prefix 0: only r = 0, weight log 1 = 0
    for j in range(m):
        tj = theta[j]
        length = int(row.shape[0])  # currently covers r = 0 .. length-1
        head = row[0:1]  # r = 0: exclude-only
        middle = torch.logaddexp(row[1:length], row[0:length - 1] + tj) if length >= 2 else row[0:0]
        if length < cap:  # the row grows: a new top cell r = length is include-only
            row = torch.cat([head, middle, row[length - 1:length] + tj])
        else:  # capped at r = b: the would-be include into r = b+1 is dropped
            row = torch.cat([head, middle])
    return torch.logsumexp(row, dim=0)


def _dp_table(theta: Tensor, budget: int) -> tuple[Tensor, Tensor]:
    """``(log Z, table)`` with ``table[j, r] = log sum_{|S|=r, S subseteq first j edges} exp(theta(S))``
    (shape ``[m+1, R+1]``, unreachable cells ``-inf``). For SAMPLING only -- the caller wraps it in
    ``no_grad`` so the ``-inf`` cells are harmless (no backward pass)."""
    m = int(theta.shape[-1])
    R = min(int(budget), m)
    rows = []
    row = theta.new_full((R + 1,), _NEG_INF)
    row[0] = 0.0
    rows.append(row)
    for j in range(m):
        tj = theta[j]
        include = torch.cat([theta.new_full((1,), _NEG_INF), row[:R] + tj])
        row = torch.logaddexp(row, include)
        rows.append(row)
    table = torch.stack(rows, dim=0)
    return torch.logsumexp(table[m], dim=-1), table


def subset_logp(theta: Tensor, subset_indices, budget: int) -> Tensor:
    """``log pi(S) = sum_{e in S} theta_e - log Z``. ``subset_indices`` is order-IRRELEVANT."""
    idx = torch.as_tensor(sorted(set(int(i) for i in subset_indices)), dtype=torch.long, device=theta.device)
    if idx.numel() > int(budget):
        return theta.new_full((), _NEG_INF)  # |S| > b -> outside the support
    theta_sum = theta[idx].sum() if idx.numel() else theta.new_zeros(())
    return theta_sum - log_partition(theta, budget)


def inclusion_marginals(theta: Tensor, budget: int) -> Tensor:
    """``mu_e = P(e in S) = d log Z / d theta_e`` (the per-edge inclusion marginal), via autograd."""
    if min(int(budget), int(theta.shape[-1])) == 0:
        return torch.zeros_like(theta)  # only the empty subset is legal -> no edge is ever included
    t = theta.detach().clone().requires_grad_(True)
    (mu,) = torch.autograd.grad(log_partition(t, budget), t)
    return mu.detach()


def subset_entropy(theta: Tensor, budget: int) -> Tensor:
    """Exact Shannon entropy ``H = log Z - sum_e theta_e mu_e``, ``mu_e = d log Z / d theta_e``.

    Differentiable w.r.t. ``theta`` (``create_graph`` retains the marginal graph) so it can serve as
    the PPO entropy bonus; returns a detached scalar when ``theta`` carries no grad.
    """
    if min(int(budget), int(theta.shape[-1])) == 0:
        # only the empty subset is legal (b=0 or m=0) -> H = 0 deterministically. Return a
        # theta-connected zero (zero gradient) so the entropy bonus stays differentiable; the b<m DP
        # would otherwise yield a graph-disconnected constant logZ and crash autograd.grad.
        return theta.sum() * 0.0
    if theta.requires_grad:
        logZ = log_partition(theta, budget)
        (mu,) = torch.autograd.grad(logZ, theta, create_graph=True)
        return logZ - (theta * mu).sum()
    t = theta.detach().clone().requires_grad_(True)
    logZ = log_partition(t, budget)
    (mu,) = torch.autograd.grad(logZ, t)
    return (logZ - (t * mu).sum()).detach()


def normalized_entropy(theta: Tensor, budget: int) -> Tensor:
    """``H / log|A_i|`` -- entropy normalized by the legal subset-action count (Spec S7.9), so the
    entropy coefficient is comparable across degree/budget. 0 when only one action is legal."""
    m = int(theta.shape[-1])
    h = subset_entropy(theta, budget)
    actions = num_subset_actions(m, budget)
    if actions <= 1:
        return h * 0.0
    return h / math.log(actions)


def map_subset(theta: Tensor, budget: int) -> list[int]:
    """``argmax_{|S| <= b} sum_{e in S} theta_e`` = the positive-``theta`` edges, top ``b`` (ties by
    index). EXACTLY the deployed ``local_mutual_assemble`` per-node rule (logit >= 0, local top-b)."""
    b = int(budget)
    positive = [(float(theta[i]), i) for i in range(int(theta.shape[-1])) if float(theta[i]) >= 0.0]
    positive.sort(key=lambda vi: (-vi[0], vi[1]))
    return sorted(i for _v, i in positive[:b])


def sample_subset(theta: Tensor, budget: int, *, generator: torch.Generator | None = None) -> list[int]:
    """Draw ``S ~ pi`` exactly: sample the cardinality ``K=r`` from ``P(K=r)=exp(A[m,r]-log Z)`` then
    backward-sample each edge with ``P(x_j=1 | r) = exp(theta_j + A[j-1,r-1] - A[j,r])``. Returns the
    sorted selected indices (the action; not differentiable -- score it with :func:`subset_logp`)."""
    m = int(theta.shape[-1])
    b = int(budget)
    with torch.no_grad():
        if b >= m:
            probs = torch.sigmoid(theta)
            draws = torch.rand(m, generator=generator, device=theta.device)
            return sorted((draws < probs).nonzero(as_tuple=False).flatten().tolist())
        logZ, table = _dp_table(theta, b)
        R = min(b, m)
        card_probs = (table[m] - logZ).exp().clamp_min(0.0)
        r = int(torch.multinomial(card_probs, 1, generator=generator).item())
        selected: list[int] = []
        for i in range(m - 1, -1, -1):
            if r == 0:
                break
            log_p1 = theta[i] + table[i][r - 1] - table[i + 1][r]
            p1 = float(log_p1.exp().clamp(0.0, 1.0))
            if float(torch.rand(1, generator=generator, device=theta.device).item()) < p1:
                selected.append(i)
                r -= 1
        return sorted(selected)
