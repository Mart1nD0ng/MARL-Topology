"""Phase 11a (Technical-Spec S7.12 "PNA aggregation"): Principal Neighbourhood Aggregation primitives.

PNA (Corso et al., 2020) replaces a single neighbour aggregator with the concatenation of MULTIPLE
aggregators, each multiplied by MULTIPLE degree scalers -- a strictly more expressive permutation-
invariant neighbourhood readout (provably distinguishes multisets a single aggregator cannot):

  aggregators  A = {mean, max, min, std}                        (4)
  scalers      S(d, alpha) = (log(1+d) / delta)^alpha,  alpha in {+1, 0, -1}   (amplify / identity /
               attenuate); delta = average of log(1+d) over the TRAINING degrees (a fixed scalar).

The PNA readout for a node is ``[ S(d,alpha) * agg(messages) ]`` over the 4x3 = 12 (aggregator, scaler)
pairs, concatenated. These are pure, permutation-invariant, device-preserving torch ops used inside the
DEPLOYED actor (models/ -- torch is allowed in the actor; only the decode path is torch-free). No node
IDs, no global state.
"""

from __future__ import annotations

import torch
from torch import Tensor

PNA_SCALER_ALPHAS = (1.0, 0.0, -1.0)   # amplification, identity, attenuation


def pna_aggregators(messages: Tensor) -> Tensor:
    """The 4 PNA aggregators over neighbour messages ``[k, F]`` -> ``[4, F]`` (mean, max, min, std), all
    permutation-invariant. ``k == 0`` (isolated node) -> zeros; ``k == 1`` -> std row is 0 (population
    std). Device/dtype-preserving."""
    if messages.dim() != 2:
        raise ValueError("messages must be [num_neighbours, feat]")
    k, f = messages.shape
    if k == 0:
        return messages.new_zeros(4, f)
    mean = messages.mean(dim=0)
    mx = messages.max(dim=0).values
    mn = messages.min(dim=0).values
    std = messages.std(dim=0, unbiased=False) if k > 1 else messages.new_zeros(f)
    return torch.stack([mean, mx, mn, std], dim=0)


def pna_degree_scalers(degree: Tensor, delta: float, alphas=PNA_SCALER_ALPHAS) -> Tensor:
    """PNA degree scalers ``S(d, alpha) = (log(1+d)/delta)^alpha`` (Corso 2020). ``degree`` is ``[N]``;
    returns ``[N, len(alphas)]``. ``alpha=0`` -> identity (1); ``+1`` amplifies high-degree nodes; ``-1``
    attenuates them. ``delta`` (the training log-degree mean) must be > 0. Device/dtype-preserving."""
    if delta <= 0.0:
        raise ValueError("delta (mean log(1+degree) over training) must be > 0")
    base = torch.log1p(degree.to(torch.get_default_dtype())) / delta            # [N]
    # NEGATIVE powers (attenuation) would give 0^alpha = +inf at an isolated node (degree 0 -> base 0),
    # and even a clamped 1e12 leaves a huge/ill-conditioned BACKWARD path. An isolated node has ZERO
    # aggregation, so its scaler should be exactly 0 with ZERO gradient: mask the attenuation column by
    # (degree > 0). amplification/identity at degree 0 stay exactly 0/1 (the true PNA values).
    pos = (base > 0).to(base.dtype)
    cols = [((base.clamp_min(1e-12) ** float(a)) * pos if a < 0 else base ** float(a)) for a in alphas]
    return torch.stack(cols, dim=-1)                                            # [N, len(alphas)]


def pna_combine(aggregated: Tensor, degree: Tensor, delta: float, alphas=PNA_SCALER_ALPHAS) -> Tensor:
    """Full PNA readout: scale each aggregator by each degree scaler and concatenate. ``aggregated`` is
    the per-node stacked aggregators ``[N, 4, F]`` (e.g. from :func:`pna_aggregators` per node);
    ``degree`` ``[N]``. Returns ``[N, 4 * len(alphas) * F]`` = the (aggregator x scaler) outer product,
    flattened. Permutation-invariant in the neighbours (the aggregators already are)."""
    if aggregated.dim() != 3:
        raise ValueError("aggregated must be [N, n_agg, F]")
    n, n_agg, f = aggregated.shape
    scalers = pna_degree_scalers(degree, delta, alphas)                         # [N, S]
    # outer product over (aggregator, scaler): [N, n_agg, 1, F] * [N, 1, S, 1] -> [N, n_agg, S, F]
    out = aggregated.unsqueeze(2) * scalers.unsqueeze(1).unsqueeze(-1)
    return out.reshape(n, n_agg * scalers.shape[-1] * f)


def training_degree_delta(degrees) -> float:
    """``delta = mean(log(1 + d))`` over the TRAINING degrees (Corso 2020) -- the fixed scaler
    normalizer. Pass an iterable of node degrees seen in training."""
    ds = [float(d) for d in degrees]
    if not ds:
        return 1.0
    import math
    return sum(math.log1p(d) for d in ds) / len(ds)
