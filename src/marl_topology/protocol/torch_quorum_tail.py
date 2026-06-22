"""Torch-differentiable heterogeneous quorum-tail (Phase 1c, Technical-Spec S4.4-4.5).

The reference DP ``protocol.quorum_tail.heterogeneous_quorum_tail`` is a pure-Python,
non-differentiable evaluator. The closed-form PBFT reward and the SCQ counterfactual
supervision need the *same* quorum tail as a differentiable Torch op so that gradients
flow into per-edge delivery probabilities.

This module implements the identical capped generating-polynomial DP in Torch:

    buckets[..., k] = P(exactly k successes so far)   for k < q
    buckets[..., q] = P(>= q successes so far)         (absorbing tail bucket)

processing the ``n`` heterogeneous Bernoullis one at a time. It is batched over all
leading dimensions (the last dimension is the committee), uses only differentiable ops
(no in-place writes into the graph), and matches the reference DP to float tolerance.

By Spec S4.5 the gradient is the quorum sensitivity::

    dQ_q / dp_i = P( sum_{j != i} Y_j = q - 1 )

i.e. message ``i`` matters most when the others are exactly one short of a quorum. This
is exactly what autograd returns (pinned by the tests).

This file is a standalone submodule (NOT re-exported from ``protocol/__init__``) so that
``import marl_topology.protocol`` stays free of a hard Torch import.
"""

from __future__ import annotations

import torch


def torch_quorum_tail(probabilities: torch.Tensor, quorum_size: int) -> torch.Tensor:
    """Return ``P(at least quorum_size of the last-dim Bernoullis succeed)``.

    ``probabilities``: shape ``(..., n)`` with entries in ``[0, 1]``. Returns shape
    ``(...)``. Differentiable w.r.t. ``probabilities``; numerically equal to
    :func:`marl_topology.protocol.quorum_tail.heterogeneous_quorum_tail` per committee.
    """

    if isinstance(quorum_size, bool) or not isinstance(quorum_size, int):
        raise TypeError("quorum_size must be an integer")
    if quorum_size < 0:
        raise ValueError("quorum_size must be nonnegative")
    if probabilities.shape[-1:] == torch.Size([0]) and quorum_size > 0:
        return torch.zeros(probabilities.shape[:-1], dtype=probabilities.dtype, device=probabilities.device)

    n = probabilities.shape[-1]
    batch_shape = probabilities.shape[:-1]
    dtype, device = probabilities.dtype, probabilities.device

    if quorum_size == 0:
        return torch.ones(batch_shape, dtype=dtype, device=device)
    if quorum_size > n:
        return torch.zeros(batch_shape, dtype=dtype, device=device)

    # buckets: (..., quorum_size + 1), initialized to P(0 successes) = 1 (out-of-place).
    leading_ones = torch.ones((*batch_shape, 1), dtype=dtype, device=device)
    trailing_zeros = torch.zeros((*batch_shape, quorum_size), dtype=dtype, device=device)
    buckets = torch.cat([leading_ones, trailing_zeros], dim=-1)

    for index in range(n):
        success = probabilities[..., index : index + 1]      # (..., 1)
        failure = 1.0 - success
        # shift right by one along the bucket axis: shifted[k] = buckets[k-1], shifted[0] = 0.
        shifted = torch.cat([torch.zeros_like(buckets[..., :1]), buckets[..., :-1]], dim=-1)
        advanced = buckets * failure + shifted * success     # next[k] = b[k]*qf + b[k-1]*p
        # the top bucket is absorbing: next[q] = b[q]*1 + b[q-1]*p (add back b[q]*p).
        absorbing = advanced[..., -1:] + buckets[..., -1:] * success
        buckets = torch.cat([advanced[..., :-1], absorbing], dim=-1)

    return buckets[..., quorum_size]
