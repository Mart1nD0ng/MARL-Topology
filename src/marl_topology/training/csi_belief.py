"""R2 (Belief-Guided Residual PPO) — CSI belief prediction auxiliary (TechSpec §3).

Supervises the policy actor's GRU to recover the TRUE current link psucc from stale/partial history. The
true current psucc is a TRAINING-ONLY LABEL (the deployed actor never reads it; its input ef carries the
STALE observed value). Leak-free by the same construction Q2 verified: the target reads ``scene.context(t)``
(the true current channel) while the actor input reads the staleified ef.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

_EPS = 1e-4


def leak_free_motion_features(scene, t: int, edge_ids) -> Tensor:
    """Per-edge [relative_velocity_along_link, distance_delta] in ``edge_ids`` order -- PURE GEOMETRY from
    node positions + velocities (a node knows its own + neighbour position/velocity via broadcast). This is
    LEAK-FREE: unlike the env ``_motion_edge_block`` it does NOT include the ``csi_delta = cur-prev`` term,
    which is computed from the TRUE current channel (a true-CSI leak). It gives the belief head the velocity
    signal needed to extrapolate the current channel from the stale observation (the Q2 recovery signal)
    WITHOUT exposing the true current CSI."""
    ctx = scene.context(t)
    pos = {n.node_id: n.position for n in scene.scenes[t].nodes}
    vel = scene.velocities
    uv = {e.edge_id: (e.node_u, e.node_v) for e in ctx.graph.edges}
    rows = []
    for eid in edge_ids:
        u, v = uv[eid]
        pu, pv = pos[u], pos[v]
        vux, vuy, _u = vel.get(u, (0.0, 0.0, 0.0))
        vvx, vvy, _w = vel.get(v, (0.0, 0.0, 0.0))
        dx, dy = pu.x_m - pv.x_m, pu.y_m - pv.y_m
        dist = math.hypot(dx, dy) or 1e-9
        rel_vel = ((vux - vvx) * dx + (vuy - vvy) * dy) / dist
        rows.append([float(rel_vel), float(rel_vel * scene.dt_s)])
    return torch.tensor(rows, dtype=torch.float32)


def stale_echo_floor_mse(scene, t: int, edge_ids, observed_psucc: Tensor) -> float:
    """The honest baseline (the verifier's diagnostic): MSE of trivially predicting the STALE observed psucc
    instead of recovering the current. A belief head that does not beat this is a no-op CSI predictor."""
    if observed_psucc.numel() == 0:
        return 0.0
    true_p = torch.sigmoid(belief_target_logits(scene, t, edge_ids))
    return float((observed_psucc.reshape(-1) - true_p).pow(2).mean())


def _logit(p: Tensor) -> Tensor:
    p = p.clamp(_EPS, 1.0 - _EPS)
    return torch.log(p / (1.0 - p))


def belief_target_logits(scene, t: int, edge_ids) -> Tensor:
    """TRUE current-frame link psucc (logit) per edge, in ``edge_ids`` order -- the training-only belief
    label. Reads ``scene.context(t).link_records`` (the true current channel), NOT the observed ef. The
    clamp + logit are done in float64 (the logit of a near-1 psucc is sensitive to the clamp boundary in
    float32) and cast to float32."""
    recs = scene.context(t).link_records
    p = torch.tensor([float(recs[eid].link_success_probability) for eid in edge_ids], dtype=torch.float64)
    return _logit(p).to(torch.float32)


def belief_weights(edge_ids, prev_topo=None, anchor=None, candidates=None, criticality=None,
                   *, alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0, gamma: float = 1.0) -> Tensor:
    """``w = 1 + alpha*1[prev] + beta*1[anchor] + eta*1[candidate] + gamma*criticality`` -- emphasize the
    decision-relevant edges (TechSpec §3.2) so the belief signal is not diluted by many irrelevant links."""
    prev = set(prev_topo or [])
    anc = set(anchor or [])
    cand = set(candidates or [])
    w = torch.ones(len(edge_ids), dtype=torch.float32)
    for i, eid in enumerate(edge_ids):
        if eid in prev:
            w[i] += alpha
        if eid in anc:
            w[i] += beta
        if eid in cand:
            w[i] += eta
    if criticality is not None:
        w = w + gamma * torch.as_tensor(criticality, dtype=torch.float32).reshape(-1)
    return w


def csi_belief_loss(belief_logits: Tensor, target_logits: Tensor, weights: Tensor | None = None,
                    *, delta: float = 1.0) -> Tensor:
    """Weighted Huber in logit space (TechSpec §3.1). ``belief_logits`` is the actor's predicted current-psucc
    logit; ``target_logits`` the true-current logit label."""
    if belief_logits.numel() == 0:
        return belief_logits.new_zeros(())
    huber = F.huber_loss(belief_logits, target_logits, reduction="none", delta=delta)
    if weights is None:
        return huber.mean()
    weights = weights.to(huber.dtype)
    return (weights * huber).sum() / weights.sum().clamp_min(1e-9)


def belief_mse(belief_logits: Tensor, target_logits: Tensor) -> float:
    """Held metric: MSE of the predicted vs true current psucc in PROBABILITY space (comparable to Q2)."""
    if belief_logits.numel() == 0:
        return 0.0
    return float((torch.sigmoid(belief_logits) - torch.sigmoid(target_logits)).pow(2).mean())
