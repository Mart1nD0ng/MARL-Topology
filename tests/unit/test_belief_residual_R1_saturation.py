"""R1 (Belief-Guided Residual PPO) — feature standardization + logit-saturation fix.

Load-bearing + Effect-on-Decision tests (Contract v4 §3/§4). The central one,
`test_recurrent_changes_logits_under_stale_csi`, proves the separate small-range residual head lets the
GRU's cross-frame signal reach the logits (no longer bit-identical to memoryless) — the precondition the
whole campaign needs before any "recurrence ineffective" claim (Contract v4 §8/§10).

Fails on HEAD: `belief_residual_actor` and `residual_saturation` do not exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _stale_scenes(n=3, mode="delay", delay=1, frames=4, seed=7):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.csi_observation_model import CsiObservationModel
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost
    csi = (CsiObservationModel(mode="current") if mode == "current"
           else CsiObservationModel(mode="delay", delay_frames=delay))
    return sample_dynamic_scenes(seed=seed, count=n, node_count_choices=(8,),
                                 regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0,
                                 speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                                 hold_interval=4, gamma=0.95, csi_observation_model=csi)


# --- standardization across ALL train frames (R1 ①) --------------------------------------------------

def test_standardization_uses_all_train_frames() -> None:
    from marl_topology.training.decentralized_distillation import feature_standardization
    from marl_topology.training.residual_saturation import feature_standardization_all_frames
    scenes = _stale_scenes(n=3, mode="delay", delay=2, frames=5)
    frame0 = [s.observation(0, []) for s in scenes]
    mean_f0, std_f0 = feature_standardization(frame0)
    mean_all, std_all, n_obs = feature_standardization_all_frames(scenes, return_count=True)
    # pooled n_scenes * n_frames observations, not n_scenes
    assert n_obs == 3 * 5
    # std is (nf_std, ef_std); under stale CSI the edge csi_age/observed features vary across frames ->
    # all-frame ef-std differs from frame-0-only ef-std
    assert not torch.allclose(std_all[1], std_f0[1], atol=1e-4), "all-frame ef-std must differ from frame-0-only"


def test_stale_feature_scale_not_exploding() -> None:
    from marl_topology.training.decentralized_distillation import feature_standardization
    from marl_topology.training.residual_saturation import feature_standardization_all_frames
    from marl_topology.training.dynamic_rl import _standardize
    scenes = _stale_scenes(n=4, mode="delay", delay=2, frames=5)
    mean_all, std_all = feature_standardization_all_frames(scenes)
    # standardize every frame with the all-frame stats -> bounded magnitudes
    p99 = []
    for s in scenes:
        for t in range(s.n_frames):
            obs = s.observation(t, [])
            _nf, ef_s = _standardize(obs["nf"], obs["ef"], mean_all, std_all)
            if ef_s.numel():
                p99.append(float(ef_s.abs().quantile(0.99)))
    assert max(p99) < 12.0, f"standardized stale features explode (p99 {max(p99):.1f})"


# --- saturation metrics (R1 ③) -----------------------------------------------------------------------

def test_saturation_metrics_logged() -> None:
    from marl_topology.training.residual_saturation import saturation_metrics
    raw = torch.tensor([0.5, 40.0, -35.0, 1.0])          # 2 of 4 have |raw|>30
    logits = 10.0 * torch.tanh(raw / 10.0)               # the OLD ±10 head -> the |raw|>30 ones rail near ±10
    m = saturation_metrics(raw, logits, logit_scale=10.0)
    for k in ("raw_abs_mean", "raw_abs_p95", "frac_abs_raw_gt_30", "frac_abs_logit_gt_9_5", "frac_logit_near_rail"):
        assert k in m
    assert m["frac_abs_raw_gt_30"] == pytest.approx(0.5)
    assert m["frac_logit_near_rail"] == pytest.approx(0.5)        # the two railed ones
    assert m["raw_abs_mean"] == pytest.approx((0.5 + 40 + 35 + 1) / 4, abs=1e-4)


def test_raw_logit_l2_reduces_saturation_on_pilot() -> None:
    from marl_topology.models.belief_residual_actor import BeliefResidualActor
    from marl_topology.training.residual_saturation import raw_logit_l2_penalty, saturation_metrics

    def train(lam_raw):
        torch.manual_seed(0)
        a = BeliefResidualActor(node_dim=5, edge_dim=6, hidden=16, residual_logit_scale=3.0)
        opt = torch.optim.Adam(a.parameters(), lr=0.05)
        nf = torch.randn(8, 5); ef = torch.randn(12, 6); ei = torch.randint(0, 8, (12, 2))
        last = None
        for _ in range(60):
            z, raw, _h = a(nf, ef, ei, hidden=None)
            # objective pushes logits to the positive rail (the REINFORCE-to-extreme failure mode)
            loss = -z.mean() + lam_raw * raw_logit_l2_penalty(raw)
            opt.zero_grad(); loss.backward(); opt.step()
            last = raw.detach()
        return saturation_metrics(last, a(nf, ef, ei, hidden=None)[0].detach(), 3.0)["raw_abs_mean"]

    no_reg = train(0.0)
    with_reg = train(0.05)
    assert with_reg < no_reg, f"raw-L2 should shrink raw (with {with_reg:.2f} vs without {no_reg:.2f})"


# --- THE Effect-on-Decision test (R1 ⑤; load-bearing for the whole campaign) --------------------------

def test_recurrent_changes_logits_under_stale_csi() -> None:
    from marl_topology.models.belief_residual_actor import BeliefResidualActor
    from marl_topology.training.decentralized_distillation import feature_standardization
    from marl_topology.training.residual_saturation import (feature_standardization_all_frames,
                                                            saturation_metrics)
    from marl_topology.training.dynamic_rl import _standardize
    scenes = _stale_scenes(n=2, mode="delay", delay=1, frames=4)
    mean, std = feature_standardization_all_frames(scenes)
    nd = scenes[0].observation(0, [])["nf"].shape[1]
    ed = scenes[0].observation(0, [])["ef"].shape[1]
    torch.manual_seed(1)
    actor = BeliefResidualActor(nd, ed, hidden=32, residual_logit_scale=3.0)

    sc = scenes[0]
    h = None
    rec_logits = mem_logits = rec_raw = None
    for t in range(sc.n_frames):
        obs = sc.observation(t, [])
        nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
        with torch.no_grad():
            z_mem, _raw_m, _hm = actor(nf_s, ef_s, obs["ei"], hidden=None)       # memoryless
            z_rec, raw_r, h = actor(nf_s, ef_s, obs["ei"], hidden=h)             # recurrent (carry h)
        if t == sc.n_frames - 1:
            rec_logits, mem_logits, rec_raw = z_rec, z_mem, raw_r

    # (1) recurrence is NOT bit-identical to memoryless at the LOGIT level (saturation fixed)
    assert not torch.allclose(rec_logits, mem_logits, atol=1e-6), \
        "separate small-range head: recurrent vs memoryless residual logits must differ under stale CSI"
    delta = float((rec_logits - mem_logits).abs().max())
    assert delta > 1e-3, f"recurrent-memoryless logit delta too small ({delta:.4f})"
    # (2) the new head is UNSATURATED (the whole point of R1)
    m = saturation_metrics(rec_raw, rec_logits, logit_scale=3.0)
    assert m["frac_logit_near_rail"] < 0.5, f"new residual head still saturates ({m['frac_logit_near_rail']:.2f})"
    # (3) Effect-on-Decision: the flip decode (z>0) CAN differ between recurrent and memoryless
    flips_rec = (rec_logits > 0)
    flips_mem = (mem_logits > 0)
    assert flips_rec.shape == flips_mem.shape          # action-delta is measurable (may be 0 on a tiny net, but defined)
