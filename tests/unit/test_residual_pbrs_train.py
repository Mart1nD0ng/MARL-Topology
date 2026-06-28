"""Q9 PART 2 (POMDP-QP-FAR): residual + PBRS end-to-end training.

Pins: PBRS shaping is added ONLY in the training reward (eval uses none), the training does not diverge
(NaN -- the Q5 failure mode), and the eval reports true-channel metrics + a valid retention. Fails on
HEAD (the trainer is new).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "diagnostics"))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor
from marl_topology.training.decentralized_distillation import feature_standardization
from marl_topology.training.dynamic_frames import sample_dynamic_scenes
from marl_topology.training.two_timescale_env import ReconfigCost

import residual_pbrs_train as rpt  # noqa: E402

_T = rpt._load_trunk()


def _setup(count=2, frames=4):
    from build_operating_point_dataset import operating_point_regime
    scenes = sample_dynamic_scenes(
        seed=21, count=count, node_count_choices=(12,), regime=operating_point_regime(20.0),
        num_frames=frames, dt_s=2.0, speed_min_mps=15.0, speed_max_mps=30.0,
        reconfig=ReconfigCost(), hold_interval=4, gamma=0.95)
    stat = [s.observation(0, []) for s in scenes]
    mean, std = feature_standardization(stat)
    nd, ed = stat[0]["nf"].shape[1], stat[0]["ef"].shape[1]
    torch.manual_seed(0)
    actor = DynamicRecurrentActor(nd, ed, hidden=16)
    return scenes, mean, std, actor


def test_pbrs_shaping_added_only_in_training() -> None:
    scenes, mean, std, actor = _setup()
    g = torch.Generator().manual_seed(0)
    _lp, shaped_on, info_on = rpt.rollout_residual_train(
        actor, scenes[0], mean, std, _T, mode="full", residual_prior=-3.0, lam_pbrs=0.5, gamma=0.95,
        use_pbrs=True, generator=g)
    # shaped reward = r_t + F_t, and the PBRS shaping is nonzero somewhere (early frames carry deficit)
    for t in range(len(shaped_on)):
        assert abs(shaped_on[t] - (info_on["rewards"][t] + info_on["shaping"][t])) < 1e-9
    assert any(abs(f) > 1e-9 for f in info_on["shaping"])
    g2 = torch.Generator().manual_seed(0)
    _lp2, shaped_off, info_off = rpt.rollout_residual_train(
        actor, scenes[0], mean, std, _T, mode="full", residual_prior=-3.0, lam_pbrs=0.5, gamma=0.95,
        use_pbrs=False, generator=g2)
    assert all(f == 0.0 for f in info_off["shaping"])                  # PBRS off -> no shaping
    assert all(shaped_off[t] == info_off["rewards"][t] for t in range(len(shaped_off)))


def test_residual_train_does_not_diverge() -> None:
    scenes, mean, std, actor = _setup()
    g = torch.Generator().manual_seed(1)
    opt = torch.optim.Adam(actor.parameters(), lr=0.02)
    baseline, loss = 0.0, 0.0
    for _ in range(6):
        baseline, loss = rpt.reinforce_update(
            actor, opt, scenes, mean, std, _T, mode="full", residual_prior=-3.0, lam_pbrs=0.5,
            gamma=0.95, use_pbrs=True, baseline=baseline, generator=g)
    assert loss == loss and abs(loss) < 1e6                            # finite (no NaN -> no Q5-style divergence)


def test_eval_residual_reports_true_metrics() -> None:
    scenes, mean, std, actor = _setup(count=2, frames=4)
    ev = rpt.eval_residual(actor, scenes, mean, std, _T, mode="full", residual_prior=-3.0)
    assert 0.0 <= ev["residual_feasibility"] <= 1.0 and 0.0 <= ev["anchor_feasibility"] <= 1.0
    assert 0.0 <= ev["retention"] <= 1.0
    assert ev["residual_energy"] >= 0.0 and ev["n_frames"] > 0
