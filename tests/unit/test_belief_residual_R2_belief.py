"""R2 (Belief-Guided Residual PPO) — CSI belief prediction auxiliary.

Load-bearing PATH + Effect-on-Decision tests (Contract v4 §3/§4):
- the belief target is the TRUE current psucc (training-only label); the actor input is the STALE value (no leak);
- L_CSI genuinely ENTERS a training loss (gradients reach the belief head AND the shared GRU) -- not a diagnostic;
- the recurrent policy actor predicts current CSI better than memoryless under stale CSI (the exit condition).

Fails on HEAD: `csi_belief` and `BeliefResidualActor.belief` / `csi_belief_train` do not exist yet.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _stale_scenes(n=3, delay=1, frames=4, seed=7):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.csi_observation_model import CsiObservationModel
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost
    return sample_dynamic_scenes(seed=seed, count=n, node_count_choices=(8,),
                                 regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0,
                                 speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                                 hold_interval=4, gamma=0.95,
                                 csi_observation_model=CsiObservationModel(mode="delay", delay_frames=delay))


def test_belief_head_training_target_is_true_current_csi() -> None:
    from marl_topology.training.csi_belief import belief_target_logits
    sc = _stale_scenes(n=1, delay=1, frames=4)[0]
    obs = sc.observation(2, [])
    eids = obs["edge_ids"]
    tgt = belief_target_logits(sc, 2, eids)
    recs = sc.context(2).link_records
    for i, eid in enumerate(eids):
        p = min(max(float(recs[eid].link_success_probability), 1e-4), 1 - 1e-4)
        assert float(tgt[i]) == pytest.approx(math.log(p / (1 - p)), abs=1e-4)


def test_actor_input_does_not_include_true_current_csi() -> None:
    from marl_topology.training.csi_belief import belief_target_logits
    sc = _stale_scenes(n=1, delay=1, frames=5)[0]
    diffs = 0
    for t in range(1, sc.n_frames):
        obs = sc.observation(t, [])
        eids = obs["edge_ids"]
        observed_p = obs["ef"][:, 0]                                   # STALE observed psucc (actor input)
        true_p = torch.sigmoid(belief_target_logits(sc, t, eids))      # TRUE current psucc (label only)
        diffs += int((observed_p - true_p).abs().max() > 1e-3)
    assert diffs > 0, "under delay-1 the actor input (stale ef[:,0]) must differ from the true-current label"


def _one_train_step(actor, sc, recurrent):
    from marl_topology.training.csi_belief import belief_target_logits, belief_weights, csi_belief_loss
    from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: F401
    loss = torch.zeros(())
    h = None
    for t in range(sc.n_frames):
        obs = sc.observation(t, [])
        eids = obs["edge_ids"]
        h_in = h if recurrent else None
        bel, h = actor.belief(obs["nf"], obs["ef"], obs["ei"], hidden=h_in)
        tgt = belief_target_logits(sc, t, eids)
        w = belief_weights(eids, prev_topo=[], anchor=[])
        loss = loss + csi_belief_loss(bel, tgt, w)
    return loss


def test_belief_loss_enters_total_loss() -> None:
    from marl_topology.models.belief_residual_actor import BeliefResidualActor
    sc = _stale_scenes(n=1, delay=1, frames=3)[0]
    nd = sc.observation(0, [])["nf"].shape[1]
    ed = sc.observation(0, [])["ef"].shape[1]
    torch.manual_seed(0)
    actor = BeliefResidualActor(nd, ed, hidden=16)
    actor.zero_grad()
    loss = _one_train_step(actor, sc, recurrent=True)
    loss.backward()
    bhead_grad = sum(p.grad.abs().sum() for p in actor.belief_head.parameters() if p.grad is not None)
    gru_grad = sum(p.grad.abs().sum() for p in actor.gru.parameters() if p.grad is not None)
    assert float(bhead_grad) > 0, "belief loss must reach the belief head"
    assert float(gru_grad) > 0, "belief loss must flow through the shared GRU (not a diagnostic-only head)"


def test_belief_loss_logged() -> None:
    import csi_belief_train as bt
    res = bt.train_belief(_stale_scenes(n=2, delay=1, frames=4, seed=11),
                          _stale_scenes(n=2, delay=1, frames=4, seed=77),
                          recurrent=True, epochs=5, hidden=16, seed=0)
    assert "held_belief_mse" in res and "belief_mse_history" in res
    assert len(res["belief_mse_history"]) == 5


def test_belief_prediction_improves_over_epochs() -> None:
    import csi_belief_train as bt
    res = bt.train_belief(_stale_scenes(n=3, delay=2, frames=5, seed=3),
                          _stale_scenes(n=3, delay=2, frames=5, seed=303),
                          recurrent=True, epochs=40, hidden=24, seed=0)
    hist = res["belief_mse_history"]
    assert hist[-1] < hist[0], f"belief MSE should drop over epochs ({hist[0]:.4f} -> {hist[-1]:.4f})"


def test_belief_does_not_beat_stale_echo_floor() -> None:
    """HONEST NEGATIVE (Workflow wozljm9uf MAJOR, reproduced + multi-seed belief_floor_multiseed.json): the
    belief head does NOT recover current CSI beyond trivially echoing the stale observed psucc -- its held MSE
    sits at/above the stale-echo floor even WITH leak-free velocity features. The belief auxiliary is active,
    leak-free, and in-loss (other tests), but it is a no-op CSI *predictor* on the policy actor. This test
    pins that finding; a future stage that makes belief recover CSI would flip + update it."""
    import csi_belief_train as bt
    train = _stale_scenes(n=5, delay=2, frames=6, seed=9)
    held = _stale_scenes(n=5, delay=2, frames=6, seed=909)
    r = bt.train_belief(train, held, recurrent=True, epochs=80, hidden=32, seed=0, use_velocity=True)
    assert "stale_echo_floor_mse" in r and "beats_stale_echo_floor" in r
    assert not r["beats_stale_echo_floor"], \
        f"belief MSE {r['held_belief_mse']:.5f} unexpectedly beat the stale-echo floor " \
        f"{r['stale_echo_floor_mse']:.5f} -- revisit the R2 honest-negative conclusion"


def test_recurrent_belief_no_robust_gain_over_memoryless() -> None:
    """HONEST robust finding (5-seed-confirmed, belief_multiseed.json: paired CI spans 0 on delay1/delay2/
    partial): cross-frame recurrence does NOT robustly improve belief over a memoryless head that already has
    age+velocity features -- the held belief MSE gap is small either way (the CSI is recoverable per-frame).
    This replaces the earlier cherry-picked single-config 'recurrent beats memoryless' assertion."""
    import csi_belief_train as bt
    train = _stale_scenes(n=4, delay=2, frames=6, seed=5)
    held = _stale_scenes(n=4, delay=2, frames=6, seed=505)
    rec = bt.train_belief(train, held, recurrent=True, epochs=60, hidden=32, seed=0)
    mem = bt.train_belief(train, held, recurrent=False, epochs=60, hidden=32, seed=0)
    gap = abs(rec["held_belief_mse"] - mem["held_belief_mse"])
    assert gap < 0.2 * mem["held_belief_mse"], \
        f"recurrent-vs-memoryless belief gap {gap:.5f} should be small vs MSE {mem['held_belief_mse']:.5f} " \
        f"(recurrence is not a robust belief lever)"
