"""R7 (Belief-Guided Residual PPO) — adaptive anchor-KL / safety constraint (replace the fixed flip penalty).

Load-bearing tests (Contract v4 §3/§4): the anchor-KL term is the mean flip probability over candidate edges
(deviation from the flip-nothing anchor) and its gradient PULLS the residual logits toward the anchor; the
adaptive controller TIGHTENS beta when retention drops and LOOSENS it when validation improves at stable
retention; the adaptive path shares the R3 PPO/critic (one variable = the anchor pull, fixed → adaptive) and
still calls graph_mappo.ppo_clip_actor_loss; the eval decode is the deployed MAP decode (0 eval). The empirical
"does adaptive beat the anchor?" question is the R7 DECISION (measured by the multi-seed), NOT asserted here.

Fails on HEAD: `residual_ppo_train.anchor_kl_penalty` / `update_beta` / the adaptive_anchor_kl path do not exist.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _scenes(n=1, frames=3, seed=4):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_scenes
    from marl_topology.training.two_timescale_env import ReconfigCost
    return sample_dynamic_scenes(seed=seed, count=n, node_count_choices=(8,),
                                 regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0,
                                 speed_min_mps=15.0, speed_max_mps=30.0, reconfig=ReconfigCost(),
                                 hold_interval=4, gamma=0.95)


def test_anchor_kl_penalty_is_mean_flip_prob() -> None:
    import residual_ppo_train as rp3
    z = torch.tensor([2.0, -2.0, 0.0, 5.0, -5.0], requires_grad=True)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])            # last two are non-candidates
    pen = rp3.anchor_kl_penalty(z, mask)
    # mean sigmoid(z) over the 3 candidate edges: (0.8808 + 0.1192 + 0.5)/3
    assert float(pen) == pytest.approx((0.880797 + 0.119203 + 0.5) / 3.0, abs=1e-4)
    pen.backward()
    assert float(z.grad[3]) == 0.0 and float(z.grad[4]) == 0.0     # non-candidates contribute 0 grad
    assert float(z.grad[2]) > 0.0                                   # increasing z increases flip prob


def test_anchor_kl_gradient_pulls_toward_anchor() -> None:
    """A descent step on the anchor-KL penalty REDUCES the mean flip probability (pulls toward flip-nothing)."""
    import residual_ppo_train as rp3
    from marl_topology.models.belief_residual_actor import BeliefResidualActor
    torch.manual_seed(0)
    a = BeliefResidualActor(node_dim=5, edge_dim=6, hidden=16, residual_logit_scale=3.0)
    nf, ef, ei = torch.randn(8, 5), torch.randn(12, 6), torch.randint(0, 8, (12, 2))
    mask = torch.ones(12)
    z0 = a(nf, ef, ei, hidden=None)[0]
    before = float((torch.sigmoid(z0) * mask).sum() / mask.sum())
    opt = torch.optim.SGD(a.parameters(), lr=0.5)
    for _ in range(20):
        z = a(nf, ef, ei, hidden=None)[0]
        loss = rp3.anchor_kl_penalty(z, mask)
        opt.zero_grad(); loss.backward(); opt.step()
    after = float((torch.sigmoid(a(nf, ef, ei, hidden=None)[0]) * mask).sum() / mask.sum())
    assert after < before - 1e-3                                    # the pull genuinely lowers flip prob


def test_update_beta_tightens_and_loosens() -> None:
    import residual_ppo_train as rp3
    # retention below target -> TIGHTEN (beta up)
    b = rp3.update_beta(1.0, retention=0.5, target=0.95, val_return=0.0, prev_val=0.0,
                        up=1.5, down=0.7, lo=0.0, hi=10.0)
    assert b == pytest.approx(1.5)
    # retention at/above target AND val improved -> LOOSEN (beta down)
    b = rp3.update_beta(1.0, retention=0.98, target=0.95, val_return=0.2, prev_val=0.1,
                        up=1.5, down=0.7, lo=0.0, hi=10.0)
    assert b == pytest.approx(0.7)
    # retention ok but val NOT improving -> hold
    b = rp3.update_beta(1.0, retention=0.98, target=0.95, val_return=0.05, prev_val=0.1,
                        up=1.5, down=0.7, lo=0.0, hi=10.0)
    assert b == pytest.approx(1.0)
    # clamps
    assert rp3.update_beta(9.0, 0.5, 0.95, 0.0, 0.0, 1.5, 0.7, 0.0, 10.0) == pytest.approx(10.0)
    assert rp3.update_beta(0.0, 0.98, 0.95, 0.2, 0.1, 1.5, 0.7, 0.0, 10.0) == pytest.approx(0.0)


def test_adaptive_path_calls_ppo_and_returns_beta_history(monkeypatch) -> None:
    import residual_ppo_train as rp3
    from marl_topology.training import graph_mappo
    calls = {"n": 0}
    real = graph_mappo.ppo_clip_actor_loss
    monkeypatch.setattr(graph_mappo, "ppo_clip_actor_loss",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    m = rp3.train_residual_ppo(_scenes(1, 3), epochs=3, ppo_epochs=2, hidden=16, seed=0,
                               adaptive_anchor_kl=True, residual_prior=0.0)
    assert calls["n"] >= 1                                          # shares the R3 PPO path (one variable)
    assert "beta_anchor_history" in m and len(m["beta_anchor_history"]) >= 1
    assert m["critic_parameter_delta"] > 0.0                        # CTDE critic still trains
    assert "residual_feasibility" in m and "anchor_feasibility" in m and "edit_rate" in m


def test_adaptive_is_deployable_no_flip_penalty() -> None:
    import residual_ppo_train as rp3
    # the adaptive path exposes the controller knobs (one variable = the anchor pull)
    params = set(inspect.signature(rp3.train_residual_ppo).parameters)
    for k in ("adaptive_anchor_kl", "target_retention", "beta_anchor_init"):
        assert k in params
    src = (_ROOT / "scripts" / "diagnostics" / "residual_ppo_train.py").read_text(encoding="utf-8")
    assert "anchor_kl_penalty" in src and "update_beta" in src      # the adaptive anchor-KL is the protection
    assert "residual_decode_from_flips" in src                       # eval decode is the deployed MAP decode
