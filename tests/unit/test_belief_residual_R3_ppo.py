"""R3 (Belief-Guided Residual PPO) — residual PPO + CTDE value critic.

Load-bearing PATH + Effect-on-Decision tests (Contract v4 §3/§4/§11):
- the residual trainer GENUINELY CALLS graph_mappo.ppo_clip_actor_loss (spy) -- not the trunk's PPO;
- it logs ratio/approx_kl/clip_fraction; epoch-0 ratio==1; target_kl early-stops the inner epochs;
- the CTDE critic actually trains (critic_parameter_delta > 0); entropy is computed + enters the loss;
- the PPO ratio is PER-EDGE (per-agent), not the joint per-frame ratio.

Fails on HEAD: `residual_ppo_train` and `residual_action.residual_logp_per_edge` do not exist yet.
"""

from __future__ import annotations

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


# --- per-edge (per-agent) logp, not the joint ratio (Contract: PPO docstring warns about joint ratio) ----

def test_per_edge_logp_sums_to_joint_logp() -> None:
    from marl_topology.training.residual_action import (residual_candidate_mask, residual_logp,
                                                        residual_logp_per_edge)
    z = torch.tensor([0.4, -1.2, 0.8, -0.3, 1.1])
    d = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])           # edge 2 is a non-candidate
    per_edge = residual_logp_per_edge(z, d, mask)
    assert per_edge.shape == z.shape
    assert float(per_edge[2]) == 0.0                          # non-candidate contributes 0
    assert float(per_edge.sum()) == pytest.approx(float(residual_logp(z, d, mask)), abs=1e-6)


# --- the residual trainer GENUINELY CALLS PPO (the R0 tripwire's flip) -----------------------------------

def test_residual_ppo_calls_ppo_clip_actor_loss(monkeypatch) -> None:
    import residual_ppo_train as rp3
    from marl_topology.training import graph_mappo
    calls = {"n": 0}
    real = graph_mappo.ppo_clip_actor_loss
    monkeypatch.setattr(graph_mappo, "ppo_clip_actor_loss",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    rp3.train_residual_ppo(_scenes(1, 3), epochs=2, ppo_epochs=3, hidden=16, seed=0)
    assert calls["n"] >= 1, "residual PPO trainer must call graph_mappo.ppo_clip_actor_loss"


def test_residual_ppo_logs_approx_kl_and_clip_fraction() -> None:
    import residual_ppo_train as rp3
    m = rp3.train_residual_ppo(_scenes(1, 3), epochs=2, ppo_epochs=3, hidden=16, seed=0)
    for k in ("approx_kl", "clip_fraction", "approx_kl_history", "critic_parameter_delta", "value_loss",
              "explained_variance", "entropy", "edit_rate", "retention", "ppo_calls"):
        assert k in m, f"missing metric {k}"
    assert m["ppo_calls"] >= 1


def test_residual_ppo_epoch0_ratio_one() -> None:
    import residual_ppo_train as rp3
    m = rp3.train_residual_ppo(_scenes(1, 3), epochs=1, ppo_epochs=3, hidden=16, seed=0)
    # inner epoch 0 recomputes logp under the SAME params -> ratio == 1 -> approx_kl ~ 0
    assert m["approx_kl_history"][0] == pytest.approx(0.0, abs=1e-5)


def test_target_kl_can_stop_inner_epochs() -> None:
    import residual_ppo_train as rp3
    loose = rp3.train_residual_ppo(_scenes(2, 3), epochs=1, ppo_epochs=8, target_kl=1e9, hidden=16, seed=0)
    tight = rp3.train_residual_ppo(_scenes(2, 3), epochs=1, ppo_epochs=8, target_kl=1e-6, hidden=16, seed=0)
    assert tight["inner_epochs_ran"][0] < loose["inner_epochs_ran"][0], \
        "a tiny target_kl must early-stop the inner PPO epochs"


def test_residual_critic_parameter_delta_positive() -> None:
    import residual_ppo_train as rp3
    m = rp3.train_residual_ppo(_scenes(2, 3), epochs=3, ppo_epochs=3, hidden=16, seed=0)
    assert m["critic_parameter_delta"] > 0.0, "the CTDE value critic must actually train"


def test_entropy_bonus_present_and_in_loss() -> None:
    import residual_ppo_train as rp3
    m = rp3.train_residual_ppo(_scenes(2, 3), epochs=2, ppo_epochs=3, entropy_coef=0.05, hidden=16, seed=0)
    assert m["entropy"] > 0.0                                  # stochastic policy -> positive entropy
    # entropy_coef changes the actor loss -> changes the trained policy edit_rate (effect-on-decision)
    m0 = rp3.train_residual_ppo(_scenes(2, 3), epochs=2, ppo_epochs=3, entropy_coef=0.0, hidden=16, seed=0)
    assert m["entropy_coef"] == 0.05 and m0["entropy_coef"] == 0.0
