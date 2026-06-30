"""R0 (Belief-Guided Residual PPO) — residual-trainer PATH AUDIT (Contract v4 §3 load-bearing path tests).

These pin the AUDIT conclusion in executable code: the PPO clip / approx_kl machinery EXISTS in the
Graph-MAPPO trunk but is NOT_PRESENT in the residual trainer path, and the residual trainer has no value
critic / entropy / CSI-belief auxiliary. This encodes the Q14 scope boundary (Q14 tested REINFORCE-residual
+ flip-penalty, NOT residual PPO / belief-augmented policy) and acts as the R3 tripwire: when R3 wires
`ppo_clip_actor_loss` into the residual trainer, `test_residual_update_does_not_call_ppo_clip_spy` FLIPS.

Per Contract v4 §7 (No Substitution) — the existence of PPO in graph_mappo does NOT prove residual PPO.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "train", _ROOT / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --- (1) PPO machinery EXISTS in the Graph-MAPPO trunk (the "PPO exists" half) -----------------------

def test_graph_mappo_defines_ppo_clip_and_approx_kl() -> None:
    from marl_topology.training.graph_mappo import approx_kl, ppo_clip_actor_loss
    # epoch-0 invariant: logp_new == logp_old -> ratio == 1 -> approx_kl == 0, loss == -mean(A)
    logp = torch.tensor([-0.5, -1.0, -0.2])
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss, info = ppo_clip_actor_loss(logp, logp, adv, clip_eps=0.2)
    assert float(info["approx_kl"]) == pytest.approx(0.0, abs=1e-7)
    assert float(info["clip_fraction"]) == pytest.approx(0.0, abs=1e-7)
    assert float(loss) == pytest.approx(-adv.mean().item(), abs=1e-6)
    assert float(approx_kl(logp, logp)) == pytest.approx(0.0, abs=1e-7)


# --- (2) the residual trainer SOURCE is REINFORCE-only (No Substitution; Contract v4 §11) ------------

def test_residual_trainer_source_is_reinforce_no_ppo_no_critic_no_csi() -> None:
    src = (_ROOT / "scripts" / "diagnostics" / "residual_pbrs_train.py").read_text(encoding="utf-8")
    # present: REINFORCE + moving baseline + flip-penalty trust region
    assert "REINFORCE" in src and "baseline" in src and "flip_pen" in src
    # absent: PPO clip / approx_kl / value critic / entropy bonus / CSI-belief auxiliary
    for forbidden in ("ppo_clip_actor_loss", "approx_kl", "clip_fraction", "target_kl",
                      "value_loss", "critic", "belief", "L_CSI", "entropy"):
        assert forbidden not in src, f"residual trainer unexpectedly references {forbidden!r} (audit stale)"
    # the only 'clip' is gradient-norm clipping, NOT PPO clip
    assert "clip_grad_norm_" in src


# --- (3) LOAD-BEARING SPY: a residual update does NOT call ppo_clip_actor_loss (the R3 tripwire) ------

def _tiny_args():
    return SimpleNamespace(dyn_nodes=[8], tx_power=20.0, frames=2, gamma=0.95, hidden=16,
                           residual_prior=-3.0, lam_pbrs=0.5)


def test_residual_update_does_not_call_ppo_clip_spy(monkeypatch) -> None:
    import residual_pbrs_train as rp
    from marl_topology.training import graph_mappo
    from marl_topology.training.decentralized_distillation import feature_standardization

    calls = {"ppo": 0}
    real = graph_mappo.ppo_clip_actor_loss

    def spy(*a, **k):
        calls["ppo"] += 1
        return real(*a, **k)

    monkeypatch.setattr(graph_mappo, "ppo_clip_actor_loss", spy)

    args = _tiny_args()
    T = rp._load_trunk()
    scenes = rp._build("random", 12345, 1, args)            # 1 tiny random scene, N=8, 2 frames
    stat = [s.observation(0, []) for s in scenes]
    mean, std = feature_standardization(stat)
    nd, ed = stat[0]["nf"].shape[1], stat[0]["ef"].shape[1]
    actor = rp.make_actor("mlp", nd, ed, args.hidden)
    opt = torch.optim.Adam(actor.parameters(), lr=0.02)
    gen = torch.Generator().manual_seed(1)
    rp.reinforce_update(actor, opt, scenes, mean, std, T, mode="full", residual_prior=args.residual_prior,
                        lam_pbrs=args.lam_pbrs, gamma=args.gamma, use_pbrs=True, baseline=0.0,
                        generator=gen, anchor_reg=0.5)
    assert calls["ppo"] == 0, "residual trainer called ppo_clip_actor_loss -- R0 audit is stale (PPO now wired)"
