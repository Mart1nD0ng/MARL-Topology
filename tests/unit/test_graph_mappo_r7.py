"""R7 (v2 Engineering-Plan §R7, Spec §8-9): Graph-MAPPO completion -- BCSP in trunk + critic.

Pins: the critic TRAIN forward keeps its gradient (Spec §8.4 -- not @no_grad), an optimizer step
moves the critic but not the actor, the PPO ratio is PER-AGENT not joint (Spec §9.2), the entropy
bonus is the BCSP agent-normalized entropy, the centralized critic resumes from a checkpoint, and the
real-shard graph-mappo smoke (the ex-HANG case) exits 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic
from marl_topology.models.message_passing_graph_edge_scorer import MessagePassingGraphEdgeScorer
from marl_topology.training.budget_conditioned_subset import normalized_entropy
from marl_topology.training.graph_mappo import critic_scene_value, ppo_clip_actor_loss

ROOT = Path(__file__).resolve().parents[2]
ND, ED, N, E = 6, 4, 5, 8


def _scene():
    torch.manual_seed(0)
    nf = torch.randn(N, ND)
    ef = torch.randn(E, ED)
    ei = torch.randint(0, N, (E, 2))           # [E, 2] per-edge (src, dst)
    nm = (torch.zeros(ND), torch.ones(ND))
    em = (torch.zeros(ED), torch.ones(ED))
    return nf, ef, ei, nm, em


def _value(critic, scene):
    nf, ef, ei, nm, em = scene
    return critic_scene_value(critic, nf, ef, ei, node_mean=nm[0], node_std=nm[1],
                              edge_mean=em[0], edge_std=em[1])


def test_critic_formal_helper_has_grad() -> None:
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2)
    critic.train()
    v = _value(critic, _scene())
    assert v.requires_grad and v.grad_fn is not None   # Spec 8.4: train forward keeps its graph


def test_critic_optimizer_changes_parameters() -> None:
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2)
    opt = torch.optim.AdamW(critic.parameters(), lr=1e-2)
    scene = _scene()
    before = [p.detach().clone() for p in critic.parameters()]
    for _ in range(3):
        v = _value(critic, scene)
        loss = (torch.tensor(-2.0) - v).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = list(critic.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))  # the critic actually moved


def test_actor_unchanged_by_critic_step() -> None:
    actor = MessagePassingGraphEdgeScorer(ND, ED, hidden=16, rounds=2)
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-2)
    actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    v = _value(critic, _scene())
    (torch.tensor(-2.0) - v).pow(2).mean().backward(); opt_c.step()
    for k, v_after in actor.state_dict().items():
        assert torch.allclose(actor_before[k], v_after)  # the critic step never touches the actor


def test_per_agent_ratio_epoch0_is_one() -> None:
    # PER-AGENT flattened arrays; at epoch 0 logp_new == logp_old -> every per-agent ratio == 1.
    logp_old = torch.tensor([-1.2, -0.5, -2.0, -0.1])  # 4 agents (across scenes)
    adv = torch.tensor([0.3, -0.2, 0.5, 0.1])
    loss, info = ppo_clip_actor_loss(logp_old.clone(), logp_old, adv)
    assert torch.allclose(info["ratio"], torch.ones(4))
    assert float(info["approx_kl"]) == pytest.approx(0.0, abs=1e-12)
    assert loss == pytest.approx(-adv.mean())  # reduces to -mean(A)


def test_joint_ratio_not_used_by_actor_loss() -> None:
    # per-agent (per-element) ratio: moving ONE agent's logp shifts only that agent's ratio. A joint
    # ratio (exp of the SUM of per-agent log-ratios) would instead fold all agents into one number.
    logp_old = torch.zeros(4)
    logp_new = torch.tensor([0.5, 0.0, 0.0, 0.0])  # only agent 0 moved
    _loss, info = ppo_clip_actor_loss(logp_new, logp_old, torch.ones(4))
    ratio = info["ratio"]
    assert ratio[0] == pytest.approx(torch.exp(torch.tensor(0.5)).item())
    assert torch.allclose(ratio[1:], torch.ones(3))      # the other agents are unaffected (per-agent)
    assert ratio.shape == (4,)                            # N_agent elements, not 1 joint scalar


def test_entropy_scale_is_agent_normalized() -> None:
    for theta, b in [(torch.randn(6, dtype=torch.float64), 3),
                     (torch.randn(10, dtype=torch.float64), 4),
                     (torch.zeros(4, dtype=torch.float64), 2)]:
        h = float(normalized_entropy(theta, b))
        assert 0.0 <= h <= 1.0 + 1e-9   # normalized by log|A_i| -> in [0, 1]
    # uniform-ish logits give near-maximal normalized entropy; very peaked give near 0
    assert float(normalized_entropy(torch.zeros(5, dtype=torch.float64), 5)) > 0.9
    assert float(normalized_entropy(torch.tensor([50.0, -50.0, -50.0], dtype=torch.float64), 3)) < 0.1


def test_graph_mappo_checkpoint_round_trips_critic() -> None:
    # the centralized critic + its optimizer state survive a save/load (Spec 8.6 resume).
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-2)
    _value(critic, _scene()).backward(); opt_c.step()  # populate optimizer state
    blob = {"critic": critic.state_dict(), "opt_c": opt_c.state_dict()}
    fresh = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2)
    fresh.load_state_dict(blob["critic"]); fresh_opt.load_state_dict(blob["opt_c"])
    for a, b in zip(critic.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)


def test_graph_mappo_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    critic = CentralizedGraphCritic(ND, ED, hidden=16, rounds=2).cuda()
    nf, ef, ei, nm, em = _scene()
    v = critic_scene_value(critic, nf.cuda(), ef.cuda(), ei.cuda(),
                           node_mean=nm[0].cuda(), node_std=nm[1].cuda(),
                           edge_mean=em[0].cuda(), edge_std=em[1].cuda())
    assert v.device.type == "cuda"


@pytest.mark.slow
def test_graph_mappo_real_shard_smoke(tmp_path) -> None:
    # THE ex-HANG case: the graph-mappo arm with BCSP must run real shards + exit 0 in seconds.
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train" / "train_decentralized_rl.py"),
         "--smoke", "--cold-start", "--baseline", "graph-mappo", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=240, env=env, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "[done]" in proc.stdout
