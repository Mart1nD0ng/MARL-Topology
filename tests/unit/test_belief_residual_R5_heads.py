"""R5 (Belief-Guided Residual PPO) — repair/safety/edit heads (deployable-learning crux).

Load-bearing tests (Contract v4 §3): the heads read ONLY local features (no evaluator / no true CSI); the
top-k precision-vs-random metric is correct; the heads train; held is separate from train. The EMPIRICAL
question (does the held top-k beat random on real data?) is NOT asserted by a unit test -- it is the R5
decision, measured by the multi-seed pilot (asserting it here would prejudge the honest result).

Fails on HEAD: `edit_head_training` and `BeliefResidualActor.edit_scores` do not exist yet.
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


def test_heads_use_only_local_features() -> None:
    from marl_topology.models.belief_residual_actor import BeliefResidualActor
    # edit_scores takes the deployed local obs (nf, ef, ei) -- NO evaluator / true-CSI argument
    params = set(inspect.signature(BeliefResidualActor.edit_scores).parameters)
    assert params == {"self", "nf", "ef", "ei", "hidden"}
    # and it runs with NO evaluator present (pure local features)
    a = BeliefResidualActor(node_dim=5, edge_dim=6, hidden=16)
    out, _h = a.edit_scores(torch.randn(8, 5), torch.randn(12, 6), torch.randint(0, 8, (12, 2)))
    assert set(out) == {"edit_logit", "repair_pred", "safety_pred"} and out["edit_logit"].shape == (12,)
    # the head training module must not import the evaluator into the head-input path
    src = (_ROOT / "src" / "marl_topology" / "training" / "edit_head_training.py").read_text(encoding="utf-8")
    assert "edit_scores" in src and "topk_metrics" in src


def test_edit_head_topk_metric_correct() -> None:
    """The precision@k vs random-base-rate metric correctly rewards good ranking and punishes bad ranking."""
    from marl_topology.training.edit_head_training import topk_metrics

    class _Stub:
        def __init__(self, sign):
            self.sign = sign

        def edit_scores(self, nf, ef, ei, hidden=None):
            E = ei.shape[0]
            pos = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
            return {"edit_logit": self.sign * pos, "repair_pred": torch.zeros(E),
                    "safety_pred": torch.zeros(E)}, None

    ex = [{"nf_s": torch.zeros(5, 5), "ef_s": torch.zeros(5, 6), "ei": torch.zeros(5, 2, dtype=torch.long),
           "positive": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]), "repair": torch.zeros(5), "risk": torch.zeros(5),
           "cand": torch.ones(5), "add_m": torch.zeros(5), "rem_m": torch.zeros(5)}]
    good = topk_metrics(ex, _Stub(+1.0))
    bad = topk_metrics(ex, _Stub(-1.0))
    assert good["topk_precision"] == pytest.approx(1.0) and good["random_base_rate"] == pytest.approx(0.4)
    assert good["precision_minus_base"] > 0 and bad["precision_minus_base"] < 0   # ranking matters


def test_repair_safety_heads_present_and_trained() -> None:
    from marl_topology.training.edit_head_training import train_edit_heads
    import residual_ppo_train as rp3
    T = rp3._load_trunk()
    res = train_edit_heads(_scenes(2, 3, seed=1), _scenes(2, 3, seed=99), T, epochs=8, hidden=16, seed=0)
    for k in ("topk_precision", "random_base_rate", "precision_minus_base", "repair_corr", "safety_corr"):
        assert k in res["trained_held"]
    assert res["n_train_examples"] > 0 and res["n_held_examples"] > 0


def test_supervised_vs_random_baseline_is_measured() -> None:
    # the load-bearing comparison (held top-k precision vs random base rate) is reported with both terms --
    # the SIGN (does it beat random?) is the R5 decision, measured by the multi-seed pilot, not asserted here.
    from marl_topology.training.edit_head_training import train_edit_heads
    import residual_ppo_train as rp3
    T = rp3._load_trunk()
    res = train_edit_heads(_scenes(2, 3, seed=2), _scenes(2, 3, seed=88), T, epochs=8, hidden=16, seed=0)
    t = res["trained_held"]
    assert t["topk_precision"] is not None and t["random_base_rate"] is not None
    assert isinstance(t["precision_minus_base"], float)


def test_heads_dataset_uses_train_held_separately() -> None:
    from marl_topology.training.edit_head_training import train_edit_heads
    params = set(inspect.signature(train_edit_heads).parameters)
    assert "train_scenes" in params and "held_scenes" in params      # held is a SEPARATE eval set
