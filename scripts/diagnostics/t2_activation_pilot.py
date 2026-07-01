"""T2 (Temporal Recovery) pilot: the non-saturating (leaky-tanh) activation on REAL stale-CSI observations.

Compares the frozen tanh head (residual_leak=0) vs the leaky-tanh (residual_leak>0) on the SAME actor init
over real delay-1 stale scenes: saturation metrics + the recurrent-vs-memoryless Effect-on-Decision delta on
the real (standardized) raw distribution. The leaky head must (a) stay non-degenerate (no rail) and (b) carry
at least as much recurrent->logit signal as the tanh head. Writes mechanism_activation.json (the activation
artifact, Contract v4 §15). Enabling-mechanism pilot, NOT a 5-seed headline (topology conversion is T6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from r8_stale_csi_gen import _A, build_csi_scenes  # noqa: E402
from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: E402
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.residual_saturation import (feature_standardization_all_frames,  # noqa: E402
                                                        recurrent_vs_memoryless_delta, saturation_metrics)

_LEAK = 0.1


def _run(actor, scenes, mean, std) -> dict:
    raws, logits, dl, da = [], [], [], []
    for sc in scenes:
        prev: list = []
        h = None
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            if len(obs["ei"]) == 0:
                continue
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            lr, raw, h = actor.forward(nf_s, ef_s, obs["ei"], hidden=h)          # recurrent (carry h)
            lm, _rawm, _hm = actor.forward(nf_s, ef_s, obs["ei"], hidden=None)   # memoryless (h from zeros)
            d = recurrent_vs_memoryless_delta(lr, lm)
            raws.append(raw.detach()); logits.append(lr.detach())
            dl.append(d["recurrent_memoryless_logit_delta"]); da.append(d["recurrent_memoryless_action_delta"])
            prev = list(obs["edge_ids"])[: max(1, len(obs["edge_ids"]) // 2)]
    raw = torch.cat(raws) if raws else torch.zeros(0)
    lg = torch.cat(logits) if logits else torch.zeros(0)
    return {"saturation": saturation_metrics(raw, lg, actor.residual_logit_scale),
            "recurrent_logit_delta_mean": float(sum(dl) / len(dl)) if dl else 0.0,
            "recurrent_logit_delta_max": float(max(dl)) if dl else 0.0,
            "recurrent_action_delta_mean": float(sum(da) / len(da)) if da else 0.0}


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    scenes = build_csi_scenes("urban", 777, 6, _A(), csi)
    mean, std = feature_standardization_all_frames(scenes)
    ex = scenes[0].observation(0, [])
    nd = _standardize(ex["nf"], ex["ef"], mean, std)[0].shape[1]
    ed = _standardize(ex["nf"], ex["ef"], mean, std)[1].shape[1]

    torch.manual_seed(0)
    tanh_actor = BeliefResidualActor(node_dim=nd, edge_dim=ed, hidden=64, residual_logit_scale=3.0,
                                     residual_leak=0.0)
    leak_actor = BeliefResidualActor(node_dim=nd, edge_dim=ed, hidden=64, residual_logit_scale=3.0,
                                     residual_leak=_LEAK)
    leak_actor.load_state_dict(tanh_actor.state_dict())          # identical init; ONLY leak differs

    r_tanh = _run(tanh_actor, scenes, mean, std)
    r_leak = _run(leak_actor, scenes, mean, std)
    report = {
        "scope": "T2 non-saturating leaky-tanh activation pilot on real delay-1 stale urban scenes; untrained "
                 "actor, identical init, only residual_leak differs. Enabling mechanism (not a 5-seed headline).",
        "residual_leak": _LEAK,
        "tanh_head": {"boundary_report": tanh_actor.boundary_report(), **r_tanh},
        "leak_head": {"boundary_report": leak_actor.boundary_report(), **r_leak},
        "activation_non_saturating": leak_actor.boundary_report()["non_saturating_activation"],
        "leak_carries_ge_signal": r_leak["recurrent_logit_delta_mean"] >= r_tanh["recurrent_logit_delta_mean"],
    }
    out = ROOT / "result_save" / "temporal_recovery" / "T2" / "mechanism_activation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"tanh rail {r_tanh['saturation']['frac_logit_near_rail']} raw_abs_mean {r_tanh['saturation']['raw_abs_mean']:.2f} rec_logit_delta {r_tanh['recurrent_logit_delta_mean']:.4f}")
    print(f"leak rail {r_leak['saturation']['frac_logit_near_rail']} raw_abs_mean {r_leak['saturation']['raw_abs_mean']:.2f} rec_logit_delta {r_leak['recurrent_logit_delta_mean']:.4f}")
    print(f"leak_carries_ge_signal={report['leak_carries_ge_signal']} wrote {out}")


if __name__ == "__main__":
    main()
