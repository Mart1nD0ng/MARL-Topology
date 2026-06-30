"""R1 pilot: does the separate small-range head + raw-L2 fix the saturation that made recurrence inert?

Compares the OLD shared +-10 head (DynamicRecurrentActor) vs the NEW small-range residual head
(BeliefResidualActor, z_max=3 + raw-L2) on a REAL stale (delay-1) scene, AFTER an identical brief
logit-pushing optimization (loss = -mean(logit), the REINFORCE-to-extreme failure mode that Q14 hit).
Reports, on held frames: saturation (frac_logit_near_rail, raw_abs_mean) and the Effect-on-Decision delta
(recurrent vs memoryless logit/action). Eval-only diagnostic; no production-path change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import torch  # noqa: E402

from marl_topology.models.belief_residual_actor import BeliefResidualActor  # noqa: E402
from marl_topology.models.dynamic_recurrent_actor import DynamicRecurrentActor  # noqa: E402
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.residual_saturation import (  # noqa: E402
    feature_standardization_all_frames, raw_logit_l2_penalty, recurrent_vs_memoryless_delta,
    saturation_metrics)
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402


def _scenes(seed, count, frames):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import sample_dynamic_urban_scenes
    return sample_dynamic_urban_scenes(
        rsu_count=4, blocks_per_side=3, seed=seed, count=count, node_count_choices=(8, 12, 16),
        regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0, speed_min_mps=15.0,
        speed_max_mps=30.0, reconfig=ReconfigCost(), hold_interval=4, gamma=0.95,
        csi_observation_model=CsiObservationModel(mode="delay", delay_frames=1))


def _push_logits(actor, train, mean, std, *, is_new, lam_raw, steps=60):
    """Brief optimization pushing logits to the positive rail (Q14's REINFORCE-to-extreme failure mode)."""
    opt = torch.optim.Adam(actor.parameters(), lr=0.03)
    for _ in range(steps):
        loss = torch.zeros(())
        for s in train:
            obs = s.observation(0, [])
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            if is_new:
                z, raw, _h = actor(nf_s, ef_s, obs["ei"], hidden=None)
                loss = loss - z.mean() + lam_raw * raw_logit_l2_penalty(raw)
            else:
                z, _h = actor(nf_s, ef_s, obs["ei"], hidden=None)
                loss = loss - z.mean()
        opt.zero_grad(); loss.backward(); opt.step()


def _measure(actor, held, mean, std, *, is_new, scale):
    """On held frame>=1, measure saturation + recurrent-vs-memoryless delta."""
    sat, deltas = [], []
    for s in held:
        h = None
        for t in range(s.n_frames):
            obs = s.observation(t, [])
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            with torch.no_grad():
                if is_new:
                    z_mem, _rm, _hm = actor(nf_s, ef_s, obs["ei"], hidden=None)
                    z_rec, raw_r, h = actor(nf_s, ef_s, obs["ei"], hidden=h)
                else:
                    z_mem, _hm = actor(nf_s, ef_s, obs["ei"], hidden=None)
                    z_rec, h = actor(nf_s, ef_s, obs["ei"], hidden=h)
                    raw_r = torch.atanh((z_rec / scale).clamp(-0.999999, 0.999999)) * scale   # invert head
            if t >= 1 and z_rec.numel():
                sat.append(saturation_metrics(raw_r, z_rec, scale))
                deltas.append(recurrent_vs_memoryless_delta(z_rec, z_mem))
    def avg(key, src):
        vals = [d[key] for d in src]
        return round(sum(vals) / max(1, len(vals)), 5)
    return {"frac_logit_near_rail": avg("frac_logit_near_rail", sat),
            "raw_abs_mean": avg("raw_abs_mean", sat),
            "recurrent_memoryless_logit_delta": avg("recurrent_memoryless_logit_delta", deltas),
            "recurrent_memoryless_action_delta": avg("recurrent_memoryless_action_delta", deltas),
            "behaviorally_equal_to_baseline_frac": round(
                sum(d["behaviorally_equal_to_baseline"] for d in deltas) / max(1, len(deltas)), 3)}


def main() -> None:
    torch.manual_seed(0)
    train = _scenes(1001, 4, 5)
    held = _scenes(1777, 4, 5)
    mean, std = feature_standardization_all_frames(train)
    nd = train[0].observation(0, [])["nf"].shape[1]
    ed = train[0].observation(0, [])["ef"].shape[1]

    old = DynamicRecurrentActor(nd, ed, hidden=64)
    _push_logits(old, train, mean, std, is_new=False, lam_raw=0.0)
    old_m = _measure(old, held, mean, std, is_new=False, scale=10.0)

    torch.manual_seed(0)
    new = BeliefResidualActor(nd, ed, hidden=64, residual_logit_scale=3.0)
    _push_logits(new, train, mean, std, is_new=True, lam_raw=0.02)
    new_m = _measure(new, held, mean, std, is_new=True, scale=3.0)

    report = {"scope": "R1 saturation fix pilot: old +-10 shared head vs new +-3 separate head + raw-L2; "
                       "urban delay-1; after identical logit-pushing optimization",
              "old_head_pm10": old_m, "new_head_pm3_rawL2": new_m,
              "exit_condition": {
                  "saturation_dropped": new_m["frac_logit_near_rail"] < old_m["frac_logit_near_rail"],
                  "recurrent_not_bit_identical": new_m["recurrent_memoryless_logit_delta"] > 1e-3,
                  "old_was_inert": old_m["recurrent_memoryless_logit_delta"] < 1e-3}}
    print(json.dumps(report, indent=2))
    out = ROOT / "result_save" / "belief_residual" / "R1" / "saturation_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
