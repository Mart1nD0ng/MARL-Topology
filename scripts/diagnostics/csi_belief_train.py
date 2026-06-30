"""R2 (Belief-Guided Residual PPO) — train the POLICY actor's GRU + belief head with L_CSI.

This is the path where the CSI belief loss ENTERS a training loss (not a Q2-style standalone diagnostic):
the same BeliefResidualActor the residual policy uses (its GRU + belief_head) is supervised to predict the
TRUE current psucc from stale history. Reports held belief MSE (recurrent vs memoryless) per CSI mode. The
true current psucc is a training-only LABEL; the actor input is the staleified ef (no leak). At R3 this
L_CSI is added alongside the residual PPO loss.
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
from marl_topology.training.csi_belief import (belief_mse, belief_target_logits,  # noqa: E402
                                               belief_weights, csi_belief_loss,
                                               leak_free_motion_features, stale_echo_floor_mse)
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402
from marl_topology.training.dynamic_rl import _standardize  # noqa: E402
from marl_topology.training.residual_saturation import feature_standardization_all_frames  # noqa: E402
from marl_topology.training.two_timescale_env import ReconfigCost  # noqa: E402


def _scenes(mode, delay, seed, count, frames, data="random"):
    from build_operating_point_dataset import operating_point_regime
    from marl_topology.training.dynamic_frames import (sample_dynamic_scenes,
                                                       sample_dynamic_urban_scenes)
    csi = (CsiObservationModel(mode="current") if mode == "current"
           else CsiObservationModel(mode="partial", probe_probability=0.5, seed=seed) if mode == "partial"
           else CsiObservationModel(mode="delay", delay_frames=delay))
    common = dict(seed=seed, count=count, node_count_choices=(8, 12, 16),
                  regime=operating_point_regime(20.0), num_frames=frames, dt_s=2.0, speed_min_mps=15.0,
                  speed_max_mps=30.0, reconfig=ReconfigCost(), hold_interval=4, gamma=0.95,
                  csi_observation_model=csi)
    if data == "urban":
        return sample_dynamic_urban_scenes(rsu_count=4, blocks_per_side=3, **common)
    return sample_dynamic_scenes(**common)


_VEL_SCALE = torch.tensor([30.0, 60.0])           # normalize [rel_vel m/s, distance_delta m] to ~[-2,2]


def _extra(sc, t, eids, use_velocity):
    """Leak-free per-edge velocity features (or None) for the belief head."""
    if not use_velocity:
        return None
    return leak_free_motion_features(sc, t, eids) / _VEL_SCALE


def _epoch_loss(actor, scenes, mean, std, recurrent, use_velocity):
    """Sum L_CSI over scenes (carry GRU hidden iff recurrent). Belief target = true current psucc (label)."""
    total = torch.zeros(())
    for sc in scenes:
        h = None
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            eids = obs["edge_ids"]
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            bel, h = actor.belief(nf_s, ef_s, obs["ei"], extra=_extra(sc, t, eids, use_velocity),
                                  hidden=(h if recurrent else None))
            tgt = belief_target_logits(sc, t, eids)
            w = belief_weights(eids, prev_topo=[], anchor=list(eids))   # anchor-ish emphasis (R5 refines)
            total = total + csi_belief_loss(bel, tgt, w)
    return total


def _held_metrics(actor, scenes, mean, std, recurrent, use_velocity):
    """Held belief MSE AND the stale-echo-floor MSE (predict the stale observed psucc) -- the honest baseline."""
    bel_vals, floor_vals = [], []
    for sc in scenes:
        h = None
        for t in range(sc.n_frames):
            obs = sc.observation(t, [])
            eids = obs["edge_ids"]
            nf_s, ef_s = _standardize(obs["nf"], obs["ef"], mean, std)
            with torch.no_grad():
                bel, h = actor.belief(nf_s, ef_s, obs["ei"], extra=_extra(sc, t, eids, use_velocity),
                                      hidden=(h if recurrent else None))
            if bel.numel():
                bel_vals.append(belief_mse(bel, belief_target_logits(sc, t, eids)))
                floor_vals.append(stale_echo_floor_mse(sc, t, eids, obs["ef"][:, 0]))
    return (sum(bel_vals) / max(1, len(bel_vals)), sum(floor_vals) / max(1, len(floor_vals)))


def train_belief(train_scenes, held_scenes, *, recurrent: bool, epochs: int = 60, hidden: int = 32,
                 lr: float = 0.02, seed: int = 0, use_velocity: bool = True) -> dict:
    """Train BeliefResidualActor's GRU + belief_head on L_CSI; return held belief MSE vs the stale-echo floor.

    ``use_velocity`` feeds the LEAK-FREE per-edge velocity/distance (no csi_delta) so the head can extrapolate
    the current channel from the stale observation -- the signal Q2 needed. Beating the stale-echo floor is
    the real test of CSI recovery (vs the trivial echo of the stale input)."""
    torch.manual_seed(seed)
    mean, std = feature_standardization_all_frames(train_scenes)
    nd = train_scenes[0].observation(0, [])["nf"].shape[1]
    ed = train_scenes[0].observation(0, [])["ef"].shape[1]
    actor = BeliefResidualActor(nd, ed, hidden=hidden, belief_extra_dim=(2 if use_velocity else 0))
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    hist, floor = [], 0.0
    for _e in range(epochs):
        loss = _epoch_loss(actor, train_scenes, mean, std, recurrent, use_velocity)
        opt.zero_grad(); loss.backward(); opt.step()
        bmse, floor = _held_metrics(actor, held_scenes, mean, std, recurrent, use_velocity)
        hist.append(bmse)
    return {"recurrent": recurrent, "epochs": epochs, "use_velocity": use_velocity,
            "held_belief_mse": hist[-1] if hist else float("nan"),
            "stale_echo_floor_mse": round(floor, 6),
            "beats_stale_echo_floor": (hist[-1] < floor) if hist else False,
            "belief_mse_history": hist, "n_params": sum(p.numel() for p in actor.parameters())}


def main() -> None:
    report = {"scope": "R2 CSI belief auxiliary on the POLICY actor; recurrent vs memoryless held belief MSE; "
                       "true current psucc = training-only label", "by_mode": {}}
    for mode, delay in [("current", 0), ("delay1", 1), ("delay2", 2), ("partial", 0)]:
        train = _scenes(mode, delay, 1001, 5, 6)
        held = _scenes(mode, delay, 1777, 5, 6)
        rec = train_belief(train, held, recurrent=True, epochs=80, hidden=32, seed=0)
        mem = train_belief(train, held, recurrent=False, epochs=80, hidden=32, seed=0)
        report["by_mode"][mode] = {
            "recurrent_held_belief_mse": round(rec["held_belief_mse"], 6),
            "memoryless_held_belief_mse": round(mem["held_belief_mse"], 6),
            "recurrent_beats_memoryless": rec["held_belief_mse"] < mem["held_belief_mse"],
            "improvement": round(mem["held_belief_mse"] - rec["held_belief_mse"], 6)}
        r = report["by_mode"][mode]
        print(f"  [{mode:8}] recurrent {r['recurrent_held_belief_mse']:.5f}  memoryless "
              f"{r['memoryless_held_belief_mse']:.5f}  rec<mem={r['recurrent_beats_memoryless']} "
              f"(improvement {r['improvement']:+.5f})")
    out = ROOT / "result_save" / "belief_residual" / "R2" / "csi_belief_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
