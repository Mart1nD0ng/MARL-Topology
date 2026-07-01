"""T1 (Temporal Recovery) generator — the CSI-recovery UPPER-BOUND oracle (THE GATE).

Question: is the stale-CSI feasibility drop (R8: urban ~0.165 / random ~0.055) REALIZABLY recoverable from
stale t-1 + leak-free geometry, or reachable only by a clairvoyant oracle that reads the true current channel?

The ``local_hysteresis`` anchor decides PURELY on psucc (col 0, ``dynamic_baselines._PSUCC_COL``). So we feed
the SAME anchor four different psucc vectors and score the produced topology with the TRUE evaluator (0 eval at
decision -- the evaluator scores ONLY the produced topology):
  A floor     stale psucc                     (= R8 stale_anchor; the deployed regime)
  B direct    sigmoid(net(feats))             (a DIRECT-neural predictor of current psucc -- current-arch style)
  C physics   clamp(stale + net(feats), 0, 1) (a physics-RESIDUAL/correction predictor -- realizable-info ceiling)
  D oracle    true current psucc              (= R8 current_anchor; clairvoyant upper bound)

B and C are small MLPs trained (per seed) on TRAIN scenes with the true-current psucc LABEL (training-only).
Their INFERENCE reads ONLY leak-free features [stale psucc/latency/energy, CURRENT distance (col 5 is NOT
staleified), relative velocity, distance_delta, csi_age] -- never the true current CSI. B predicts the ABSOLUTE
current psucc (no echo floor); C predicts a CORRECTION on the stale value (echo is its zero-baseline, the T3
inductive bias). Reports per seed + 95% CI:
  ceiling      = D - A   (the clairvoyant headroom; cross-checks R8 stale_drop)
  realizable   = C - A   (THE GATE: does a realizable geometry-informed predictor beat the stale floor?)
  arch_gap     = C - B   (does the correction/physics structure recover more than direct prediction?)
  model        = B - A   (where a current-style direct predictor sits)
  MSE landscape: stale-echo (= A error) / B / C / 0 (= D) in psucc probability space.
Decision (Contract v4): C-A CI>0 and C<D -> realizable headroom exists -> proceed T2-T5. C-A spans/below 0 ->
recheck oracle design, then pivot to leak-safe env temporal hidden features (task-1 fallback).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import r6_evidence_gate_gen as g6  # noqa: E402
from r8_stale_csi_gen import _A, _ci95, _TAU_FEAS, build_csi_scenes  # noqa: E402
from marl_topology.training.csi_belief import belief_target_logits, leak_free_motion_features  # noqa: E402
from marl_topology.training.edit_head_training import _anchor  # noqa: E402

_ARMS = ("A_stale", "B_direct", "C_physics", "D_true")


# --------------------------------------------------------------------------- features / labels
def _edge_features(sc, t: int, obs) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-edge LEAK-FREE features in obs['edge_ids'] order + the stale psucc column (arm A / the residual base).
    Under the delay overlay ef cols are: 0,1 stale psucc | 2 stale latency | 3 stale energy | 4 prev-topo |
    5 CURRENT distance (NOT staleified) | 6,7 roles | 8 csi_age | 9 csi_observed_mask. We use the stale channel
    (0,2,3) + the CURRENT geometry (distance col 5, relative velocity, distance_delta) + csi_age -- the exact
    info a deployed node has (its own + neighbour position/velocity by broadcast). NO true current CSI.
    NOTE: relative velocity and distance_delta are computed OUT-OF-BAND via ``leak_free_motion_features``
    (node positions/velocities only) -- they are NOT read from ef, and the env's leaky ``_motion_edge_block``
    (which contains a csi_delta term derived from the TRUE current channel) is NEVER appended here
    (build_csi_scenes leaves motion_features off). The only true-CSI reader is ``_true_psucc`` (the label)."""
    ef = obs["ef"]
    edge_ids = obs["edge_ids"]
    stale_psucc = ef[:, 0]
    stale_lat = ef[:, 2]
    stale_en = ef[:, 3]
    cur_dist = ef[:, 5]
    motion = leak_free_motion_features(sc, t, edge_ids)              # [E,2]: rel_vel, distance_delta
    age = ef[:, 8] if ef.shape[1] > 8 else ef.new_zeros(ef.shape[0])
    feats = torch.stack([stale_psucc, stale_lat, stale_en, cur_dist, motion[:, 0], motion[:, 1], age], dim=1)
    return feats, stale_psucc


def _true_psucc(sc, t: int, edge_ids) -> torch.Tensor:
    """The TRAINING-ONLY label / the arm-D oracle value: true current per-edge psucc from the true channel."""
    return torch.sigmoid(belief_target_logits(sc, t, edge_ids))


def _anchor_with_psucc(obs, prev, psucc: torch.Tensor):
    """Run the deployable local_hysteresis anchor with psucc col 0 (and its duplicate col 1) OVERWRITTEN by the
    arm's psucc vector -- everything else (budgets, edges, decoder) identical. 0 evaluator calls."""
    ef2 = obs["ef"].clone()
    ef2[:, 0] = psucc.to(ef2.dtype)                # the anchor sorts ONLY on col 0 (dynamic_baselines._PSUCC_COL)
    if ef2.shape[1] > 1:
        ef2[:, 1] = psucc.to(ef2.dtype)            # col 1 is the psucc duplicate in graph_payload; kept consistent
    obs2 = dict(obs)
    obs2["ef"] = ef2
    return _anchor(obs2, prev)


# --------------------------------------------------------------------------- predictors (B/C)
class _MLP(nn.Module):
    def __init__(self, fin: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_predictor(X: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, mode: str, seed: int,
                     epochs: int = 400):
    """Fit a small MLP predicting the true current psucc. mode='direct' -> sigmoid(net) (absolute, no echo
    floor); mode='physics' -> clamp(stale + net, 0, 1) (a correction on the stale value; echo is net==0)."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xn = (X - mu) / sd
    net = _MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        raw = net(Xn)
        pred = torch.sigmoid(raw) if mode == "direct" else (S + raw).clamp(0.0, 1.0)
        F.mse_loss(pred, Y).backward()
        opt.step()
    return net, mu, sd


def _predict(net, mu, sd, X: torch.Tensor, S: torch.Tensor, mode: str) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        raw = net((X - mu) / sd)
        return torch.sigmoid(raw) if mode == "direct" else (S + raw).clamp(0.0, 1.0)


def _collect(scenes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather (features, true-psucc label, stale psucc) over the anchor rollout (deployment distribution)."""
    X, Y, S = [], [], []
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            feats, stale = _edge_features(sc, t, obs)
            X.append(feats)
            Y.append(_true_psucc(sc, t, obs["edge_ids"]))
            S.append(stale)
            prev = list(_anchor(obs, prev))
    return torch.cat(X), torch.cat(Y), torch.cat(S)


# --------------------------------------------------------------------------- arm evaluation
def _eval_arms(scenes, preds: dict) -> dict:
    """Roll each arm independently over held scenes (each arm keeps its own prev trajectory). Report per-arm
    feasibility (true evaluator) and the psucc-recovery MSE landscape (stale-echo / B / C in probability space)."""
    feas = {a: [] for a in _ARMS}
    se_echo, se_b, se_c = [], [], []
    for sc in scenes:
        prev = {a: [] for a in _ARMS}
        for t in range(sc.n_frames):
            for a in _ARMS:
                obs = sc.observation(t, prev[a])
                feats, stale = _edge_features(sc, t, obs)
                if a == "A_stale":
                    p = stale
                elif a == "D_true":
                    p = _true_psucc(sc, t, obs["edge_ids"])
                elif a == "B_direct":
                    p = _predict(*preds["B"], feats, stale, "direct")
                else:
                    p = _predict(*preds["C"], feats, stale, "physics")
                topo = _anchor_with_psucc(obs, prev[a], p)
                ev = obs["context"].evaluator
                feas[a].append(1.0 if g6._cons(ev, topo) >= _TAU_FEAS else 0.0)
                prev[a] = list(topo)
            # MSE landscape (prev-independent features; use arm A's fresh obs)
            obs0 = sc.observation(t, prev["A_stale"])
            feats0, stale0 = _edge_features(sc, t, obs0)
            true0 = _true_psucc(sc, t, obs0["edge_ids"])
            pb = _predict(*preds["B"], feats0, stale0, "direct")
            pc = _predict(*preds["C"], feats0, stale0, "physics")
            se_echo += ((stale0 - true0) ** 2).tolist()
            se_b += ((pb - true0) ** 2).tolist()
            se_c += ((pc - true0) ** 2).tolist()

    def _m(x):
        return sum(x) / len(x) if x else 0.0
    return {"feas": {a: _m(feas[a]) for a in _ARMS},
            "mse": {"echo": _m(se_echo), "B_direct": _m(se_b), "C_physics": _m(se_c)}}


def main() -> None:
    from marl_topology.training.csi_observation_model import CsiObservationModel
    seeds = [0, 1, 2, 3, 4]
    csi = CsiObservationModel(mode="delay", delay_frames=1)
    report = {"scope": "T1 CSI-recovery upper-bound oracle (delay-1): 4 anchor arms stale/direct/physics/true, "
                       "5 seeds x {random,urban}; predictors trained on stale features + current geometry, true "
                       "current psucc = training-only label; deployed 0-eval, true evaluator scores.",
              "csi": csi.manifest(), "arms": list(_ARMS), "by_data": {}}
    for data in ["random", "urban"]:
        per: dict = {f"{a}_feas": [] for a in _ARMS}
        per.update({"ceiling_D_minus_A": [], "realizable_C_minus_A": [], "arch_C_minus_B": [],
                    "model_B_minus_A": [], "mse_echo": [], "mse_B": [], "mse_C": []})
        for s in seeds:
            tr = build_csi_scenes(data, s * 1000 + 1, 5, _A(), csi)
            hd = build_csi_scenes(data, s * 1000 + 777, 10, _A(), csi)
            X, Y, S = _collect(tr)
            netB = _train_predictor(X, Y, S, "direct", s)
            netC = _train_predictor(X, Y, S, "physics", s)
            r = _eval_arms(hd, {"B": netB, "C": netC})
            f = r["feas"]
            for a in _ARMS:
                per[f"{a}_feas"].append(f[a])
            per["ceiling_D_minus_A"].append(f["D_true"] - f["A_stale"])
            per["realizable_C_minus_A"].append(f["C_physics"] - f["A_stale"])
            per["arch_C_minus_B"].append(f["C_physics"] - f["B_direct"])
            per["model_B_minus_A"].append(f["B_direct"] - f["A_stale"])
            per["mse_echo"].append(r["mse"]["echo"])
            per["mse_B"].append(r["mse"]["B_direct"])
            per["mse_C"].append(r["mse"]["C_physics"])
        d = {k: _ci95(v) for k, v in per.items()}
        d["realizable_recovery"] = (d["realizable_C_minus_A"]["lo"] is not None
                                    and d["realizable_C_minus_A"]["lo"] > 0)
        report["by_data"][data] = d
        print(f"[{data}] A_stale {d['A_stale_feas']['mean']} B_direct {d['B_direct_feas']['mean']} "
              f"C_physics {d['C_physics_feas']['mean']} D_true {d['D_true_feas']['mean']} | "
              f"ceiling(D-A) {d['ceiling_D_minus_A']['mean']} realizable(C-A) {d['realizable_C_minus_A']['mean']} "
              f"CI[{d['realizable_C_minus_A']['lo']},{d['realizable_C_minus_A']['hi']}] "
              f"recoverable={d['realizable_recovery']} | MSE echo {d['mse_echo']['mean']} "
              f"B {d['mse_B']['mean']} C {d['mse_C']['mean']}")
    out = ROOT / "result_save" / "temporal_recovery" / "T1" / "oracle_recovery_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
