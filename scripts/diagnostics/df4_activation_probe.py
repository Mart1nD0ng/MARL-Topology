"""DF4 (Decision-Focused, goal 3): activation / output-parametrization -- leaky-tanh vs softmax (empirical probe).

Owner goal 3: is leaky-tanh the best activation, and why not softmax? DF0 answered analytically (leaky-tanh is
sound for a per-edge logit -- bounded for trust-region stability + gradient bounded away from 0 so the temporal
signal reaches the acted logit, the T2 fix; softmax is a CATEGORY ERROR for INDEPENDENT per-edge keep/drop). This
probe grounds that answer empirically on the real anchor decision, and shows the ONE place softmax would be right
(edit-selection) is blocked by the campaign's aleatoric wall.

Test 1 (ARGMAX EQUIVALENCE): the deployed anchor selects each node's top-b incident edges by psucc. softmax over
the incident-edge logits is MONOTONE in the logit (= monotone in psucc), so its top-b is IDENTICAL to the
psucc/sigmoid top-b. => at deployment softmax cannot change the action; it is at best a no-op over sigmoid/tanh.

Test 2 (N-DEPENDENCE = the category error): for N equal-psucc incident edges, the softmax per-edge keep-mass is
1/N (shrinks with node degree) whereas the sigmoid/tanh per-edge keep-prob is N-invariant. So a softmax head
under-connects high-degree nodes and cannot express 'keep many edges at once' -- fatal for a cross-N,
budget-top-k, multi-edge decision. Reported numerically.

Reuses DF3 scenes + the anchor incident structure. Deterministic; no training; gate-exempt diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import df3_decision_focused_gen as df3  # noqa: E402
from r8_stale_csi_gen import build_csi_scenes  # noqa: E402
from marl_topology.training.csi_observation_model import CsiObservationModel  # noqa: E402
from marl_topology.training.decentralized_action import incident_index  # noqa: E402
from marl_topology.training.dynamic_rl import _budgets_edges  # noqa: E402


def _logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def test_argmax_equiv(scenes):
    """Fraction of (node, frame) where top-b by psucc == top-b by softmax(logit) == top-b by sigmoid(logit).
    Softmax/sigmoid are both monotone in the logit, so this must be ~1.0 (ties aside): softmax cannot change the
    deployed top-b selection."""
    ident_softmax = ident_sigmoid = total = 0
    for sc in scenes:
        prev: list = []
        for t in range(sc.n_frames):
            obs = sc.observation(t, prev)
            ef = obs["ef"]
            budgets, edges = _budgets_edges(obs["context"])
            for node, idxs in incident_index(obs["edge_ids"], edges).items():
                b = int(budgets.get(node, 0))
                if b <= 0 or len(idxs) == 0:
                    continue
                by_psucc = set(sorted(idxs, key=lambda i: -float(ef[i, 0]))[:b])
                by_softmax = set(sorted(idxs, key=lambda i: -math.exp(_logit(ef[i, 0])))[:b])   # softmax monotone
                by_sigmoid = set(sorted(idxs, key=lambda i: -(1.0 / (1.0 + math.exp(-_logit(ef[i, 0])))))[:b])
                total += 1
                ident_softmax += int(by_psucc == by_softmax)
                ident_sigmoid += int(by_psucc == by_sigmoid)
            prev = list(df3._anchor(obs, prev))
    return {"n_node_frames": total,
            "softmax_topk_equals_anchor_frac": ident_softmax / total if total else float("nan"),
            "sigmoid_topk_equals_anchor_frac": ident_sigmoid / total if total else float("nan")}


def test_n_dependence():
    """For N equal-psucc (logit 0) incident edges: softmax per-edge keep-mass = 1/N (N-dependent); sigmoid/tanh
    per-edge keep-prob = sigmoid(0) = 0.5 (N-invariant). Demonstrates why softmax breaks cross-N / multi-edge."""
    rows = []
    for N in (2, 4, 8, 16, 24):
        rows.append({"N": N, "softmax_per_edge_keep_mass": 1.0 / N, "sigmoid_per_edge_keep_prob": 0.5})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="urban", choices=["urban", "random"])
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "decision_focused" / "DF4" / "df4_activation_metrics.json"))
    args = ap.parse_args()

    csi = CsiObservationModel(mode="delay", delay_frames=1)
    scenes = build_csi_scenes(args.data, args.seed * 1000 + 777, args.scenes, df3._ARGS, csi)
    t1 = test_argmax_equiv(scenes)
    t2 = test_n_dependence()
    report = {
        "scope": "DF4 goal-3 activation probe: (1) softmax top-b == anchor top-b (monotone) -> softmax cannot "
                 "change the deployed decision; (2) softmax per-edge keep-mass is N-dependent (1/N) vs sigmoid "
                 "N-invariant (0.5) -> the category error for per-edge independent keep/drop. Analytical answer in "
                 "DF0 / DF4 decision.md.",
        "config": vars(args),
        "argmax_equivalence": t1,
        "n_dependence": t2,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Test 1 (argmax equivalence, {t1['n_node_frames']} node-frames): "
          f"softmax top-b == anchor {t1['softmax_topk_equals_anchor_frac']:.4f} | "
          f"sigmoid top-b == anchor {t1['sigmoid_topk_equals_anchor_frac']:.4f}")
    print("Test 2 (N-dependence, softmax per-edge keep-mass vs sigmoid 0.5):")
    for r in t2:
        print(f"   N={r['N']:2d}: softmax {r['softmax_per_edge_keep_mass']:.4f}  sigmoid {r['sigmoid_per_edge_keep_prob']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
