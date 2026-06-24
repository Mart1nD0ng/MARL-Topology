"""8b vs R7 credit-RESOLUTION diagnostic (Phase 8 mechanism; Spec S9.4 / S13).

HONEST SCOPE. This is a MECHANISM diagnostic, NOT a corrected-env-math headline (the op shards are
pre-corrected-env-math; the corrected 8b-vs-R7 training headline is gated on the dataset rebuild). It
answers one question: does the COMA per-agent counterfactual credit carry per-agent information that the
R7 shared scene advantage A_s = r - V structurally cannot?

  Part A (always): on real scenes, the TRUE per-agent marginal contribution
      Delta_i = r(S) - E_{S~_i ~ pi_i}[ r(decode(S~_i, S_-i)) ]
  computed with the REAL evaluator (this is COMA with Q = the true reward). We report the mean
  within-scene variance of Delta_i. The R7 shared advantage gives EVERY agent the same scalar -> its
  within-scene variance is exactly 0. So any positive Delta_i variance is per-agent credit the shared
  advantage cannot represent. The evaluator calls here are DIAGNOSTIC-ONLY and never enter training
  (training stays at 1 evaluator call/scene -- reported separately below).

  Part B (only with --q-ckpt): the LEARNED-Q COMA advantage A_i (the budget-neutral training signal)
  vs Delta_i, by Spearman rank correlation pooled over (scene, agent) -- Spec S13 counterfactual rank
  correlation. On pre-corrected data a weak correlation is reported honestly as Q-quality-limited
  (deferred to the rebuild), NOT spun as a headline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_topology.training.counterfactual_credit import (  # noqa: E402
    counterfactual_advantages,
    per_agent_counterfactual_credit,
    within_scene_credit_variance,
)


def _load_trunk():
    spec = importlib.util.spec_from_file_location(
        "trunk", ROOT / "scripts" / "train" / "train_decentralized_rl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _spearman(xs, ys) -> float:
    """Spearman rank correlation (manual; ties -> average ranks). Returns 0.0 for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--k-cf", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--warm-start", action="store_true",
                    help="use the frozen BC actor (realistic near-feasible topologies) instead of cold-start")
    ap.add_argument("--artifacts", default=None, help="BC actor artifacts (default: trunk DEFAULT_ARTIFACTS)")
    ap.add_argument("--warm-actor-idx", type=int, default=2)
    ap.add_argument("--q-ckpt", default=None,
                    help="optional trained Q critic ckpt (Part B learned-Q rank correlation)")
    ap.add_argument("--out", default=str(ROOT / "result_save" / "coma_credit_resolution.json"))
    args = ap.parse_args()

    T = _load_trunk()
    torch.manual_seed(args.seed)
    shards = args.shards or T.OP_SHARDS[:2]
    pool = T.load_pool(shards)
    samples = T.build_samples(pool)
    mean, std = T.feature_standardization(samples)
    node_dim, edge_dim = samples[0]["nf"].shape[1], samples[0]["ef"].shape[1]
    critic = None
    if args.q_ckpt:
        # Part B: load the MATCHED actor + Q critic + standardization from the SAME run, so the learned
        # Q is scored on-distribution (the actions its actor actually produces).
        ck = torch.load(args.q_ckpt, map_location="cpu", weights_only=False)
        assert ck.get("critic_sees_action"), "--q-ckpt must be an action-conditioned Q critic (8b)"
        mean, std = ck["mean"], ck["std"]
        actor = T.load_actor_from_state(ck["actor"], node_dim, edge_dim, hidden=args.hidden, rounds=args.rounds)
        critic = T.CentralizedGraphCritic(node_dim, edge_dim, hidden=args.hidden, rounds=args.rounds,
                                          critic_sees_action=True)
        critic.load_state_dict(ck["critic"])
        critic.eval()
        actor_tag = "q-ckpt matched actor+Q (on-distribution)"
    elif args.warm_start:
        # frozen BC actor: realistic near-feasible topologies (its own normalization)
        art = torch.load(args.artifacts or T.DEFAULT_ARTIFACTS, map_location="cpu", weights_only=False)
        a = art["actors"][args.warm_actor_idx]
        mean, std = a["mean"], a["std"]
        actor = T.load_actor_from_state(a["state"], node_dim, edge_dim, hidden=args.hidden, rounds=args.rounds)
        actor_tag = f"warm-start BC[{args.warm_actor_idx}]"
    else:
        # cold-start (random-init) actor: it only defines pi_i for sampling; Delta_i is evaluator-based.
        actor = T.MessagePassingGraphEdgeScorer(node_dim, edge_dim, hidden=args.hidden,
                                                rounds=args.rounds, dropout=0.0)
        actor_tag = "cold-start random-init"
    actor.eval()

    eval_calls = {"n": 0}

    def reward_raw(sample, e_ref, active_edge_ids):
        eval_calls["n"] += 1
        # dense, lam_c=lam_b=beta=0 -> the raw consensus-margin objective (r = c - tau); the additive
        # constant tau cancels in every Delta_i, so this isolates per-agent topology quality.
        return T.reward_of(sample, active_edge_ids, e_ref, 0.0, 0.0, 0.0, "dense", False)[0]

    coma_vars, shared_vars = [], []
    learned_all, true_all = [], []
    n_used = 0
    gen = torch.Generator().manual_seed(args.seed + 1)
    for s in samples:
        if n_used >= args.n_scenes:
            break
        edge_ids = s["edge_ids"]
        edges = {e.edge_id: (e.node_u, e.node_v) for e in s["context"].graph.edges}
        budgets = dict(T.node_budgets_for_scene(s["context"].evaluator.scene))
        if not edges:
            continue
        e_ref = T._ref_energy(s)
        with torch.no_grad():
            logits = T.forward_logits(actor, s, mean, std)
            act = T.sample_decentralized_bcsp_action(logits, edge_ids, edges=edges, budgets=budgets,
                                                     temperature=args.temp, compute_entropy=False)
        if not any(pa.incident_edge_indices for pa in act.per_agent):
            continue

        def q_true(active_indices):
            return reward_raw(s, e_ref, [edge_ids[j] for j in active_indices])

        true_cf = per_agent_counterfactual_credit(
            q_true, per_agent_actions=act.per_agent, edge_ids=edge_ids, edges=edges,
            logits=logits, temperature=args.temp, k_cf=args.k_cf, generator=gen)
        # only agents with an action (>=1 incident edge) carry credit
        active_nodes = [pa.node_id for pa in act.per_agent if pa.incident_edge_indices]
        delta = {nid: true_cf.advantages[nid] for nid in active_nodes}
        coma_vars.append(within_scene_credit_variance(delta))
        shared_vars.append(0.0)   # A_s = r - V is one scalar for the whole scene -> within-scene var 0

        if critic is not None:
            learned_cf = counterfactual_advantages(
                critic, node_features=s["nf"], edge_features=s["ef"], edge_index=s["ei"],
                edge_ids=edge_ids, edges=edges, per_agent_actions=act.per_agent, logits=logits,
                temperature=args.temp, k_cf=args.k_cf, node_mean=mean[0], node_std=std[0],
                edge_mean=mean[1], edge_std=std[1], generator=gen)
            for nid in active_nodes:
                learned_all.append(learned_cf.advantages[nid])
                true_all.append(delta[nid])
        n_used += 1

    mean_coma = sum(coma_vars) / len(coma_vars) if coma_vars else 0.0
    frac_pos = sum(1 for v in coma_vars if v > 1e-9) / len(coma_vars) if coma_vars else 0.0
    report = {
        "scope": "MECHANISM diagnostic (NOT a corrected headline; corrected 8b-vs-R7 gated on rebuild)",
        "actor": actor_tag, "temp": args.temp,
        "shards": [Path(s).name for s in shards], "n_scenes": n_used, "k_cf": args.k_cf,
        "part_A_credit_resolution": {
            "mean_within_scene_variance_COMA_true_marginal": mean_coma,
            "mean_within_scene_variance_shared_advantage": 0.0,
            "frac_scenes_with_nonzero_COMA_resolution": frac_pos,
            "note": "shared A_s=r-V is one scalar/scene -> within-scene variance is 0 BY CONSTRUCTION; "
                    "any positive COMA variance is per-agent credit the shared advantage cannot represent",
        },
        "training_evaluator_calls_per_scene": 1,  # the TRAINING path (counterfactuals are critic forwards)
        "diagnostic_ground_truth_evaluator_calls": eval_calls["n"],  # DIAGNOSTIC-ONLY, not in training
    }
    if critic is not None and len(true_all) >= 2:
        report["part_B_learned_Q_rank_correlation"] = {
            "spearman_A_i_vs_true_Delta_i": _spearman(learned_all, true_all),
            "n_agent_points": len(true_all),
            "note": "pre-corrected data: a weak rho is Q-quality-limited (deferred to rebuild), not a headline",
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
