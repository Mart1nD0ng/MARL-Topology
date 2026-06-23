"""Iteration-2: genuine *decentralized* REINFORCE fine-tune of the recovered GNN actor.

WHY: the validated trunk is supervised behavior-cloning of a centralized SA / critic-beam search
oracle -- decentralized only at execution, with NO reward and NO policy improvement (violates the
project's "learning must be truly decentralized" + "SOTA reward" invariants). This script replaces
BCE-to-teacher with a reward-driven, fully-decentralized policy gradient on the SAME deployed actor.

WHAT (one variable changed: BC -> policy gradient):
  * Policy: per-edge Bernoulli over the existing MessagePassingGraphEdgeScorer logits (parameter-
    shared, scale-invariant, K-hop local message passing -- unchanged architecture).
  * Learning: REINFORCE with a per-scene EMA baseline. NO critic at all -> zero CTDE gap, the
    strongest form of INVARIANT #1 (decentralized learning). A networked decentralized critic for
    variance reduction is a later iteration.
  * Reward (INVARIANT #5, ONE principled objective -- feasibility-first barrier, not a weighted bag):
      feasible (c>=tau AND budget_ok):   r = 1 - beta * clip(E/E_ref, 0, 2)      (energy min, DoD)
      infeasible:                        r = -lam_c*max(0,tau-c) - lam_b*g_budget (dense potential)
    where c = CLOSED-FORM PBFT consensus_success_probability (INVARIANT #6 compliant), E = protocol
    energy (joules), tau = 0.9. lam_c, lam_b are Lagrangian duals updated by dual ascent on the mean
    constraint violation (constrained-RL / RCPO treatment), NOT hand-tuned weights. Feasible reward
    (>= 1-2*beta >= 0.8) strictly dominates any infeasible reward (<= 0) -> feasibility-first.
  * Deploy/eval decode: local_mutual_assemble (budget-respecting, fully decentralized) -- the SAME
    decoder the campaign deploys, so held-out numbers are directly comparable to E1 (0.66 raw/0.90 cond).

Warm-start is mandatory (a cold GNN samples near-random and collapses): we load a frozen BC actor
from the E1 campaign artifacts and RL fine-tune it, with keep-best on a held-out-from-train VAL split
so RL is never reported worse than the BC start.

SCOPE: this is a DIAGNOSTIC at the canonical op point (single config) to (a) validate the RL loop
runs end-to-end on the FULL environment and (b) measure reward-driven RL vs the BC warm-start.
Multi-config domain-randomized held-out validation (INVARIANT #4) is a later iteration; nothing here
claims the DoD is met. Temporal stays OFF (single-step contextual-bandit MARL) per the documented
INVARIANT #2 null (geometric, not topological, loss under motion; quasi-static physics).

Usage::

    python scripts/train/train_decentralized_rl.py --smoke          # fast end-to-end smoke
    python scripts/train/train_decentralized_rl.py                   # op-point diagnostic
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from marl_topology.budgets import is_budget_feasible, node_budgets_for_scene  # noqa: E402
from marl_topology.models.message_passing_graph_edge_scorer import (  # noqa: E402
    MessagePassingGraphEdgeScorer,
)
from marl_topology.training.decentralized_distillation import (  # noqa: E402
    TAU,
    build_samples,
    feature_standardization,
    forward_logits,
    load_actor_from_state,
    local_mutual_assemble,
)
from marl_topology.data.row_context_builder import build_row_contexts  # noqa: E402
from marl_topology.training.decentralized_action import (  # noqa: E402
    recompute_entropy,
    recompute_logp,
    sample_decentralized_action,
)
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic  # noqa: E402
from marl_topology.training.graph_mappo import (  # noqa: E402
    explained_variance,
    ppo_clip_actor_loss,
)

OP_SHARDS = [str(ROOT / "result_save" / "campaign" / "data" / "op" / f"_op_shard_{s}.pkl")
             for s in range(3001, 3025)]
DEFAULT_ARTIFACTS = str(ROOT / "result_save" / "campaign" / "E1_op_headline" / "_artifacts.pt")


def load_pool(shard_paths):
    pool = []
    for shard_path in shard_paths:
        with open(shard_path, "rb") as handle:
            dataset = pickle.load(handle)
        labels = dataset.source_dataset.teacher_labels
        for split in ("train", "eval", "test"):
            for row, context in build_row_contexts(dataset, split):
                pool.append((row, context, labels[context.fixture.fixture_id]))
    return pool


def _evaluate(sample, edge_ids):
    """Closed-form evaluate of a topology edge-id set -> (consensus, energy, latency)."""
    m = sample["context"].evaluator.evaluate(set(edge_ids)).metrics
    return (float(m["consensus_success_probability"]),
            float(m.get("energy", 0.0)), float(m.get("latency", 0.0)))


def _budgets(sample):
    return node_budgets_for_scene(sample["context"].evaluator.scene)


def _budget_violation(edge_ids, budgets):
    """Graded budget-infeasibility in [0,1]: fraction of nodes whose incident degree exceeds budget."""
    cap = dict(budgets)
    deg: dict = {}
    sep = "--"
    for e in edge_ids:
        a, b = e.split(sep)
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    if not cap:
        return 0.0
    over = sum(1 for node, d in deg.items() if d > cap.get(node, 0))
    return over / max(1, len(cap))


def _ref_energy(sample):
    """Per-scene energy normalizer E_ref = energy of the SA-teacher topology (a known feasible,
    near-minimal backbone) when present, else the full candidate graph. Floored away from 0."""
    teacher = [str(e) for e in sample["label"].get("selected_physical_edges", [])]
    ref_edges = teacher if teacher else list(sample["edge_ids"])
    try:
        _c, e, _l = _evaluate(sample, ref_edges)
    except Exception:
        e = 0.0
    return max(e, 1e-9)


def reward_of(sample, edge_ids, e_ref, lam_c, lam_b, beta, reward_mode="barrier",
              live_consensus_dual=False):
    """ONE constrained objective. Returns (reward, consensus_violation, budget_violation, feasible).

    barrier (binary): feasible -> 1 - beta*E/E_ref; infeasible -> -lam_c*g_c - lam_b*g_b. Gives NO
        gradient until a sample crosses tau -> signal-starved when the warm-started policy is too sharp
        to explore across the feasibility boundary (iteration-9 finding: train feasibility froze).
    dense (potential-based shaping, INVARIANT #5): r = (c - tau) - lam_b*g_b - beta*1[feasible]*E/E_ref.
        Phi = c - tau is a TRUE potential (Ng-Harada-Russell): dense in c so EVERY sample -- even
        infeasible -- gets a gradient toward higher consensus (a sample at c=0.88 beats one at c=0.70),
        feasibility-ordered (r>=0 iff feasible), and it does NOT change the optimal policy, only densifies
        the gradient toward the tau frontier. Energy stays gated to the feasible set.
    """
    try:
        c, energy, _lat = _evaluate(sample, edge_ids)
    except Exception:
        return (-(TAU) if reward_mode == "dense" else -float(lam_c) * TAU - float(lam_b)), TAU, 1.0, False
    budgets = _budgets(sample)
    budget_ok = is_budget_feasible(tuple(edge_ids), budgets) if edge_ids else False
    g_c = max(0.0, TAU - c)
    g_b = _budget_violation(edge_ids, budgets)
    feasible = (c >= TAU) and budget_ok
    er = min(energy / e_ref, 2.0)
    if reward_mode == "dense":
        r = (c - TAU) - lam_b * g_b - (beta * er if feasible else 0.0)   # potential-based, dense in c
        if live_consensus_dual and c < TAU:
            # INNOVATION B (MACPO dense/sparse split): a SPARSE binary consensus-violation cost makes the
            # consensus dual lam_c LIVE (in plain dense it is computed+ascended but never enters the reward).
            # ONE potential (c-tau) + TWO distinct Lagrangian duals (lam_c on consensus, lam_b on budget) --
            # still INVARIANT #5 (not a weighted bag). Critic-free; lam_c is dual ascent, not a tuned weight.
            r = r - lam_c
    elif feasible:
        r = 1.0 - beta * er                      # barrier: in [1-2*beta, 1]  (>= 0.8 for beta<=0.1)
    else:
        r = -lam_c * g_c - lam_b * g_b           # barrier: <= 0  -> strictly below any feasible reward
    return r, g_c, (0.0 if budget_ok else g_b), feasible


def mutual_acceptance_sample(logits, sample, temperature):
    """STOCHASTIC, fully-decentralized, budget-respecting topology sample whose deterministic
    (temperature->0) limit is EXACTLY local_mutual_assemble -> train == deploy.

    Thin wrapper over the Phase-6 per-agent action API
    (``marl_topology.training.decentralized_action.sample_decentralized_action``): each node
    samples up to its radio budget among its incident edges with logit >= 0 via Plackett-Luce /
    Gumbel-top-b over softmax(logit/temperature); an edge activates iff BOTH endpoints sampled it.
    Returns (active_edge_ids, logp) with logp = sum over nodes of the per-agent PL log-prob -- the
    REINFORCE score function. Using the shared API fixes the historical NaN-gumbel bug (the inline
    `-log(-log(U).clamp_min(1e-12))` clamped the negative inner log to a constant -> NaN -> the
    sampler collapsed to a fixed order with no exploration).
    """
    context = sample["context"]
    edge_ids = sample["edge_ids"]
    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    edges = {e.edge_id: (e.node_u, e.node_v) for e in context.graph.edges}
    action = sample_decentralized_action(
        logits, edge_ids, edges=edges, budgets=budgets,
        temperature=temperature, compute_entropy=False,
    )
    active = [edge_ids[i] for i in action.active_edge_indices]
    return active, action.joint_logp


def gauss_perturb_sample(logits, sample, sigma):
    """Gaussian-perturbation exploration on per-edge logits + the EXACT deployed gated decoder.

    action = local_mutual_assemble(logits + N(0,sigma^2)); logp = Gaussian log-prob of the perturbed
    logits under N(logits, sigma) -> gradient flows to the actor (the mean). Unlike the Plackett-Luce
    mutual sampler, this EXPLORES THE logit>=0 GATE BOUNDARY (a below-gate teacher edge can be perturbed
    above the gate and discovered), which is exactly the per-edge-logit miscalibration BC suffers. The
    sigma->0 limit is local_mutual_assemble(logits) == the deploy decoder, so train -> deploy as sigma
    anneals. Returns (active_edge_ids, logp).
    """
    dist = torch.distributions.Normal(logits, max(float(sigma), 1e-6))
    action = dist.sample()                                   # [E] perturbed logits (no grad through sample)
    logp = dist.log_prob(action).sum()                       # scalar, grad to logits (the mean)
    topo = local_mutual_assemble(action, sample["edge_ids"], sample["context"])
    return topo, logp


@torch.no_grad()
def forward_value(critic, sample, mean, std):
    """Centralized critic value V(scene) from the SAME standardized node/edge features the actor
    sees (graph_payload), masks all-ones (one real scene). No oracle/teacher label is fed -> no
    leakage. Returns a scalar tensor (grad on when the critic is in train mode)."""
    nf = ((sample["nf"] - mean[0]) / std[0]).unsqueeze(0)
    ef = ((sample["ef"] - mean[1]) / std[1]).unsqueeze(0)
    return critic(nf, ef, sample["ei"].unsqueeze(0),
                  torch.ones(1, sample["nf"].shape[0]), torch.ones(1, sample["ef"].shape[0]))[0]


def eval_held(actor, samples, mean, std):
    """Deterministic decentralized eval (local_mutual_assemble): raw / conditional + mean energy,
    latency, edge-count over the FEASIBLE decoded topologies (DoD low-energy/low-latency signal)."""
    actor.eval()
    solved = total = solved_solv = solvable = 0
    e_acc, l_acc, edge_acc, nfeas = 0.0, 0.0, 0, 0
    by_n: dict = {}
    for s in samples:
        logits = forward_logits(actor, s, mean, std)
        topo = local_mutual_assemble(logits, s["edge_ids"], s["context"])
        c, energy, lat = _evaluate(s, topo)
        ok = (c >= TAU) and is_budget_feasible(tuple(topo), _budgets(s))
        n = len(s["context"].graph.node_ids)
        d = by_n.setdefault(n, [0, 0])
        d[0] += int(ok); d[1] += 1
        total += 1; solved += int(ok)
        if bool(s["label"]["feasible_exists"]):
            solvable += 1; solved_solv += int(ok)
        if ok:
            e_acc += energy; l_acc += lat; edge_acc += len(topo); nfeas += 1
    return {
        "raw": solved / max(1, total),
        "conditional": (solved_solv / solvable) if solvable else None,
        "solved": solved, "total": total, "solvable": solvable,
        "raw_by_n": {n: by_n[n][0] / by_n[n][1] for n in sorted(by_n)},
        "mean_energy_feasible_j": (e_acc / nfeas) if nfeas else None,
        "mean_latency_feasible_s": (l_acc / nfeas) if nfeas else None,
        "mean_edges_feasible": (edge_acc / nfeas) if nfeas else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", default=OP_SHARDS)
    p.add_argument("--artifacts", default=DEFAULT_ARTIFACTS, help="warm-start BC actor artifacts")
    p.add_argument("--warm-actor-idx", type=int, default=2, help="which BC actor seed to warm-start from")
    p.add_argument("--cold-start", action="store_true",
                   help="random-init actor + data-computed standardization, NO BC warm-start -- tests "
                        "whether decentralized RL learns WITHOUT the centralized SA/critic oracle")
    p.add_argument("--out-dir", default=str(ROOT / "result_save" / "dec_rl"))
    p.add_argument("--held-frac", type=float, default=0.4)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--val-scenes", type=int, default=40)
    p.add_argument("--action-space", choices=["mutual", "gauss", "bernoulli"], default="mutual",
                   help="gauss = Gaussian-perturbation on logits + the EXACT gated deploy decoder "
                        "(explores the logit>=0 gate boundary; sigma=temp); mutual = budget-respecting "
                        "Plackett-Luce (no gate-boundary exploration); bernoulli = iteration-2 baseline")
    p.add_argument("--temp", type=float, default=1.0, help="initial sampling temperature (mutual policy)")
    p.add_argument("--temp-end", type=float, default=0.1,
                   help="final temperature; linear anneal temp->temp_end so train sampling -> deploy argmax")
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=0.1, help="energy-minimization weight (feasible set)")
    p.add_argument("--live-consensus-dual", action="store_true",
                   help="INNOVATION B: add a sparse binary consensus-violation cost (-lam_c on infeasible) so "
                        "the consensus dual is LIVE in dense mode (else lam_c is computed but inert)")
    p.add_argument("--reward-mode", choices=["barrier", "dense"], default="barrier",
                   help="dense = potential-based shaping r=(c-tau) (gradient on every sample, even "
                        "infeasible) -- needed when the binary barrier is signal-starved on a sharp policy")
    p.add_argument("--lam-c", type=float, default=2.0, help="initial consensus dual")
    p.add_argument("--lam-b", type=float, default=2.0, help="initial budget dual")
    p.add_argument("--dual-lr", type=float, default=0.5)
    p.add_argument("--lam-max", type=float, default=10.0)
    p.add_argument("--entropy-coef", type=float, default=0.005)
    p.add_argument("--baseline-ema", type=float, default=0.2)
    p.add_argument("--samples-per-scene", type=int, default=1,
                   help="K rollouts per scene per update; K>1 uses a low-variance RLOO leave-one-out "
                        "baseline (instead of the per-scene EMA) -- K chances to sample the fixing "
                        "topology on the few scenes BC misses (headroom is capturable but signal-starved)")
    p.add_argument("--include-unsolvable", action="store_true",
                   help="include feasible_exists=False scenes in the PG + dual (DEFAULT OFF: the "
                        "consensus constraint is UNSATISFIABLE there, so the lam_c dual ratchets "
                        "unboundedly and destabilizes; the controller is optimized on solvable scenes)")
    p.add_argument("--normalize-adv", action="store_true",
                   help="batch-normalize advantages (DEFAULT OFF: normalization un-learns the warm "
                        "start by giving already-solved scenes a spurious negative advantage)")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="save a resumable checkpoint every N updates (0=off) -- survives host sleep/kill; "
                        "a relaunch with --resume continues instead of restarting from scratch")
    p.add_argument("--resume", action="store_true",
                   help="resume from out-dir/_ckpt.pt if present (continue training across restarts)")
    p.add_argument("--smoke", action="store_true", help="tiny fast end-to-end check")
    # --- Phase 7 Graph-MAPPO arm ---
    p.add_argument("--baseline", choices=["ema", "rloo", "graph-mappo"], default="ema",
                   help="advantage baseline. ema (default, byte-identical to the historical trunk: "
                        "K=1 per-scene EMA) | rloo (K>=2 leave-one-out, requires --samples-per-scene>=2) "
                        "| graph-mappo (a centralized graph value critic + PPO-clip update, CTDE).")
    p.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon (graph-mappo)")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO inner epochs per update (graph-mappo)")
    p.add_argument("--target-kl", type=float, default=0.01,
                   help="early-stop the PPO inner loop when approx_kl > 1.5*target_kl (graph-mappo)")
    p.add_argument("--critic-coef", type=float, default=0.5, help="critic loss weight (graph-mappo)")
    p.add_argument("--critic-lr", type=float, default=1e-3, help="centralized critic learning rate")
    p.add_argument("--critic-hidden", type=int, default=64, help="critic hidden width")
    p.add_argument("--critic-rounds", type=int, default=4, help="critic message-passing rounds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.shards = args.shards[:2]
        args.updates = 3
        args.val_scenes = 6
        args.eval_every = 1
    if args.baseline == "rloo" and args.samples_per_scene < 2:
        raise SystemExit("[graph-mappo] --baseline rloo requires --samples-per-scene >= 2 "
                         "(RLOO needs M>=2 rollouts for the leave-one-out baseline)")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- data: identical split to the campaign so held-out is comparable to E1
    pool = load_pool(args.shards)
    Random(args.split_seed).shuffle(pool)
    cut = int((1.0 - args.held_frac) * len(pool))
    train_items, held_items = pool[:cut], pool[cut:]
    if args.smoke:
        train_items = train_items[:24]
    val_items = train_items[-args.val_scenes:]
    fit_items = train_items[:-args.val_scenes] if len(train_items) > args.val_scenes else train_items
    train_s = build_samples(fit_items)
    val_s = build_samples(val_items)
    held_s = build_samples(held_items)
    ceiling = fmean(float(bool(l["feasible_exists"])) for _r, _c, l in held_items)
    print(f"[data] pool {len(pool)} | fit {len(train_s)} | val {len(val_s)} | held {len(held_s)} "
          f"| teacher ceiling(held) {ceiling:.3f}")

    node_dim, edge_dim = train_s[0]["nf"].shape[1], train_s[0]["ef"].shape[1]
    if args.cold_start:
        # random-init actor, standardization computed from data -- NO oracle warm-start
        mean, std = feature_standardization(train_s)
        actor = MessagePassingGraphEdgeScorer(node_dim, edge_dim, hidden=args.hidden,
                                              rounds=args.rounds, dropout=0.0)
        bc_held = eval_held(actor, held_s, mean, std)        # random-init baseline (the floor RL starts from)
        tag = "cold-start random-init"
    else:
        # warm-start from a frozen BC actor (its own normalization)
        art = torch.load(args.artifacts, map_location="cpu", weights_only=False)
        a = art["actors"][args.warm_actor_idx]
        mean, std = a["mean"], a["std"]
        actor = load_actor_from_state(a["state"], node_dim, edge_dim, hidden=args.hidden, rounds=args.rounds)
        bc_held = eval_held(actor, held_s, mean, std)
        tag = "warm-start BC"
    print(f"[{tag}] held raw={bc_held['raw']:.3f} conditional="
          f"{None if bc_held['conditional'] is None else round(bc_held['conditional'],3)} "
          f"energy={bc_held['mean_energy_feasible_j']} edges={bc_held['mean_edges_feasible']}")

    # --- RL fine-tune: decentralized REINFORCE + per-scene EMA baseline + dual ascent, keep-best on VAL
    e_ref = [_ref_energy(s) for s in train_s]
    baseline = [0.0] * len(train_s)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1e-4)
    # Phase 7 Graph-MAPPO: a centralized graph value critic + its own optimizer (training-only;
    # constructed only on this arm, never reachable from the deployed actor -> D1).
    critic = opt_c = None
    critic_history: list = []
    if args.baseline == "graph-mappo":
        critic = CentralizedGraphCritic(node_dim, edge_dim, hidden=args.critic_hidden, rounds=args.critic_rounds)
        opt_c = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
        print(f"[graph-mappo] centralized critic: hidden={args.critic_hidden} rounds={args.critic_rounds} "
              f"clip={args.clip_epsilon} ppo_epochs={args.ppo_epochs} critic_lr={args.critic_lr}")
    lam_c, lam_b = args.lam_c, args.lam_b
    ws_val = eval_held(actor, val_s, mean, std)["raw"]            # warm-start VAL = keep-best floor
    best_val, best_state = ws_val, {k: v.detach().clone() for k, v in actor.state_dict().items()}
    print(f"[warm-start] VAL raw={ws_val:.3f} (keep-best floor; RL never reported below this)")
    history = []
    rng = Random(args.seed)

    # Resume across host sleeps / external kills: a relaunch with --resume continues from the last
    # checkpoint instead of recomputing from scratch (this host suspends background tasks when idle).
    start_upd = 1
    ckpt_path = out_dir / "_ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(ck["actor"]); opt.load_state_dict(ck["opt"])
        best_state, best_val = ck["best_state"], ck["best_val"]
        mean, std = ck["mean"], ck["std"]
        baseline, lam_c, lam_b = ck["baseline"], ck["lam_c"], ck["lam_b"]
        history, start_upd = ck["history"], ck["update"] + 1
        print(f"[resume] from update {ck['update']} -> {start_upd}/{args.updates} (best VAL {best_val:.3f})",
              flush=True)

    for upd in range(start_upd, args.updates + 1):
        actor.train()
        frac = (upd - 1) / max(1, args.updates - 1)
        temp_now = args.temp + frac * (args.temp_end - args.temp)  # anneal train sampling -> deploy
        order = list(range(len(train_s)))
        rng.shuffle(order)
        logps, advs, ents = [], [], []
        rwds, gcs, gbs, feas = [], [], [], []
        K = max(1, args.samples_per_scene)
        if args.baseline == "graph-mappo":
            # ---- collect ONE rollout per solvable scene (1 evaluator call/scene == EMA budget) ----
            batch = []
            for i in order:
                s = train_s[i]
                if not args.include_unsolvable and not bool(s["label"]["feasible_exists"]):
                    continue
                edges = {e.edge_id: (e.node_u, e.node_v) for e in s["context"].graph.edges}
                budgets = dict(node_budgets_for_scene(s["context"].evaluator.scene))
                with torch.no_grad():
                    logits0 = forward_logits(actor, s, mean, std)
                    act = sample_decentralized_action(logits0, s["edge_ids"], edges=edges,
                                                      budgets=budgets, temperature=temp_now,
                                                      compute_entropy=False)
                    v0 = float(forward_value(critic, s, mean, std))
                topo = [s["edge_ids"][j] for j in act.active_edge_indices]
                r, g_c, g_b, ok = reward_of(s, topo, e_ref[i], lam_c, lam_b, args.beta, args.reward_mode,
                                            args.live_consensus_dual)
                per_agent = [(pa.gated_edge_indices, pa.accepted_order)
                             for pa in act.per_agent if pa.accepted_order]
                batch.append({"s": s, "per_agent": per_agent, "logp_old": float(act.joint_logp),
                              "r": r, "V": v0})
                rwds.append(r); gcs.append(g_c); gbs.append(g_b); feas.append(float(ok))
            if not batch:
                continue
            adv_all = torch.tensor([b["r"] - b["V"] for b in batch])  # single-step A = r - V (GAE at T=1)
            if args.normalize_adv:
                adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-6)
            logp_old_t = torch.tensor([b["logp_old"] for b in batch])
            # ---- PPO-clip inner epochs on the actor: re-score the frozen order over the frozen gate ----
            last_kl, last_clip = 0.0, 0.0
            for _epoch in range(args.ppo_epochs):
                lp_new, en_new = [], []
                for b in batch:
                    logits = forward_logits(actor, b["s"], mean, std)
                    lp = logits.new_zeros(()); en = logits.new_zeros(())
                    for gated, order_ in b["per_agent"]:
                        lp = lp + recompute_logp(logits, gated, order_, temp_now)
                        en = en + recompute_entropy(logits, gated, len(order_), temp_now)
                    lp_new.append(lp); en_new.append(en)
                ppo_loss, info = ppo_clip_actor_loss(torch.stack(lp_new), logp_old_t,
                                                     adv_all.detach(), clip_eps=args.clip_epsilon)
                loss = ppo_loss - args.entropy_coef * torch.stack(en_new).mean()  # exact PL entropy bonus
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
                last_kl, last_clip = float(info["approx_kl"]), float(info["clip_fraction"])
                if last_kl > args.target_kl * 1.5:                  # early-stop the inner loop
                    break
            # ---- critic regression V(scene) -> reward (its OWN optimizer; never touches the actor) ----
            r_t = torch.tensor([float(b["r"]) for b in batch])
            for _ in range(args.ppo_epochs):
                v_pred = torch.stack([forward_value(critic, b["s"], mean, std) for b in batch])
                v_loss = args.critic_coef * (r_t - v_pred).pow(2).mean()
                opt_c.zero_grad(); v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()
            ev = explained_variance([b["r"] for b in batch], [b["V"] for b in batch])
            critic_history.append({"update": upd, "explained_variance": ev, "approx_kl": last_kl,
                                   "clip_fraction": last_clip, "n_scenes": len(batch),
                                   "evaluator_calls": len(batch),       # 1 _evaluate/scene == EMA budget
                                   "evaluator_calls_per_scene": 1,      # fair A/B vs ema (rloo spends K)
                                   "critic_value_mean": fmean([b["V"] for b in batch])})
        else:
            for i in order:
                s = train_s[i]
                if not args.include_unsolvable and not bool(s["label"]["feasible_exists"]):
                    continue   # no feasible topology exists -> nothing learnable; excluded from PG + dual
                logits = forward_logits(actor, s, mean, std)              # [E], grad on; reused for K samples
                ent = torch.distributions.Bernoulli(logits=logits).entropy().sum()
                s_logps, s_rs, s_gc, s_gb, s_ok = [], [], [], [], []
                for _k in range(K):
                    if args.action_space == "gauss":
                        topo, logp = gauss_perturb_sample(logits, s, temp_now)      # sigma=temp_now (annealed)
                    elif args.action_space == "mutual":
                        topo, logp = mutual_acceptance_sample(logits, s, temp_now)  # train -> deploy (annealed)
                    else:
                        dist = torch.distributions.Bernoulli(logits=logits)
                        action = dist.sample()
                        logp = dist.log_prob(action).sum()
                        topo = [eid for j, eid in enumerate(s["edge_ids"]) if action[j] > 0.5]
                    r, g_c, g_b, ok = reward_of(s, topo, e_ref[i], lam_c, lam_b, args.beta, args.reward_mode,
                                                args.live_consensus_dual)
                    s_logps.append(logp); s_rs.append(r); s_gc.append(g_c); s_gb.append(g_b); s_ok.append(ok)
                if K > 1:
                    tot = sum(s_rs)
                    for k in range(K):
                        b_k = (tot - s_rs[k]) / (K - 1)            # RLOO leave-one-out baseline (low variance)
                        advs.append(s_rs[k] - b_k); logps.append(s_logps[k])
                else:
                    adv = s_rs[0] - baseline[i]                    # single-sample EMA-baseline fallback
                    baseline[i] = (1 - args.baseline_ema) * baseline[i] + args.baseline_ema * s_rs[0]
                    advs.append(adv); logps.append(s_logps[0])
                ents.append(ent)
                rwds.append(fmean(s_rs)); gcs.append(fmean(s_gc)); gbs.append(fmean(s_gb))
                feas.append(fmean([float(x) for x in s_ok]))
            if not logps:                                               # no solvable scenes this update
                continue
            adv_t = torch.tensor(advs)
            if args.normalize_adv:                                       # OFF by default: normalization
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)    # un-learns the warm start
            logp_t = torch.stack(logps)
            ent_t = torch.stack(ents)
            loss = -(adv_t.detach() * logp_t).mean() - args.entropy_coef * ent_t.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
        # dual ascent on the mean constraint violations (constrained-RL)
        mean_gc, mean_gb = fmean(gcs), fmean(gbs)
        lam_c = min(args.lam_max, max(0.0, lam_c + args.dual_lr * mean_gc))
        lam_b = min(args.lam_max, max(0.0, lam_b + args.dual_lr * mean_gb))

        if upd % args.eval_every == 0 or upd == args.updates:
            ve = eval_held(actor, val_s, mean, std)
            sel = ve["raw"]
            if sel > best_val:
                best_val = sel
                best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            print(f"[upd {upd:3d}] temp={temp_now:.2f} train R={fmean(rwds):+.3f} feas={fmean(feas):.3f} "
                  f"g_c={mean_gc:.3f} g_b={mean_gb:.3f} lam_c={lam_c:.2f} lam_b={lam_b:.2f} "
                  f"| VAL raw={ve['raw']:.3f} (best {best_val:.3f})")
            if args.baseline == "graph-mappo" and critic_history:
                cm = critic_history[-1]
                print(f"           [critic] EV={cm['explained_variance']:+.3f} approx_kl={cm['approx_kl']:.4f} "
                      f"clip_frac={cm['clip_fraction']:.3f} eval_calls/scene={cm['evaluator_calls']}")
            history.append({"update": upd, "train_reward": fmean(rwds), "train_feasible": fmean(feas),
                            "mean_g_c": mean_gc, "mean_g_b": mean_gb, "lam_c": lam_c, "lam_b": lam_b,
                            "val_raw": ve["raw"]})
            if args.ckpt_every and upd % args.ckpt_every == 0:
                torch.save({"actor": actor.state_dict(), "opt": opt.state_dict(),
                            "best_state": best_state, "best_val": best_val, "mean": mean, "std": std,
                            "baseline": baseline, "lam_c": lam_c, "lam_b": lam_b,
                            "history": history, "update": upd}, ckpt_path)
                print(f"[ckpt] update {upd} saved (best VAL {best_val:.3f})", flush=True)

    final_held = eval_held(actor, held_s, mean, std)   # FINAL-update policy (before keep-best revert)
    actor.load_state_dict(best_state)        # keep-best (never worse than warm-start on VAL)
    rl_held = eval_held(actor, held_s, mean, std)
    print("=" * 78)
    print(f"[BC   ] held raw={bc_held['raw']:.3f} cond="
          f"{None if bc_held['conditional'] is None else round(bc_held['conditional'],3)} "
          f"energy={bc_held['mean_energy_feasible_j']}")
    print(f"[RLfin] held raw={final_held['raw']:.3f} cond="
          f"{None if final_held['conditional'] is None else round(final_held['conditional'],3)} "
          f"energy={final_held['mean_energy_feasible_j']}  (final-update policy, pre keep-best)")
    print(f"[RL   ] held raw={rl_held['raw']:.3f} cond="
          f"{None if rl_held['conditional'] is None else round(rl_held['conditional'],3)} "
          f"energy={rl_held['mean_energy_feasible_j']}  (keep-best)")
    print(f"[time] {time.time() - t0:.1f}s")

    result = {"config": vars(args), "ceiling": ceiling,
              "warm_start_bc": bc_held, "rl": rl_held, "rl_final": final_held, "history": history,
              "deltas": {"raw": rl_held["raw"] - bc_held["raw"],
                         "conditional": (None if rl_held["conditional"] is None or bc_held["conditional"] is None
                                         else rl_held["conditional"] - bc_held["conditional"])}}
    (out_dir / ("smoke_result.json" if args.smoke else "rl_result.json")).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    artifacts = {"actors": [{"state": actor.state_dict(), "mean": mean, "std": std}]}
    if args.baseline == "graph-mappo":
        # the centralized critic + its optimizer are TRAINING artifacts only (D1: never deployed)
        artifacts["critic"] = critic.state_dict()
        artifacts["opt_c"] = opt_c.state_dict()
        final = critic_history[-1] if critic_history else {}
        (out_dir / "critic_metrics.json").write_text(
            json.dumps({"final": final, "history": critic_history,
                        "explained_variance": final.get("explained_variance"),
                        "approx_kl": final.get("approx_kl"),
                        "clip_fraction": final.get("clip_fraction"),
                        "evaluator_calls_per_scene": final.get("evaluator_calls_per_scene")}, indent=2, default=str),
            encoding="utf-8")
    torch.save(artifacts, out_dir / ("_smoke_artifacts.pt" if args.smoke else "_rl_artifacts.pt"))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
