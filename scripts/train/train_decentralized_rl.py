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
    recompute_bcsp_entropy,
    recompute_bcsp_logp,
    sample_decentralized_action,
    sample_decentralized_bcsp_action,
)
from marl_topology.models.centralized_graph_critic import CentralizedGraphCritic  # noqa: E402
from marl_topology.models.pna_directional_actor import PNADirectionalActor  # noqa: E402
from marl_topology.models.pna_aggregation import training_degree_delta  # noqa: E402
from marl_topology.training.graph_mappo import (  # noqa: E402
    critic_q_value,
    critic_scene_value,
    explained_variance,
    ppo_clip_actor_loss,
)
from marl_topology.training.counterfactual_credit import counterfactual_advantages  # noqa: E402
from marl_topology.training.scq_supervision import (  # noqa: E402
    scq_counterfactual_targets,
    scq_loss_from_targets,
)
from marl_topology.training.reliability_constraints import (  # noqa: E402
    chance_dual_update,
    pareto_archive_select,
)
from marl_topology.training.run_instrumentation import (  # noqa: E402
    dataset_manifest,
    mechanism_activation,
    shard_generator_config,
    split_manifest,
)
from marl_topology.solvability.status import WITNESS_FEASIBLE  # noqa: E402
from marl_topology.training.tristate_training import (  # noqa: E402
    assert_split_isolation,
    solvability_status_for_label,
    trainable_under_tristate,
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
                label = dict(labels[context.fixture.fixture_id])  # copy; never mutate the shared label
                # R3: stamp tri-state solvability (back-compat for shards built before the field).
                # A finite-search miss is `unknown`, never certified_infeasible (#11).
                label["solvability_status"] = solvability_status_for_label(label)
                pool.append((row, context, label))
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
              live_consensus_dual=False, lam_chance=0.0):
    """ONE constrained objective. Returns (reward, consensus_violation, budget_violation, feasible).

    lam_chance (Phase 10a, Spec S6.2): the CHANCE-constraint Lagrangian dual. When >0 it adds the
    per-scene chance penalty -lam_chance*1[c<tau] -- the scene's contribution to the constraint
    Pr(C<tau)<=delta (the -delta term is a constant, omitted from r). Default 0.0 -> byte-identical.

    barrier (binary): feasible -> 1 - beta*E/E_ref; infeasible -> -lam_c*g_c - lam_b*g_b. Gives NO
        gradient until a sample crosses tau -> signal-starved when the warm-started policy is too sharp
        to explore across the feasibility boundary (iteration-9 finding: train feasibility froze).
    dense (feasibility-margin reward): r = (c - tau) - lam_b*g_b - beta*1[feasible]*E/E_ref.
        r is the consensus success probability c minus the fixed threshold tau, dense in c so EVERY
        sample -- even infeasible -- gets a gradient toward higher consensus (a sample at c=0.88 beats
        one at c=0.70) and is feasibility-ordered (r>=0 iff feasible). NOTE (honest framing, v2 R0): the
        -tau term is a CONSTANT offset, not Ng-Harada-Russell potential-based shaping. This is a single-
        step (T=1) bandit with no states/transitions, so there is no F=gamma*Phi(s')-Phi(s) shaping; the
        constant only shifts the reward (a fixed baseline; E[grad log pi * const]=0), it does not change
        the argmax. Energy stays gated to the feasible set.
    """
    try:
        c, energy, _lat = _evaluate(sample, edge_ids)
    except Exception:    # evaluator failure == infeasible (c<tau) -> chance penalty applies too
        base = -(TAU) if reward_mode == "dense" else -float(lam_c) * TAU - float(lam_b)
        return base - lam_chance, TAU, 1.0, False
    budgets = _budgets(sample)
    budget_ok = is_budget_feasible(tuple(edge_ids), budgets) if edge_ids else False
    g_c = max(0.0, TAU - c)
    g_b = _budget_violation(edge_ids, budgets)
    feasible = (c >= TAU) and budget_ok
    er = min(energy / e_ref, 2.0)
    if reward_mode == "dense":
        r = (c - TAU) - lam_b * g_b - (beta * er if feasible else 0.0)   # feasibility-margin, dense in c
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
    if lam_chance and c < TAU:                   # Phase 10a chance-constraint penalty (-delta is const)
        r = r - lam_chance
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


def forward_value(critic, sample, mean, std):
    """Centralized critic value V(scene) from the SAME standardized node/edge features the actor sees
    (graph_payload), all-ones masks (one real scene). No oracle/teacher label is fed -> no leakage (D1).

    R7 / Spec S8.4: this is NOT wrapped in no_grad -- the critic TRAIN forward must keep its gradient
    so opt_c.step() actually moves the critic. The ROLLOUT caller wraps it in `with torch.no_grad()`
    for the detached baseline value; the UPDATE caller calls it directly (grad on). (The prior
    @torch.no_grad() decorator silently froze the critic -- its loss had no grad_fn.)"""
    return critic_scene_value(critic, sample["nf"], sample["ef"], sample["ei"],
                              node_mean=mean[0], node_std=std[0], edge_mean=mean[1], edge_std=std[1])


def forward_q(critic, sample, active_indices, mean, std):
    """Action-conditioned Q(s, S) (Phase 8b counterfactual arm): forward_value conditioned on the
    realized active-edge one-hot. Requires a critic built with critic_sees_action=True. Grad-on (Spec
    S8.4) -- the UPDATE caller trains the Q critic by regressing Q(s, S_actual) -> reward; the ROLLOUT
    caller / counterfactual baseline wrap their own no_grad."""
    oh = sample["ef"].new_zeros(sample["ef"].shape[0])
    if active_indices:
        oh[list(active_indices)] = 1.0
    return critic_q_value(critic, sample["nf"], sample["ef"], sample["ei"], oh,
                          node_mean=mean[0], node_std=std[0], edge_mean=mean[1], edge_std=std[1])


def eval_held(actor, samples, mean, std):
    """Deterministic decentralized eval (local_mutual_assemble): raw / conditional + mean energy,
    latency, edge-count over the FEASIBLE decoded topologies (DoD low-energy/low-latency signal)."""
    actor.eval()
    solved = total = solved_solv = solvable = 0
    unknown_total = unknown_discovered = 0
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
        # R3 tri-state: witness recall is over the KNOWN-solvable (witness_feasible) held scenes;
        # solving an `unknown` scene is a WITNESS DISCOVERY (U->W), reported separately -- it has
        # no recall denominator (we never certified it solvable/infeasible).
        status = solvability_status_for_label(s["label"])
        if status == WITNESS_FEASIBLE:
            solvable += 1; solved_solv += int(ok)
        else:  # unknown (no certified_infeasible exists yet)
            unknown_total += 1; unknown_discovered += int(ok)
        if ok:
            e_acc += energy; l_acc += lat; edge_acc += len(topo); nfeas += 1
    witness_recall = (solved_solv / solvable) if solvable else None
    return {
        "raw": solved / max(1, total),
        "witness_recall": witness_recall,
        "conditional": witness_recall,  # back-compat alias (== witness_recall)
        "witness_discovered": unknown_discovered, "unknown_total": unknown_total,
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
                   help="dense = feasibility-margin reward r=(c-tau) (gradient on every sample, even "
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
    p.add_argument("--exclude-unknown", action="store_true",
                   help="R3 ABLATION: exclude `unknown` scenes from training (reproduce the old "
                        "feasible-only training). DEFAULT OFF -- per D8/#8 an unknown scene is NOT "
                        "deleted; it enters exploration (the dense reward gives a gradient toward "
                        "higher consensus and the policy may DISCOVER a witness, U->W). Only a PROVEN "
                        "certified_infeasible scene is always excluded (none exist yet -- no UB proof).")
    p.add_argument("--include-unsolvable", action="store_true",
                   help="DEPRECATED (R3): superseded by tri-state -- `unknown` scenes now train by "
                        "default. Accepted as a no-op for back-compat; use --exclude-unknown to opt out.")
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
    # --- Phase 8b Graph-Counterfactual PPO arm ---
    p.add_argument("--counterfactual", action="store_true",
                   help="Phase 8b: per-agent COMA counterfactual credit A_i=Q(s,S)-E[Q(s,S~_i,S_-i)] "
                        "(action-conditioned Q critic) replacing the shared scene advantage. The "
                        "counterfactual Q evals are critic forwards -> the evaluator budget is unchanged "
                        "(1/scene). Requires --baseline graph-mappo.")
    p.add_argument("--k-cf", type=int, default=4,
                   help="counterfactual subset samples per agent for the COMA baseline (Phase 8b)")
    # --- Phase 9 SCQ critic supervision ---
    p.add_argument("--scq", action="store_true",
                   help="Phase 9: SCQ closed-form counterfactual supervision -- calibrate the Q critic "
                        "with EXACT evaluator differences DeltaR_i=R(S)-R(S~_i,S_-i) (Spec S10). "
                        "Requires --counterfactual. NOT budget-neutral: spends up to --scq-m extra "
                        "evaluator calls/scene (logged as scq_evaluator_calls_per_scene).")
    p.add_argument("--scq-m", type=int, default=2,
                   help="SCQ counterfactuals per scene (extra evaluator calls/scene); Spec S10.4 top-M")
    p.add_argument("--scq-coef", type=float, default=0.5,
                   help="weight of the SCQ consistency loss in the critic objective (Phase 9)")
    p.add_argument("--scq-select", choices=["simple", "topM"], default="simple",
                   help="SCQ counterfactual selection: simple (9a, BCSP samples) or topM (9b, Spec "
                        "S10.4 sensitivity-guided add/remove/swap -- the most informative counterfactuals)")
    # --- Phase 10a chance constraint ---
    p.add_argument("--chance", action="store_true",
                   help="Phase 10a: distribution-level CHANCE constraint Pr(C<tau)<=delta (Spec S6.2) -- "
                        "a sign-flexible dual lam_chance ascends on (frac scenes below tau - delta) and "
                        "penalizes the reward -lam_chance*1[c<tau]. Opt-in (default off -> byte-identical).")
    p.add_argument("--chance-delta", type=float, default=0.1,
                   help="allowed reliability failure rate delta for the chance constraint (Phase 10a)")
    p.add_argument("--chance-lr", type=float, default=0.0,
                   help="chance dual learning rate (Phase 10a); 0 -> use --dual-lr")
    # --- Phase 10c Pareto checkpoint archive ---
    p.add_argument("--pareto-archive", action="store_true",
                   help="Phase 10c: select the FINAL checkpoint from a validation Pareto archive (Spec "
                        "S6.4: reliability-risk -> min violation -> energy-latency non-dominated -> "
                        "hypervolume -> stability) instead of raw VAL feasibility alone. Opt-in (default "
                        "off -> the keep-best-on-VAL-raw selection is byte-identical).")
    p.add_argument("--pareto-risk-budget", type=float, default=0.0,
                   help="max reliability_violation (1 - VAL raw) for an archive entry to count as "
                        "reliability-risk satisfied (Phase 10c)")
    # --- Phase 11 actor architecture ---
    p.add_argument("--actor", choices=["mlp", "pna"], default="mlp",
                   help="actor architecture: mlp (default = MessagePassingGraphEdgeScorer, byte-identical) "
                        "or pna (Phase 11 preference-conditioned directional PNA actor). PNA is "
                        "cold-start only (warm-start loads an MLP checkpoint).")
    # --- Dynamic (two-timescale, T>1) episode RL (owner-authorized; reverses the R5 deferral) ---
    p.add_argument("--dynamic", action="store_true",
                   help="route to the DYNAMIC T>1 episode arm (two_timescale_env): moving-vehicle "
                        "trajectories, per-frame channel, reconfiguration cost, gamma-return, recurrent "
                        "PPO. Leaves the default T=1 path byte-identical (this branch returns early).")
    p.add_argument("--dynamic-actor", choices=["recurrent", "memoryless"], default="recurrent",
                   help="dynamic arm: recurrent (per-node hidden carries across frames) vs memoryless "
                        "(same architecture, hidden reset each frame) -- a controlled cross-frame-memory ablation")
    p.add_argument("--frames", type=int, default=8, help="episode length T (macro frames)")
    p.add_argument("--dt", type=float, default=2.0, help="seconds per macro frame (mobility step)")
    p.add_argument("--speed-min", type=float, default=15.0, help="min vehicle speed m/s (mobility)")
    p.add_argument("--speed-max", type=float, default=30.0, help="max vehicle speed m/s (mobility)")
    p.add_argument("--hold-interval", type=int, default=4, help="PBFT micro-rounds a macro topology is held (H_PBFT)")
    p.add_argument("--gamma", type=float, default=0.95, help="episode discount (dynamic arm)")
    p.add_argument("--reconfig-e", type=float, default=0.1, help="reconfiguration energy per toggled edge")
    p.add_argument("--reconfig-l", type=float, default=0.0, help="reconfiguration latency per toggled edge")
    p.add_argument("--dyn-train", type=int, default=24, help="dynamic train trajectories")
    p.add_argument("--dyn-held", type=int, default=24, help="dynamic held trajectories (disjoint seed)")
    p.add_argument("--dyn-nodes", type=int, nargs="+", default=[8, 12, 16], help="node-count choices (dynamic)")
    p.add_argument("--dyn-eval-every", type=int, default=5,
                   help="dynamic arm: run the (heavy) decoded val eval every N updates + on the last "
                        "update (keep-best selection); per-update eval is pathological under the N<=16 "
                        "operating-point evaluator")
    p.add_argument("--dyn-warmstart", type=int, default=0,
                   help="dynamic arm: supervised warm-start epochs toward the per-frame myopic-greedy "
                        "teacher BEFORE RL (0=off). Cold-start RL alone cannot find the feasible backbone "
                        "at N<=16; both arms get the SAME warm-start so the cross-frame-memory ablation "
                        "stays the only difference.")
    p.add_argument("--dyn-warmstart-lr", type=float, default=0.0, help="warm-start lr (0 -> use --lr)")
    p.add_argument("--tx-power", type=float, default=20.0, help="tx power dBm (operating-point regime)")
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
    if args.counterfactual and args.baseline != "graph-mappo":
        raise SystemExit("[counterfactual] --counterfactual (Phase 8b) requires --baseline graph-mappo "
                         "(the action-conditioned Q critic lives in the graph-mappo arm)")
    if args.scq and not args.counterfactual:
        raise SystemExit("[scq] --scq (Phase 9) requires --counterfactual (SCQ supervises the action-"
                         "conditioned Q critic, which exists only in the counterfactual arm)")
    if args.actor == "pna" and not args.cold_start:
        raise SystemExit("[actor] --actor pna (Phase 11) is cold-start only (warm-start loads an MLP "
                         "BC checkpoint); pass --cold-start")
    if args.dynamic:
        # DYNAMIC T>1 episode arm (two_timescale_env). Reached only here; the T=1 path below is
        # untouched (byte-identical). The trunk's ONE constrained objective (reward_of + helpers) is
        # passed in so the dynamic arm shares the exact reward definition (no fork).
        if args.smoke:
            args.updates = 3
            args.dyn_train = 4
            args.dyn_held = 4
            args.frames = 4
        from marl_topology.training.dynamic_rl import run_dynamic_training
        args._root = str(ROOT)
        run_dynamic_training(args, reward_of=reward_of, _evaluate=_evaluate, _budgets=_budgets,
                             _ref_energy=_ref_energy, TAU=TAU)
        return
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

    # R3 tri-state: (a) witness-memory isolation -- fit/val/held are DISJOINT pool-item slices, so a
    # held/val witness discovery can never feed training (Spec 5.3; the loop reads only train_s). The
    # guard keys on item identity, NOT scenario_id (which is shard-local: the same proc name in two
    # shards is a different geometry, not a leak); (b) mechanism-activation log -- `unknown` scenes
    # now ENTER training (D8).
    assert_split_isolation({id(it) for it in fit_items}, {id(it) for it in val_items},
                           {id(it) for it in held_items})
    n_wf = sum(1 for s in train_s if s["label"]["solvability_status"] == WITNESS_FEASIBLE)
    n_unk = len(train_s) - n_wf
    n_train = sum(1 for s in train_s
                  if trainable_under_tristate(s["label"]["solvability_status"],
                                              exclude_unknown=args.exclude_unknown))
    print(f"[tri-state] train pool {len(train_s)}: {n_wf} witness_feasible + {n_unk} unknown -> "
          f"{n_train} enter training (exclude_unknown={args.exclude_unknown})")

    # Full-Integration-Audit instrumentation (checklist §11): emit reproducible provenance now that
    # the split is fixed -- shard SHA256 + env-math generator config, per-split scenario IDs +
    # node-count / solvability / trajectory-length(=1, T=1 bandit) distributions, seed/split. The
    # mechanism_activation.json (with the MEASURED critic-parameter delta) is written after training.
    data_man = dataset_manifest(args.shards)
    dataset_env_math = data_man.get("generator_config", {}).get("env_math_regime", {})
    (out_dir / "data_manifest.json").write_text(json.dumps(data_man, indent=2, default=str),
                                                encoding="utf-8")
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest(fit_items, val_items, held_items), indent=2, default=str),
        encoding="utf-8")
    (out_dir / "seed_manifest.json").write_text(json.dumps(
        {"seed": args.seed, "split_seed": args.split_seed, "held_frac": args.held_frac,
         "val_scenes": args.val_scenes, "cold_start": bool(args.cold_start),
         "warm_actor_idx": None if args.cold_start else args.warm_actor_idx}, indent=2), encoding="utf-8")

    node_dim, edge_dim = train_s[0]["nf"].shape[1], train_s[0]["ef"].shape[1]
    if args.cold_start:
        # random-init actor, standardization computed from data -- NO oracle warm-start
        mean, std = feature_standardization(train_s)
        if args.actor == "pna":   # Phase 11: preference-conditioned directional PNA actor (opt-in)
            degs = []
            for s in train_s:
                ei = s["ei"]; deg = [0] * s["nf"].shape[0]
                for e in range(ei.shape[0]):
                    deg[int(ei[e, 0])] += 1; deg[int(ei[e, 1])] += 1
                degs.extend(deg)
            pna_delta = training_degree_delta(degs)
            actor = PNADirectionalActor(node_dim, edge_dim, hidden=args.hidden, rounds=args.rounds,
                                        delta=pna_delta, dropout=0.0)
            print(f"[actor] Phase 11 PNA directional actor ACTIVE: hidden={args.hidden} rounds={args.rounds} "
                  f"delta={pna_delta:.3f} (preference-conditioned; decentralized D1)")
        else:
            actor = MessagePassingGraphEdgeScorer(node_dim, edge_dim, hidden=args.hidden,
                                                  rounds=args.rounds, dropout=0.0)
        bc_held = eval_held(actor, held_s, mean, std)        # random-init baseline (the floor RL starts from)
        tag = f"cold-start random-init ({args.actor})"
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
    critic_init_params = None    # snapshot for the audited critic_parameter_delta (§5.1)
    critic_history: list = []
    if args.baseline == "graph-mappo":
        critic = CentralizedGraphCritic(node_dim, edge_dim, hidden=args.critic_hidden,
                                        rounds=args.critic_rounds, critic_sees_action=args.counterfactual)
        opt_c = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
        critic_init_params = torch.cat([p.detach().flatten().clone() for p in critic.parameters()])
        print(f"[graph-mappo] centralized critic: hidden={args.critic_hidden} rounds={args.critic_rounds} "
              f"clip={args.clip_epsilon} ppo_epochs={args.ppo_epochs} critic_lr={args.critic_lr}")
        if args.counterfactual:
            assert critic.critic_sees_action, "Phase 8b needs an action-conditioned Q critic"
            cf_gen = torch.Generator().manual_seed(args.seed)
            print(f"[counterfactual] Phase 8b COMA per-agent credit ACTIVE: K_cf={args.k_cf} "
                  f"(action-conditioned Q critic; counterfactuals are critic forwards -> "
                  f"evaluator budget unchanged 1/scene)")
        if args.scq:
            scq_gen = torch.Generator().manual_seed(args.seed + 7)
            print(f"[scq] Phase 9 SCQ critic supervision ACTIVE: scq_m={args.scq_m} scq_coef={args.scq_coef} "
                  f"select={args.scq_select} -> calibrates Q with EXACT evaluator diffs; budget = 1 + up "
                  f"to {args.scq_m} evaluator calls/scene (NOT budget-neutral, logged)")
    lam_c, lam_b = args.lam_c, args.lam_b
    lam_chance = 0.0                                              # Phase 10a chance dual (0 unless --chance)
    chance_lr = args.chance_lr if args.chance_lr > 0 else args.dual_lr
    if args.chance:
        print(f"[chance] Phase 10a chance constraint ACTIVE: Pr(C<tau)<=delta={args.chance_delta} "
              f"-> dual lam_chance ascends on (frac_below - delta), reward -lam_chance*1[c<tau] "
              f"(lr={chance_lr})")
    ws_val = eval_held(actor, val_s, mean, std)["raw"]            # warm-start VAL = keep-best floor
    best_val, best_state = ws_val, {k: v.detach().clone() for k, v in actor.state_dict().items()}
    pareto_archive: list = []                                     # Phase 10c (Spec S6.4); used iff --pareto-archive
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
        lam_chance = ck.get("lam_chance", lam_chance)            # Phase 10a (back-compat default)
        history, start_upd = ck["history"], ck["update"] + 1
        if critic is not None and "critic" in ck:  # R7: resume the centralized critic + its optimizer
            saved_csa = ck.get("critic_sees_action", False)   # 8b: the ckpt's critic was V or Q
            if saved_csa != critic.critic_sees_action:
                raise SystemExit(
                    f"[resume] checkpoint critic_sees_action={saved_csa} but this run is "
                    f"{'--counterfactual' if args.counterfactual else 'plain graph-mappo'} "
                    f"(critic_sees_action={critic.critic_sees_action}); the Q vs V critic architecture "
                    f"differs -- relaunch {'WITH' if saved_csa else 'WITHOUT'} --counterfactual to resume.")
            critic.load_state_dict(ck["critic"]); opt_c.load_state_dict(ck["opt_c"])
            critic_history = ck.get("critic_history", critic_history)
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
                if not trainable_under_tristate(s["label"]["solvability_status"],
                                                exclude_unknown=args.exclude_unknown):
                    continue   # R3: certified_infeasible never trains; unknown trains by default (D8)
                edges = {e.edge_id: (e.node_u, e.node_v) for e in s["context"].graph.edges}
                budgets = dict(node_budgets_for_scene(s["context"].evaluator.scene))
                with torch.no_grad():
                    logits0 = forward_logits(actor, s, mean, std)
                    act = sample_decentralized_bcsp_action(logits0, s["edge_ids"], edges=edges,
                                                           budgets=budgets, temperature=temp_now,
                                                           compute_entropy=False)
                    if not args.counterfactual:
                        v0 = float(forward_value(critic, s, mean, std))   # rollout baseline (no_grad)
                topo = [s["edge_ids"][j] for j in act.active_edge_indices]
                r, g_c, g_b, ok = reward_of(s, topo, e_ref[i], lam_c, lam_b, args.beta, args.reward_mode,
                                            args.live_consensus_dual, lam_chance=lam_chance)
                # PER-AGENT records for the per-agent PPO ratio (Spec S9.2 -- NOT a joint sum). Each
                # agent with >=1 incident edge contributes (incident, accepted-subset, budget, logp_old).
                filtered = [pa for pa in act.per_agent if pa.incident_edge_indices]
                per_agent = [(pa.incident_edge_indices, pa.accepted_local_indices, pa.budget, float(pa.logp))
                             for pa in filtered]
                per_agent_adv = None
                if args.counterfactual:
                    # Phase 8b: per-agent COMA advantage A_i = Q(s,S) - E[Q(s,S~_i,S_-i)]. The
                    # counterfactual Q evals are critic forwards (no_grad inside) -- NO evaluator call.
                    cf = counterfactual_advantages(
                        critic, node_features=s["nf"], edge_features=s["ef"], edge_index=s["ei"],
                        edge_ids=s["edge_ids"], edges=edges, per_agent_actions=act.per_agent,
                        logits=logits0, temperature=temp_now, k_cf=args.k_cf,
                        node_mean=mean[0], node_std=std[0], edge_mean=mean[1], edge_std=std[1],
                        generator=cf_gen)
                    per_agent_adv = [cf.advantages[pa.node_id] for pa in filtered]
                    v0 = cf.q_actual                                  # action-conditioned value of S
                rec = {"s": s, "per_agent": per_agent, "per_agent_adv": per_agent_adv,
                       "r": r, "V": v0, "active": act.active_edge_indices}
                if args.scq:  # Phase 9: SCQ needs the full per-agent actions + rollout logits + edges
                    rec.update({"act_per_agent": act.per_agent, "logits0": logits0.detach(),
                                "edges": edges, "e_ref": e_ref[i]})
                batch.append(rec)
                rwds.append(r); gcs.append(g_c); gbs.append(g_b); feas.append(float(ok))
            if not batch:
                continue
            # ---- PPO-clip inner epochs on the actor: PER-AGENT ratio (Spec S9.2 -- NOT a joint sum).
            #      Flatten (scene, agent): each agent re-scores its recorded BCSP subset under fresh
            #      logits. The advantage is per (scene, agent) and FIXED across epochs (recorded at
            #      rollout): the shared single-step A_s = r - V (graph-mappo) OR the per-agent COMA
            #      counterfactual A_i = Q(s,S) - E[Q(s,S~_i,S_-i)] (Phase 8b, --counterfactual). ----
            logp_old_flat = torch.tensor([lo for b in batch for (_inc, _acc, _bud, lo) in b["per_agent"]])
            if args.counterfactual:
                adv_flat = torch.tensor([a for b in batch for a in b["per_agent_adv"]])
            else:
                adv_all = torch.tensor([b["r"] - b["V"] for b in batch])   # single-step A = r - V (GAE at T=1)
                adv_flat = torch.tensor([float(adv_all[bi]) for bi, b in enumerate(batch)
                                         for _ in b["per_agent"]])
            if args.normalize_adv:
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-6)
            last_kl, last_clip = 0.0, 0.0
            last_actor_loss, last_actor_gnorm, last_entropy = 0.0, 0.0, 0.0  # §6 audit logging
            for _epoch in range(args.ppo_epochs):
                lp_new, en_new = [], []
                for b in batch:
                    logits = forward_logits(actor, b["s"], mean, std)
                    for (incident, accepted, bud, _lo) in b["per_agent"]:
                        lp_new.append(recompute_bcsp_logp(logits, incident, accepted, temp_now, bud))
                        en_new.append(recompute_bcsp_entropy(logits, incident, temp_now, bud))
                if not lp_new:
                    break
                ppo_loss, info = ppo_clip_actor_loss(torch.stack(lp_new), logp_old_flat,
                                                     adv_flat.detach(), clip_eps=args.clip_epsilon)
                ent_bonus = torch.stack(en_new).mean()
                loss = ppo_loss - args.entropy_coef * ent_bonus  # BCSP normalized-entropy bonus
                opt.zero_grad(); loss.backward()
                # clip_grad_norm_ returns the TOTAL norm pre-clip -> actor gradient-norm audit log (§4.2)
                actor_gnorm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
                opt.step()
                last_kl, last_clip = float(info["approx_kl"]), float(info["clip_fraction"])
                last_actor_loss, last_actor_gnorm, last_entropy = float(ppo_loss), actor_gnorm, float(ent_bonus)
                if last_kl > args.target_kl * 1.5:                  # early-stop the inner loop
                    break
            # ---- critic regression -> reward (its OWN optimizer; never touches the actor). The
            #      graph-mappo critic regresses V(scene); the Phase-8b Q critic regresses the action-
            #      conditioned Q(s, S_actual) (Spec S9.7) -- same target reward, grad-on. ----
            r_t = torch.tensor([float(b["r"]) for b in batch])
            # Phase 9 SCQ: the EXACT evaluator differences are computed ONCE per update (here), so the
            # extra evaluator calls are paid once -- NOT per critic epoch (Spec S10.2 single-step).
            scq_targets, scq_calls, scq_res = None, 0, 0.0
            if args.scq:
                scq_targets = []
                for b in batch:
                    def _reward_fn(active, _b=b):
                        return reward_of(_b["s"], [_b["s"]["edge_ids"][j] for j in active], _b["e_ref"],
                                         lam_c, lam_b, args.beta, args.reward_mode,
                                         args.live_consensus_dual, lam_chance=lam_chance)[0]
                    tgt = scq_counterfactual_targets(
                        _reward_fn, per_agent_actions=b["act_per_agent"], edge_ids=b["s"]["edge_ids"],
                        edges=b["edges"], logits=b["logits0"], temperature=temp_now, scq_m=args.scq_m,
                        r_actual=b["r"], generator=scq_gen, selection=args.scq_select)
                    scq_targets.append(tgt)
                    scq_calls += tgt.counterfactual_calls
            last_critic_loss, last_critic_gnorm = 0.0, 0.0      # §5.1 audit logging
            for _ in range(args.ppo_epochs):
                if args.counterfactual:
                    v_pred = torch.stack([forward_q(critic, b["s"], b["active"], mean, std) for b in batch])
                else:
                    v_pred = torch.stack([forward_value(critic, b["s"], mean, std) for b in batch])
                v_loss = args.critic_coef * (r_t - v_pred).pow(2).mean()
                if scq_targets is not None:   # + SCQ consistency loss (recomputed per epoch; Q changes)
                    scq_terms, mars = [], []
                    for b, tgt in zip(batch, scq_targets):
                        def _q_of(active, _b=b):
                            return forward_q(critic, _b["s"], active, mean, std)
                        l, mar = scq_loss_from_targets(_q_of, tgt)
                        scq_terms.append(l); mars.append(mar)
                    v_loss = v_loss + args.critic_coef * args.scq_coef * torch.stack(scq_terms).mean()
                    scq_res = fmean(mars)
                opt_c.zero_grad(); v_loss.backward()
                # clip_grad_norm_ returns the TOTAL norm pre-clip -> critic gradient-norm audit log (§5.1)
                critic_gnorm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0))
                opt_c.step()
                last_critic_loss, last_critic_gnorm = float(v_loss), critic_gnorm
            ev = explained_variance([b["r"] for b in batch], [b["V"] for b in batch])
            scq_calls_per_scene = (scq_calls / len(batch)) if (args.scq and batch) else 0.0
            # §4/§5 audit logging: subset cardinality |S_i| and active-edge count per scene.
            card = [len(acc) for b in batch for (_inc, acc, _bud, _lo) in b["per_agent"]]
            mean_subset_card = fmean(card) if card else 0.0
            mean_active_edges = fmean([len(b["active"]) for b in batch]) if batch else 0.0
            critic_history.append({"update": upd, "explained_variance": ev, "approx_kl": last_kl,
                                   "clip_fraction": last_clip, "n_scenes": len(batch),
                                   "actor_loss": last_actor_loss, "actor_grad_norm": last_actor_gnorm,
                                   "entropy": last_entropy, "critic_loss": last_critic_loss,
                                   "critic_grad_norm": last_critic_gnorm,
                                   "mean_subset_cardinality": mean_subset_card,
                                   "mean_active_edges": mean_active_edges,
                                   "evaluator_calls": len(batch) + scq_calls,  # rollout(1/scene) + SCQ
                                   "evaluator_calls_per_scene": 1 + scq_calls_per_scene,  # SCQ NOT free
                                   "counterfactual": bool(args.counterfactual),  # Phase 8b COMA credit
                                   "k_cf": args.k_cf if args.counterfactual else 0,  # critic forwards, not evals
                                   "scq": bool(args.scq), "scq_evaluator_calls_per_scene": scq_calls_per_scene,
                                   "scq_critic_difference_error": scq_res,  # Spec S10.5 audit
                                   "critic_value_mean": fmean([b["V"] for b in batch])})
        else:
            for i in order:
                s = train_s[i]
                if not trainable_under_tristate(s["label"]["solvability_status"],
                                                exclude_unknown=args.exclude_unknown):
                    continue   # R3: certified_infeasible never trains; unknown trains by default (D8)
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
                                                args.live_consensus_dual, lam_chance=lam_chance)
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
        chance_frac_below, chance_res = 0.0, 0.0
        if args.chance and gcs:  # Phase 10a: sign-flexible dual on the chance residual Pr(c<tau)-delta
            chance_frac_below = fmean([1.0 if g > 0.0 else 0.0 for g in gcs])  # 1[c<tau] == 1[g_c>0] exactly
            # (g_c = max(0, tau-c), so g_c>0 <=> c<tau -- byte-identical to chance_residual + the c<TAU
            # reward indicator; no 1e-9 guard, which would mis-handle the measure-zero c in (tau-1e-9, tau))
            chance_res = chance_frac_below - args.chance_delta
            lam_chance = chance_dual_update(lam_chance, chance_res, chance_lr, args.lam_max)

        if upd % args.eval_every == 0 or upd == args.updates:
            ve = eval_held(actor, val_s, mean, std)
            sel = ve["raw"]
            if sel > best_val:
                best_val = sel
                best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            if args.pareto_archive:   # Phase 10c (Spec S6.4): collect a validation Pareto entry per eval
                pareto_archive.append({
                    "update": upd, "reliability_violation": 1.0 - ve["raw"],
                    "energy": ve["mean_energy_feasible_j"] if ve["mean_energy_feasible_j"] is not None else 1e9,
                    "latency": ve["mean_latency_feasible_s"] if ve["mean_latency_feasible_s"] is not None else 1e9,
                    "hypervolume": 0.0, "stability": ve["raw"],
                    "state": {k: v.detach().clone() for k, v in actor.state_dict().items()}})
            chance_tag = (f" | chance frac<tau={chance_frac_below:.3f} res={chance_res:+.3f} "
                          f"lam_chance={lam_chance:.2f}") if args.chance else ""
            print(f"[upd {upd:3d}] temp={temp_now:.2f} train R={fmean(rwds):+.3f} feas={fmean(feas):.3f} "
                  f"g_c={mean_gc:.3f} g_b={mean_gb:.3f} lam_c={lam_c:.2f} lam_b={lam_b:.2f} "
                  f"| VAL raw={ve['raw']:.3f} (best {best_val:.3f}){chance_tag}")
            if args.baseline == "graph-mappo" and critic_history:
                cm = critic_history[-1]
                cf_tag = f" cf(K={cm['k_cf']})" if cm.get("counterfactual") else ""
                print(f"           [critic] EV={cm['explained_variance']:+.3f} approx_kl={cm['approx_kl']:.4f} "
                      f"clip_frac={cm['clip_fraction']:.3f} "
                      f"eval_calls={cm['evaluator_calls']}({cm['evaluator_calls_per_scene']}/scene){cf_tag}")
            history.append({"update": upd, "train_reward": fmean(rwds), "train_feasible": fmean(feas),
                            "mean_g_c": mean_gc, "mean_g_b": mean_gb, "lam_c": lam_c, "lam_b": lam_b,
                            "lam_chance": lam_chance, "chance_frac_below": chance_frac_below,
                            "val_raw": ve["raw"]})
            if args.ckpt_every and upd % args.ckpt_every == 0:
                ckpt = {"actor": actor.state_dict(), "opt": opt.state_dict(),
                        "best_state": best_state, "best_val": best_val, "mean": mean, "std": std,
                        "baseline": baseline, "lam_c": lam_c, "lam_b": lam_b, "lam_chance": lam_chance,
                        "history": history, "update": upd}
                if critic is not None:  # R7: the centralized critic + its optimizer resume too (Spec 8.6)
                    ckpt["critic"] = critic.state_dict()
                    ckpt["opt_c"] = opt_c.state_dict()
                    ckpt["critic_history"] = critic_history
                    ckpt["critic_sees_action"] = critic.critic_sees_action  # 8b: V vs Q critic arch
                torch.save(ckpt, ckpt_path)
                print(f"[ckpt] update {upd} saved (best VAL {best_val:.3f})", flush=True)

    final_held = eval_held(actor, held_s, mean, std)   # FINAL-update policy (before keep-best revert)
    if args.pareto_archive and pareto_archive:   # Phase 10c (Spec S6.4): pick the checkpoint by the
        # reliability-risk -> min-violation -> non-dominated -> hypervolume -> stability order, NOT raw
        # feasibility alone. Reported alongside the raw-best for an honest comparison.
        sel_entry = pareto_archive_select(pareto_archive, risk_budget=args.pareto_risk_budget)
        print(f"[pareto] S6.4 selected checkpoint from update {sel_entry['update']} "
              f"(reliability_violation={sel_entry['reliability_violation']:.3f} "
              f"E={sel_entry['energy']:.3g} L={sel_entry['latency']:.3g}, stability={sel_entry['stability']:.3f}) "
              f"-- vs raw-best-VAL {best_val:.3f}")
        best_state = sel_entry["state"]
    actor.load_state_dict(best_state)        # keep-best (never worse than warm-start on VAL)
    rl_held = eval_held(actor, held_s, mean, std)
    print("=" * 78)
    print(f"[BC   ] held raw={bc_held['raw']:.3f} cond="
          f"{None if bc_held['conditional'] is None else round(bc_held['conditional'],3)} "
          f"energy={bc_held['mean_energy_feasible_j']}")
    print(f"[RLfin] held raw={final_held['raw']:.3f} cond="
          f"{None if final_held['conditional'] is None else round(final_held['conditional'],3)} "
          f"energy={final_held['mean_energy_feasible_j']}  (final-update policy, pre keep-best)")
    print(f"[RL   ] held raw={rl_held['raw']:.3f} wit_recall="
          f"{None if rl_held['witness_recall'] is None else round(rl_held['witness_recall'],3)} "
          f"wit_disc={rl_held['witness_discovered']}/{rl_held['unknown_total']} "
          f"energy={rl_held['mean_energy_feasible_j']}  (keep-best; wit_recall=success on "
          f"witness_feasible held, wit_disc=witnesses discovered on unknown held)")
    print(f"[time] {time.time() - t0:.1f}s")

    result = {"config": vars(args), "ceiling": ceiling,
              "warm_start_bc": bc_held, "rl": rl_held, "rl_final": final_held, "history": history,
              "deltas": {"raw": rl_held["raw"] - bc_held["raw"],
                         "conditional": (None if rl_held["conditional"] is None or bc_held["conditional"] is None
                                         else rl_held["conditional"] - bc_held["conditional"])}}
    (out_dir / ("smoke_result.json" if args.smoke else "rl_result.json")).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    # Full-Integration-Audit: the MEASURED critic-parameter delta (§5.1 critic_parameter_delta>0)
    # + the honest mechanism_activation.json (dynamic_task=False at T=1; the env-math read from data).
    critic_param_delta = None
    if critic is not None and critic_init_params is not None:
        final_params = torch.cat([p.detach().flatten() for p in critic.parameters()])
        critic_param_delta = float((final_params - critic_init_params).norm())
    scq_cps = critic_history[-1].get("scq_evaluator_calls_per_scene", 0.0) if critic_history else 0.0
    (out_dir / "mechanism_activation.json").write_text(json.dumps(
        mechanism_activation(args, critic_parameter_delta=critic_param_delta,
                             scq_calls_per_scene=scq_cps, dataset_env_math=dataset_env_math),
        indent=2, default=str), encoding="utf-8")

    artifacts = {"actors": [{"state": actor.state_dict(), "mean": mean, "std": std}]}
    if args.baseline == "graph-mappo":
        # the centralized critic + its optimizer are TRAINING artifacts only (D1: never deployed)
        artifacts["critic"] = critic.state_dict()
        artifacts["opt_c"] = opt_c.state_dict()
        artifacts["critic_parameter_delta"] = critic_param_delta
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
