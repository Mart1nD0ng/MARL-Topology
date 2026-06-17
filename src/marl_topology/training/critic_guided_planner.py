"""Critic-guided planner — the teacher-upgrade arm of the recovered production pipeline.

This is the second half of the validated recipe (`docs/URBAN_V2X_RESEARCH_LOG.md` Phase-2 /
phase2p / Step-3): a centralized graph critic (CTDE, training-only) is trained on
(topology -> ACTUAL consensus/feasibility), hardened on the dense candidate region via DAgger,
then used to guide a budget-aware beam search whose top candidates are evaluator-verified. The
resulting feasible topologies are the planner TARGETS that the decentralized actor is BC-distilled
on (`training/decentralized_distillation.py`). The critic-planner targets beat the SA-teacher
targets and lift the actor to the project's best result (~0.82 held-out at the operating point).

Recovered verbatim from the deleted `logs/` research path (train_urban_critic.py,
phase2_critic_hardening.py, rigorous_ctde_rerun.py). Pure / I/O-free: dataset-shard loading lives
in the runnable drivers. The critic is never deployed (CTDE); only its planner targets reach the
actor, and the deployed decode stays decentralized (local mutual acceptance).
"""

from __future__ import annotations

from random import Random
from statistics import fmean

import torch
import torch.nn.functional as F

from marl_topology.budgets import is_budget_feasible, node_budgets_for_scene
from marl_topology.models.centralized_message_passing_graph_critic import (
    CentralizedMessagePassingGraphCritic,
    CentralizedMessagePassingGraphCriticConfig,
    GraphCriticBatch,
)
from marl_topology.training.mappo.stage28_repaired_critic_pilot import (
    _graph_batch_from_payloads,
    _graph_payload,
)

TAU = 0.9
DEFAULT_BEAM_WIDTH = 6
DEFAULT_VERIFY_K = 20
DEFAULT_CRITIC_EPOCHS = 40


# ----------------------------------------------------------------- payloads / features
def payload_for(row, context, edges):
    return _graph_payload(row=row, context=context, previous_selected_edges=list(edges), step_index=0)


def sample_topologies(all_edges, teacher_edges, rng):
    """A spread of topologies per scene (teacher, full, empty, random densities, +/-1 edge) so the
    critic learns to rank the dense near-boundary region where final selection happens."""

    topos = [("teacher", list(teacher_edges)), ("full", list(all_edges)), ("empty", [])]
    for p in (0.2, 0.3, 0.4, 0.5, 0.6):
        topos.append((f"rand{p}", [e for e in all_edges if rng.random() < p]))
    if teacher_edges:
        topos.append(("teacher-1", list(teacher_edges)[:-1]))
    extra = [e for e in all_edges if e not in set(teacher_edges)]
    if extra:
        topos.append(("teacher+1", list(teacher_edges) + [rng.choice(extra)]))
    return topos


def build_critic_samples(rows, rng):
    """(payload, actual_consensus, feasible_float) over sampled topologies per (row, context)."""

    samples = []
    for row, context in rows:
        evaluator = context.evaluator
        all_edges = list(context.graph.edge_ids)
        for _name, edges in sample_topologies(all_edges, row.selected_physical_edges, rng):
            actual = float(evaluator.evaluate(set(edges)).metrics["consensus_success_probability"])
            samples.append((payload_for(row, context, edges), actual, 1.0 if actual >= TAU else 0.0))
    return samples


def standardization_state(payloads):
    node_rows = [list(map(float, n)) for p in payloads for n in p["node_features"]]
    edge_rows = [list(map(float, e)) for p in payloads for e in p["edge_features"]]
    nt = torch.tensor(node_rows)
    et = torch.tensor(edge_rows)
    return {
        "node_mean": nt.mean(0), "node_std": nt.std(0).clamp_min(1e-6),
        "edge_mean": et.mean(0), "edge_std": et.std(0).clamp_min(1e-6),
    }


def std_batch(payloads, state):
    batch = _graph_batch_from_payloads(payloads)
    return GraphCriticBatch(
        node_features=(batch.node_features - state["node_mean"]) / state["node_std"],
        edge_features=(batch.edge_features - state["edge_mean"]) / state["edge_std"],
        edge_index=batch.edge_index, node_mask=batch.node_mask, edge_mask=batch.edge_mask,
    )


# ----------------------------------------------------------------- critic fit (CTDE, training-only)
def fit_critic(samples, state, seed, epochs=DEFAULT_CRITIC_EPOCHS):
    """Train the centralized graph critic on (payload -> actual consensus + feasibility)."""

    torch.manual_seed(seed)
    critic = CentralizedMessagePassingGraphCritic(CentralizedMessagePassingGraphCriticConfig())
    opt = torch.optim.AdamW(critic.parameters(), lr=2e-3, weight_decay=1e-5)
    payloads = [s[0] for s in samples]
    cons = torch.tensor([s[1] for s in samples])
    feas = torch.tensor([s[2] for s in samples])
    n = len(samples)
    critic.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 32):
            idx = perm[start:start + 32].tolist()
            out = critic(std_batch([payloads[i] for i in idx], state))
            loss = (F.mse_loss(out.consensus_proxy, cons[idx])
                    + F.binary_cross_entropy_with_logits(out.feasibility_logit, feas[idx]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
    critic.eval()
    return critic


def predict_consensus(critic, state, row, context, edges):
    with torch.no_grad():
        out = critic(std_batch([payload_for(row, context, edges)], state))
    return float(out.consensus_proxy.reshape(-1)[0])


# ----------------------------------------------------------------- budget-aware beam + verify
def beam(score_fn, context, max_edges, width):
    """Budget-aware beam search over topologies; returns {topology_tuple: score} visited."""

    budgets = dict(node_budgets_for_scene(context.evaluator.scene))
    all_edges = list(context.graph.edge_ids)
    ends = {e: (context.graph.get_edge(e).node_u, context.graph.get_edge(e).node_v) for e in all_edges}
    beams = [((), {}, score_fn(()))]
    visited = {}
    for _ in range(max_edges):
        pool = {}
        for topo, deg, _p in beams:
            tset = set(topo)
            for e in all_edges:
                if e in tset:
                    continue
                a, b = ends[e]
                if deg.get(a, 0) >= budgets[a] or deg.get(b, 0) >= budgets[b]:
                    continue
                new = tuple(sorted(topo + (e,)))
                if new in pool:
                    continue
                nd = dict(deg); nd[a] = nd.get(a, 0) + 1; nd[b] = nd.get(b, 0) + 1
                pool[new] = (nd, score_fn(new))
        if not pool:
            break
        ranked = sorted(pool.items(), key=lambda kv: kv[1][1], reverse=True)[:width]
        beams = [(topo, nd, sc) for topo, (nd, sc) in ranked]
        for topo, _nd, sc in beams:
            visited[topo] = sc
    return visited


def verify_best(visited, context, k):
    """Evaluator-verify the top-k beam candidates; return the best budget-feasible one."""

    ranked = sorted(visited.items(), key=lambda kv: kv[1], reverse=True)[:k]
    best, best_key = (), (0, -1.0)
    for topo, _s in ranked:
        if not is_budget_feasible(topo, node_budgets_for_scene(context.evaluator.scene)):
            continue
        p = float(context.evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
        key = (int(p >= TAU), p)
        if key > best_key:
            best_key, best = key, topo
    return best


def pearson(pairs):
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    if not xs:
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    cov = fmean([(a - mx) * (b - my) for a, b in pairs])
    sx = (fmean([(a - mx) ** 2 for a in xs]) ** 0.5) or 1e-9
    sy = (fmean([(b - my) ** 2 for b in ys]) ** 0.5) or 1e-9
    return cov / (sx * sy)


def beam_for(critic, state, row, context, *, beam_width=DEFAULT_BEAM_WIDTH):
    n = len(context.graph.node_ids)
    return beam(lambda t: predict_consensus(critic, state, row, context, list(t)), context, 2 * n, beam_width)


# ----------------------------------------------------------------- DAgger hard negatives + eval
def collect_dense_hard_negatives(critic, state, items, *, beam_width=DEFAULT_BEAM_WIDTH):
    """Label the beam's visited DENSE candidates with TRUE consensus (the region the critic is
    weak on, where final selection ranks) -> hard-negative/positive critic training samples."""

    extra = []
    for row, context, _label in items:
        n = len(context.graph.node_ids)
        for topo in beam_for(critic, state, row, context, beam_width=beam_width):
            if len(topo) < n - 1:
                continue
            actual = float(context.evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
            extra.append((payload_for(row, context, list(topo)), actual, 1.0 if actual >= TAU else 0.0))
    return extra


def controller_feasibility(critic, state, items, *, verify_k=DEFAULT_VERIFY_K, beam_width=DEFAULT_BEAM_WIDTH):
    """(pure, verified) held-out feasibility of the critic-guided controller: pure = the beam's
    peak-scored topology; verified = evaluator-verify the top-k."""

    pure = verify = 0
    for row, context, _label in items:
        visited = beam_for(critic, state, row, context, beam_width=beam_width)
        budgets = node_budgets_for_scene(context.evaluator.scene)
        peak = max(visited.items(), key=lambda kv: kv[1])[0] if visited else ()
        p = float(context.evaluator.evaluate(set(peak)).metrics["consensus_success_probability"])
        pure += int(p >= TAU and is_budget_feasible(peak, budgets))
        vtopo = verify_best(visited, context, verify_k)
        vp = float(context.evaluator.evaluate(set(vtopo)).metrics["consensus_success_probability"])
        verify += int(vp >= TAU and is_budget_feasible(tuple(vtopo), budgets))
    n = max(1, len(items))
    return pure / n, verify / n


def planner_targets(critic, state, items, *, verify_k=DEFAULT_VERIFY_K, beam_width=DEFAULT_BEAM_WIDTH):
    """Generate critic-planner BC targets: for each scene, the evaluator-verified best feasible
    topology from the critic-guided beam. Returns (retargeted_items, feasible_count) where each
    retargeted label['selected_physical_edges'] is the planner topology when feasible (else the
    original teacher label is kept)."""

    retargeted, feasible = [], 0
    for row, context, label in items:
        visited = beam_for(critic, state, row, context, beam_width=beam_width)
        budgets = node_budgets_for_scene(context.evaluator.scene)
        topo = verify_best(visited, context, verify_k)
        p = float(context.evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
        new_label = dict(label)
        if p >= TAU and is_budget_feasible(tuple(topo), budgets):
            feasible += 1
            new_label["selected_physical_edges"] = tuple(topo)
        retargeted.append((row, context, new_label))
    return retargeted, feasible
