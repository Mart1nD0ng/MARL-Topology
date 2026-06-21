"""Recovered production training/eval pipeline for the decentralized message-passing actor.

This is the validated path that produced the project's best result (held-out decentralized
feasibility 0.82 under the full TR 37.885 stochastic stack at the 4-RSU / 20 dBm operating
point; ``docs/URBAN_V2X_RESEARCH_LOG.md`` Step-3). Recovered verbatim from the ``logs`` research
scripts (``phase1_decentralized_curve`` / ``phase2p_pipeline``) into src so it is the single
trunk, not scratch.

Pipeline: Stage-33 operating-point dataset shards -> per-scene actor-safe graph payloads ->
BC-distil the (budget-aware SA or critic-planner) teacher into the K-round message-passing actor
(weight decay + early stopping + keep-best) -> GENUINELY DECENTRALIZED execution via local
mutual-acceptance assembly (``policies/decentralized_mutual_acceptance``). The centralized graph
critic (CTDE, training-only) supplies the planner-target arm; it never reaches the deployed actor.
"""

from __future__ import annotations

import math
from collections import defaultdict
from random import Random
from statistics import fmean, pstdev

import torch
import torch.nn.functional as F

from marl_topology.budgets import is_budget_feasible, node_budgets_for_scene
from marl_topology.models.message_passing_graph_edge_scorer import MessagePassingGraphEdgeScorer
from marl_topology.policies.decentralized_mutual_acceptance import (
    global_argsort_assemble,
    local_mutual_assemble,
)
from marl_topology.data.graph_payload import graph_payload as _graph_payload

TAU = 0.9
DEFAULT_ROUNDS = 4
DEFAULT_HIDDEN = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LR = 2e-3
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 15
DEFAULT_POS_WEIGHT = 3.0


# ----------------------------------------------------------------- per-scene tensors
# NB: dataset-shard loading (pickle / file I/O) lives in the runnable drivers/tests, not here
# -- src stays I/O-free. A loaded "pool" is a list of (row, context, teacher_label) where each
# (row, context) comes from ``Stage33ProductionMappoAdapter.build_row_contexts``.
def scene_tensors(row, context):
    payload = _graph_payload(row=row, context=context, previous_selected_edges=[], step_index=0)
    nf = torch.tensor([list(map(float, n)) for n in payload["node_features"]])
    ef = torch.tensor([list(map(float, e)) for e in payload["edge_features"]])
    ei = torch.tensor([list(map(int, p)) for p in payload["edge_index"]], dtype=torch.long)
    edge_ids = [e.edge_id for e in context.graph.edges]
    return nf, ef, ei, edge_ids


def build_samples(items):
    """(tensors, target_vector, context) per scene; target = teacher's selected physical edges."""

    out = []
    for row, context, label in items:
        nf, ef, ei, edge_ids = scene_tensors(row, context)
        target_edges = set(str(e) for e in label["selected_physical_edges"])
        tgt = torch.tensor([1.0 if e in target_edges else 0.0 for e in edge_ids])
        out.append({"nf": nf, "ef": ef, "ei": ei, "edge_ids": edge_ids, "tgt": tgt,
                    "context": context, "label": label})
    return out


# ----------------------------------------------------------------- train / eval
def forward_logits(actor, sample, mean, std):
    nf = ((sample["nf"] - mean[0]) / std[0]).unsqueeze(0)
    ef = ((sample["ef"] - mean[1]) / std[1]).unsqueeze(0)
    return actor(nf, ef, sample["ei"].unsqueeze(0),
                 torch.ones(1, sample["nf"].shape[0]), torch.ones(1, sample["ef"].shape[0]))[0]


def feasible_rate(actor, samples, mean, std, assemble_fn=local_mutual_assemble):
    """Fraction of scenes whose decoded topology clears tau AND is radio-budget feasible."""

    actor.eval()
    feas = 0
    for sample in samples:
        with torch.no_grad():
            logits = forward_logits(actor, sample, mean, std)
        topo = assemble_fn(logits, sample["edge_ids"], sample["context"])
        p = float(sample["context"].evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
        ok = p >= TAU and is_budget_feasible(tuple(topo), node_budgets_for_scene(sample["context"].evaluator.scene))
        feas += int(ok)
    return feas / max(1, len(samples))


def feasible_rate_by_n(actor, samples, mean, std, assemble_fn=local_mutual_assemble):
    agg = defaultdict(lambda: [0, 0])
    actor.eval()
    for sample in samples:
        with torch.no_grad():
            logits = forward_logits(actor, sample, mean, std)
        topo = assemble_fn(logits, sample["edge_ids"], sample["context"])
        p = float(sample["context"].evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
        ok = p >= TAU and is_budget_feasible(tuple(topo), node_budgets_for_scene(sample["context"].evaluator.scene))
        n = len(sample["context"].graph.node_ids)
        agg[n][0] += int(ok)
        agg[n][1] += 1
    overall = sum(a for a, _ in agg.values()) / max(1, sum(b for _, b in agg.values()))
    return overall, {n: a / b for n, (a, b) in sorted(agg.items())}


def feasible_breakdown(actor, samples, mean, std, assemble_fn=local_mutual_assemble):
    """Raw and solvable-conditional feasibility, overall and per-N.

    ``raw`` = solved / all held scenes (bounded by scene-generation: the dataset is
    deliberately part-infeasible). ``conditional`` = solved-on-solvable / solvable, where
    a scene is solvable when the teacher label reports ``feasible_exists`` (some budget-
    feasible topology clears tau). The conditional rate is the honest controller-quality
    number -- it isolates "of the scenes that CAN be solved, how many does the
    decentralized planner solve" from the fraction the scene distribution makes
    unsolvable. Returns a dict with both, plus per-N splits and the solvable counts.
    """

    actor.eval()
    # per N: [solved, total, solved_on_solvable, solvable]
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for sample in samples:
        with torch.no_grad():
            logits = forward_logits(actor, sample, mean, std)
        topo = assemble_fn(logits, sample["edge_ids"], sample["context"])
        p = float(sample["context"].evaluator.evaluate(set(topo)).metrics["consensus_success_probability"])
        ok = p >= TAU and is_budget_feasible(tuple(topo), node_budgets_for_scene(sample["context"].evaluator.scene))
        solvable = bool(sample["label"]["feasible_exists"])
        n = len(sample["context"].graph.node_ids)
        agg[n][0] += int(ok)
        agg[n][1] += 1
        if solvable:
            agg[n][2] += int(ok)
            agg[n][3] += 1
    solved = sum(a[0] for a in agg.values())
    total = sum(a[1] for a in agg.values())
    solved_solv = sum(a[2] for a in agg.values())
    solvable = sum(a[3] for a in agg.values())
    return {
        "raw": solved / max(1, total),
        "conditional": (solved_solv / solvable) if solvable else None,
        "solved": solved, "total": total, "solvable": solvable,
        "raw_by_n": {n: a[0] / a[1] for n, a in sorted(agg.items())},
        "conditional_by_n": {n: (a[2] / a[3] if a[3] else None) for n, a in sorted(agg.items())},
        "solvable_by_n": {n: a[3] for n, a in sorted(agg.items())},
    }


def feature_standardization(train_samples):
    allnf = torch.cat([s["nf"] for s in train_samples])
    allef = torch.cat([s["ef"] for s in train_samples])
    mean = (allnf.mean(0), allef.mean(0))
    std = (allnf.std(0).clamp_min(1e-6), allef.std(0).clamp_min(1e-6))
    return mean, std


def train_actor(train_samples, val_samples, seed, *, hidden=DEFAULT_HIDDEN, rounds=DEFAULT_ROUNDS,
                dropout=DEFAULT_DROPOUT, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                max_epochs=DEFAULT_MAX_EPOCHS, patience=DEFAULT_PATIENCE, pos_weight=DEFAULT_POS_WEIGHT):
    """BC-distil the teacher into the K-round message-passing actor with early stopping + keep-best."""

    torch.manual_seed(seed)
    node_dim = train_samples[0]["nf"].shape[1]
    edge_dim = train_samples[0]["ef"].shape[1]
    mean, std = feature_standardization(train_samples)
    actor = MessagePassingGraphEdgeScorer(node_dim, edge_dim, hidden=hidden, rounds=rounds, dropout=dropout)
    opt = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor(float(pos_weight))
    best_val, best_state, since = float("inf"), None, 0
    rng = Random(seed)
    epoch = 0
    for epoch in range(max_epochs):
        actor.train()
        order = list(range(len(train_samples)))
        rng.shuffle(order)
        for i in order:
            sample = train_samples[i]
            logits = forward_logits(actor, sample, mean, std)
            loss = F.binary_cross_entropy_with_logits(logits, sample["tgt"], pos_weight=pw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
        actor.eval()
        with torch.no_grad():
            val = fmean(
                float(F.binary_cross_entropy_with_logits(forward_logits(actor, s, mean, std), s["tgt"], pos_weight=pw))
                for s in val_samples
            )
        if val < best_val - 1e-4:
            best_val, since = val, 0
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        actor.load_state_dict(best_state)
    actor.eval()
    return actor, mean, std, epoch + 1, best_val


def load_actor_from_state(state, node_dim, edge_dim, *, hidden=DEFAULT_HIDDEN, rounds=DEFAULT_ROUNDS):
    """Rebuild a frozen actor from a saved state dict (e.g. ``_artifacts_step3.pt``)."""

    actor = MessagePassingGraphEdgeScorer(node_dim, edge_dim, hidden=hidden, rounds=rounds, dropout=0.0)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def ci95(values):
    n = len(values)
    m = fmean(values)
    if n < 2:
        return m, 0.0
    sd = pstdev(values) * math.sqrt(n / (n - 1))
    t = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    return m, t * sd / math.sqrt(n)


__all__ = [
    "TAU", "scene_tensors", "build_samples", "forward_logits",
    "feasible_rate", "feasible_rate_by_n", "feasible_breakdown", "feature_standardization", "train_actor",
    "load_actor_from_state", "ci95", "local_mutual_assemble", "global_argsort_assemble",
]
