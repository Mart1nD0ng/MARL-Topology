"""Diagnostic: is the sparse2 in-range RL headroom CAPTURABLE or DECODER-CAPPED?

Question (from URBAN_V2X_RESEARCH_LOG iteration 7/8): RL fine-tuning matches but never
beats the BC actor. On sparse2 (2 RSU) BC conditional = 0.864 (misses ~14% of solvable
scenes) yet RL can't capture it. Is that residual headroom an OPTIMIZATION problem an RL
policy could in principle capture, or is it a REPRESENTATION limit of the deployed
local_mutual_assemble decoder (no per-edge-logit assignment can output the needed
tau-clearing topology, so NO policy with this decoder could ever capture it)?

Steps:
  1. Load sparse2 pool (load_pool pattern), replicate the campaign split
     (Random(7).shuffle, cut=int(0.6*len), held=pool[cut:]).
  2. build_samples; load BC actor[0] from E9_sparse2 _artifacts.pt (state/mean/std), rounds=4 hidden=64.
  3. For each HELD scene with feasible_exists==True, decode BC via local_mutual_assemble;
     check tau=0.9 AND is_budget_feasible. Count SOLVABLE scenes BC FAILS.
  4. For those BC-missed-solvable scenes:
     (a) re-evaluate the SA-teacher topology set(label.selected_physical_edges):
         does it clear tau AND budget?
     (b) DECODER-REACHABILITY: force logits = +10 on teacher edges, -10 elsewhere; run
         local_mutual_assemble. Does its output equal / clear tau like the teacher topology?
         If the budget-respecting mutual-acceptance decoder CANNOT reproduce a tau-clearing
         topology for these scenes -> the headroom is DECODER-CAPPED.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from marl_topology.budgets import is_budget_feasible, node_budgets_for_scene  # noqa: E402
from marl_topology.training.decentralized_distillation import (  # noqa: E402
    TAU,
    build_samples,
    forward_logits,
    load_actor_from_state,
    local_mutual_assemble,
)
from marl_topology.data.row_context_builder import build_row_contexts  # noqa: E402

SPARSE2_SHARDS = [str(ROOT / "result_save" / "campaign" / "data" / "sparse2" / f"_op_shard_{s}.pkl")
                  for s in range(3001, 3025)]
ARTIFACTS = str(ROOT / "result_save" / "campaign" / "E9_sparse2" / "_artifacts.pt")


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


def evaluate(sample, edge_ids):
    m = sample["context"].evaluator.evaluate(set(edge_ids)).metrics
    return (float(m["consensus_success_probability"]),
            float(m.get("energy", 0.0)), float(m.get("latency", 0.0)))


def budgets_of(sample):
    return dict(node_budgets_for_scene(sample["context"].evaluator.scene))


def clears(sample, edge_ids):
    """tau AND budget feasibility of an edge-id set."""
    c, _e, _l = evaluate(sample, list(edge_ids))
    ok_budget = is_budget_feasible(tuple(edge_ids), node_budgets_for_scene(sample["context"].evaluator.scene))
    return (c >= TAU) and ok_budget, c, ok_budget


def teacher_edges(sample):
    return [str(e) for e in sample["label"].get("selected_physical_edges", [])]


def forced_teacher_logits(sample, hi=10.0, lo=-10.0):
    """Per-edge logits that strongly favor EXACTLY the teacher edges: +hi for teacher edges, lo else."""
    tset = set(teacher_edges(sample))
    return torch.tensor([hi if eid in tset else lo for eid in sample["edge_ids"]], dtype=torch.float32)


def teacher_degree_within_budget(sample):
    """Does the teacher topology itself respect per-node budgets? (necessary for decoder reachability)"""
    buds = budgets_of(sample)
    ends = {e.edge_id: (e.node_u, e.node_v) for e in sample["context"].graph.edges}
    deg: dict = {}
    for eid in teacher_edges(sample):
        if eid not in ends:
            continue
        a, b = ends[eid]
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    over = {node: (d, buds.get(node, 0)) for node, d in deg.items() if d > buds.get(node, 0)}
    return (len(over) == 0), over


def main() -> None:
    pool = load_pool(SPARSE2_SHARDS)
    Random(7).shuffle(pool)
    cut = int(0.6 * len(pool))
    held_items = pool[cut:]
    held = build_samples(held_items)

    art = torch.load(ARTIFACTS, map_location="cpu", weights_only=False)
    a = art["actors"][0]
    mean, std = a["mean"], a["std"]
    node_dim, edge_dim = held[0]["nf"].shape[1], held[0]["ef"].shape[1]
    actor = load_actor_from_state(a["state"], node_dim, edge_dim, hidden=64, rounds=4)
    actor.eval()

    n_held = len(held)
    solvable = [s for s in held if bool(s["label"]["feasible_exists"])]
    n_solvable = len(solvable)

    bc_solved_solvable = 0
    missed = []  # solvable scenes BC FAILS to clear
    for s in solvable:
        with torch.no_grad():
            logits = forward_logits(actor, s, mean, std)
        topo = local_mutual_assemble(logits, s["edge_ids"], s["context"])
        ok, _c, _b = clears(s, topo)
        if ok:
            bc_solved_solvable += 1
        else:
            missed.append((s, topo, _c, _b))

    bc_conditional = bc_solved_solvable / max(1, n_solvable)

    # --- analyze the BC-missed-solvable scenes ---
    teacher_clears_count = 0
    teacher_within_budget_count = 0
    decoder_reaches_clearing = 0      # forced-teacher-logit decode clears tau+budget
    decoder_equals_teacher = 0        # forced-teacher-logit decode == exact teacher edge set
    per_scene = []
    for s, bc_topo, bc_c, bc_budget_ok in missed:
        tedges = teacher_edges(s)
        t_ok, t_c, t_budget_ok = clears(s, tedges)
        teacher_clears_count += int(t_ok)
        within, over = teacher_degree_within_budget(s)
        teacher_within_budget_count += int(within)

        # DECODER-REACHABILITY: force logits to exactly the teacher edges, run the deployed decoder
        flogits = forced_teacher_logits(s)
        dec_topo = local_mutual_assemble(flogits, s["edge_ids"], s["context"])
        d_ok, d_c, d_budget_ok = clears(s, dec_topo)
        decoder_reaches_clearing += int(d_ok)
        decoder_equals_teacher += int(set(dec_topo) == set(tedges))

        n_nodes = len(s["context"].graph.node_ids)
        per_scene.append({
            "n_nodes": n_nodes,
            "bc_consensus": round(bc_c, 4), "bc_budget_ok": bc_budget_ok, "bc_n_edges": len(bc_topo),
            "teacher_consensus": round(t_c, 4), "teacher_budget_ok": t_budget_ok,
            "teacher_clears": t_ok, "teacher_n_edges": len(tedges),
            "teacher_within_budget": within, "teacher_over_budget_nodes": over,
            "decoder_forced_consensus": round(d_c, 4), "decoder_forced_budget_ok": d_budget_ok,
            "decoder_forced_clears": d_ok, "decoder_forced_n_edges": len(dec_topo),
            "decoder_equals_teacher": (set(dec_topo) == set(tedges)),
        })

    n_missed = len(missed)
    result = {
        "n_held": n_held,
        "n_solvable_held": n_solvable,
        "bc_solved_on_solvable": bc_solved_solvable,
        "bc_conditional": round(bc_conditional, 4),
        "n_bc_missed_solvable": n_missed,
        "teacher_clears_count": teacher_clears_count,
        "teacher_within_budget_count": teacher_within_budget_count,
        "decoder_reaches_tau_clearing_count": decoder_reaches_clearing,
        "decoder_equals_teacher_count": decoder_equals_teacher,
        "per_scene": per_scene,
    }
    out = ROOT / "result_save" / "dec_rl_sparse2" / "headroom_capturability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("=" * 78)
    print(f"HELD scenes: {n_held} | solvable (feasible_exists): {n_solvable}")
    print(f"BC conditional (solved/solvable): {bc_solved_solvable}/{n_solvable} = {bc_conditional:.4f}")
    print(f"BC-MISSED-SOLVABLE scenes: {n_missed}")
    print("-" * 78)
    print(f"  teacher topology clears tau+budget : {teacher_clears_count}/{n_missed}")
    print(f"  teacher within per-node budget      : {teacher_within_budget_count}/{n_missed}")
    print(f"  DECODER (forced teacher logits) reaches a tau+budget-clearing topology : "
          f"{decoder_reaches_clearing}/{n_missed}")
    print(f"  DECODER forced-output == exact teacher edge set                        : "
          f"{decoder_equals_teacher}/{n_missed}")
    print("-" * 78)
    for i, ps in enumerate(per_scene):
        print(f"  [{i}] N={ps['n_nodes']:2d} bc(c={ps['bc_consensus']},bud={ps['bc_budget_ok']},e={ps['bc_n_edges']}) "
              f"teach(c={ps['teacher_consensus']},bud={ps['teacher_budget_ok']},clears={ps['teacher_clears']},"
              f"e={ps['teacher_n_edges']},inbud={ps['teacher_within_budget']}) "
              f"DEC(c={ps['decoder_forced_consensus']},bud={ps['decoder_forced_budget_ok']},"
              f"clears={ps['decoder_forced_clears']},==teach={ps['decoder_equals_teacher']})")
        if ps["teacher_over_budget_nodes"]:
            print(f"        teacher OVER budget nodes (deg,budget): {ps['teacher_over_budget_nodes']}")
    print("=" * 78)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
