# MARL-Topology — Decentralized constrained-RL for V2X topology control

A single-trunk research codebase for **fully decentralized** multi-agent topology
planning in urban V2X networks. A graph neural-network actor learns, per node, which
physical links to activate so that the network reaches **PBFT consensus** reliably
under a radio-budget constraint — with **no central critic, no global-state leakage,
and no oracle supervision**.

## The one trunk

The project has exactly **one** trunk: the decentralized cold-start constrained-RL
trunk, driven by

```
scripts/train/train_decentralized_rl.py     # the trunk
scripts/train/evaluate_actor_on_dataset.py  # eval-only tool (cross-N / OOD / baselines)
scripts/train/build_operating_point_dataset.py  # dataset shard builder
```

The old **centralized-critic MAPPO trunk** and the **planner / critic teacher arm**
were **removed on 2026-06-21** (owner-authorized full removal). What remains is the
critic-free decentralized learner.

## What the trunk does

- **Learning — critic-free REINFORCE + RLOO.** Per-edge policy over the
  `MessagePassingGraphEdgeScorer` logits (parameter-shared, scale-invariant, K-hop
  local message passing). A leave-one-out (RLOO) baseline reduces variance. There is
  **no critic at all** → zero CTDE gap, the strongest form of the "learning must be
  truly decentralized" invariant.
- **Reward — one principled objective, not a weighted bag.** A feasibility-first
  potential `r = (c − τ)` (Ng–Harada shaping) where `c` is the **closed-form PBFT
  quorum-tail consensus success probability** (`protocol/quorum_tail.py`, a
  Poisson-binomial whole-network quorum tail — no Monte Carlo). The budget constraint
  is handled by a **Lagrangian dual** updated by dual ascent (constrained-RL / RCPO),
  not by hand-tuned weights. `τ = 0.9`, hard-frozen across training and evaluation.
- **Decode — feasibility by construction.** `local_mutual_assemble`
  (`policies/decentralized_mutual_acceptance.py`): each node ranks only its own
  incident edges within its radio budget; an edge activates iff both endpoints accept.
  Per-node computable, zero global state. `global_argsort_assemble` is kept only as the
  centralized-decode ablation.
- **Actor — `MessagePassingGraphEdgeScorer`** (`models/message_passing_graph_edge_scorer.py`):
  a K-round bidirectional message-passing GNN edge scorer (decentralized-with-communication).

Warm-start (BC from a frozen artifact) is the default; `--cold-start` runs from random
init. `keep-best` on a held-out-from-train VAL split guarantees RL is never reported
below the BC start.

## Headline result

A **cold-start, oracle-free** decentralized learner **beats the centralized
Simulated-Annealing oracle at out-of-range scale (N = 24)**: held-out raw feasibility
~0.769–0.808 vs the SA-teacher ceiling 0.577, with a 4-seed CI95 of **[+0.067, +0.279]**
on `raw_mean − 0.577` — i.e. the lower bound is above zero. This is decisive evidence
that the decentralized learner generalizes past the scale its centralized teacher was
built for. See `docs/URBAN_V2X_RESEARCH_LOG.md`.

## Quick start

```bash
# fast end-to-end smoke (random init, ~10s)
python scripts/train/train_decentralized_rl.py --smoke --cold-start

# operating-point diagnostic
python scripts/train/train_decentralized_rl.py

# eval a frozen actor on an arbitrary dataset (cross-N / OOD / baselines)
python scripts/train/evaluate_actor_on_dataset.py \
    --artifacts result_save/.../_artifacts.pt --shards result_save/.../_op_shard_*.pkl \
    --eval-split all --baselines
```

## Repository layout

- `src/marl_topology/` — the importable package:
  - `protocol/quorum_tail.py` — closed-form PBFT quorum-tail consensus reliability.
  - `policies/decentralized_mutual_acceptance.py` — the decentralized decoder.
  - `models/message_passing_graph_edge_scorer.py` — the production actor.
  - `training/decentralized_distillation.py` — BC distillation + decentralized eval helpers.
  - `data/row_context_builder.py` — critic-free per-split (row, context) builder.
  - `data/graph_payload.py` — critic-free node/edge graph featurization.
- `scripts/train/` — the trunk + dataset/eval tooling (see above).
- `tests/` — `unit/` and `contract/` suites.
- `docs/` — `TRUNK_MAP.md`, `URBAN_V2X_RESEARCH_LOG.md`,
  `FOUR090_CAMPAIGN_PLAN.md`, `NEXT_LOOP_INSTRUCTION.md`,
  `REVIEW_2026-06-21_EVIDENCE_PASS.md`.

## Hard invariants (must never be downgraded)

- **I1** fully decentralized *execution AND learning* (no global-state leakage, no central critic).
- **I3** consensus threshold `τ ≥ 0.9`, identical in training reward and evaluation.
- **I4** generalization: domain randomization + held-out + varying N + multi-seed with CI.
- **I5** one principled reward: potential shaping + constrained-RL dual (no stacked weighted terms).
- **I6** closed-form whole-network consensus failure probability (Poisson-binomial quorum tail), not Monte Carlo.
