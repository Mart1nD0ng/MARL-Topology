# MARL-Topology — Decentralized constrained-RL for V2X topology control

A single-trunk research codebase for multi-agent topology planning in urban V2X
networks. A graph neural-network actor learns, per node, which physical links to
activate so that the network reaches **PBFT consensus** reliably under a radio-budget
constraint. The architecture is **CTDE** — *Centralized Training with fully
Decentralized Execution*: training may use a centralized critic + global PBFT
evaluator, but the **deployed actor uses only local observations + physical-neighbour
messages + public protocol parameters + a torch-free local decoder** — no critic, no
global state, no global decoder, no solver/evaluator at inference.

> **Authority of record:** `AGENTS.md` (the working agreement + hard invariants),
> `docs/CURRENT_HEAD_STATUS.md` (where HEAD actually is), and the technical spec +
> engineering plan it cites. Where this README conflicts with those, **they win.**
> The earlier "critic-free is the project identity" framing is **retired** (2026-06-23):
> the binding invariant is *deployment-decentralization*, not critic-freeness — a
> central critic in training is legal; a central deployment decoder is not.

## The one trunk

The project has exactly **one** trunk, driven by

```
scripts/train/train_decentralized_rl.py     # the trunk (static T=1 + dynamic T>1 arms)
scripts/train/evaluate_actor_on_dataset.py  # eval-only tool (cross-N / OOD / baselines)
scripts/train/build_operating_point_dataset.py  # dataset shard builder
```

The old separate **centralized-critic MAPPO trunk** and the **planner / critic teacher
arm** were removed on 2026-06-21. The trunk's **recommended production arm** is
`--baseline graph-mappo --actor mlp` (a shared-advantage Graph-MAPPO spine + MLP actor +
the torch-free local mutual-acceptance decoder). The original **critic-free REINFORCE +
RLOO** path is retained as a registered baseline (`--baseline ema`, the byte-identical
historical default).

## What the trunk does

- **Learning — CTDE Graph-MAPPO (critic-free REINFORCE/RLOO kept as a baseline).** A
  per-edge / per-agent policy over the `MessagePassingGraphEdgeScorer` logits
  (parameter-shared, scale-invariant, K-hop local message passing). Training may use a
  **centralized graph critic** + global PBFT evaluator (CTDE); the legacy critic-free
  REINFORCE with a leave-one-out (RLOO) baseline remains available. Opt-in CTDE
  mechanisms (all verified, default-off, byte-identical when off): `--counterfactual`
  (COMA per-agent credit), `--scq` (closed-form counterfactual supervision), `--chance`
  / CVaR / `--pareto-archive` (reliability constraints), `--actor pna` (directional PNA).
- **Reward — one principled objective, not a weighted bag.** A feasibility-margin reward
  `r = (c − τ)` where `c` is the **closed-form PBFT quorum-tail consensus success
  probability** (`protocol/quorum_tail.py`, a Poisson-binomial whole-network quorum tail
  — no Monte Carlo). The budget constraint is a **Lagrangian dual** (constrained-RL /
  RCPO), not hand-tuned weights. `τ = 0.9`, hard-frozen across training and evaluation.
- **Decode — feasibility by construction, fully local.** `local_mutual_assemble`
  (`policies/decentralized_mutual_acceptance.py`): each node ranks only its own incident
  edges within its radio budget; an edge activates iff both endpoints accept. Per-node
  computable, zero global state, torch-free. `global_argsort_assemble` is kept only as
  the centralized-decode ablation (training-only / never deployed).
- **Actor — `MessagePassingGraphEdgeScorer`** (`models/message_passing_graph_edge_scorer.py`):
  a K-round bidirectional message-passing GNN edge scorer (decentralized-with-communication).
- **Dynamic (T>1) arm — `--dynamic`.** A two-timescale episode rollout (moving-vehicle
  trajectories, per-frame channel, BCSP per-agent action, recurrent or memoryless actor,
  per-frame centralized critic, discounted return). Real 4-RSU urban-grid data via
  `--dyn-data urban` (buildings + road-constrained motion); single-RSU random geometry is
  the `--dyn-data random` ablation. Default-off; the T=1 path is byte-identical.

Warm-start (BC from a frozen artifact / decoder-aware BCSP teacher) is supported;
`--cold-start` runs from random init. `keep-best` on a held-out-from-train **VAL** split
guarantees RL is never reported below the warm-start.

## Headline result — what the campaigns actually showed

> ⚠️ Several **earlier** headline numbers are **retired/superseded** (kept for history in
> the docs): the "beats the SA oracle by +0.177 at N = 24" claim was produced under the
> pre-recalibration (incorrect) environment math; the pre-CTDE recalibration "RL ≈ SA
> oracle in-range (0.610 vs 0.655)" predates Graph-MAPPO + Phase 8–13. See
> `docs/CURRENT_HEAD_STATUS.md` and `docs/URBAN_V2X_RESEARCH_LOG.md` for the current ledger.

**The central, adversarially-verified result of both completed campaigns is the same
honest finding:** at the realistic scale (N ≤ 16), every sophisticated CTDE mechanism
(COMA, SCQ, chance/CVaR/Pareto, PNA, cross-frame recurrence) is **verified-correct but
gives no headline gain** over the simple baseline, and a non-learned per-frame heuristic
is hard to beat — so the mechanisms ship **opt-in** and the simple baseline is the
default. The binding limit is **RL feasibility-region learning** (and large-N
generalization), not credit assignment, reliability shaping, actor architecture, data
realism, or temporal modeling.

- **v2 static campaign (R0–R7 + Phase 8–13, complete):** the simple Graph-MAPPO + MLP
  actor + closed-form decoder is best in-range AND best-generalizing at N ≤ 16.
- **Dynamic-repair campaign (D0–D14, complete):** the dynamic (T>1) task is built,
  corrected (10 gaps closed), and tested on **real 4-RSU urban** + single-RSU random data
  (5 seeds, N{8,12,16}); all mechanism A/B CIs span 0; learned arms sit below the
  zero-eval-call deployable heuristics on both data. Ledger: the `Dynamic-Repair` entries
  in `docs/URBAN_V2X_RESEARCH_LOG.md` + `docs/dynamic_repair/`.

Open frontier (deferred): large-N (N ≥ 24) generalization, which needs a cheaper
exact-fault evaluator.

## Quick start

```bash
# fast end-to-end smoke (random init, ~10s)
python scripts/train/train_decentralized_rl.py --smoke --cold-start

# operating-point diagnostic (recommended production arm)
python scripts/train/train_decentralized_rl.py --baseline graph-mappo --actor mlp

# dynamic (T>1) arm on real 4-RSU urban data
python scripts/train/train_decentralized_rl.py --dynamic --dyn-data urban --cold-start

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
  - `training/` — Graph-MAPPO + critic, counterfactual/SCQ/chance/Pareto mechanisms,
    the dynamic (T>1) episode RL, and BC distillation + decentralized eval helpers.
  - `data/row_context_builder.py` — per-split (row, context) builder.
  - `data/graph_payload.py` — node/edge graph featurization.
- `scripts/train/` — the trunk + dataset/eval tooling (see above).
- `tests/` — `unit/` and `contract/` suites.
- `docs/` — `AGENTS.md`, `CURRENT_HEAD_STATUS.md`, `URBAN_V2X_RESEARCH_LOG.md`,
  `CURRENT_DYNAMIC_REPAIR_STATUS.md`, the technical spec + engineering plans.

## Hard invariants (must never be downgraded — see `AGENTS.md`)

- **D1 — deployment decentralization.** The deployed actor uses local + physical-neighbour
  info only; no central critic, global state, global decoder/argsort, or
  solver/searcher/evaluator at inference. Training MAY use a centralized critic + global
  evaluator (marked training-only). Rollout == deployment local mutual-acceptance.
- **τ ≥ 0.9**, identical in training reward and evaluation.
- **Reliability** = the closed-form whole-network PBFT quorum tail (Poisson-binomial),
  never Monte Carlo or a local proxy.
- **One principled reward** — a feasibility-margin reward `(c − τ)` + constrained-RL dual
  (no stacked weighted terms).
- **Generalization** — domain randomization + held-out + varying N + multi-seed with CI.
