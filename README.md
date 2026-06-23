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
- **Reward — one principled objective, not a weighted bag.** A feasibility-margin
  reward `r = (c − τ)` where `c` is the **closed-form PBFT
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

## Headline result (under the corrected environment math)

> ⚠️ The project's environment math was reconstructed (P0–P4: safe PBFT quorum, fixed
> Byzantine fault set, single-correct route/relay, tri-state solvability, timeout-aware
> latency) and the production dataset re-built under it. The **earlier** "beats the SA
> oracle by +0.177 at out-of-range N = 24" claim was produced under the **pre-recalibration
> (incorrect) math** — unsafe quorum + incoherent remove-largest fault model + degenerate
> latency + double-counted relay — and is **retired** (`retired_due_to_protocol_metric_change`).

Under the **corrected** math **and a corrected sampler** (a latent NaN-gumbel bug that had
collapsed the training sampler to a no-exploration fixed order was found and fixed, 2026-06-23 —
the trunk now samples through the verified Plackett-Luce action API), an oracle-free **cold-start
decentralized learner approximately matches the centralized Simulated-Annealing oracle in-range**
(N ∈ {8, 12, 16}): a 5-seed held-out comparison gives RL keep-best **0.610** vs SA ceiling
**0.655**, margin **−0.045, 95% CI [−0.083, −0.006]** (final-update policy margin **−0.045,
[−0.101, +0.011]**). The learner sits **at parity-to-marginally-below** the near-optimal
centralized oracle in-range. (The sampler fix is a genuine train/deploy-alignment correctness fix
but is **neutral on this headline** — the pre-fix 0.624/−0.031 is statistically indistinguishable;
a single-shard pilot gain did not generalize to the full pool. Beating the oracle is the job of the
CTDE method, Graph-MAPPO, now in progress.)

Whether a learned policy **generalizes past the oracle's build scale out-of-range (N = 24)**
under the corrected math — the legacy claim's actual setting — is the **open question**, gated
on a `fixed_set` cost optimization + a heavy N = 24 build. See `docs/URBAN_V2X_RESEARCH_LOG.md`
and `docs/CURRENT_HEAD_STATUS.md`.

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
- **I5** one principled reward: a feasibility-margin reward `(c − τ)` + constrained-RL dual (no stacked weighted terms).
- **I6** closed-form whole-network consensus failure probability (Poisson-binomial quorum tail), not Monte Carlo.
