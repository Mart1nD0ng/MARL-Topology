# AGENTS.md — working agreement for this repo

This is a **single-trunk** decentralized constrained-RL codebase for V2X topology
control. Read this before changing anything.

## The trunk (the only one)

`scripts/train/train_decentralized_rl.py` — the decentralized cold-start
constrained-RL trunk:

- **critic-free REINFORCE + RLOO** over `MessagePassingGraphEdgeScorer` edge logits;
- reward = Ng–Harada potential `r = (c − τ)` on the **closed-form PBFT quorum-tail
  consensus success probability** (`protocol/quorum_tail.py`) **+ a Lagrangian dual**
  on the radio-budget constraint (constrained-RL, not hand-tuned weights);
- `τ = 0.9`, hard-frozen in both training and evaluation;
- decode via `local_mutual_assemble` (`policies/decentralized_mutual_acceptance.py`),
  feasibility-by-construction, fully decentralized.

The shared, **critic-free** data path the trunk depends on:
`data/row_context_builder.py` (`build_row_contexts`) and `data/graph_payload.py`
(`graph_payload`). These were extracted so the trunk never imports any critic / MAPPO
code.

## What was removed (2026-06-21, owner-authorized)

The old **centralized-critic MAPPO trunk** and the **planner / critic teacher arm**
were fully removed:

- `training/production_mappo_adapter.py`, the entire `training/mappo/` and
  `training/policy_gradient/` packages, `training/critic_*` (dataset, features,
  repair trainer, guided planner);
- `models/centralized_*_critic.py`, `models/enriched_centralized_mlp_critic.py`,
  `models/quorum_tail_pool.py`;
- the planner/critic drivers (`train_recovered_decentralized.py`,
  `run_4090_campaign.py`, `reproduce_recovered_step3.py`, the `stage23/24/25/27/28/33`
  MAPPO drivers) and their tests/contracts.

Do **not** reintroduce a central critic, a CTDE value function, or global-state
decoding into the learning loop — that breaks invariant I1.

## Hard invariants (do not downgrade, proxy, or substitute)

- **I1** decentralized *execution AND learning* — local/neighbor info only; no central
  critic, no CTDE gap, no global-matrix compile.
- **I3** `τ ≥ 0.9`, same constant in reward and eval.
- **I4** generalize: domain randomization + held-out + varying N + multi-seed with CI;
  never report success on the train config / single operating point only.
- **I5** one principled reward: potential shaping + constrained-RL dual; no stacked
  weighted reward terms.
- **I6** closed-form whole-network consensus failure (Poisson-binomial quorum tail);
  not Monte Carlo, empirical frequency, or a local proxy. It is both a constraint and
  the evaluation metric.

## Workflow per change

1. Locate the bottleneck → single hypothesis (one variable) → minimal implementation.
2. Train multi-config + held-out + multi-seed + baselines → evaluate on real data.
3. Record in `docs/URBAN_V2X_RESEARCH_LOG.md` → keep or roll back.
4. If stuck 3–5 rounds, or success requires violating an invariant, stop and report.

## Verification gate

Before claiming a change works, run the smoke and the unit suite:

```bash
python scripts/train/train_decentralized_rl.py --smoke --cold-start   # must exit 0
python scripts/train/evaluate_actor_on_dataset.py --help              # must import
python -m pytest tests/unit -q
```

The smoke must exit 0, print a `[data] pool ...` line, and run a few updates.

## Headline result (the thing to protect)

Cold-start, oracle-free decentralized learning **beats the SA oracle at out-of-range
scale N = 24** (4-seed CI95 on `raw_mean − 0.577` = **[+0.067, +0.279]**, lower bound
> 0). Don't regress this; if a change moves it, say so honestly in the research log.
