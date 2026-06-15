# Stage 23 - Policy-Gradient Pilot Report

Stage verdict: `stage23_pass_pg_pilot_landed_sampler_promoted`.

Selected sampler: `physical_plackett_luce_top_k_sampler`.

## Preflight

| Gate | Result |
| --- | --- |
| Active action semantics is `undirected_physical_link_v1` | pass |
| Discarded action semantics not active | pass |
| Full GNN v2 actor active | pass |
| Actor-safe input boundary | pass |
| Physical-link assembler active | pass |
| Stage 3/4 objective evaluator active | pass |
| Reward config read-only | pass |
| Run manifest validator available | pass |

## Before And After For Winner

| Metric | Before PG | After PG |
| --- | ---: | ---: |
| tau_feasible_rate | 0.3333 | 0.3333 |
| violation_rate | 0.6667 | 0.6667 |
| mean consensus_success_probability | 0.3333 | 0.3333 |
| mean latency | 0.0036002032 | 0.0036002032 |
| mean energy | 0.0048000000 | 0.0048000000 |
| mean selected_edge_count | 2.0000 | 2.0000 |
| top_proposal_rejection_rate | 0.1111 | 0.1111 |
| above_threshold_rejection_rate | 0.0000 | 0.0000 |
| mean entropy | 2.7319 | 2.7286 |
| mean reward_surrogate | -56.5134 | -56.5134 |
| empty_graph_rate | 0.0000 | 0.0000 |
| full_graph_rate | 0.0000 | 0.0000 |

The micro update preserved tau-feasible rate and did not improve latency or
energy. This is acceptable for Stage 23 because the landing gate requires a
controlled pilot and safe sampler promotion, not scale-up performance.

## Baselines

| Baseline | tau_feasible_rate | Mean latency | Mean energy | Mean selected edges |
| --- | ---: | ---: | ---: | ---: |
| Objective-aware teacher | 0.6667 | 0.0024001177 | 0.0051200000 | 2.6667 |
| Projected greedy baseline | 0.6667 | 0.0030002127 | 0.0060800000 | 2.3333 |
| Projected full graph baseline | 0.6667 | 0.0030002127 | 0.0054400000 | 3.0000 |

These baselines remain diagnostic. Full graph is not an oracle and fixed top-k
is not promoted as final policy.

## Boundary Results

- Reward config unchanged before and after pilot.
- No reward weight tuning.
- No final tau selection.
- No COMA.
- No Transformer.
- No new GNN/GRU/LSTM architecture.
- No scale-up training.
- No checkpoint.
- No uncontrolled artifact.
- No `v5` modification.

## Stage 24 Recommendation

Recommended next task:
`stage_24_policy_gradient_pilot_analysis_and_scale_readiness_review`.

Owner approval is required before Stage 24. Stage 23 does not authorize
scale-up training.
