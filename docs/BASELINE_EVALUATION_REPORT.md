# Stage 2.4 Baseline Evaluation Report

This document records the Stage 2.4 baseline evaluation report contract and the current demo-scenario sensor. It is not a training result and it does not authorize reward, actor, critic, COMA, GNN, LSTM, or v5 migration work.

## Controlled Object

The controlled object is the minimal Stage 2 evaluation feedback loop:

```text
Scene3D -> CandidateGraph -> LinkModel -> TopologyEvaluator
-> MinimalDecPOMDPEnv -> non-learning baselines -> report
```

## Desired State

The project can compare simple non-learning topology choices through registered metrics before adding learning. The report must keep global baselines, decentralized baselines, and oracle references semantically separate.

## Report Generator

Run:

```powershell
python scripts\replay\baseline_evaluation_report.py
```

The script prints JSON and writes no result files.

Stage 2.5 scenario-fixture report:

```powershell
python scripts\replay\scenario_fixture_report.py
```

This script prints a nested report across all active deterministic fixtures and writes no result files.

## Registered Metric Names

The report may output only these registered metric names:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Every metric row must include `scenario_id`, `topology_id`, `metric_name`, `metric_level`, `metric_value`, and `used_for`.

## Baseline Families

Global non-learning baselines:

- `empty`
- `full`
- `greedy_reliability`
- `random_seed_7`

Decentralized non-learning baselines:

- `decentralized_no_edges`
- `decentralized_all_local_edges`
- `decentralized_top1_reliability`
- `decentralized_threshold_p65`
- `decentralized_random_seed_13_p40`

The decentralized rows consume `ActorObservation`, emit `EdgeActionDecision`, and are evaluated through `MinimalDecPOMDPEnv.step`.

## Oracle Boundary

The exhaustive oracle remains a reference sensor for the small demo graph. It is not a deployment actor input.

Important boundary: full graph is a baseline, not an oracle.

## Current Demo Signal

For the current `demo_stage2` scenario:

- `empty` and `decentralized_no_edges` fail consensus because quorum is unreachable.
- `full` and `decentralized_all_local_edges` succeed but carry the highest edge count and energy among the deterministic baselines.
- `greedy_reliability` and `decentralized_top1_reliability` succeed with fewer selected edges than full graph.
- The exhaustive oracle is feasible and remains separate from all baseline rows.

These observations are a smoke-test signal only. They do not prove general policy quality.

## Negative Gates

- No unregistered metric may enter the report.
- Diagnostic fields may not be promoted to objectives.
- The full graph row must have `is_oracle: false`.
- Oracle output must not enter `ActorObservation`.
- The report must remain no training and no v5 code migration.
- The report must not copy old reward, old phase scripts, or old metric naming.

## Residual Risks

- The Stage 2.4 report covers one deterministic demo scenario.
- The Stage 2.5 fixture report covers a small deterministic fixture suite.
- Random baseline evidence is seeded but still only a single smoke-test sample.
- More scenarios should be added only through `docs/SCENARIO_FIXTURE_CONTRACT.md`.
