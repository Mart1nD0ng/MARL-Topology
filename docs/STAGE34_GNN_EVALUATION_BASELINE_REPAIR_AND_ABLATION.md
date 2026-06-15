# Stage 34 - GNN Evaluation Baseline Repair and Ablation Diagnostics

## Controlled Object

The controlled object is the GNN evaluation baseline: Stage34 graph-necessity dataset contract, strict graph-necessity metrics, official mappo adapter path, seven GNN ablation registry, logit/KL/gradient diagnostic schema, report artifacts, active model registry, and project state.

## Desired State

Stage34 should not reuse the tiny Stage33 baseline for architecture judgment. It should require a credible 350+ scenario dataset, metric-derived graph-necessity labels, a full official-adapter diagnostic protocol, and exactly seven GNN ablations. One production GNN may be selected only after the full protocol passes.

## Baseline

Stage33 failed but was not conclusive: only seven scenarios were used, graph-necessity metrics were weak, training had one update, and the active v3 collapse could have reflected the evaluation baseline rather than true architecture failure.

## Implemented Actuators

- `src/marl_topology/data/stage34_graph_necessity_dataset.py`
- `src/marl_topology/evaluation/graph_necessity_metrics.py`
- `src/marl_topology/evaluation/stage34_gnn_diagnostics.py`
- `src/marl_topology/models/gnn_ablation_registry.py`
- `scripts/train/stage34_gnn_ablation_diagnostic_training.py`
- `scripts/replay/stage34_gnn_failure_attribution_report.py`
- `harness/tasks/stage34_gnn_baseline_repair_and_ablation_diagnostics.yaml`

## Stage34 Result

Verdict: `stage34_gnn_ablation_blocked_awaiting_owner_decision`.

Gate issues:

- `baseline_not_repaired`
- `full_protocol_not_completed`
- `no_selected_gnn`
- `selected_gnn_collapse_rate_above_20_percent`

The code now defines the repaired baseline and ablations, but the full materialized dataset plus seven-ablation official-adapter training run was not executed. The required transition estimate is 3,136,000 rollout transitions before eval/test overhead, so architecture selection is blocked pending owner compute/protocol decision.

## Completion Gate

Completion Gate passed? No.

Next-stage readiness gate passed? No.

No production GNN was selected. MLP was not promoted. Stage35 is not allowed.

## Verification

- Stage34 focused tests: `19 passed`.
- `python scripts\train\stage34_gnn_ablation_diagnostic_training.py`: completed with blocked verdict.
- `python scripts\replay\stage34_gnn_failure_attribution_report.py`: completed with owner-decision failure attribution.
- Full pytest and harness validation are recorded in the Stage34 self-review after final validation.

## Residual Risk

The Stage34 baseline is repaired as a contract and diagnostic scaffold, but not yet repaired as a completed empirical evaluation. Owner approval is required to materialize the full dataset and run the full official-adapter protocol.
