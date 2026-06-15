# Stage 34 Graph-Necessity Dataset

## Contract

Implementation: `src/marl_topology/data/stage34_graph_necessity_dataset.py`.

Default config:

- required families: 7
- samples_per_family: 50
- total_scenarios: 350
- node_count_choices: 6, 7, 8, 9, 10
- tau_requirement_min: 0.9
- split: family-stratified train/eval/test

Families:

- `bridge_node_family`
- `weak_primary_repair_family`
- `rsu_hub_role_structured_family`
- `redundant_local_quality_family`
- `multi_hop_sparse_backbone_family`
- `near_threshold_structural_family`
- `full_graph_resource_failure_sparse_feasible_family`

## Stage34 Diagnostic Run

The default script produced a planned dataset contract report with:

- scenario_count: 350
- samples_per_family_min: 50
- duplicate_context_rate: 0.0
- split_leakage_overlap_count: 0
- all required families present: true
- materialized: false

## Materialization Status

Full materialization was not run in the default validation path because Stage34 also requires a full seven-ablation official-adapter training protocol. The script blocks rather than silently using a small dataset.

## Residual Risk

The dataset builder can materialize test-sized samples through the Stage3/4 evaluator, but the full 350-scenario dataset remains pending owner-approved compute.
