# Stage 33 Graph-Structure-Necessity Data

## Controlled Object

The controlled object is the procedural training/evaluation data surface used to test whether graph message passing is necessary.

## Dataset

Implementation: `src/marl_topology/data/stage33_graph_structure_dataset.py`.

Dataset id: `stage33_graph_structure_necessity_dataset_v1`.

Families:

- `bridge_node`
- `weak_primary_repair`
- `rsu_hub_role_structured`
- `redundant_local_quality`
- `multi_hop_sparse_backbone`
- `near_threshold_structural`
- `full_graph_resource_failure_sparse_feasible`

Stage33 uses the Stage31 procedural generator and Stage3/4-backed evaluator. Family labels describe the intended structural stressor. Feasibility, local-edge heuristic gaps, full-graph gaps, teacher feasibility, and graph-necessity scores are measured.

## Run Evidence

The Stage33 bounded run used 7 scenarios, covering all seven families.

Quality report:

- duplicate_context_rate: `0.0`
- split_leakage_overlap_count: `0`
- node_counts_present: `6, 8, 9, 10`
- stage3_stage4_evaluator_used: `true`
- simple_link_model_fallback_used: `false`
- hardcoded_fake_feasibility: `false`
- mean_graph_necessity_score: `8.3`
- max_graph_necessity_score: `16.25`

## Acceptance

Graph-structure data requirement passed for Stage33 instrumentation. It did not rescue GNN stability.

## Residual Risk

The bounded seven-scenario run is a sensor for integration and collapse. A larger graph-structure dataset may be needed after the optimization/architecture blocker is understood, but Stage33 does not authorize uncontrolled scaling.
