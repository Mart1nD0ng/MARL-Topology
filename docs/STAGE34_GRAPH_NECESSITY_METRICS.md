# Stage 34 Graph-Necessity Metrics

## Rule

Graph necessity is metric-derived. A sample is not graph-necessary merely because it belongs to a named family.

Implementation: `src/marl_topology/evaluation/graph_necessity_metrics.py`.

## Metrics

- `local_heuristic_gap`
- `local_quality_ambiguity`
- `bridge_sensitivity`
- `weak_primary_sensitivity`
- `role_sensitivity`
- `graph_structure_rank_gap`
- `mlp_hardness_diagnostic`

If the local heuristic equals the teacher and no other sensitivity/rank/ambiguity signal is present, `graph_necessary` is false.

## Tests

Tests verify:

- graph necessity is false when local heuristic equals teacher;
- bridge, weak-primary, and role sensitivity can trigger graph necessity;
- ambiguity and rank gap are computed from edge rows;
- thresholds are documented in the payload.

## Residual Risk

The thresholds are now explicit and test-covered, but they still need calibration on a materialized full Stage34 dataset.
