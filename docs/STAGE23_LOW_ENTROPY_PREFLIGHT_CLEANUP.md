# Stage 23 - Low-Entropy Preflight Cleanup

Controlled object: executable action-semantics and model paths after Stage 22.

Desired state: only `undirected_physical_link_v1` remains callable from active
source, scripts, and tests. The discarded Stage 22 A/B branch and old GNN
identifier are removed from executable code rather than marked archived.

## Cleanup

Removed executable Stage 22 A/B entry points:

- `src/marl_topology/evaluation/stage22_action_semantics_ab_evaluation.py`
- `src/marl_topology/training/stage22_action_semantics_supervised.py`
- `scripts/train/stage22_action_semantics_ab_full_gnn_report.py`

Cleaned active registries and tests:

- `src/marl_topology/policies/action_semantics.py` now exposes only
  `undirected_physical_link_v1`.
- Stage 22 evidence now builds selected physical-link evidence only.
- Stage 22 action-semantics tests now cover physical-link aggregation and
  selected physical evidence only.
- The GNN boundary report now identifies the active full message-passing GNN
  without carrying the old model identifier.

## Protected Behavior

Historical Stage 22 reports still record that A/B comparison happened. They are
documentation, not executable action-semantics code. Stage 8 directed edge-score
records remain because local endpoint proposals still use directed ids before
physical-link aggregation.

## Preflight Verdict

Cleanup passed. The active executable path is selected physical-link topology
control under `undirected_physical_link_v1`.

policy-gradient was not run during this preflight.
