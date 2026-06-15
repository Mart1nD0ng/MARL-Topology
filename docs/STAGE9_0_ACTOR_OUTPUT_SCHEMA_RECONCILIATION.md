# Stage 9 Actor Output Schema Reconciliation

The active neural actor output schema is
`actor_policy_local_edge_score_output_v1`.

`activate` is not an active neural actor output. It remains valid only in the
legacy local edge-decision schema used by older non-learning baselines or in
the topology selected after environment-side projection.

The Stage 9 model package therefore exposes actor logits and `EdgeScoreBatch`
records only. Final topology selection belongs to the assembler, not the
actor.
