# Stage 23 - Policy-Gradient Sampler Comparison

All three stochastic proposal samplers ran under `undirected_physical_link_v1`.
The active actor was `local_message_passing_gnn_edge_scorer_v2`; the environment
projection was `stage22_physical_link_conflict_aware_greedy_max_endpoint`.

| Sampler | Pass gate | Tau-feasible rate after PG | Top proposal rejection | Empty graph rate | Mean reward surrogate | Closeout state |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `physical_bernoulli_proposal_sampler` | fail | 0.0000 | 0.0667 | 0.5000 | -81.8383 | archived, inactive |
| `physical_plackett_luce_top_k_sampler` | pass | 0.3333 | 0.1111 | 0.0000 | -56.5134 | promoted active |
| `endpoint_budgeted_physical_proposal_sampler` | pass | 0.5000 | 0.1250 | 0.0000 | -43.0900 | archived, inactive |

Selection rule order: no boundary violation, no collapse, no tau-feasible
degradation relative to the sampler's supervised-before-PG evaluation, lower
projection mismatch, then reward/resource tradeoff and implementation
simplicity.

`physical_plackett_luce_top_k_sampler` was selected because it passed safety
gates, avoided empty/full collapse, had the lowest projection mismatch among
safety-passing samplers, and has clean sequential without-replacement logprob
semantics aligned with budgeted topology selection.

The endpoint-budgeted sampler had better mean reward in this micro sample but
slightly higher projection mismatch and approximate aggregate physical-proposal
logprob semantics. It remains a documented trial result, not an active policy
path.

The Bernoulli sampler was removed from the active registry because it produced
a 0.5 empty-graph rate in the fixed seed trial, matching its known too-few or
too-many edge risk.

Active registry after closeout:

```text
physical_plackett_luce_top_k_sampler
```

Historical trial implementations are excluded from active sampler lookup.
