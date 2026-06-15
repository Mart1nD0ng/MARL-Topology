# Stage 24 Endpoint Sampler Logprob Repair

## Repaired Semantics

The endpoint-budgeted sampler policy action is the joint endpoint proposal set, not the assembled topology.

For endpoint `i`, incident physical links are sampled without replacement from the actor logits using a tractable sequential categorical distribution. The endpoint action is `S_i`; the joint action is all endpoint proposal sets:

```text
A = {S_i for all endpoints i}
log pi(A) = sum_i log pi_i(S_i)
```

The aggregate physical proposal set is deterministic given `A`. The environment-side physical-link assembler then projects that proposal into the selected topology.

## Recorded Fields

The sampler now records:

- `endpoint_budget_config`
- `endpoint_proposal_sets`
- `endpoint_logprobs`
- `joint_proposal_logprob`
- `proposal_entropy`
- `aggregated_physical_proposals`
- `logprob_semantics = exact_endpoint_proposal_logprob`
- `projected_topology_logprob_exact = false`

Rollout transitions add `projected_selected_physical_edges` after assembler execution.

## Explicit Limitation

The endpoint logprob is exact for endpoint proposal actions. It is not the exact marginal probability of the final projected topology, because multiple endpoint proposal sets may aggregate and project to the same selected physical-link topology.

## Evidence

Tests cover finite endpoint logprob and entropy, reproducibility under seed, local budget compliance, mask respect, deterministic aggregation from endpoint proposals, exact proposal-level semantics, no projected-topology exactness claim, and no objective/reward/oracle leakage.

The Stage 24 micro comparison showed the repaired endpoint sampler passed safety and critic-integration gates, but it remained out of the active registry because the Plackett-Luce sampler had lower projection rejection and lower resource cost in the closeout run.
