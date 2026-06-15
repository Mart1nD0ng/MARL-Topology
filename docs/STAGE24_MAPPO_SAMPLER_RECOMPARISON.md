# Stage 24 MAPPO Sampler Re-Comparison

## Protocol

Both candidate samplers ran under the same Stage 24 micro-loop:

- action semantics: `undirected_physical_link_v1`
- actor: `local_message_passing_gnn_edge_scorer_v2`
- critic: `centralized_mlp_critic_baseline_v1`
- assembler: `stage22_physical_link_conflict_aware_greedy_max_endpoint`
- evaluator: Stage 3 URLLC finite-blocklength + Stage 4 expected-initiator PBFT + Stage 5 objective/surrogate
- rollout: 8 scenarios x 8 steps = 64 transitions
- minibatch size: 16
- update epochs: 3
- policy updates: 5
- reward config: unchanged Stage 5.3 surrogate references and weights

## Micro Results

| Sampler | Before tau feasible | After tau feasible | After violation | Mean CSP | Mean latency | Mean energy | Mean selected edges | Top rejection | Entropy | Max KL | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_budgeted_physical_proposal_sampler` | 0.46875 | 0.46875 | 0.53125 | 0.50625 | 0.0076532783 | 0.0139788083 | 3.171875 | 0.0854166667 | 4.7111780047 | 0.0000609718 | yes |
| `physical_plackett_luce_top_k_sampler` | 0.234375 | 0.234375 | 0.765625 | 0.346875 | 0.0056450862 | 0.0070864889 | 2.578125 | 0.0572916667 | 4.8294744417 | 0.0000463538 | yes |

Both samplers preserved before/after tau feasibility, avoided collapse, kept finite logprob/entropy/loss values, kept approximate KL below the Stage 24 threshold, and updated both actor and critic parameters.

## Winner

Selected active sampler:

```text
physical_plackett_luce_top_k_sampler
```

Reason: the endpoint sampler repair passed and had higher sampled tau-feasible rate in this micro seed, but it also selected more edges, had higher latency and energy, and had higher projection rejection. Stage 24 selection put projection mismatch and resource cost ahead of replacing the already-active sampler when both candidates passed safety gates. The simpler Plackett-Luce sampler therefore remains the only active policy-gradient sampler.

## Cleanup State

Active registry:

```text
active_policy_gradient_sampler_id = physical_plackett_luce_top_k_sampler
active_sampler_count = 1
```

Archived comparison-only samplers:

- `endpoint_budgeted_physical_proposal_sampler`
- `physical_bernoulli_proposal_sampler`

The endpoint sampler implementation remains available only through the Stage 24 candidate comparison registry and is excluded from the active policy-gradient sampler registry.
