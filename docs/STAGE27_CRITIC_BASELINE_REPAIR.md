# Stage 27 Critic Baseline Repair

Stage 27 is critic-only. It collected frozen-policy critic data, trained critic candidates, and selected one future value baseline without actor updates or policy-gradient updates.

## Result

- verdict: `stage27_pass_critic_repaired`
- pass gate: `True`
- selected critic: `centralized_message_passing_graph_value_critic_v1`
- selection reason: graph critic had materially higher eval explained variance
- actor update performed: `False`
- reward weights changed: `False`
- sampler switched: `False`
- checkpoint created: `False`

## Candidate Table

| Candidate | Gate | Eval EV | Eval Corr | Bias Reduction | Advantage Var Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| `enriched_centralized_mlp_value_critic_v1` | `True` | 0.2445 | 0.4945 | 0.9531 | 0.2445 |
| `centralized_message_passing_graph_value_critic_v1` | `True` | 0.5972 | 0.7752 | 0.9819 | 0.5972 |

## Acceptance

The selected critic passed the minimum gate and preferred explained-variance/correlation gate. Stage 28 is recommended only with owner approval.
