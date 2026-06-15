# Stage 23 - Selected-Physical Policy-Gradient Landing

Controlled object: selected-physical policy-gradient pilot harness for
`undirected_physical_link_v1`.

Desired state: actor emits physical-link proposal logits only, stochastic
samplers propose undirected physical links, the environment-side physical-link
assembler executes projection, Stage 3/4 evaluation produces objective metrics,
and the frozen Stage 5 surrogate supplies the training signal.

## Why Stage 15 Is Not Reused

The Stage 15 pilot predates the Stage 22 action-semantics repair. It sampled
directed-edge proposal indicators and projected them through the older directed
assembler path. Stage 23 uses `undirected_physical_link_v1`, so replaying Stage
15 would put the policy-gradient logprob on the wrong action object and would
not test the selected physical-link assembler.

## Policy-Gradient Semantics

Stage 23 policy action:

```text
policy_action = stochastic proposal over undirected physical-link candidates
environment_transition = proposal set -> selected physical-link topology
logprob = proposal set logprob
```

The policy-gradient estimate is approximate under projection. It does not claim
that the proposal logprob is the exact probability of the final projected
topology.

The policy tries to preserve or improve selected physical-link reliability and
resource tradeoff after environment projection. It is not optimizing final tau,
not tuning reward weights, and not running scale-up training.

## Mandatory Diagnostics

Projection diagnostics are mandatory because the actor proposes links but the
assembler can reject proposals for resource, duplicate, invalid-candidate, or
conflict reasons. Stage 23 records pre-projection proposals, proposal logprob,
proposal entropy, post-projection selected physical edges, rejection reasons,
top-proposal rejection rate, above-threshold rejection rate, selected edge
count, reward surrogate, consensus probability, latency, energy, tau-feasible
status, and violation indicator.

## Reward Rule

Stage 23 uses the existing Stage 5.2 surrogate interface with the Stage 5.3
selected normalization references:

```text
config_id = stage5_3_surrogate_config_with_selected_references
tau = 0.9
reliability_weight = 100.0
latency_weight = 1.0
energy_weight = 1.0
```

These values are reused as the existing active pilot surrogate configuration.
Stage 23 does not add, tune, calibrate, or rename reward weights. The reward
surrogate is training-only and is not actor input.

## Sampler Comparison

The trial compares:

- `physical_bernoulli_proposal_sampler`
- `physical_plackett_luce_top_k_sampler`
- `endpoint_budgeted_physical_proposal_sampler`

All samplers use the same selected Stage 22 physical evidence, same in-memory
full-GNN actor warm start, same Stage 3/4 evaluator, same physical-link
assembler, same reward config, and same fixed seed group.

## Low-Entropy Cleanup

Stage 23 promotes only one active policy-gradient sampler:

```text
active_policy_gradient_sampler_id = physical_plackett_luce_top_k_sampler
```

The Bernoulli and endpoint-budgeted samplers remain historical trial
implementations for evidence and tests only. They are excluded from the active
policy-gradient sampler registry, so the active registry has exactly one entry
after closeout.

## Verification

Required commands:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python scripts\train\stage23_selected_physical_policy_gradient_pilot.py
```

Completion gate passed for the bounded pilot only. Scale-up remains blocked
until owner-approved Stage 24 analysis.
