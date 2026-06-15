# Stage 24 Critic-Integrated MAPPO Micro-Loop

## Control Model

Controlled object: selected-physical policy-gradient training surface for `undirected_physical_link_v1`.

Desired state: actor-safe decentralized logits drive stochastic physical-link proposals; a centralized training-only critic supplies value predictions for GAE; PPO/MAPPO-style clipped policy loss and critic value loss both update; the Stage 5 surrogate configuration remains frozen; exactly one active sampler remains at closeout.

Sensors: Stage 24 preflight, endpoint logprob tests, rollout/GAE/loss/critic-integration tests, sampler cleanup contract, smoke script, micro script, full pytest, and harness validation.

Actuators: repaired endpoint sampler metadata/logprob semantics, added rollout batch, GAE, clipped actor/value loss, trainer, script, docs, tests, harness task, and project state.

Disturbances: projection changes the executed topology after the sampled policy action, short horizons make value estimates noisy, and stochastic sampler evidence can vary by seed.

## Why Stage 23 Was Not Enough

Stage 23 landed a selected-physical policy-gradient pilot and selected `physical_plackett_luce_top_k_sampler`, but its baseline was REINFORCE-style with a batch-mean baseline. That was enough to test sampler feasibility and projection diagnostics, but not enough to claim a complete CTDE/MAPPO-style loop because the centralized critic was not used to compute returns or advantages and critic value loss was not optimized in the policy loop.

## Centralized Critic Boundary

Stage 24 uses `centralized_mlp_critic_baseline_v1` as a training-only value baseline. The actor still receives only the Stage 22 actor-safe local rows and outputs edge logits. The critic may receive centralized training-only fields such as candidate edges, selected topology state, node ids, and evaluator-side diagnostics. These critic inputs are recorded in rollout transitions but are not actor inputs.

## Endpoint Logprob Repair

The endpoint sampler action is the joint endpoint proposal set. For each endpoint, it samples local incident links without replacement, then sums endpoint-local log probabilities:

```text
log pi(A) = sum_i log pi_i(S_i)
```

The aggregate physical proposal set is a deterministic environment-side aggregation of endpoint proposals. The final selected topology is still produced by the physical-link assembler. Stage 24 therefore records exact endpoint proposal logprob, and explicitly does not claim exact projected-topology logprob.

## Micro-Loop, Not Scale-Up

Stage 24 has two modes:

- `smoke`: 4 scenarios x 4 rollout steps = 16 transitions, minibatch size 8, 2 update epochs, 1 policy update.
- `micro`: 8 scenarios x 8 rollout steps = 64 transitions, minibatch size 16, 3 update epochs, 5 policy updates.

Smoke mode checks loop completeness. Micro mode is the closeout decision mode. Neither mode writes checkpoints, exports datasets, performs hyperparameter search, selects final tau, or authorizes scale-up.

## Frozen Reward Rule

Stage 24 uses the existing Stage 5.2/5.3 surrogate configuration via the Stage 23 adapter. The config fingerprint is checked before and after the run. No reward weights, normalization references, tau value, timeout reward, quorum reward, density reward, or reliability bonus above tau are changed.

## Sampler Comparison

The compared samplers are:

- `physical_plackett_luce_top_k_sampler`
- `endpoint_budgeted_physical_proposal_sampler`

Both use the same actor initialization, critic initialization policy, scenario set, seed policy, assembler, evaluator, rollout sizes, minibatches, update epochs, and reward-surrogate config. Bernoulli remains historical Stage 23 evidence only and is not an active Stage 24 candidate.

Winner selection prioritizes boundary safety, no collapse, no before/after degradation, lower projection mismatch, reliability-resource tradeoff, stable KL/entropy, clean logprob semantics, and implementation simplicity when evidence is close.

## Pass / Fail

PASS requires a complete critic-integrated micro-loop, GAE or equivalent advantages, clipped actor loss, critic value loss, actor and critic parameter updates, unchanged reward config, no actor leakage, no COMA/Transformer, no scale-up, no checkpoint, and exactly one active sampler.

FAIL produces `docs/STAGE24_FAILURE_REVIEW.md` and leaves policy-gradient blocked pending owner decision.
