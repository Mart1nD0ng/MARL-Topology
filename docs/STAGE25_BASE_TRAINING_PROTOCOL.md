# Stage 25 Base Training Protocol

Stage 25 uses one fixed protocol, `stage25_pilot_base_config`, for the
small-scale formal MAPPO pilot. The protocol is not a tuning surface. Stage 24
smoke and micro-loop metrics remain implementation evidence only and are not
formal Stage 25 comparison metrics.

## Cybernetic Model

- Controlled object: selected-physical MAPPO training loop using
  `local_message_passing_gnn_edge_scorer_v2`, the centralized training-only
  critic, `physical_plackett_luce_top_k_sampler`, the selected physical-link
  assembler, and the frozen Stage 5 reward surrogate.
- Desired state: run a bounded train/eval pilot that compares MAPPO fine-tuning
  against the supervised GNN baseline without changing sampler, reward weights,
  action semantics, model architecture family, tau, or artifact policy.
- State variables: train/eval scenario slots, seed id, rollout step, update id,
  reward, tau-feasible rate, violation rate, consensus success probability,
  latency, energy, selected edge count, projection rejection, entropy, KL,
  clip fraction, policy loss, value loss, explained variance, gradient norm,
  actor score moments, empty graph rate, and full graph rate.
- Sensors: `run_stage25_preflight`, the run manifest validator, per-update and
  per-eval metrics, reward-fingerprint comparison, source-scan contracts,
  visualization output checks, pytest, and harness task validation.
- Actuators: bounded MAPPO actor/critic gradient updates, manifest-approved
  report writes, visualization report generation, and source hygiene edits for
  torch import discipline.
- Disturbances: small scenario count, stochastic sampler variance, projection
  rejection, critic baseline variance, reward/objective mismatch, and possible
  overfit between train and eval slots.
- Coupling: actor logits affect sampler proposals; sampler proposals are
  projected by the assembler; projected topology drives Stage 3/4 objective
  metrics; objective metrics drive the frozen Stage 5 surrogate; GAE/value loss
  updates couple actor and critic during training.
- Feedback loop: evaluate baseline on eval slots, train with fixed protocol,
  evaluate at fixed intervals, stop on safety regressions, compare final MAPPO
  to supervised GNN on eval, then use visualization and reward-surface sensors
  to decide whether Stage 26 review is safe.

## Fixed Config

```yaml
config_id: stage25_pilot_base_config
train_scenarios: 16
eval_scenarios: 8
seeds: [2501, 2502, 2503]
rollout_steps: 16
transitions_per_update: 256
minibatch_size: 64
update_epochs: 4
max_updates: 20
eval_every: 5
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
actor_lr: 0.0001
critic_lr: 0.0001
entropy_coef: 0.01
value_coef: 0.5
max_grad_norm: 0.5
tau_requirement_min: 0.9
```

The current Stage 21/22 evidence provides 10 unique source contexts. Stage 25
therefore uses disjoint source rows expanded cyclically into 16 train scenario
slots and 8 eval scenario slots. This preserves the fixed slot protocol while
making the limited source-count disturbance explicit.

## Contract Checks

- `transitions_per_update = train_scenarios * rollout_steps = 256`.
- `minibatch_size <= transitions_per_update`.
- At least two minibatches per update.
- `update_epochs >= 2`.
- `eval_every <= max_updates`.
- Seed count is at least two.
- Reward weights and `tau_requirement_min` stay unchanged.
- Active sampler remains `physical_plackett_luce_top_k_sampler`.

## Boundary Rules

No COMA, Transformer, GRU/LSTM, recurrent PPO, reward weight tuning, sampler
switching, final tau selection, checkpoint creation, v5 migration, scale-up
training, or unmanifested artifact writes are part of Stage 25.

## Verification

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
- `python scripts\train\stage25_small_scale_mappo_training_pilot.py`
- `python scripts\replay\stage25_training_visualization_report.py`

Residual risk: the source context set is still small, and one seed can stop on
a safety condition. Stage 26 must review scale readiness and failure modes
before any broader training is authorized.
