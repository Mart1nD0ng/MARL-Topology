# Stage 34 Training Protocol Baseline

## Required Protocol

Implementation: `Stage34TrainingProtocolConfig` in `src/marl_topology/evaluation/stage34_gnn_diagnostics.py`.

Defaults:

- train_scenarios: 280
- eval_scenarios: 70
- test_scenarios: 70
- seeds: 5
- rollout_steps: 16
- transitions_per_update: 4480
- minibatch_size: 64
- update_epochs: 4
- max_updates: 20
- eval_every: 5
- smoke_mode: false
- tau_requirement_min: 0.9

The seven-ablation rollout transition estimate is 3,136,000 before eval/test overhead.

## Stage34 Decision

The default diagnostic script does not run this full protocol. It writes a blocker report because silently downgrading to smoke mode would violate the owner decision.

## Acceptance

Protocol contract implemented. Full empirical protocol not completed.
