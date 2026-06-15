# Learning Target Replay Contract

## Current Status

The active learning-target replay contract is design-only.

Authoritative Stage 5.8 document:

- `docs/STAGE5_8_LEARNING_TARGET_REPLAY_CONTRACT.md`

Structured manifest:

- `src/marl_topology/data/learning_target_contract.py`

## Execution Boundary

Stage 5.8 does not implement a dataset writer, replay buffer, learner batch,
target derivation, checkpoint, training loop, model code, reward-weight
calibration, final `tau_consensus` selection, or v5 migration.

## Planned Columns

Future learning-target columns are defined but not active:

- `return`
- `advantage`
- `value_target`
- `episode_id`
- `trajectory_id`
- `transition_index`
- `discount_factor`
- `bootstrap_value`

Future training diagnostics are also planned, not active:

- `policy_log_probability`
- `policy_entropy`
- `action_sparsity`
- `constraint_violation`

## Actor Boundary

None of the planned columns may enter deployment actor inputs. Actor projection
must remain a subset of `ActorObservation`.

## Active Schema Boundary

The active replay schema still rejects `return`, `advantage`, and
`value_target`. Stage 5.8 is a contract freeze, not an implementation.

## Future Required Work

Before planned columns become active, a later owner-approved implementation
must define:

- target derivation equations;
- episode boundary semantics;
- discount and bootstrap policy;
- writer and reader validation;
- actor projection negative tests;
- artifact and run-manifest policy.
