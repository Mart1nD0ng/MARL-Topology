# Stage 2.6 Replay Dataset Column Contract

This contract defines column boundaries for future replay rows and dataset exports. It does not create a replay buffer, write dataset files, train models, compute rewards, or authorize actor/critic implementation.

## Controlled Object

The controlled object is the column boundary between:

```text
MinimalDecPOMDPEnv -> replay/report row -> actor input projection
                                      -> centralized training view
                                      -> evaluation-only evidence
```

## Desired State

Future replay and evaluation data can contain environment-side evidence without leaking global, oracle, metric, reward, or critic-only fields into deployment actor inputs.

## Column Classes

### Deployment Actor Input

Allowed deployment actor input columns mirror `ActorObservation`:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- `local_neighbor_observations`
- `local_messages`
- `local_history`

An actor-input batch must contain only these columns.

### Local Action Columns

Local action columns may exist in replay rows, but they are not actor observations:

- `action_agent_id`
- `action_neighbor_id`
- `action_activate`

### Centralized Training Only

These columns are allowed only through training-only interfaces:

- `scenario_id`
- `node_ids`
- `candidate_edge_ids`
- `selected_edge_ids`
- `metric_names`
- `oracle_status`
- `global_graph`
- `global_topology`
- `joint_action`

They must not enter deployment actor input batches.

### Evaluation Only

Evaluation/report-only columns include registered metrics and report identifiers:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`
- `metric_name`
- `metric_level`
- `metric_value`
- `used_for`
- `topology_id`
- `evaluation_topology_id`
- `baseline_name`
- `fixture_id`
- `is_oracle`
- `oracle_name`
- `oracle_selected_edge_ids`
- `searched_topology_count`
- `is_exhaustive`

These columns may appear in reports or evaluation rows, but not in deployment actor inputs.

## Unsupported Until Future Contracts

The following columns are unsupported in replay datasets until a future reward/training contract explicitly admits them:

- `reward`
- `return`
- `advantage`
- `value_target`
- `full_committee_success`
- `future_channel_state`
- `future_consensus_outcome`
- `future_mobility`

Stage 5.1 plans future reward columns but does not admit them yet. A later
owner-approved implementation must classify any reward-surrogate fields as
training-only or evaluation-only before they appear in replay data, and must
keep them out of deployment actor input projections.

Rule shorthand: must keep them out of deployment actor input projections.

Stage 5.2 admits the first reward-surrogate diagnostic columns as
training-only columns:

- `reward_surrogate`
- `reward_reliability_penalty`
- `reward_latency_penalty`
- `reward_energy_penalty`
- `reward_config_id`

These fields may appear only in mixed replay/training rows after validation.
They must not appear in deployment actor input batches. Generic `reward`, `return`, `advantage`, and `value_target` remain unsupported until future training contracts admit them.

Stage 5.5 Training Preflight Review confirms that training execution is still
blocked. It does not admit new replay columns. In particular, generic `reward`,
`return`, `advantage`, and `value_target` remain unsupported until a future
owner-approved learning-target and replay-column contract is written.

The Stage 5.5 report may list gate ids, blocked tasks, and recommended next
tasks as review metadata. Those fields are not replay dataset columns and must
not be treated as actor inputs, centralized critic features, or evaluation
metrics.

Stage 5.6 Training Design Contract lists future learning-target fields but
does not admit them as active replay columns. The following remain planned
until a later owner-approved replay-column update:

- `return`
- `advantage`
- `value_target`
- `episode_id`
- `trajectory_id`
- `discount_factor`

Stage 5.6 adds no dataset writer, replay buffer, learner batch, checkpoint
loader, or artifact writer.

Stage 5.7 Policy Architecture Contract adds no replay columns. It requires
future actor-batch construction and actor export to exclude centralized
training views, surrogate diagnostics, oracle labels, registered evaluation
metrics, per-primary reliability diagnostics, and future outcomes.

Future architecture implementation must prove replay actor projection excludes
training-only columns before any policy code is accepted.

Stage 5.8 Learning Target And Replay Contract freezes future target-column
semantics in:

- `docs/STAGE5_8_LEARNING_TARGET_REPLAY_CONTRACT.md`
- `docs/LEARNING_TARGET_REPLAY_CONTRACT.md`
- `src/marl_topology/data/learning_target_contract.py`

Stage 5.8 still does not admit new active replay columns. The following are
defined as `planned_not_active` only:

- `return`
- `advantage`
- `value_target`
- `episode_id`
- `trajectory_id`
- `transition_index`
- `discount_factor`
- `bootstrap_value`

The active replay schema must still reject `return`, `advantage`, and
`value_target`. Stage 5.8 adds no dataset writer, replay buffer, learner batch,
target derivation, checkpoint loader, or artifact writer.

## Required Gates

- All replay columns must be registered by class before use.
- Deployment actor input columns must be a subset of `ActorObservation`.
- Metric columns must map to `docs/METRIC_CONTRACT.md`.
- Oracle labels and global topology may be stored only as centralized/evaluation evidence, never as actor input.
- Local action columns are labels or sampled decisions, not observations.
- Unknown columns fail validation.

## V5 Anti-Inheritance Rules

- Do not import old v5 replay columns, phase CSV fields, `P_eff` variants, reward labels, or actor/critic tensors as defaults.
- Do not treat full topology, oracle labels, future outcomes, or global metrics as deployment actor features.
- Do not add reward, return, advantage, value-target, or checkpoint columns before reward/training contracts exist.

## Verification

Run:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

The Stage 2.6 tests must show:

- actor-safe columns pass;
- centralized, oracle, metric, action, reward, and future columns fail as actor input;
- registered mixed replay rows can be classified;
- actor projection drops non-actor columns;
- unknown columns fail.

## Residual Risks

- This contract validates column names, not the semantic content of nested objects.
- Future dataset writers must call these validators at write time and before actor batch construction.
- Future replay storage format remains undecided.
