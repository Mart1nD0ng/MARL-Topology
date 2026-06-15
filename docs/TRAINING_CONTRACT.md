# Training Contract

## Current Status

The active training contract is design-only.

Authoritative Stage 5.6 document:

- `docs/STAGE5_6_TRAINING_DESIGN_CONTRACT.md`

Authoritative Stage 5.7 policy architecture documents:

- `docs/STAGE5_7_POLICY_ARCHITECTURE_CONTRACT.md`
- `docs/POLICY_ARCHITECTURE_CONTRACT.md`

Authoritative Stage 5.8 learning-target replay documents:

- `docs/STAGE5_8_LEARNING_TARGET_REPLAY_CONTRACT.md`
- `docs/LEARNING_TARGET_REPLAY_CONTRACT.md`

Authoritative Stage 5.9 run-manifest artifact documents:

- `docs/STAGE5_9_TRAINING_RUN_MANIFEST_ARTIFACT_CONTRACT.md`
- `docs/RUN_MANIFEST_ARTIFACT_CONTRACT.md`

Authoritative Stage 5.10 run-manifest validator document:

- `docs/STAGE5_10_RUN_MANIFEST_VALIDATOR.md`

Authoritative Stage 6.0 minimal stack document:

- `docs/STAGE6_0_MINIMAL_TRAINING_STACK_WITH_MANIFEST_GUARD.md`

Authoritative Stage 6.1 actor-safe batch and closure documents:

- `docs/STAGE6_1_ACTOR_SAFE_BATCH_BUILDER.md`
- `docs/STAGE6_COMPLETION_REVIEW.md`

Structured manifest:

- `src/marl_topology/training/design_contract.py`
- `src/marl_topology/policies/architecture_contract.py`
- `src/marl_topology/data/learning_target_contract.py`
- `src/marl_topology/training/run_manifest_contract.py`
- `src/marl_topology/training/run_manifest_validator.py`
- `src/marl_topology/training/minimal_stack.py`
- `src/marl_topology/data/actor_batch.py`

## Execution Boundary

Stage 5.6 does not authorize training execution.

Stage 5.7 does not authorize policy or critic implementation.

Stage 5.8 does not authorize dataset writer, replay buffer, learner batch, or
target derivation implementation.

Stage 5.9 does not authorize artifact writes, manifest writers, checkpoint
creation, dataset export, or training execution.

Stage 5.10 adds only a dry-run run-manifest validator and closes Stage 5. It
does not authorize artifact writes, manifest writers, checkpoint creation,
dataset export, model implementation, or training execution.

Stage 6.0 adds only a minimal dry-run training-stack guard that consumes the
Stage 5.10 validator. It does not authorize artifact writes, manifest writers,
checkpoint creation, dataset export, model implementation, reward-weight
calibration, final tau selection, or training execution.

Stage 6.1 adds only an in-memory actor-safe batch builder. It does not
authorize dataset export, replay storage, model implementation, checkpoint
creation, artifact writing, reward-weight calibration, final tau selection, or
training execution. Stage 6 is closed after Stage 6.1.

Blocked until a later owner-approved task:

- training runs;
- learner implementation;
- actor or critic model code;
- checkpoint creation;
- reward weight calibration;
- final `tau_consensus` selection;
- v5 code migration.

## Run Manifest Artifact Boundary

Stage 5.9 defines future run-manifest and artifact semantics as planned, not
active. Future artifacts must stay under `result_save`, and the current
scaffold baseline remains `.gitkeep` only.

Future run manifests must record owner approval id, stage id, config id,
scenario set id, split id, seed, seed group id, code version marker, contract
ids, metric registry version, physics regime id, protocol model id, objective
contract id, surrogate config id, normalization reference id, architecture
contract id, replay schema version, and artifact policy id.

Stage 5.10 validates those fields in memory and validates artifact-root
containment under `result_save`. It is an exit gate, not execution permission.

Stage 6.0 requires this validator before the minimal stack can report
readiness. Invalid manifests, missing Stage 6 approval markers, missing
contract markers, and blocked operation requests keep the stack blocked.

Stage 6.1 consumes actor observations or projected mixed rows and emits
actor-safe in-memory rows only. It is not a replay writer and not a model input
contract beyond local field safety.

## Learning Target Replay Boundary

Stage 5.8 defines future target and replay columns as planned, not active.

`return`, `advantage`, `value_target`, `episode_id`, `trajectory_id`,
`transition_index`, `discount_factor`, and `bootstrap_value` remain
training-only future fields. They must not enter deployment actor inputs and
must not be treated as evaluation metrics.

The active replay schema still rejects `return`, `advantage`, and
`value_target` until a later owner-approved implementation task changes the
schema with tests.

## Policy Architecture Boundary

Stage 5.7 freezes the architecture boundary without implementation.

Deployment actors remain local-observation only. Future centralized training
views are allowed only for training-only critic interfaces and must not enter
actor input batches, actor export, checkpoint actor serialization, or
deployment evaluation.

Future actor options may be reviewed only as local-input designs. Stage 5.7
does not choose a final actor family, critic family, credit mechanism, or
checkpoint format.

## Required Future Contract Before Execution

Before training can run, a later task must provide:

- concrete actor and critic architecture contract;
- scalarization weights as explicit config, not copied from v5;
- learning-target and replay-column contract;
- artifact, seed, and run-manifest policy;
- multi-seed evaluation protocol;
- stop-condition rules;
- owner approval for execution.

## Evaluation Evidence

Training evidence must report registered evaluation concepts separately from
training-only diagnostics:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

Surrogate components, losses, entropy, action sparsity, and seed variance are
diagnostics unless a future metric registration promotes them.

## Dec-POMDP Boundary

Deployment actors remain local-observation only. Centralized information is
training-only and must not leak into deployment actor inputs, replay actor
projection, or checkpoint actor serialization.

## Stage 7 Learning Evidence Boundary

Stage 7.0 adds only learning evidence dataset generation.

It may create in-memory evidence rows and, after owner approval plus run
manifest validation, evidence-only dataset artifacts. It must not create
checkpoints, training logs, model outputs, replay buffers for learner execution,
or any training run.

Stage 7.0 separates:

- actor-safe rows;
- centralized critic context;
- learning targets;
- diagnostics.

Actor policy interface work is deferred to Stage 8 after a Stage 7 data quality
report.

Stage 7.1 audits data quality and closes Stage 7 for interface-design purposes.
It reports the current minimal evidence as structurally valid but not training
sufficient because feasibility class diversity is low at tau 0.9. Training
execution remains blocked.

## Stage 8.0 Actor Policy Interface Boundary

Stage 8.0 defines the deployment actor policy input and output schemas without
implementing a model or learner.

The actor input schema is exactly the Stage 6.1 actor-safe batch field set.
After Stage 9.0 schema unification, the active actor output schema is local
directed edge-score batches only. `activate` belongs only to the legacy
edge-decision schema or to assembler-selected topology. Centralized critic
views, learning targets, objective metrics, oracle labels, surrogate
diagnostics, and future outcomes remain training-only or diagnostic-only.

Stage 8.0 still does not authorize actor implementation, critic
implementation, training execution, checkpoint creation, artifact export,
reward-weight calibration, final tau selection, or v5 migration.

## Stage 9.0 Local MLP Edge Scorer Boundary

Stage 9.0 adds a minimal PyTorch local MLP edge scorer scaffold for inference
over actor-safe local observations. It is not a training stage.

Stage 9.0 still blocks:

- training execution;
- optimizer construction or `optimizer.step`;
- checkpoint creation or loading;
- training artifact writes;
- PPO/MAPPO/COMA implementation;
- GNN, GRU, LSTM, or Transformer implementation;
- reward-weight calibration;
- final tau selection;
- v5 migration.
