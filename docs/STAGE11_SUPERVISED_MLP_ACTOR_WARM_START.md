# Stage 11 Supervised Local MLP Actor Warm Start

## Controlled Object

The controlled object is the `LocalMLPEdgeScorer` actor parameter state under a
supervised warm-start update.

## Desired State

Train only the Local MLP actor against Stage 7 supervised edge labels. Record
train/validation losses, BCE, precision/recall, ranking diagnostics,
assembler-level diagnostics, leakage checks, and tiny-batch overfit evidence.
Do not train the critic, do not run RL, and do not write checkpoints.

## State Variables

- Actor parameter checksum before and after supervised updates.
- Train and validation loss components.
- Edge decision precision and recall.
- Ranking pairwise accuracy and Spearman diagnostic.
- Tiny-batch overfit initial/final loss.
- Assembler projected topology diagnostics.
- Manifest validation status and artifact location.

## Sensors

- Unit tests for actor-only parameter updates, tiny-batch overfit, manifest
  fields, and no critic/RL/checkpoint route.
- Smoke script: `python scripts\train\stage11_supervised_mlp_actor.py`.
- Full pytest and harness validation.

## Actuators

- `src/marl_topology/training/supervised_actor_trainer.py`
- `scripts/train/stage11_supervised_mlp_actor.py`
- Stage 11 docs, tests, manifest-validated report artifacts, and self-review.

## Disturbances

Stage 7 topology variants can produce contradictory labels for identical local
edge observations. Tiny-batch overfit is therefore measured on one deterministic
mixed-label topology row, while full train/validation metrics remain diagnostic.

## Coupling Map

Stage 7 actor-safe rows become actor tensors. Edge labels train only the Local
MLP actor. Actor scores feed the Stage 8 recommended assembler, and projected
topology metrics are evaluated environment-side. Critic and RL paths remain
inactive.

## Acceptance Criteria

- Actor-only training completes on the controlled small dataset.
- Tiny-batch overfit passes on deterministic mixed labels.
- Assembler consumes trained actor scores.
- Manifest validation passes; report artifact writing remains optional and is
  disabled by the default smoke run to keep historical scaffold tests stable.
- No checkpoint, critic training, RL update, COMA, GNN, GRU, LSTM,
  Transformer, final tau selection, reward-weight calibration, or v5 migration.

## Verification

```powershell
python scripts\train\stage11_supervised_mlp_actor.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

The Stage 7 evidence is small and partially contradictory across topology
variants. Stage 12 must test critic and edge-delta fidelity before any policy
gradient pilot can rely on centralized value or credit signals.
