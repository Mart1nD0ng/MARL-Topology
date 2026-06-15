# Stage 10 Supervised Loss Dry-Run

## Controlled Object

The controlled object is supervised loss alignment between Stage 7 learning
evidence, Stage 9 actor logits, and Stage 9 centralized critic heads.

## Desired State

Compute finite supervised actor and critic loss components without calling
`backward`, creating an optimizer, updating parameters, writing checkpoints, or
writing artifacts.

## State Variables

- Actor logits and edge decision targets.
- Optional actor edge-delta diagnostic targets.
- Critic scalar targets: value placeholder, feasibility, consensus success
  probability, latency, and energy.
- Critic edge-delta targets: add, remove, and keep.
- Parameter checksum before and after dry-run.

## Sensors

- Unit tests for finite scalar losses, missing-target rejection, target/input
  separation, and source scans.
- Smoke script:
  `python scripts\replay\stage10_supervised_loss_dry_run_report.py`.
- Full pytest and harness validation.

## Actuators

- `src/marl_topology/training/supervised_batching.py`
- `src/marl_topology/training/supervised_losses.py`
- Stage 10 smoke script and tests.

## Disturbances

Stage 7 evidence contains training-only targets near actor-safe rows. The
dry-run must use targets for loss computation only and must not feed them into
actor tensorization.

## Coupling Map

Actor-safe evidence rows become actor edge tensors. Selected-edge labels and
edge-delta rows become target tensors. Centralized critic views become critic
tensors. Loss functions consume model outputs plus targets, but no loss result
is backpropagated.

## Acceptance Criteria

- Loss report contains finite scalar components.
- Missing target fields are rejected.
- Actor-safe fields remain separated from learning targets.
- No backward call, optimizer, checkpoint, artifact write, COMA, GNN, GRU, LSTM,
  PPO/MAPPO, Transformer, or v5 migration is introduced.

## Verification

```powershell
python scripts\replay\stage10_supervised_loss_dry_run_report.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

Finite loss alignment does not prove trainability. Stage 11 must include a
controlled tiny-batch overfit check before treating supervised warm start as
usable.
