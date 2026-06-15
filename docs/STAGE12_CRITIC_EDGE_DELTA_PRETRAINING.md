# Stage 12 Critic And Edge-Delta Pretraining

## Controlled Object

The controlled object is the centralized MLP critic parameter state and its
direct edge-delta heads.

## Desired State

Train only the centralized critic on Stage 7 evidence. Produce a fidelity
report for scalar heads and edge-delta heads before any policy-gradient pilot.
Actor training and RL updates remain inactive.

## State Variables

- Critic loss before and after pretraining.
- Critic parameter checksum before and after updates.
- Add/remove edge-delta ranking metrics.
- Balanced sign accuracy.
- Helpful-edge precision and harmful-edge recall.
- Feasibility classification accuracy.
- Consensus, latency, and energy regression errors.
- Calibration errors for probability heads.
- Collapse diagnostics.

## Sensors

- Unit tests for critic-only updates, fidelity report fields, and no checkpoint
  or actor/RL update route.
- Smoke script: `python scripts\train\stage12_critic_pretraining.py`.
- Full pytest and harness validation.

## Actuators

- `src/marl_topology/training/critic_pretrainer.py`
- `scripts/train/stage12_critic_pretraining.py`
- Stage 12 docs, tests, and self-review.

## Disturbances

The Stage 7 evidence is small and edge-delta targets are imbalanced. The
fidelity gate is a basic sanity gate, not proof of deployment-quality credit
assignment.

## Coupling Map

Stage 7 centralized training views become critic tensors. Critic targets train
scalar heads and add/remove/keep edge-delta heads. Critic outputs remain
training-only and never become actor inputs.

## Acceptance Criteria

- Critic-only parameter update completes.
- Fidelity report is produced.
- Predictions are not collapsed.
- Edge-delta sign/rank sanity passes or the stage blocks later policy-gradient
  work.
- No actor RL fine-tune, COMA, GNN actor, temporal actor, Transformer,
  checkpoint, final tau selection, reward-weight calibration, or v5 migration.

## Verification

```powershell
python scripts\train\stage12_critic_pretraining.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

Passing this small fidelity gate does not prove robust credit assignment. It
only allows a controlled pilot to proceed with explicit comparison and stop
conditions.
