# Stage 15 Controlled MAPPO/PPO Pilot

## Controlled Object

The controlled object is a short online policy-gradient fine-tune pilot using
the best supervised actor candidate and Stage 12 critic evidence as a gate.

## Desired State

Run a small clipped policy update with explicit log-probability semantics:
policy actions are sampled proposal indicators from actor logits, while the
environment transition uses the assembler-projected topology. The pilot reports
pre/post diagnostics and may stop with a reason.

## State Variables

- Before/after consensus success probability, violation rate, latency, energy,
  topology diagnostics, and edge count.
- Policy loss, value loss diagnostic, KL proxy, entropy, clip fraction.
- Actor score distribution.
- Projection rejection rate.
- Collapse and safety-degradation flags.

## Sensors

- Unit tests for log-probability semantics, metrics, stop conditions, and
  forbidden COMA/Transformer/checkpoint routes.
- Smoke script: `python scripts\train\stage15_controlled_ppo_pilot.py`.
- Full pytest and harness validation.

## Actuators

- `src/marl_topology/training/ppo_pilot.py`
- Stage 15 script, docs, tests, and self-review.

## Disturbances

The demo fixture is a small single-rollout pilot and the reward-surrogate
weights are an explicit non-calibrated pilot config, not final calibration.

## Coupling Map

Stage 13 local GNN actor produces proposal logits. Proposal samples are scored
under the actor distribution. The assembler projects active proposals into a
topology. The evaluator and Stage 5.2 surrogate interface produce diagnostics
for a clipped policy update. Projected topology log-probability is not claimed
to be exact.

## Acceptance Criteria

- Pilot completes or stops with a reason.
- Before/after supervised actor comparison is reported.
- Projection diagnostics and log-probability semantics are explicit.
- No COMA, Transformer, large sweep, final tau selection, reward-weight
  calibration, checkpoint, v5 migration, or convergence claim.

## Verification

```powershell
python scripts\train\stage15_controlled_ppo_pilot.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

The pilot is a smoke test only. Any scale-up requires Stage 16 owner approval
and stronger seed/scenario evidence.
