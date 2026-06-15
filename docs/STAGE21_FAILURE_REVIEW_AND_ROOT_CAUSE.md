# Stage 21 Failure Review And Root Cause

Stage 21 status: blocked awaiting owner decision.

Policy-gradient pilot: skipped.

## Failed Gate

Only one hard gate failed:

`actor_performance_sufficient_for_pilot`

Evidence:

- best actor: `GNN`
- best actor projected tau-feasible rate: `0.4714285714`
- full graph projected tau-feasible rate: `0.4571428571`
- projected greedy tau-feasible rate: `0.4857142857`
- Stage 20 best actor tau-feasible rate: `0.5593220339`
- GNN top-proposal rejection rate: `0.5128571429`
- GNN above-threshold rejection rate: `0.6889880952`
- GNN empty collapse rate: `0.0857142857`
- GNN full-graph collapse rate: `0.1714285714`

## Gates That Passed

- Objective-stack alignment passed. Main evidence uses Stage 3
  finite-blocklength network records and Stage 4 expected-initiator PBFT.
- Actor target quality passed. Targets have high/mid/low spread and pairwise
  rankings; hard labels are not actor training targets.
- Fair assembler evaluation passed. Main baseline comparison uses projected
  baselines under the same assembler constraints.
- Projection mismatch improved. Top-proposal rejection improved from the Stage
  20 baseline `0.6647058824` to `0.5128571429`.
- Boundary safety passed. No actor leakage, `v5` modification, COMA,
  Transformer, reward-weight tuning, final tau selection, checkpoint, or
  uncontrolled artifact write occurred.

## Root Cause

Stage 21 fixed the evaluator and target mismatch enough to reduce top-proposal
rejection, but the supervised actor still does not produce a sufficiently good
deployment proposal distribution under the final objective stack.

The strongest remaining signal is actor feature/training mismatch under
assembler constraints:

- GNN is slightly better than MLP after projection, but both remain below the
  Stage 20 best actor tau-feasible threshold.
- GNN beats projected full graph narrowly but remains below projected greedy.
- Above-threshold rejection remains high, so high-scored actor proposals are
  still often incompatible with tx/rx budget or conflict constraints.
- The target now has high/mid/low spread, so the root cause is no longer a
  uniformly high target distribution.
- The evidence stack is now final-stack aligned, so another repair should not
  revert to the early `SimpleLinkModel` or min-link evaluator.

## Recommended Repair

Recommended next task:

`stage_22_actor_feature_or_supervised_training_repair_based_on_stage21_report`

Repair scope:

- improve actor feature representation for assembler resource and conflict
  constraints;
- improve supervised objective weighting or calibration without reward-weight
  tuning;
- inspect why above-threshold actor proposals still hit tx/rx and conflict
  rejection;
- keep global edge-delta targets critic-only;
- keep projected baselines as the main comparison sensor.

Do not run policy-gradient until the actor performance gate passes.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

Result: `745 passed`; task validation passed for `78` tasks.
