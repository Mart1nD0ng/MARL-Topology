# Post-Task Self-Review - Stage 17 Actor Observability And Label Disambiguation

## Completed Task

Stage 17 diagnosed whether actor-safe local observations can support the
current hard edge labels. It added an actor-observation signature builder,
contradiction detector, cause classifier, report script, documentation,
harness task, tests, and PROJECT_STATE closeout. It did not train, rerun
Stage 11-15, implement a new model, introduce COMA or Transformer, tune reward
weights, modify v5, create checkpoints, or write training artifacts.

## Intended Desired State

The desired state was an evidence-generation and exit-gate diagnostic that
groups identical actor-safe local observation signatures, detects conflicting
add/remove/keep and metric-delta labels, classifies likely causes, and decides
whether training reruns are safe.

## Actual Achieved State

Stage 17 analyzed the Stage 16 evidence set with an actor-safe signature over
agent identity/type, time step, local position, incident neighbor observation,
local messages, and local history. It found:

- `850` edge-label samples analyzed.
- `142` actor observation signatures.
- `142` contradiction clusters.
- `850` samples inside contradictory clusters.
- contradiction rate by signature: `1.0`.
- actor local observations sufficient for current hard labels: `false`.

## Evidence

- Unit and contract tests cover synthetic conflicts, forbidden field rejection,
  Stage 16 data analysis, report-only script behavior, and source boundary
  scans.
- The report script prints JSON and records `artifact_written: false`.
- PROJECT_STATE records Stage 17 completion, Stage 18 recommendation, and
  blocked training/model/rerun tasks.

## Gates Passed

- Actor-safe signature construction gate.
- Forbidden global/oracle/reward/critic/future field rejection gate.
- Contradiction cluster detection gate.
- Cause classification gate.
- Documentation and harness registry gate.
- No training/model/checkpoint/v5 modification gate.

## Gates Deferred

- Training rerun readiness gate.
- Stage 11-15 rerun gate.
- Scale-up training gate.
- Final target-semantics decision gate.
- Dec-POMDP feature-boundary owner decision gate.

## New Risks

The report shows that the current hard labels are not functions of the current
actor-local edge observations. Reusing them for supervised or policy-gradient
reruns would mostly measure label ambiguity rather than actor improvement.

## Regressions Checked

The Stage 17 code is under `src/marl_topology/data/` and replay reporting only.
It does not modify `D:\PhD_works\v5`, does not write artifacts, and does not add
training, optimizer, checkpoint, COMA, PPO/MAPPO, Transformer, or reward tuning
paths.

## Required Stage 17 Questions

Actor local observations sufficient for current hard labels? No. Identical
actor-safe signatures still map to contradictory add/remove/keep and metric
delta directions.

Main contradiction source? The dominant sources are missing previous/current
topology or local projection history, missing local resource context, and hard
labels that encode global counterfactual or oracle topology effects.

Add features, change targets, or expand history? Do all in a controlled Stage
18 rebuild: add actor-safe local history/resource/message summaries where
allowed, transform hard labels into soft/ranking targets where topology state
drives ambiguity, and keep global counterfactual value in critic-only targets.

Is rerun of Stage 11-15 allowed? No. Stage 11-15 reruns remain blocked until
Stage 18 rebuilds disambiguated evidence or the owner changes the Dec-POMDP
feature boundary.

## Candidate Next Tasks

- `stage_18_evidence_rebuild_with_disambiguated_features_or_targets`
- `owner_decision_on_dec_pomdp_feature_boundary`

## Recommended Next Task

`stage_18_evidence_rebuild_with_disambiguated_features_or_targets`

## Owner Decision Required

Yes. Stage 17 does not self-authorize Stage 18, training reruns, scale-up
training, COMA, Transformer, or reward-weight tuning.
