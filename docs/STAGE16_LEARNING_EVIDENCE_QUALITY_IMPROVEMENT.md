# Stage 16 Learning Evidence Quality Improvement

## Controlled Object

The controlled object is the learning evidence and edge-delta target quality
gate used by the Stage 9-15 model stack. The boundary is data construction,
quality sensing, and documentation only. It does not include model expansion,
optimizer execution, policy-gradient training, checkpointing, reward-weight
calibration, final tau selection, COMA, Transformer, or v5 migration.

## Desired State

Stage 16 improves evidence observability after the Stage 15 pilot showed no
violation-rate worsening and no actor collapse, but still failed
`tau_requirement_min=0.9`. The desired observable state is an in-memory fixture
set that covers feasible examples, near-threshold examples, infeasible hard
examples, sparse feasible topology rows, full graph resource-dominated
topology rows, weak-primary / center-primary contrast, interference penalty
examples, and at least one real multi-step actor-safe sequence.

## State Variables

- Evidence rows: `59`.
- Edge-delta targets: `850`.
- Tau requirement: `tau_requirement_min=0.9`.
- Feasible rows at tau: `30`.
- Near-threshold rows: `17`.
- Hard infeasible rows: `5`.
- Sparse feasible rows: `8`.
- Full graph resource-dominated rows: `2`.
- Interference/resource penalty proxy rows: `6`.
- Real multi-step actor-safe sequence: one sequence with `time_step = 0, 1, 2`.
- Identical local observations with contradictory labels: still present.

## Sensors

- `build_stage16_learning_evidence_report()` builds the in-memory report.
- `evaluate_learning_evidence_quality()` checks view coverage, actor leakage,
  topology-family coverage, metric ranges, target actions, and feasibility.
- `tests/unit/test_stage16_learning_evidence_quality.py` checks fixture
  coverage, edge-delta balance, actor-safe multi-step evidence, and no
  artifact writes.
- `tests/contract/test_stage16_learning_evidence_quality_contract.py` checks
  documentation, PROJECT_STATE repair, harness registry, and forbidden
  implementation patterns.

## Actuators

- Added `src/marl_topology/data/learning_evidence_stage16.py`.
- Added Stage 16 unit and contract tests.
- Added harness task records for Stage 9-15 model-stack review and Stage 16.
- Updated PROJECT_STATE to repair Stage 10-15 history and record the Stage 16
  no-scale-up boundary.

## Evidence Coverage

The Stage 16 fixture set is generated from deterministic scenario geometry,
candidate graphs, link estimates, topology evaluations, and actor-safe local
observations. It remains in memory and writes no `result_save` artifact.

- tau feasible examples: `30` rows meet or exceed `tau_requirement_min=0.9`.
- near-threshold examples: `17` rows sit within `0.05` of the tau boundary.
- infeasible hard examples: `5` rows cover quorum-unreachable or scheduled but
  requirement-infeasible cases.
- sparse feasible topology: `8` sparse quorum rows are feasible under tau.
- full graph resource-dominated topology: `2` full graph rows are explicitly
  tagged as dense resource-dominated topology examples.
- weak-primary / center-primary contrast: center-primary sparse probability is
  about `0.9414`, weak-primary sparse probability is about `0.6290`, giving a
  center-minus-weak delta of about `0.3125`.
- interference penalty examples: represented as a declared dense
  resource/interference proxy in the current evaluator, not as a new Stage 3
  physical interference simulation rerun.
- real multi-step actor-safe sequence: `stage16_real_mobility_actor_safe_sequence_v1`
  uses scenario mobility and `time_step` values `0`, `1`, and `2`; actor rows
  are produced from `build_local_observations`, not synthetic temporal tensors.

## Edge-Delta Target Rebuild

Stage 16 rebuilds edge-delta targets for add edge, remove edge, and keep edge
actions. The targets include feasibility-changing deltas and metric deltas.

- add edge targets: `275`.
- remove edge targets: `150`.
- keep edge targets: `425`.
- positive delta consensus_success_probability targets: `52`.
- negative delta consensus_success_probability targets: `134`.
- helpful edge targets: `143`.
- harmful edge targets: `282`.
- feasibility-changing deltas: `90`.
- delta consensus_success_probability nonzero targets: `186`.
- delta latency nonzero targets: `223`.
- delta energy nonzero targets: `425`.
- delta surrogate diagnostic nonzero targets: `425`.
- rare safety samples: `32`.
- oracle gap comparable rows: `54`; max probability gap about `0.9879`.

The delta surrogate diagnostic is a learning-target diagnostic only. It is not
reward execution, reward-weight calibration, or deployment actor input.

## Data Quality Answers

Current data is not sufficient to continue supervised training. It is improved
enough for smoke and dry-run sensors, but not enough to rerun Stage 11-15 or
claim scale-up readiness.

current data is not sufficient to continue supervised training.

There are tau>=0.9 feasible examples. The report contains `30` such rows, so
the previous all-low-consensus blocker is improved. This is still fixture-level
coverage, not training-scale evidence.

There are sparse feasible examples. The report contains `8` sparse feasible
rows and sparse/full tradeoff cases where sparse topologies reduce latency and
energy compared with full graph.

There is real multi-step evidence. The sequence is generated from deterministic
scenario mobility and time_step values, not from the synthetic temporal fixture
used in Stage 14.

Actor-observable features are not yet sufficient to distinguish all edge
labels. Identical local observations with contradictory labels still exist
because multiple topology variants share the same actor-local view while
global feasibility/resource consequences differ.

identical local observations with contradictory labels remain present.

The recommended next step is to continue evidence expansion before rerunning
Stage 11-15. Rerunning supervised or policy-gradient stages now would mainly
re-measure known contradictory-label behavior.

continue evidence expansion before rerunning Stage 11-15.

Scale-up training remains blocked. scale-up training remains blocked.

## Boundaries Preserved

- No training execution.
- No PPO/MAPPO.
- No COMA.
- No Transformer.
- No new neural model.
- No checkpoint.
- No training artifact.
- No reward-weight calibration.
- No final tau selection.
- No v5 modification or migration.
- No oracle label, reward surrogate, consensus metric, or critic field enters
  actor-safe observations.

## Acceptance Criteria

- Stage 16 evidence improves coverage for tau, near-threshold, hard
  infeasible, sparse/full, primary contrast, interference/resource proxy, and
  real multi-step actor-safe cases.
- Edge-delta targets include add edge, remove edge, keep edge, feasibility
  changes, delta consensus_success_probability, delta latency, delta energy,
  and delta surrogate diagnostic.
- Actor-safe rows pass leakage checks.
- The builder writes no artifacts.
- Validation commands pass:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risk

The fixture set is still small and partly deterministic. It improves the data
quality sensor but does not solve label ambiguity. The next safe actuator is
additional scenario/mobility evidence expansion, followed by an owner decision
on whether to rerun Stage 11-15 against the expanded evidence.
