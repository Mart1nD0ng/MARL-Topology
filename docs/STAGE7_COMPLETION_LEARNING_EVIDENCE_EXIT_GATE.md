# Stage 7 Completion - Learning Evidence Dataset Build + Quality Exit Gate

## Why This Completion Task Exists

Stage 7.0 and Stage 7.1 proved the data boundary and a minimal quality sensor,
but they closed the stage too early. A smoke dataset is not enough to answer the
pre-model learning questions:

- Which topologies are good under the current simulator and consensus target?
- Which edges are useful to add, remove, keep, or swap?
- Whether actor-local observations plausibly contain the information needed to
  predict local edge decisions?
- Which centralized critic fields are available for future value learning?

This task reopens Stage 7 as a completion exit gate and closes it only after the
three stage deliverables are present.

## Control Model

Controlled object: the Stage 7 learning evidence data layer.

Desired state: a reproducible evidence dataset and quality report that can
support Stage 8 actor policy interface contract design, without training or
model implementation.

State variables:

- scenario diversity;
- topology variant coverage;
- feasible/infeasible class balance under `tau_requirement_min = 0.9`;
- sparse-vs-full trade-off visibility;
- oracle and heuristic label availability;
- add/remove/keep/swap edge-delta target coverage;
- surrogate diagnostic target separation;
- actor-safe leakage count;
- critic centralized field coverage;
- artifact manifest validity.

Sensors:

- `build_stage7_completion_dataset`;
- `build_stage7_completion_report`;
- `scripts/replay/stage7_completion_evidence_dataset_report.py`;
- manifest validator;
- unit and contract tests;
- harness validation;
- post-task self-review.

Actuators:

- Stage 7 completion dataset builder;
- Stage 7 completion edge-delta target builder;
- Stage 7 completion quality report;
- evidence-only artifact writer;
- PROJECT_STATE and harness task updates.

Disturbances:

- premature actor/critic/model work;
- treating oracle labels as actor inputs;
- treating diagnostic surrogate components as actor observations;
- smoke-only evidence being mistaken for learning evidence;
- default artifact writes bypassing manifest guard;
- copying v5 reward, metrics, phase scripts, or checkpoint loading.

## Detailed Execution Plan

1. Build scenario/topology evidence rows.
   - Use the existing deterministic scenario fixtures.
   - Add Stage 7 close-reference and sparse-vs-full trade-off fixtures.
   - Evaluate `empty`, `random`, `sparse_heuristic`, `full_graph`, `greedy`,
     and feasible `oracle_candidate` topologies where available.
   - Record consensus success probability, latency, energy, topology
     diagnostics, and feasibility under `tau_requirement_min = 0.9`.

2. Build oracle and edge-delta learning targets.
   - For each topology, hold scenario state fixed.
   - Evaluate add edge, remove edge, keep edge, and swap edge counterfactuals.
   - Record delta consensus success probability, delta latency, delta energy,
     delta feasibility, and training-only surrogate diagnostic deltas.
   - Keep these labels out of actor observations.

3. Build replay projection and quality report.
   - Separate `actor_safe_view`, `critic_centralized_view`,
     `learning_target_view`, and `diagnostics_view`.
   - Report class balance, feasible/infeasible ratio, sparse-vs-full trade-off,
     edge-delta distribution, rare safety samples, oracle gap, scenario
     diversity, and actor-observable predictability risk.

4. Write an evidence-only artifact.
   - Use owner-approved manifest fields.
   - Validate artifact root under `result_save`.
   - Write only manifest, learning evidence dataset, and quality report under
     `result_save/evidence_dataset_only/stage7_completion_learning_evidence_dataset_v1`.
   - Do not write checkpoints, model outputs, training logs, or v5 paths.

5. Close Stage 7 only if the exit criteria pass.

## Exit Criteria

1. Reproducible scenario/topology evidence rows exist.
2. At least one oracle or heuristic topology label exists.
3. Edge-delta learning targets exist.
4. Actor-safe, critic-view, target-view, and diagnostics-only data are clearly
   separated.
5. The data quality report shows a learnable signal.
6. Oracle labels do not enter actor input.
7. Surrogate diagnostics do not enter actor observations.
8. `result_save` artifact writing passes manifest validation.
9. No training is run.
10. No actor, critic, GNN, LSTM, COMA, checkpoint, or model implementation is
    added.

## Current Completion Result

The completion report produces:

- 5 scenarios;
- 29 scenario/topology evidence rows;
- 367 edge-delta learning targets;
- 8 feasible rows and 21 infeasible rows under tau 0.9;
- add/remove/keep/swap target coverage;
- 234 nonzero edge-delta targets;
- sparse-vs-full trade-off cases;
- oracle-comparable rows;
- rare safety samples;
- actor-safe leakage count of zero.

Stage 7 is ready to close only after validation commands pass.

## Verification Commands

```powershell
python scripts\replay\stage7_completion_evidence_dataset_report.py --write-artifact
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Boundary

Stage 8 may begin only with owner approval. Stage 8 should start with an actor
policy interface contract, not a model implementation. Training remains blocked.
