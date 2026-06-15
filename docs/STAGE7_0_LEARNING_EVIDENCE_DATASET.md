# Stage 7.0 Learning Evidence Dataset Generation

## Scope

Stage 7.0 starts the learning evidence phase. It replaces the previous
recommendation to jump directly from Stage 6 actor-safe batches to an actor
policy interface. Actor policy interface work is deferred to Stage 8.

This stage creates a minimal evidence dataset boundary and in-memory builder.
It does not train, implement actor or critic models, create checkpoints,
calibrate reward weights, select final tau, or migrate legacy code.

## Control Model

Controlled object: evidence rows derived from current simulator, topology
evaluation, objective metrics, and actor-safe observations.

Desired state: a reproducible dataset shape with actor-safe input rows separated
from centralized critic context, learning targets, and diagnostics.

State variables:

- actor-safe leakage risk
- evidence row completeness
- edge-delta target coverage
- artifact path containment
- manifest approval state
- Stage 7 to Stage 8 boundary

Sensors:

- unit tests for evidence rows and edge-delta targets
- negative leakage tests
- artifact writer guard tests
- contract tests for docs, harness task, script output, and source scan
- harness task validation

Actuators:

- `src/marl_topology/data/learning_evidence.py`
- `scripts/replay/stage7_0_learning_evidence_report.py`
- Stage 7 contract documents
- Stage 7 harness task
- PROJECT_STATE update

Disturbances:

- objective metrics accidentally entering actor inputs
- oracle diagnostic labels becoming deployment labels
- artifact writing before manifest guard approval
- over-fragmenting Stage 7 into more plan-only tasks
- premature actor/critic/model implementation

Feedback loop:

`ActorObservation` -> actor-safe projection -> topology evaluation evidence ->
edge-delta target builder -> contract tests -> harness validation -> post-task
self-review -> owner decision.

Acceptance criteria:

- evidence rows exist in memory
- `actor_safe_view` excludes objective, oracle, future, and global fields
- `learning_target_view` contains edge-delta rows marked learning-target only
- artifact writer is blocked without approval or valid manifest
- project report script runs without writing project artifacts
- `result_save` remains scaffold-only unless an explicit evidence export is
  owner-approved
- Stage 8 remains blocked pending a data quality report

## Implemented Interfaces

`LearningEvidenceRow` separates:

- `actor_safe_rows`: deployment actor observations only
- `critic_view`: centralized training-only context
- `learning_targets`: edge-delta targets
- `diagnostics`: report-only metadata and topology family labels

`EdgeDeltaTarget` supports:

- `add_edge`
- `remove_edge`
- `keep_edge`

Each target carries deltas for consensus success probability, latency, energy,
and feasibility under the declared tau requirement.

## Evidence Export Boundary

`write_learning_evidence_artifact` is available but guarded. It requires:

- owner approval flag
- valid run manifest
- `artifact_scope = evidence_dataset_only`
- artifact root contained under `result_save`
- write destination contained under `result_save/evidence_dataset_only`

The Stage 7.0 replay script intentionally does not write artifacts. Unit tests
exercise the writer in a temporary project root so the repository scaffold
stays clean.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

## Residual Risks

Stage 7.0 proves data separation and minimal edge-delta evidence shape. It does
not prove that the generated evidence is statistically sufficient for learning.
The next sensor should be a Stage 7 data quality report before Stage 8 actor
policy interface work.
