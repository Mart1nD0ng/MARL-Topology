# Engineering Planning Lessons

This document captures planning failures exposed by Stage 5-16 and turns them
into standing workflow constraints. It is about the engineering control system,
not model behavior.

## Lesson A: Stage Sprawl / Design-Only Loop

Phenomenon:

Stage work can be split indefinitely into plan, contract, readiness, review,
and calibration sub-stages while producing no new executable deliverable. Stage
5 exposed this risk through many `5.0a`, `5.0b`, and later sub-stage labels.
The safety constraints were useful, but Codex used subdivision as a way to stay
safe instead of converging the stage.

Risk:

- The project remains safe but does not move forward.
- Process entropy rises and owner review becomes harder.
- The owner cannot tell whether the project is genuinely closer to training,
  evaluation, or deployment goals.
- A plan-of-plan loop can make workflow artifacts look complete while the
  controlled system gains no new sensor, data path, interface, or capability.

Solution:

- Every stage must have a clear executable deliverable.
- Plan-only work should usually be a section inside an implementation-bearing
  stage, not a new stage number.
- A planning-only stage is allowed only when it removes a concrete blocker or
  adds a necessary sensor that an implementation stage cannot safely proceed
  without.
- Stage scope must answer: what new observation capability, data capability,
  model capability, or control capability does this stage produce?
- Small steps under the same objective should be compressed into one
  implementation-bearing stage with internal checkpoints.

Gate:

If a stage only adds documents and no runnable sensor, test, interface, data
builder, harness check, or code path, it must be marked `planning-only`.
Planning-only stages must not appear twice in a row unless the owner explicitly
approves the second one and names the blocker it removes.

## Lesson B: Weak Exit Gate / Patch Stage Cascade

Phenomenon:

Some stages proved that the assigned module existed, but not that the module
was sufficient to support the next stage. Stage 9-15 connected the model-stack
smoke path, but Stage 16 then found that evidence quality remained too weak:
`tau_requirement_min=0.9` coverage had been thin, real temporal evidence was
missing, and identical local observations with contradictory labels still
existed.

Risk:

- Later stages discover that earlier stages did not close the right risk.
- A patch-stage cascade forms, where each next stage repairs a missing
  readiness condition from the previous one.
- Tests pass but the next engineering decision is still unsafe.
- Training can start before data, target consistency, reward behavior, or
  observability are ready.

Solution:

Every stage must have two gates:

1. Completion Gate: did this stage do what it said it would do?
2. Next-Stage Readiness Gate: is the result sufficient to support the next
   stage?

Examples:

- Before model training, check whether actor observations can explain labels.
- Data stages must check contradictory labels and target consistency.
- Reward stages must check scenario and metric non-saturation.
- Model stages must check data quality, target consistency, and whether smoke
  success says anything about the proposed next scale.
- Policy-gradient stages must distinguish no-regression smoke evidence from
  convergence or scale-up readiness.

Gate:

Tests passing is not enough to enter the next stage. Each closeout must answer:

- What could make the next stage invalid?
- What evidence says the next stage is now safe?
- If the next-stage readiness gate did not pass, what blocker remains?

## Lesson C: State / Harness / Self-Review Drift

Phenomenon:

The code, docs, and tests can advance beyond the recorded project state. Stage
9-15 work had executed, while `PROJECT_STATE.md` was still effectively at
Stage 9 and Stage 10-15 harness task registration was incomplete until later
repair.

Risk:

- The engineering control system's state sensor becomes false.
- Later Codex work may choose actions from an obsolete `current_stage`.
- The owner cannot know the true stage, completed work, blocked work, or next
  approval decision.
- The feedback loop breaks because implementation, harness, and self-review
  disagree.

Solution:

Every stage closeout must synchronize:

- `docs/PROJECT_STATE.md`
- `harness/tasks`
- post-task self-review
- `completed_stages`
- `recommended_next_task`
- `blocked_tasks`
- `owner_decision_required`
- completion gate result
- next-stage readiness gate result

When `/goal` or another long-running instruction executes multiple stages,
each stage still needs individual state and harness registration. A batch goal
does not remove per-stage bookkeeping.

Gate:

If a stage has been implemented but is missing from `PROJECT_STATE.md` or the
harness task registry, its status is `implemented_unregistered`, not `closed`.
It cannot be used as a clean base for another owner-approved stage until the
state and harness drift is repaired.

## Standing Closeout Checklist

For every stage, Codex must record:

- Stage type: `planning-only`, `implementation-bearing`, `exit-gate`,
  `calibration`, `evidence-generation`, or `model-training`.
- Executable deliverable.
- Completion Gate result.
- Next-Stage Readiness Gate result.
- Evidence produced.
- What remains unproven.
- State sync checklist.
- Harness registry checklist.
- Recommended next task.
- Owner decision required.

## Non-Target Work Protected

These lessons do not authorize model implementation, training, PPO/MAPPO,
COMA, Transformer, reward-weight calibration, final tau selection, checkpoint
creation, training artifact writing, or `v5` migration.
