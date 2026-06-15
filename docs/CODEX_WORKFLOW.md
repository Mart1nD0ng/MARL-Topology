# Codex Workflow

## Required Closed Loop

Every substantial Codex task must include:

1. Controlled object.
2. Desired state.
3. State variables.
4. Sensors.
5. Actuators.
6. Disturbances.
7. Coupling map.
8. Feedback loop.
9. Acceptance criteria.
10. Verification commands.
11. Residual risks.

## Evidence Standard

Each task must report:

- Baseline evidence.
- Change summary.
- Post-change evidence.
- Regression check.
- Gaps.

## Post-Task Self-Review

After every substantial task, Codex must produce a structured self-review using `harness/templates/post-task-self-review.md.template` or the same fields:

- completed task
- intended desired state
- actual achieved state
- evidence
- tests
- gates passed
- gates deferred
- new risks
- regressions
- candidate next tasks
- recommended next task
- owner decision required

Codex may recommend next actions and update `docs/PROJECT_STATE.md`, but Codex must not self-authorize the next stage. User approval is required before executing any next task.

## Stage Closure Discipline

Codex must not turn one stage into an unlimited sequence of internal planning
steps. Substeps are justified only when they remove a concrete blocker, add a
necessary sensor, or protect a known boundary. Once an exit gate passes, Codex
must close the stage, record residual risks, recommend a next-stage task, and
wait for owner approval.

## Stage Planning Rules

### Stage Closeout Rule

Every stage closeout must:

- update `docs/PROJECT_STATE.md`;
- run the relevant tests;
- run harness validation;
- produce post-task self-review;
- update or register a harness task;
- declare whether the Completion Gate passed;
- declare whether the next-stage readiness gate passed.

### No Consecutive Planning-Only Rule

If the current stage and the previous stage are both mainly documents, plans,
contracts, readiness reviews, or review-only work with no runnable deliverable,
Codex must stop and request owner decision unless the current stage explicitly
removes a named blocker.

### Next-Stage Readiness Rule

Every closeout must answer:

- Completion Gate passed?
- Next-stage readiness gate passed?
- If not, what blocker remains?
- What could make the next stage invalid?
- What evidence says the next stage is now safe?

Tests passing alone is not an exit condition.

### Stage State Sync Rule

If a stage has been executed but is missing from `PROJECT_STATE.md` or the
harness task registry, its status is `implemented_unregistered`, not `closed`.
Codex must repair state/harness drift before using that stage as a clean base
for another owner-approved stage.

### Scope Compression Rule

If several small stages are only substeps toward the same objective, compress
them into one implementation-bearing stage with internal checkpoints. Do not
mint new stage numbers for plan-of-plan work unless it removes a concrete
blocker.

### Owner Checkpoint Rule

Codex may recommend the next task, but must not continue when
`PROJECT_STATE.md` is inconsistent, the harness task is missing, or
next-stage readiness is unclear.

## Routing

- Use `cybernetic-project-analysis` before large architecture, migration, or entropy-reduction work.
- Use `cybernetic-harness-design` before adding or revising harness tasks and rubrics.
- Use `cybernetic-stage-planning` whenever proposing, closing, repairing, or batching stages.
- Use domain skills before modifying physics, metrics, protocol, reward, Dec-POMDP, topology oracle, credit assignment, training, or entropy rules.
- Before any skill uses v5 evidence, route the recommendation through `docs/SKILL_CALIBRATION.md`.
- Use `post_task_self_review` after substantial work to update the evidence loop and next-task recommendation.

## Prohibited Work Without Contracts

- Training.
- Reward changes.
- Metric exports.
- Actor/critic model migration.
- 3D physics implementation.
- Legacy phase script promotion.
- Blind inheritance of old v5 skills, metrics, reward logic, phase structure, actor inputs, fixed thresholds, or COMA/Q defaults.
- Starting the next task or stage from a self-review without explicit user approval.
- Opening additional same-stage planning tasks after a passed exit gate.
