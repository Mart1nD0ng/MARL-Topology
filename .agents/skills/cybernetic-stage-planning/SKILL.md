---
name: cybernetic-stage-planning
description: Review stage scope, closeout, next-stage readiness, and state/harness synchronization for MARL-Topology work.
---

# Cybernetic Stage Planning

## When To Use

- Every time Codex proposes a new stage.
- Every time Codex closes a stage.
- Every time `/goal` or a similar instruction executes multiple stages.
- Every time the owner says the project is starting to sprawl.
- Every time tests pass but the project still has not made real progress.
- Every time a stage is mostly plan, contract, readiness, or review work.

## Procedure

Codex must answer:

1. What is this stage's controlled object?
2. What is this stage's executable deliverable?
3. Is this stage planning-only or implementation-bearing?
4. Does this stage remove a concrete blocker?
5. What is the Completion Gate?
6. What is the Next-Stage Readiness Gate?
7. What could make the next stage fail or become invalid?
8. Did this stage update `PROJECT_STATE.md`?
9. Did this stage register or update a harness task?
10. Did this stage produce post-task self-review?

## Hard Rules

- Do not allow two consecutive planning-only stages unless the owner explicitly
  approves the second one and names the blocker it removes.
- Do not treat "tests pass" as the only exit condition.
- Do not execute a stage and leave `PROJECT_STATE.md` stale.
- Do not batch multiple stages under `/goal` without registering each stage's
  state and harness task.
- Do not claim an unregistered stage is formally closed.
- Do not skip next-stage readiness evidence for the sake of tidy closure.
- If code, docs, and tests advanced but state/harness did not, mark the stage
  `implemented_unregistered`.

## Output Format

```text
Stage type:
Executable deliverable:
Controlled object:
Blocker removed:
Completion gate:
Next-stage readiness gate:
What could make the next stage invalid:
State sync checklist:
- PROJECT_STATE.md:
- harness task:
- post-task self-review:
- completed_stages:
- recommended_next_task:
- blocked_tasks:
- owner_decision_required:
Recommended next task:
Owner decision required:
```

## Stage Types

- `planning-only`: documents or decisions only; must remove a named blocker.
- `implementation-bearing`: adds runnable code, data builder, interface,
  tests, or harness behavior.
- `exit-gate`: closes a stage with stronger evidence and no new model/training.
- `calibration`: measures requirement feasibility or parameter ranges without
  selecting final policy unless authorized.
- `evidence-generation`: builds or audits data, labels, replay, or diagnostics.
- `model-training`: runs training and therefore requires explicit owner
  approval plus reward, metric, artifact, and safety contracts.

## Failure Modes

- stage sprawl
- premature closeout
- weak exit gate
- project state drift
- harness registry drift
- plan-of-plan loop
- model-before-data
- training-before-evidence

## Quality Gates

- Every closeout has both Completion Gate and Next-Stage Readiness Gate.
- A next task is only recommended, never self-authorized.
- Planning-only work is compressed into an implementation-bearing stage unless
  it removes a blocker.
- `PROJECT_STATE.md`, `harness/tasks`, and post-task self-review agree before a
  stage is called closed.

## V5 Anti-Inheritance Calibration

- Read `docs/SKILL_CALIBRATION.md` before any stage planning work that touches
  migration, metrics, reward, actor inputs, critics, topology oracle behavior,
  training, artifacts, or legacy evidence.
- Treat `D:\PhD_works\v5` as a read-only experience library, never as a stage
  template or code source.
- Do not let v5 phase-script structure justify stage sprawl in the clean
  project.
- Reject stage closeout that relies on old reward weights, old metric aliases,
  old checkpoint paths, fixed deployment thresholds, COMA/Q defaults, or
  global actor tensors unless a clean-project contract and tests explicitly
  authorize them.
