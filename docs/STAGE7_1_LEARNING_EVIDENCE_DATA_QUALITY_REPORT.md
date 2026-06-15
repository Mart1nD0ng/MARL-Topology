# Stage 7.1 Learning Evidence Data Quality Report

## Scope

Stage 7.1 audits the minimal Stage 7 learning evidence dataset. It is a sensor,
not a training stage. It does not implement actor, critic, GNN, LSTM, learner
batches, checkpoints, artifact export runs, reward weight calibration, final
tau selection, or v5 migration.

## Control Model

Controlled object: Stage 7.0 learning evidence rows and edge-delta targets.

Desired state: the evidence dataset is structurally valid, leakage-safe, and
useful enough to define the Stage 8 actor policy interface contract while
remaining clearly insufficient for training execution claims.

State variables:

- evidence row count
- edge-delta target count
- required view coverage
- topology family coverage
- actor-safe leakage count
- metric range sanity
- target action coverage
- nonzero edge-delta count
- feasibility class diversity

Sensors:

- `evaluate_learning_evidence_quality`
- `scripts/replay/stage7_1_learning_evidence_quality_report.py`
- unit tests for report pass and blocking cases
- contract tests for docs, harness task, script output, source scan, and
  scaffold artifact boundary

Actuators:

- `src/marl_topology/data/learning_evidence_quality.py`
- Stage 7.1 report script
- Stage 7.1 harness task
- PROJECT_STATE update
- post-task self-review

Disturbances:

- treating minimal evidence as training sufficiency
- hiding actor leakage inside dataset rows
- treating oracle candidate diagnostics as actor inputs
- confusing Stage 8 interface readiness with training readiness
- continuing Stage 7 planning after the exit sensor has passed

## Quality Gates

Blocking gates:

- required evidence views are present
- evidence rows exist
- learning targets exist
- actor-safe leakage count is zero
- weak/sparse/dense topology families are covered
- registered objective metrics are in valid ranges
- edge-delta targets cover add, remove, and keep actions
- at least one edge-delta target carries a nonzero objective delta

Warnings:

- no feasible rows at the declared tau requirement
- low feasibility class diversity

The current minimal demo evidence passes blocking gates and raises warnings for
feasibility diversity. That means Stage 7 can close for interface-design
purposes, but the dataset is not sufficient for training execution.

## Current Report Summary

The Stage 7.1 report over the minimal demo evidence returns:

- rows: 4
- edge-delta targets: 48
- actor-safe rows checked: 16
- actor leakage issues: 0
- topology family coverage: weak/disconnected, sparse, dense full graph, and
  oracle-candidate diagnostic
- nonzero edge-delta targets: 24
- blocking issues: 0
- warnings: no feasible rows at tau 0.9 and low feasibility class diversity

## Stage Boundary Decision

Stage 7 is closed for the current control objective: learning evidence exists,
is separated by view, and passes structural quality gates.

Stage 8 may begin only with owner approval and should start with:

```text
stage_8_0_actor_policy_interface_contract_with_owner_approval
```

Training execution remains blocked. A larger data generation task is still
needed before any training claim.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```
