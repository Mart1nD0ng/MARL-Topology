# Stage 2.8 Reward Contract Review

This review aligns reward semantics with the current metric, protocol,
Dec-POMDP, replay, fixture, and link-model regime gates. It does not implement a
reward function and does not authorize training.

## Controlled Object

The controlled object is the future reward-design boundary for:

```text
TopologyEvaluation metrics -> future objective contract -> future training
```

There is no reward code in this stage.

## Desired State

Future reward work can be reviewed without inheriting v5 reward ambiguity:

- consensus reliability is a constraint;
- latency and energy are objectives after reliability is feasible;
- diagnostics remain diagnostics;
- actor observations stay local and do not receive reward, oracle, or metric
  labels;
- all evaluation evidence uses registered metrics.

## Active Decision

Stage 2.8 makes these decisions:

- No active reward function exists.
- No reward weights are declared.
- No training surrogate is registered.
- No new metric is registered.
- No project-wide reliability threshold `tau` is fixed.
- Any future reward proposal must state its threshold, metric mapping,
  aggregation, and tests before implementation.

## Registered Evaluation Metrics

The only active evaluation metrics are:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

`topology_diagnostics` is diagnostic. It may explain a result or support a
declared deterministic tie-breaker, but it is not the main objective.

## Constraint And Objective Semantics

Reliability is a constraint gate. A topology that fails the declared reliability
threshold is not considered feasible merely because it has low latency or low
energy.

Within the feasible region, future reward or objective logic must prefer lower
latency and lower energy. Extra reliability beyond the threshold enters a
plateau unless a future registered metric explicitly studies safety margin or
tail risk.

## Couplings

- `METRIC_CONTRACT.md`: all reported evaluation quantities must be registered.
- `PROTOCOL_CONTRACT.md`: quorum, deadline, timeout, consensus event, and
  consensus probability remain separate.
- `LINK_MODEL_REGIME_REVIEW.md`: reward evidence must name the active physics
  regime and cannot claim cross-regime policy quality from Stage 2 deterministic
  distance links.
- `DEC_POMDP_CONTRACT.md`: reward, oracle labels, future outcomes, and
  evaluation metrics are not deployment actor observations.
- `REPLAY_DATASET_CONTRACT.md`: reward, return, advantage, and value-target
  columns remain unsupported until a future contract admits them as
  training-only.

## Excluded Routes

The following are not allowed as defaults:

- copied v5 reward formulas, weights, penalties, or sentinel constants;
- `P_eff`, soft/hard/legacy mode naming, or derived reliability aliases without
  metric registration;
- timeout or quorum as standalone reward terms;
- full graph or full mask as an implicit target;
- edge count, density, sparsity, or mask retention as primary reward;
- fixed 0.5 deployment thresholding as a reward or policy rule;
- reward values, oracle labels, or evaluation metrics inside actor
  observations.

## Required Future Tests

Before reward implementation, add tests for:

- reliability failure dominates latency and energy improvements;
- above-threshold reliability plateau prevents redundant topology preference;
- latency and energy objectives are nonnegative and separately logged;
- timeout and quorum are not reward aliases;
- diagnostics are not primary objective terms;
- full graph is baseline, not oracle or reward target;
- actor and replay projections exclude reward, oracle, future outcome, and
  evaluation metric columns;
- metric rows remain registered and separate from reward logs;
- reward evidence declares the active physics regime.

## V5 Anti-Inheritance

v5 remains a read-only experience library. This review does not copy old reward
logic, old metric names, old phase scripts, actor/critic architecture, or
training behavior.

## Acceptance

- `REWARD_CONTRACT.md` records the Stage 2.8 no-implementation boundary.
- Harness task `reward_contract_review_stage2_8` exists.
- Contract tests verify that no active reward code, training, or legacy metric
  defaults were introduced.
- Project state marks Stage 2.8 complete and waits for owner decision.

## Cybernetic Review Map

### Controlled Object Identified

Controlled object: the reward-design boundary between registered topology
evaluation metrics and any future training objective. Scope excludes reward
code, actor code, critic code, and training.

### Desired State Defined

Desired state: reward proposals can be reviewed against explicit success
criteria before implementation. The target state is reliability as constraint,
latency and energy as objectives, and registered metric evidence as the
acceptance signal.

### State Variables Defined

State variables: reward implementation status, metric registry impact,
reliability threshold declaration, latency objective, energy objective,
diagnostic-only topology fields, actor input boundary, and replay column status.
Inputs are `TopologyEvaluation` metrics. Outputs are future review decisions,
not code.

### Sensors Defined

Sensors: contract tests, harness validation, static check scans, baseline report
checks, metric registry evidence, and project-state review.

### Actuators Defined

Safe actuators: documentation, harness task metadata, project state, contract
tests, and future config interfaces. Unsafe actuators: reward code, training
runs, v5 writes, old reward copying, and deployment actor schema expansion.

### Feedback Loop Present

Feedback loop: baseline contract review -> bounded documentation change ->
post-change tests -> harness validation -> negative scan -> owner decision.

### Verification Plan Present

Verification commands:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python scripts\replay\baseline_evaluation_report.py
python harness\scripts\score_rubric.py docs\REWARD_CONTRACT_REVIEW.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\reward_contract_review_stage2_8.score.json
```

Expected evidence: tests pass, task validation passes, baseline report remains
no-training/no-v5, and negative scans find no reward implementation.

### Stability Risk Checked

Stability risk: a future scalar reward could optimize redundant topology or
hide failed reliability behind low latency and energy. Regression risk is
controlled by plateau and reward-hacking tests before implementation.

### Observability Gap Checked

Observability gap: no reward scale, threshold selection, or training dynamics
are observable yet because reward is not implemented. Missing sensor:
scenario-specific reward proposal tests.

### Controllability Checked

Controllability: owner approval is the fixed dependency before any next stage.
Safe actuator changes remain limited to contracts and tests.

### Disturbance Checked

Disturbances: v5 reward semantics, legacy metric aliases, timeout/quorum
confusion, deterministic link-regime limits, random future training noise, and
actor information leakage.

### Delay Or Async Risk Checked

Delay risk: project state can become stale after later stages. The timestamped
stage state and post-task self-review gate provide the refresh mechanism.

### Noise Or Flakiness Checked

Noise risk: baseline random rows are seeded smoke checks only; training
variance is not used as evidence in this review.

### Decoupling Checked

Coupling map: metrics, protocol, link regime, Dec-POMDP schema, replay columns,
and oracle reports are consumers of the reward contract. Non-target behavior:
existing evaluator and baseline reports should remain reward-free.

### Reliability Or Error Control Checked

Error control: independent sensors include tests, harness validation, static
negative tests, and baseline report checks. False success risk is treating
diagnostic edge count or full graph as an objective.

### Persistent Learning Update Suggested

Persistent follow-up: keep the `reward_contract_review_stage2_8` harness task
as the gate for future reward proposals, and add a focused test when a concrete
reward candidate is proposed.

## Residual Risks

- The exact scalarization, if any, remains undecided.
- The reliability threshold must be selected by a future task with scenario and
  evaluation context.
- No training-time reward scale or normalization is validated yet.
- Stage 2 deterministic link evidence may not transfer to future physical
  regimes.
