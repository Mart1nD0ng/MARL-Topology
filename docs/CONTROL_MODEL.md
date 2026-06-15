# Control Model

## Controlled Object

The controlled object is the engineering closed-loop system of the MARL-Topology project. It includes contracts, code modules, tests, harness tasks, skills, migration decisions, and verification evidence. It is not a single model, a single training run, or a single simulator component.

Boundary:

- In scope: `D:\PhD_works\MARL-Topology`.
- Read-only legacy reference: `D:\PhD_works\v5`.
- Out of scope for this scaffold: model migration, training, and full 3D physics implementation.

## Desired State

A low-entropy, testable, interpretable, and extensible 3D V2X / PBFT / MARL topology-control project.

The desired state is observable through contract tests, metric snapshots, harness rubric score, migration ledger decisions, and regression checks.

## State Variables

- Code entropy: module boundaries, script growth, duplicate semantics, and unclear ownership.
- Metric governance consistency: stable registration for consensus success, latency, energy, diagnostics, and any future metric.
- Effective-success deferral: no `P_eff` variant is introduced until a clean module needs it and registers it.
- Physical simulation correctness: geometry, LoS/NLoS, path loss, SINR, interference, latency, and energy sanity.
- Dec-POMDP information boundary: actor-local deployment observations versus centralized critic training data.
- Reward contract: reliability as constraint; latency and energy as objectives.
- Topology oracle coverage: known feasible/infeasible topology cases and counterfactual checks.
- Actor/critic fidelity: local actor compliance, centralized critic declarations, checkpoint and dataset feature boundaries.
- Training stability: variance, convergence diagnostics, reward hacking signals, and seed sensitivity.

## Sensors

- Unit tests.
- Contract tests.
- Regression tests.
- Metric snapshots.
- Harness rubric score.
- Migration ledger.
- Entropy audit.
- Simulation sanity checks.
- Topology oracle checks.
- Training diagnostics.

## Actuators

- Documentation contracts.
- Config schema.
- Module interfaces.
- Tests.
- Harness tasks.
- Skills.
- Limited code modules.
- Migration wrappers.

## Disturbances

- `v5` legacy semantic debt.
- Indicator naming ambiguity.
- RL stochasticity.
- 3D scenario randomness.
- Reward hacking.
- Actor global-information leakage.
- Overgrown phase scripts.
- Codex uncontrolled broad edits.

## Coupling Map

| Source | Coupled Target | Risk | Required Sensor |
|---|---|---|---|
| Physics contract | Link metrics and protocol success | Physically invalid reliability | Physics sanity and metric contract tests |
| Protocol contract | Reward and evaluation | Timeout/quorum/reward/metric conflation | Protocol contract tests |
| Metric contract | CSV exports and dashboards | Ambiguous comparison across modes | Field-schema tests |
| Reward contract | Training behavior | Reward hacking or proxy optimization | Reward review and oracle replay |
| Actor observation schema | Policies and replay buffers | Deployment information leakage | Dec-POMDP leakage tests |
| V5 learning ledger | Clean skeleton decisions | Unreviewed legacy semantics | Layer-specific contracts and tests |
| Harness tasks | Future Codex work | Open-loop edits | Rubric and task validation |

## Feedback Loops

- Plan -> implement -> test -> audit -> update skill/harness.
- Physics formula -> unit test -> scenario sanity check -> regression.
- Reward proposal -> oracle check -> policy replay -> training.
- Actor architecture -> local observation audit -> training diagnostics -> deployment evaluation.

## Acceptance Criteria

- Harness score is greater than or equal to the target threshold for the task.
- All contract tests pass.
- No unreviewed `v5` code migration or structural inheritance.
- No ambiguous or unregistered metric names.
- No Dec-POMDP violation.
- No reward implementation without metric governance and evaluation contract.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json
```

## Residual Risks

- Contracts are present but not yet backed by full simulator implementations.
- Harness scoring is keyword-assisted and requires human review for final judgment.
- Legacy components remain unaudited until `docs/MIGRATION_LEDGER.md` entries are resolved.

## Rubric Evidence Map

Source-awareness: this project applies software-engineering rules adapted by analogy from the local control-codex-kit, including EC-RULE-001, EC-RULE-004, EC-RULE-007, EC-RULE-010, EC-RULE-013, EC-RULE-028, and EC-RULE-034.

- controlled object identified: the controlled object, boundary, scope, subsystem, and object are named in the Controlled Object section; evidence is this control model and task plan.
- desired state defined: desired state, target state, success criteria, acceptance criteria, and metric targets are defined in Desired State and Acceptance Criteria.
- state variables defined: state variables are listed separately from inputs and outputs; evidence is the State Variables model section and future debug notes.
- sensors defined: sensors include test list, log query, metric name, trace field, static check, and evidence sources.
- actuators defined: safe actuator options include editable docs, config, interface, adapter, tests, skills, and harness tasks; diff scope and ownership notes are required per task.
- feedback loop present: every task records baseline before change, post-change evidence after change, command output summary, and CI or local test result where available.
- verification plan present: verification commands, test plan, rubric checks, expected evidence, and final summary are required.
- stability risk checked: stability, regression, rollback, oscillation, scope, and risk are tracked in the risk section and regression tests.
- observability gap checked: observability gap, blind spot, missing sensor, instrumentation, and gap list must be reported when behavior cannot be measured.
- controllability checked: controllability, safe actuator list, approval note, unsafe surfaces, and fixed dependency boundaries are checked before edits.
- disturbance checked: disturbance list includes external service, timing, environment, constraint, random scenario, and legacy semantic debt.
- delay or async risk checked: async, delay, stale result, queue, cache, CI timing, timestamp, and commit SHA freshness must be considered for delayed sensors.
- noise or flakiness checked: noise, flaky tests, random seeds, variance, repeat count, historical run data, and statistical interpretation are required for stochastic work.
- decoupling checked: coupling map, impact matrix, non-target regression, consumer, side effect, and regression checks are required for shared modules.
- reliability or error control checked: reliability, independent secondary check, redundant evidence, negative test, false success, rollback plan, and isolation are required for high-risk outputs.
- persistent learning update suggested: repeated lessons must be proposed as AGENTS.md, skill, harness backlog, or follow-up updates, not hidden in one-off task notes.
