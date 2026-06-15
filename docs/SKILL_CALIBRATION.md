# Skill Calibration Contract

This contract prevents project skills from carrying v5 error patterns into MARL-Topology. Skills are allowed to distill lessons from v5, but they must not turn v5 code structure, metric names, reward logic, phase scripts, or architecture choices into defaults.

## Controlled Object

The controlled object is the local skill system under `.agents/skills/` plus the harness tasks and tests that route future Codex work.

## Desired State

Every project skill should:

- Preserve the clean skeleton direction from `docs/GOAL_SKELETON.md`.
- Treat `D:\PhD_works\v5` as a read-only experience library.
- Apply `docs/METRIC_CONTRACT.md` as metric governance, not as a license to reintroduce metric catalogs.
- Use `docs/V5_FAILURE_LESSONS.md` and `docs/V5_DO_NOT_LEARN_BLINDLY.md` as negative controls.
- Convert any v5-inspired idea into a clean-project contract, test, or harness gate before implementation.

## State Variables

- Skill entropy: repeated or conflicting instructions across project skills.
- Metric-governance drift: use of old metric names or unregistered CSV fields.
- V5 lesson status: whether a v5 claim is `seed`, `confirmed`, `contradicted`, or `unresolved`.
- Boundary leakage: actor, critic, oracle, replay, and deployment information boundaries.
- Harness coverage: whether a skill recommendation has a test, negative check, rubric, or manual sensor.
- Debug noise: whether evidence comes from later-confirmed records or noisy debug hypotheses.

## Actuators

- Editable skill text in `.agents/skills/*/SKILL.md`.
- The calibration contract in this document.
- Harness tasks and negative checks.
- Contract tests that fail when a project skill omits calibration.
- Documentation links to `GOAL_SKELETON`, metric governance, and v5 failure lessons.

## Disturbances

- Legacy v5 semantic debt around reliability metrics, reward shaping, phase scripts, actor inputs, and credit assignment.
- Stale debug hypotheses that were later contradicted.
- Overfitting new skills to old result directories or phase reports.
- Random training outcomes being mistaken for design evidence.
- Future edits that add a new skill but forget the calibration gate.

## Coupling Map

| Skill area | Coupled risk | Required decoupling |
| --- | --- | --- |
| Metric and protocol skills | Derived reliability names can hide timeout and quorum semantics | Register metrics and keep protocol conditions separate |
| Reward and training skills | Shaped reward can hide registered evaluation failure | Report registered metrics separately from reward |
| Dec-POMDP and credit skills | Critic-only global tensors can leak into actor deployment | Negative leakage tests and separate schemas |
| Oracle and baseline skills | Full-mask can be confused with feasibility proof | Empty/full/sparse/oracle baselines with `unresolved` state |
| Entropy and harness skills | Phase scripts can become durable structure | Durable behavior goes to `src/marl_topology/` only after tests |

## Feedback Loop

1. Baseline: inventory skill text and scan for stale metric, reward, threshold, full-mask, phase, actor, and COMA assumptions.
2. Change: apply the smallest skill or harness edit that blocks the inheritance path.
3. Post-change: run contract tests and harness validation.
4. Regression: verify domain skills still give actionable output formats and do not block legitimate future design work.
5. Persistent learning: update this contract or a targeted skill when a repeated failure pattern appears.

## Global Calibration Gate

Before a project skill recommends design, code, migration, training, or evaluation work, it must check:

1. Is this suggestion tied to a current `GOAL_SKELETON` layer?
2. Does it introduce a new metric name, CSV field, reward component, actor input, or oracle output?
3. If yes, is that quantity registered or explicitly marked as future work?
4. Does any part depend on old v5 reward logic, old metric names, phase scripts, fixed 0.5 deployment thresholding, full-mask optimality, COMA/Q defaults, or global actor inputs?
5. If it uses v5 evidence, is the lesson status `seed`, `confirmed`, `contradicted`, or `unresolved`?
6. Is there a named test, negative check, harness gate, or manual sensor before the recommendation is accepted?

## Anti-Inheritance Rules

Skills must reject these as default advice:

- Importing `P_succ`, `P_eff`, hard/soft/legacy modes, log/geometric/arithmetic summary names, or any v5 CSV field without metric registration.
- Copying old reward formulas, weights, timeout rewards, quorum rewards, or edge-count objectives.
- Treating full-mask as an oracle, upper bound, or resource optimum.
- Treating fixed `0.5` thresholding as deployment policy rather than a baseline.
- Treating COMA, Q critics, direct edge-delta critics, LSTM, GNN, or GNN+LSTM as default architecture before oracle and leakage gates.
- Using global topology, oracle labels, future trajectory fields, or critic-only tensors as deployment actor inputs.
- Promoting phase scripts into durable project structure.
- Turning noisy debug hypotheses into design rules without status and evidence.

## Required Skill Output Additions

When relevant, skill outputs should include:

```text
V5 inheritance check:
Metric registry impact:
Dec-POMDP boundary impact:
Oracle/baseline evidence:
Forbidden defaults avoided:
Required negative test:
Lesson status:
```

Use `not applicable` instead of omitting a field when the task is near a known v5 failure mode.

## Sensors

- `rg` scans for stale metric, reward, phase, threshold, full-mask, oracle, actor, and COMA assumptions.
- Contract tests that ensure each project skill includes this calibration gate.
- Harness validation for task fields and negative checks.
- Rubric scoring for closed-loop control evidence.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\SKILL_CALIBRATION.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\skill_calibration.score.json
```

## Regression Checks

- Non-target behavior: skills must still support future clean implementations after contracts exist.
- Non-target behavior: v5 can still be read for evidence, but only as read-only learning input.
- Non-target behavior: metric governance may still register future derived metrics when a clean module needs them.

## Observability Gaps

- Static tests cannot prove that future Codex runs obey every skill.
- Rubric scores are conservative keyword screens and require human review.
- Some v5 evidence remains unresolved and needs new clean-project sensors before adoption.

## Controllability Notes

- Safe actuators are documentation, skill text, harness tasks, and contract tests.
- Unsafe actuators are v5 writes, training runs, old reward copying, and phase-script promotion.
- Fixed dependency: v5 evidence cannot be changed; it can only be cited, contradicted, or rejected.

## Delay, Noise, And Flakiness

- Debug records can be stale or contradicted by later audits.
- Training evidence is noisy and must not override missing contracts.
- Repeated or statistical evidence is required before promoting training lessons into skills.

## Reliability And Error Control

- Use independent sensors: a text scan plus a contract test plus harness validation.
- Include negative tests for known v5 failure routes.
- Treat missing calibration as a false-success risk because the skill may still sound plausible.

## Source-Awareness

These rules are software-engineering controls derived by analogy from the local Engineering Cybernetics workflow and from read-only v5 learning audits; they are not direct control-theory doctrine.

## Acceptance Criteria

- Every local project skill references this calibration contract.
- Every local project skill has a `V5 Anti-Inheritance Calibration` section.
- Skill wording preserves minimal metric governance and does not re-promote old `P_eff` variants as defaults.
- New harness tasks or reports can audit skill calibration without running training or v5 scripts.

## Residual Risks

- A calibrated skill can still be misused if a future task ignores its output format.
- Static tests cannot prove semantic correctness; they only prevent missing guardrails.
- Some v5 lessons remain unresolved and must not be promoted until a clean-project module needs the decision.

## Rubric Evidence Map

- controlled object identified: control model scope is `.agents/skills/`, harness tasks, and tests.
- desired state defined: acceptance criteria, test names, and metric target are listed above.
- state variables defined: state / inputs / outputs are separated in the state variable section.
- sensors defined: test list, metric name, static check, and evidence sources are listed.
- actuators defined: plan, diff scope, editable files, interface, and ownership notes are bounded.
- feedback loop present: baseline, post-change, before, after, and observe steps are named.
- verification plan present: verification commands and expected evidence are explicit.
- stability risk checked: regression, scope, rollback-by-patch, and risk are identified.
- observability gap checked: observability gap, blind spot, missing sensor, and instrumentation limits are named.
- controllability checked: safe actuator, unsafe actuator, approval boundary, and fixed dependency are named.
- disturbance checked: disturbance, environment, timing, and constraint are named.
- delay or async risk checked: stale debug records, CI/rubric delay, and timestamp sensitivity are named.
- noise or flakiness checked: noise, random training variance, repeat evidence, and statistical evidence are named.
- decoupling checked: coupling map, non-target behavior, impact matrix, consumer, and side effect are named.
- reliability or error control checked: reliability, independent sensor, negative test, false success, and isolation are named.
- persistent learning update suggested: persistent learning, skill update, lesson update, and harness backlog are named.
