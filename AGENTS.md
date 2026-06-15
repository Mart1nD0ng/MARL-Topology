# MARL-Topology Codex Operating Rules

This project follows an Engineering Cybernetics workflow: every non-trivial task must name the controlled object, desired state, state variables, sensors, actuators, disturbances, coupling, feedback loop, acceptance criteria, verification commands, and residual risks.

The stable default rules are adapted from `D:\PhD_works\Engineering Cybernetics\control-codex-kit\AGENTS.md`; project-specific MARL/V2X rules below take precedence inside this repository.

## Project Boundary

- New project: `D:\PhD_works\MARL-Topology`.
- Legacy reference: `D:\PhD_works\v5`.
- `v5` is read-only legacy reference. Do not modify, format, move, delete, or run migration-producing writes inside `v5`.
- The current scaffold is for controlled learning from legacy experience only. `v5` is an experience library, not a code template.

## Repository Consolidation (Production Main Body)

As of the 2026-06-15 consolidation, the repository is no longer an open scaffold of parallel
experiments. It has been slimmed to a single production main body. Treat the following as the
canonical surface; do not reintroduce retired alternatives into `src/` without an owner decision.

- One production deployment actor: the v3 residual-norm local message-passing GNN edge scorer
  (`local_message_passing_gnn_edge_scorer_v3_residual_norm`). It is the only registry entry gated
  active for production. The local MLP edge scorer and the v2 / role-resource GNN remain only as
  registered diagnostic baselines the production actor is measured against.
- One production MARL training flow: the Stage 33 CTDE adapter in
  `training/production_mappo_adapter.py` (behaviour-cloning warm start, Stage 27 graph value-critic
  pretraining, then a clipped on-policy actor-critic fine-tune). The `mappo/` and `policy_gradient/`
  helper modules it imports are part of this one flow, not separate pipelines.
- One production objective stack evaluator: `data/stage21_objective_stack_evidence.py` over the full
  urban 3D V2X physics. The distance-only `MinimalDecPOMDPEnv` / `topology/evaluator.py` tier is a
  declared Stage-2 skeleton kept for baselines and leakage tests only.
- Retired (recoverable from git tag `v0-full-import`): the recurrent (GRU/LSTM) edge scorers and the
  temporal and attention GNN variants; the pre-MARL supervised warm-start lineage and the Stage 21->23
  supervised fair-evaluation gate; the archived custom training loop and the readiness probe; and the
  superseded Stage 26/29/30 stage diagnostics — each removed together with its tests and driver scripts.
- The recurrent-scorer removal also clears the previously-present forbidden `nn` recurrent-cell token
  from `src/`.

Contract-coverage invariant (still in force, see Migration Rule 11): a module and the tests that pin
it are retired as a set. A kept production module must keep its contract/boundary test coverage. See
`docs/CLEAN_PROJECT_MAP.md` for the current env / configuration / architecture map.

## Mandatory Task Loop

For every task that changes files or design:

1. Baseline: capture existing state, assumptions, relevant files, and known gaps.
2. Plan: define controlled object, desired state, sensors, actuators, disturbances, coupling map, and acceptance criteria.
3. Change: keep the edit scope bounded and avoid unrelated churn.
4. Post-change evidence: run tests, contract checks, harness checks, static review, or a documented manual check.
5. Regression check: name the non-target behavior protected by the change.
6. Residual risk: state what remains unverified and the next sensor or actuator needed.

## Post-Task Self-Review

- Codex must self-review after every substantial task using the post-task self-review structure.
- The self-review must include completed task, intended desired state, actual achieved state, evidence, tests, gates passed, gates deferred, new risks, regressions, candidate next tasks, recommended next task, and owner decision required.
- Codex may recommend next actions and update project-state recommendations.
- Codex must not self-authorize the next stage or next task.
- User approval is required before executing the next task, even when Codex recommends it.

## Stage Closure Discipline

- Do not split one stage into unlimited internal planning or review tasks.
- Use substeps only when they remove a concrete blocker or add a necessary sensor.
- Once an exit gate is defined and passes, close the stage instead of opening more same-stage planning tasks.
- Remaining work after a passed exit gate must become a next-stage owner-approved task or a clearly scoped defect fix.

## Stage Planning Rules

### Stage closeout rule

Each completed stage must:

- update `docs/PROJECT_STATE.md`;
- run tests or document why a required test is deferred;
- run `python harness\scripts\validate_tasks.py`;
- produce post-task self-review;
- update or register the relevant `harness/tasks` entry;
- declare whether the Completion Gate passed;
- declare whether the next-stage readiness gate passed.

### No consecutive planning-only rule

If the current stage and previous stage are both primarily documentation,
planning, contract, readiness, or review work with no runnable deliverable,
Codex must stop and request owner decision unless the current stage removes a
specific named blocker.

### Next-stage readiness rule

Every stage closeout must answer:

- Completion Gate passed?
- Next-stage readiness gate passed?
- If not, what blocker remains?
- What could make the next stage invalid?
- What evidence says the next stage is now safe?

Passing tests is required evidence, but never the only exit condition.

### Stage state sync rule

If a stage has been implemented but is absent from `PROJECT_STATE.md` or the
harness task registry, it is `implemented_unregistered`, not closed. Do not use
an implemented-unregistered stage as the basis for another stage until state,
harness, and self-review are synchronized.

### Scope compression rule

When multiple small stages are merely substeps of one objective, combine them
into one implementation-bearing stage with internal checkpoints. Do not create
new stage numbers for plan-of-plan loops.

### Owner checkpoint rule

Codex can recommend a next step, but cannot continue if `PROJECT_STATE.md` is
inconsistent, the harness task is missing, or next-stage readiness is unknown.

## Migration Rules

1. `v5` is read-only legacy reference.
2. Inherit the goal, not the structure.
3. Consult `v5` only when a clean skeleton layer has a concrete design question.
4. Do not migrate legacy code without a clean module need, contract, test, and owner decision.
5. Do not copy the old reward as the new reward.
6. Do not import old metric names without metric registration.
7. Do not conflate consensus success, latency, energy, topology diagnostics, reward, timeout, or quorum.
8. Actor policies must preserve the Dec-POMDP local observation constraint.
9. Critics may use centralized training information, but deployment actors must not receive global information.
10. Every task must report baseline, change, and post-change evidence.
11. Every module must have a contract test before it becomes relied upon.
12. Large changes require `cybernetic-project-analysis` or `cybernetic-harness-design` first.
13. Do not grow long-term functionality in phase scripts. Durable behavior belongs in `src/marl_topology/` modules.
14. Training tasks must first declare reward, metric, and evaluation contracts.

## Metric Governance

- Current minimal metric concepts are `consensus_success`, `consensus_success_probability`, `latency`, `energy`, and `topology_diagnostics`.
- Any new metric must be registered with name, definition, range/unit, level, used_for, formula source, dependencies, and tests.
- Reward names and metric names must not be mixed unless they are explicitly the same registered quantity.
- Diagnostic metrics must not be presented as primary objectives.
- Core metrics must not be privately reimplemented in scripts.
- Future effective-success, soft/hard, tail-risk, or aggregate metrics are allowed only after registration and tests.

## Dec-POMDP Boundary

- Actor inputs may include only local observations, local history, local messages permitted by the scenario, and agent identity or type if declared in the observation contract.
- Actor inputs must not include global graph state, complete topology, future events, centralized labels, full consensus outcome, or evaluation oracle outputs.
- Training critics may use centralized state only through declared training-only interfaces.
- Tests must detect accidental deployment leakage from critic features, replay buffers, dataset columns, and model checkpoints.

## Engineering Cybernetics Defaults

- Model before substantial work. Name the controlled object, boundary, desired observable state, and acceptance signal.
- Use feedback when available. Do not claim success without a sensor signal.
- Keep edit scope bounded and switch strategy after repeated failed repair loops.
- Keep stage scope convergent. Avoid endlessly subdividing a stage when the available evidence is sufficient to close it.
- Treat observability gaps as findings.
- Identify safe actuators before changing behavior.
- Check coupling and non-target regressions for shared behavior.
- Optimize only against explicit metrics and constraints.
- Use stronger evidence for high-risk outputs.

## Verification Commands

Default scaffold checks:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json
```

Do not run training, legacy `v5` scripts, or GPU workloads unless a future task explicitly authorizes them and the contracts exist.
