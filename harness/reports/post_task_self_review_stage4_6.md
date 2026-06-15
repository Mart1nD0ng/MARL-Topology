# Post-Task Self-Review

## Completed Task

Stage 4.6 - protocol latency and energy accounting review.

## Intended Desired State

PBFT/application reliability, protocol latency, and protocol energy should stay
separate. Stage 4.6 should account latency and energy from Stage 3 network
communication records grouped by PBFT phase, while Stage 4.2/4.4 remain the
only producers of PBFT reliability.

## Actual Achieved State

Implemented `stage4_pbft_protocol_accounting_v1` in
`src/marl_topology/protocol/pbft_accounting.py`.

Stage 4.5 now uses Stage 4.6 accounting for its `latency` and `energy` fields
instead of the earlier review-only `sum_all_directed_pair_records_times_three_phases`
placeholder.

## Evidence

- Protocol accounting public interfaces exist and are exported.
- Stage 4.6 documentation records latency and energy accounting semantics.
- Protocol and metric contracts record that Stage 4.6 does not add metric
  names and does not export consensus probability.
- Stage 4.5 replay output includes `stage4_pbft_protocol_accounting_v1` under
  `topology_diagnostics`.
- Source scan over Stage 4.6 active source files found no v5 import, training,
  actor, critic, COMA, MAPPO, old metric alias, or reward implementation route.

## Tests

- `python -m pytest -q` -> `276 passed in 1.26s`
- `python harness\scripts\validate_tasks.py` -> `Task validation passed: 37 tasks`
- `python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json` -> passed
- `python scripts\replay\stage4_5_baseline_oracle_review.py` -> printed JSON report only

## Gates Passed

- metric governance gate
- consensus/protocol naming gate
- Stage 4.5 baseline/oracle review gate
- Stage 4.6 protocol accounting gate
- reward boundary gate
- training precondition gate
- v5 read-only boundary
- post-task self-review gate

## Gates Deferred

- success-conditioned latency/energy accounting
- detailed PBFT scheduler semantics
- multi-scenario Stage 4 application evaluation report
- reward objective implementation
- training and actor/critic gates

## New Risks

- `phase_max_clipped_to_budget` is a deterministic review sensor, not a full
  semi-asynchronous scheduler.
- Scheduled-attempt energy can be conservative for failed messages but is not
  success-conditioned energy.
- Stage 4.5 still uses a small deterministic reference scenario.

## Regressions

Protected non-target behavior:

- PBFT reliability still uses Stage 4.4 expected-initiator averaging.
- Stage 4.3 still maps Stage 3 delivery into phase matrices without exporting
  consensus metrics.
- Full graph remains a baseline, not an oracle.
- No training, model, reward, or v5 migration path was introduced.

## Candidate Next Tasks

- Stage 4.7 - PBFT application evaluation report over reliability, protocol
  latency, protocol energy, and baseline/oracle-candidate diagnostics.
- Stage 4.6a - protocol accounting boundary fixtures for late messages,
  zero-delivery attempted energy, and empty phases.
- Stage 5.0 - reward/objective contract freeze after Stage 4 evaluation report.

## Recommended Next Task

Stage 4.7 - PBFT application evaluation report.

Reason: Stage 4.6 now makes protocol-level latency and energy accounting
explicit. The next useful sensor is a report that jointly presents registered
reliability, latency, energy, and topology diagnostics before reward design.

## Owner Decision Required

Yes. Codex recommends Stage 4.7 but must not execute it without explicit owner
approval.
