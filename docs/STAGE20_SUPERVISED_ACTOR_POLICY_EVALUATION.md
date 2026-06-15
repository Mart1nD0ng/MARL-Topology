# Stage 20 - Supervised Actor Policy Evaluation With Assembler

Stage type: implementation-bearing evaluation stage.

Controlled object: Stage 19 supervised actor edge scores after environment-side
`ConflictAwareGreedyAssembler` projection.

Desired state: evaluate MLP, GNN, GRU, and LSTM actor scores through the
environment-side topology assembler and registered topology evaluator without
policy-gradient training, critic rerun, checkpoint writes, training artifact
writes, COMA, Transformer, reward-weight tuning, final tau selection, scale-up
training, or `v5` modification.

Source-awareness: this stage is derived from Stage 18 and Stage 19 clean-project
evidence gates and EC-RULE project controls. It does not inherit `v5` phase
scripts, fixed deployment thresholds, reward weights, COMA defaults, or global
actor tensors.

## Cybernetic Control Summary

Controlled object identified: supervised actor score projection into legal
topologies.

Desired state defined: actor outputs remain edge scores only, the assembler
owns hard topology activation, and evaluation metrics are produced after
projection.

State variables defined: actor score, selected directed edge count, selected
physical edge count, rejection reason counts, consensus success probability,
latency, energy, tau feasibility, and projection rejection rate.

Sensors defined: Stage 20 JSON script report, unit tests, contract tests,
source leakage scans, pytest, harness validation, and rubric score.

Actuators defined: Stage 20 evaluation module, report script, docs, harness
task, PROJECT_STATE update, and tests.

Feedback loop present: Stage 19 supervised loss is the baseline; Stage 20
observes post-projection topology quality and blocks policy-gradient readiness
when assembler-evaluated performance is not strong enough.

Verification plan present: run `python scripts\train\stage20_supervised_actor_policy_evaluation.py`,
`python -m pytest -q`, `python harness\scripts\validate_tasks.py`, and rubric
scoring.

Stability risk checked: the supervised actor can score edges well by loss but
still produce resource-heavy or infeasible assembled topologies.

Observability gap checked: Stage 20 observes assembler projection and topology
metrics, but it does not yet test policy rollout under online distribution
shift or multiple seeds.

Controllability checked: only in-memory supervised actor fitting and evaluation
were used; policy-gradient, checkpoint, artifact-writing, reward tuning, and
tau selection actuators remain disabled.

Disturbance checked: small evidence size, single seed, duplicate physical-edge
directions, local resource constraints, and projection conflicts remain active
disturbances.

Delay or async risk checked: no asynchronous service or delayed writer is used.

Noise or flakiness checked: fixed seed `20` is reported as gate evidence, not
final statistical proof.

Decoupling checked: actor scores, assembler projection, baseline/oracle
diagnostics, and critic-only/global targets remain separated.

Reliability or error control checked: negative tests guard against actor
global-field leakage, policy-gradient paths, checkpoint writes, COMA,
Transformer, and `v5`.

Persistent learning update suggested: Stage 21 should refine supervised actor
targets or assembler-aware calibration before any PPO/MAPPO readiness review.

## Evaluation Inputs

- Source evidence: `stage18_disambiguated_actor_evidence_v1`.
- Source actor stack: Stage 19 in-memory supervised actors.
- Actor feature schema: `stage19_stage18_actor_feature_tensor_v1`.
- Feature count: `31`.
- Full-row actor samples: `850`.
- Ranking pairs: `412`.
- Environment-side assembler: `ConflictAwareGreedyAssembler`.
- Tau requirement for diagnostics: `tau_requirement_min = 0.9`.

The deployment assembler consumed only actor scores, candidate validity, local
role metadata, local tx/rx capacity estimates, and local physical-edge conflict
groups. It did not consume oracle labels, reward surrogate, consensus
probability, global objective values, future outcomes, latency, or energy.

## Results

Full Stage 20 run:

| Policy | Rows | Tau-feasible rate | Mean consensus probability | Mean selected edges | Projection rejection rate |
|---|---:|---:|---:|---:|---:|
| MLP | `59` | `0.5423728814` | `0.6689860220` | `4.8135593220` | `0.6658823529` |
| GNN | `59` | `0.5593220339` | `0.6694933940` | `4.8305084746` | `0.6647058824` |
| GRU real multi-step subset | `18` | `0.8333333333` | `0.7714492782` | `4.6111111111` | `0.6157407407` |
| LSTM real multi-step subset | `18` | `0.8333333333` | `0.7714492782` | `4.6111111111` | `0.6157407407` |

Baseline diagnostics over the same Stage 18 rows:

| Baseline | Rows | Tau-feasible rate | Mean consensus probability | Mean energy |
|---|---:|---:|---:|---:|
| full graph baseline | `59` | `0.6101694915` | `0.8197403951` | `0.1721946579` |
| greedy reliability baseline | `59` | `0.8135593220` | `0.8424521143` | `0.0657887825` |
| source topology diagnostic | `59` | `0.5084745763` | `0.5572901495` | `0.0594784166` |
| oracle diagnostic | `54` | `0.8888888889` | `0.9204569397` | `0.0435494604` |

GNN remains the best full-row supervised actor, but the assembler-evaluated
policy does not beat the greedy reliability baseline. The temporal actors look
better on the real multi-step subset, but that subset has only `18` evaluated
rows and is not comparable to the full `59`-row MLP/GNN evaluation.

## Completion Gate

Passed:

- Stage 19 supervised actors were evaluated in memory.
- Actor outputs remained edge scores only.
- Hard topology activation was produced by the environment-side assembler.
- Assembler rejection reasons were recorded.
- Registered consensus, latency, energy, and topology diagnostics were
  computed after projection.
- Full graph was treated as a baseline, not an oracle.
- Oracle output was diagnostic only and never actor or assembler input.
- No PPO/MAPPO, COMA, Transformer, critic rerun, scale-up training,
  checkpoint, training artifact, reward tuning, final tau selection, or `v5`
  modification occurred.

## Next-Stage Readiness Gate

Policy-gradient readiness: failed.

Reason: the best full-row actor after assembler projection is GNN with
tau-feasible rate `0.5593220339`, below the greedy reliability baseline
`0.8135593220`. Projection rejection rate also remains high at
`0.6647058824`.

Recommended next task:

`stage_21_assembler_aware_supervised_target_refinement`

Stage 21 should improve the supervised actor target or calibration against
assembler-projected outcomes before any PPO/MAPPO readiness review is reopened.

## Residual Risks

- Single-seed supervised evaluation is not final evidence.
- The actor often proposes dense/resource-heavy edge sets, causing high
  projection rejection.
- Temporal results are promising but too narrow to choose GRU/LSTM for the
  full policy.
- Stage 20 does not prove deployment performance under online rollout.
