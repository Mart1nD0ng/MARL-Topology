# Stage 21 - Assembler-Aware Supervised Rerun Evaluation

Stage type: implementation-bearing evidence rebuild, target rebuild,
supervised MLP/GNN rerun, and fair same-assembler evaluation.

## Control Summary

Controlled object: objective-stack-aligned evidence, assembler-aware actor
targets, supervised MLP/GNN actor rerun, and fair projected baseline
comparison.

Desired state: evidence uses the Stage 3 finite-blocklength network stack and
Stage 4 expected-initiator PBFT reliability; actor targets have high/mid/low
priority spread; MLP/GNN train on actor-safe Stage 21 views; actors and main
baselines are evaluated through the same `ConflictAwareGreedyAssembler`.

Sensors: Stage 21 report script, Stage 21 unit tests, contract tests, full
pytest, harness validation, and manual gate review.

Actuators: added Stage 21 evidence, target, supervised rerun, evaluation, and
conditional pilot modules; added report script, docs, tests, harness task, and
state update. No `v5` writes, checkpoints, COMA, Transformer, reward-weight
tuning, or final tau selection occurred.

## Evidence Stack Answer

Yes, the Stage 21 main evidence uses the final objective stack:

- evaluator: `stage21_stage3_urlcc_stage4_expected_initiator_pbft_objective_stack_v1`
- physics regime: `urlcc_finite_blocklength_v1`
- protocol model: `stage4_expected_initiator_pbft_over_stage3_network_v1`
- objective contract: `stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1`
- `tau_requirement_min = 0.9`

The Stage 16-20 audit found that prior learning/evaluation lineage still used
`SimpleLinkModel` and the minimal `TopologyEvaluator` min-link abstraction for
main evidence. Stage 21 did not silently fall back to that stack.

Evidence readiness passed:

| Requirement | Result |
|---|---:|
| rows | `70` |
| actor edge samples | `994` |
| tau>=0.9 feasible topology rows | `30` |
| infeasible hard rows | `28` |
| near-threshold rows | `31` |
| sparse feasible rows | `8` |
| full graph baseline rows | `10` |
| projected greedy teacher rows | `10` |
| oracle diagnostic rows | `10` |
| multi-step actor-safe sequence | `time_step 0, 1, 2` |

## Target Quality Answer

The Stage 21 target distribution improved over Stage 20's uniformly high
projection-mismatch pattern.

| Target statistic | Value |
|---|---:|
| actor targets | `994` |
| min / max | `0.06 / 0.94` |
| mean | `0.5299242304` |
| variance | `0.0879605548` |
| high-priority examples | `167` |
| mid-priority examples | `513` |
| low-priority examples | `314` |
| low-priority / abstain signals | `314` |
| pairwise ranking pairs | `685` |
| uniformly high | `false` |

Global edge-delta targets remain critic-only. Hard labels are diagnostic only.
Actor target views do not include consensus probability, global latency,
global energy, reward surrogate, oracle membership, future outcome, or global
edge-delta fields.

## Supervised Rerun

MLP and GNN were rerun on Stage 21 actor-safe features and assembler-aware
soft/ranking targets. GRU/LSTM were not rerun because Stage 21 default scope is
MLP/GNN only.

| Model | Tau-feasible rate | Mean consensus probability | Mean latency | Mean energy | Mean selected edges | Top-proposal rejection |
|---|---:|---:|---:|---:|---:|---:|
| MLP projected | `0.4428571429` | `0.4885714286` | `0.0045270878` | `0.0084548029` | `3.9714285714` | `0.4157142857` |
| GNN projected | `0.4714285714` | `0.4942857143` | `0.0046292877` | `0.0089600965` | `3.9571428571` | `0.5128571429` |

GNN remained better than MLP on tau-feasible rate under the Stage 21 fair
projected evaluation.

## Fair Baselines

Raw baselines were reported as diagnostics only. Main comparison used projected
baselines under the same `ConflictAwareGreedyAssembler`.

| Projected baseline | Tau-feasible rate | Mean consensus probability | Mean latency | Mean energy | Mean selected edges |
|---|---:|---:|---:|---:|---:|
| full graph projected | `0.4571428571` | `0.4914286504` | `0.0041321162` | `0.0080218891` | `4.1` |
| greedy reliability projected | `0.4857142857` | `0.5771428571` | `0.0033432289` | `0.0069258637` | `2.3857142857` |
| random projected | `0.4` | `0.4` | `0.0056322076` | `0.0107534650` | `2.3571428571` |
| sparse heuristic projected | `0.0` | `0.0` | `0.0037806314` | `0.0032849378` | `1.5857142857` |

GNN beat `full_graph_projected` on tau-feasible rate
(`0.4714285714` vs. `0.4571428571`) but did not beat projected greedy
(`0.4857142857`).

## Projection Mismatch Answer

Projection mismatch improved relative to Stage 20 on the top-proposal sensor:

- Stage 20 GNN coarse projection rejection baseline: `0.6647058824`
- Stage 21 GNN top-proposal rejection rate: `0.5128571429`
- Stage 21 GNN accepted-vs-rejected score separation: `0.0584172961`

The above-threshold rejection rate remained high at `0.6889880952`, so the
projection repair is incomplete.

## Policy-Gradient Gate

Policy-gradient pilot is not allowed.

Passed gates:

- objective-stack alignment;
- actor target quality;
- fair assembler evaluation;
- projection mismatch improved;
- boundary safety.

Failed gate:

- actor performance sufficient for pilot.

Failure evidence:

- best actor: `GNN`
- GNN projected tau-feasible rate: `0.4714285714`
- full graph projected tau-feasible rate: `0.4571428571`
- Stage 20 best actor tau-feasible rate: `0.5593220339`
- GNN feasible actor-projected rows: `33 / 70`
- GNN empty collapse rate: `0.0857142857`
- GNN full-graph collapse rate: `0.1714285714`

The actor beat projected full graph but did not improve over the Stage 20 best
actor threshold and did not approach projected greedy closely enough to justify
policy-gradient.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```

Result: `745 passed`; task validation passed for `78` tasks.
