# Stage 22 - Action Semantics A/B Full GNN Repair

Stage type: implementation-bearing action-semantics A/B trial, objective-aware
teacher repair, full message-passing GNN replacement, supervised MLP/GNN
rerun, fair projected evaluation, and low-entropy cleanup.

## Control Summary

Controlled object: actor score semantics, assembler projection, objective-stack
evaluation, and active model/action registries.

Desired state: compare `directed_outgoing_v1` and
`undirected_physical_link_v1` under Stage 3 URLLC finite-blocklength
communication, Stage 4 expected-initiator PBFT reliability, and the Stage 5
tau/latency/energy objective contract; select one active semantics; archive
the loser; replace the toy GNN with full local message passing.

Sensors: Stage 22 report script, Stage 22 unit tests, contract tests, full
pytest, harness validation, and project-state closeout.

Actuators: added action-semantics registry, physical-link assembler, Stage 22
evidence/teacher/evaluation modules, full GNN replacement, tests, docs,
harness task, report script, and state update.

## A/B Result

`undirected_physical_link_v1` won. `directed_outgoing_v1` is archived as a
Stage 22 diagnostic path and is not returned by the active action-semantics
registry.

| Semantics | Full GNN tau-feasible | Mean consensus | Mean latency | Mean energy | Mean edges | Top rejection |
|---|---:|---:|---:|---:|---:|---:|
| `directed_outgoing_v1` | `0.0` | `0.0` | `0.0046805675` | `0.0048843056` | `3.8` | `0.67` |
| `undirected_physical_link_v1` | `0.7` | `0.7` | `0.0069618955` | `0.0140559876` | `4.0` | `0.22` |

Stage 20 historical metrics were diagnostic only and were not a hard gate.

## Evidence Stack

Both options used final-stack evidence:

- physics: `urlcc_finite_blocklength_v1`
- protocol: `stage4_expected_initiator_pbft_over_stage3_network_v1`
- objective: `stage5_tau_0_9_reliability_constraint_latency_energy_objective_v1`
- no `SimpleLinkModel` or min-link skeleton fallback

## Teacher And Targets

The objective-aware teacher selects by tau feasibility, then latency, then
energy, then sparsity. Teacher candidates are projected before selection, so
actor targets are legal under the option-specific assembler.

| Semantics | Teacher tau | Target high/mid/low | Ranking pairs | Uniformly high |
|---|---:|---:|---:|---|
| `directed_outgoing_v1` | `0.0` | `0 / 38 / 104` | `36` | `false` |
| `undirected_physical_link_v1` | `0.7` | `12 / 23 / 107` | `44` | `false` |

## Fair Baselines

Main comparisons used projected baselines under the same semantics-specific
assembler. Raw full/greedy baselines are not selection gates.

For the selected `undirected_physical_link_v1` path:

| Policy | Tau-feasible | Mean consensus | Mean latency | Mean energy | Mean edges |
|---|---:|---:|---:|---:|---:|
| full GNN actor | `0.7` | `0.7` | `0.0069618955` | `0.0140559876` | `4.0` |
| MLP actor | `0.7` | `0.7` | `0.0069618867` | `0.0135209513` | `4.0` |
| projected greedy reliability | `0.5` | `0.74` | `0.0030601904` | `0.0067388538` | `2.5` |
| projected full graph | `0.0` | `0.16` | `0.0049817850` | `0.0078066673` | `3.2` |
| objective-aware teacher | `0.7` | `0.7` | `0.0025201394` | `0.00672` | `3.0` |

## Gate Result

Stage 22 passed. policy-gradient was not run in Stage 22 and remains
owner-gated for Stage 23 readiness review.

Verification:

```powershell
python scripts\train\stage22_action_semantics_ab_full_gnn_report.py
python -m pytest -q
python harness\scripts\validate_tasks.py
```
