# Stage 4.0 PBFT / Application Consensus Contract Planning

## Controlled Object

The controlled object is the Stage 4 application consensus planning layer that
will consume Stage 3 communication records and produce registered consensus
metrics.

This is a contract and implementation plan only. It does not implement PBFT,
does not implement reward, does not train models, and does not migrate v5 code.

## Desired State

MARL-Topology needs a computationally controlled analytic reliability model for
PBFT-like consensus over heterogeneous communication topologies.

The desired state is:

- consensus semantics are separate from Stage 3 network delivery;
- PBFT three-phase communication is represented explicitly;
- semi-asynchronous deadline budgets are declared by phase;
- heterogeneous link/message success probabilities are supported;
- the calculation avoids subset enumeration, sampling prediction, and Monte
  Carlo simulation;
- the output maps to the registered `consensus_success_probability` concept;
- binary `consensus_success` remains a threshold/deadline event, not a reward.

## Stage Boundary

Stage 4 defines application/protocol consensus semantics. It may consume:

- selected topology;
- Stage 3 route or broadcast communication records;
- phase-specific message delivery probabilities;
- phase latency budgets;
- committee size and fault-tolerance parameters.

Stage 4 must not:

- change Stage 3 channel/link/network records;
- introduce reward terms or reward weights;
- train actor, critic, COMA, GNN, or LSTM models;
- copy v5 code or phase scripts;
- report legacy effective-success aliases;
- treat full graph as oracle.

## V5 Reference Used As Learning Only

Read-only v5 references:

- `D:\PhD_works\v5\core\consensus.py`
- `D:\PhD_works\v5\doc\phase35_effective_success_semantics.md`
- `D:\PhD_works\v5\result_save\phase34_effective_success_and_topology_feasibility\effective_success_audit_summary.json`

Useful ideas to distill:

- PBFT reliability should be a three-stage cascade.
- Heterogeneous link probabilities require a heterogeneous quorum calculation.
- Log-space or otherwise stable probability arithmetic is needed.
- Byzantine tolerance should be explicit.
- Deadline and quorum gates need tests.

Rejected inheritance:

- old metric names and mode taxonomy;
- old reward shaping;
- old phase-script structure;
- training surrogates as final evaluation metrics;
- v5's quorum formula as a silent default.

## Protocol Variant

Planned variant id:

```text
stage4_pbft_three_phase_closed_form_v0
```

Default PBFT assumptions:

- `n >= 3f + 1`
- initiator-as-primary with uniform primary distribution for topology-level
  reliability;
- a fixed primary is only a primary-specific helper or diagnostic;
- no view-change model in v0;
- authenticated messages;
- independent message delivery conditional on Stage 3 communication records;
- byzantine nodes may withhold or corrupt messages;
- worst-case byzantine filtering is applied before quorum-tail aggregation when
  faulty identities are unknown.

Quorum conventions:

- total certificate quorum: `2f + 1`
- external message quorum when local self-message is counted: `2f`
- global commit success: at least `2f + 1` replicas reach committed-local
  state by the round deadline.

Any deviation must create a new named protocol variant.

## Three-Phase Cascade

The model uses three PBFT communication phases:

1. `pre_prepare`: primary sends request digest to replicas.
2. `prepare`: replicas that accepted pre-prepare multicast prepare messages.
3. `commit`: replicas that reached prepared-local multicast commit messages.

For each phase, Stage 4 builds a directed message-delivery matrix:

```text
M_phase[u, v] = probability message from u to v is delivered within the phase budget
```

The matrix is derived from Stage 3 communication records and the
deadline-conditioned delivery contract. It is not a consensus metric by itself.

## Closed-Form Heterogeneous Quorum Tail

For independent heterogeneous Bernoulli message probabilities
`x_1, ..., x_m`, define:

```text
H_ge_k(x_1, ..., x_m) =
    sum_{r=k..m} coefficient[z^r] product_j ((1 - x_j) + x_j z)
```

This is the exact probability that at least `k` of the `m` messages arrive.

Implementation rule:

- evaluate the generating-polynomial coefficients using a stable truncated
  coefficient evaluator;
- do not enumerate subsets;
- do not assume identical probabilities unless a named homogeneous test case
  is being checked;
- do not use Monte Carlo or sampled prediction.

The evaluator may use log-space arithmetic for numerical stability. This is an
analytic coefficient evaluation, not a simulation.

## Byzantine Filtering

If faulty identities are unknown, Stage 4 v0 uses conservative filtering:

```text
filtered_inputs = remove the f largest incoming probabilities before H_ge_k
```

Reason: an adversary could occupy the most reliable sender positions. This is
conservative and should be tested against no-fault and known-fault variants.

If faulty identities are known in a fixture, those nodes are removed directly
instead of using worst-case filtering.

## Planned Reliability Formula

Let:

- `p` be the primary id;
- `B` be the set of backup replicas;
- `M0` be pre-prepare delivery matrix;
- `M1` be prepare delivery matrix;
- `M2` be commit delivery matrix;
- `F_i(S)` be the byzantine-filtered candidate probabilities for receiver `i`;
- `H_ge_k` be the heterogeneous quorum tail above.

Pre-prepare readiness:

```text
alpha_1[p] = 1.0
alpha_1[i] = M0[p, i] for i != p
```

Prepared-local probability for replica `i`:

```text
prepare_inputs_i[j] = alpha_1[j] * M1[j, i], j != i
alpha_2[i] = alpha_1[i] * H_ge_{2f}(F_i(prepare_inputs_i))
```

Committed-local probability for replica `i`:

```text
commit_inputs_i[j] = alpha_2[j] * M2[j, i], j != i
alpha_3[i] = alpha_2[i] * H_ge_{2f}(F_i(commit_inputs_i))
```

Primary-specific probability:

```text
Psi_p = H_ge_{2f + 1}(global_filter(alpha_3))
```

Topology-level expected initiator reliability:

```text
R_consensus(G, s) = (1 / |V|) * sum_p Psi_p(G, s)
```

This reflects PBFT's three communication rounds without enumerating success
subsets. The formula is a mean-field approximation because sender readiness and
receiver events are correlated through shared messages. Fixtures must record
this approximation boundary.

The active topology-level model id is:

```text
pbft_expected_initiator_mean_field_v1
```

`fault_filter_remove_largest` is a conservative engineering lower-bound
approximation. It is not a strict Byzantine adversary model.

## Semi-Asynchronous Deadline Semantics

Stage 4 v0 is partial-synchrony aware but does not model view change.

Inputs:

- `pre_prepare_budget_s`
- `prepare_budget_s`
- `commit_budget_s`
- `round_deadline_s`

For deterministic Stage 3 link/network records:

```text
M_phase[u, v] = delivery_probability[u, v]
                if phase_latency_s[u, v] <= phase_budget_s
                else 0.0
```

For Stage 3.6 retry-enabled links:

```text
K = floor(deadline_s / attempt_duration_s)
deadline_delivery_probability = 1 - (1 - packet_success_probability) ^ K
M_phase[u, v] = deadline_delivery_probability[u, v]
```

This is where communication reliability becomes explicitly dependent on latency.

## Latency And Energy Planning

Stage 4 must keep three quantities separate:

- `consensus_success_probability`: analytic probability of PBFT round success;
- `latency`: declared protocol round latency or deadline budget use;
- `energy`: declared scheduled, attempted, expected, or success-conditioned
  protocol energy.

Recommended v1 reporting:

- `latency` should use deterministic phase schedule latency, not zero on
  failed consensus.
- `energy` should start as scheduled or expected message energy, explicitly
  named in diagnostics.
- reward must not be implemented until a separate reward contract admits these
  semantics.

## Metric Registry Impact

No new metric name is introduced in Stage 4.0.

The planned analytic probability maps to the existing governance concept:

```text
consensus_success_probability
```

Required registration details before implementation:

- definition: analytic PBFT three-phase round success probability under the
  named protocol variant;
- range/unit: `[0, 1]`;
- level: round or topology evaluation;
- used_for: reliability constraint and evaluation;
- formula source: this contract plus implementation note;
- dependencies: Stage 3 message-delivery matrices, `n`, `f`, phase budgets,
  independence assumptions, byzantine filtering mode;
- tests: homogeneous sanity, heterogeneous monotonicity, quorum boundary,
  deadline gating, no Monte Carlo, and no subset enumeration.

## Required Implementation Plan

### Stage 4.1: Quorum Tail Utility

Implement only the heterogeneous quorum-tail coefficient evaluator.

Implementation status:

- complete in `src/marl_topology/protocol/quorum_tail.py`;
- active evaluator id: `stage4_heterogeneous_quorum_tail_v1`;
- documented in `docs/STAGE4_1_QUORUM_TAIL_UTILITY.md`;
- still not connected to PBFT three-phase reliability or Stage 3 message
  matrices.

Required tests:

- `H_ge_0 = 1`;
- `H_ge_k = 0` when `k > m`;
- all-zero inputs produce zero for positive quorum;
- all-one inputs produce one when `k <= m`;
- monotonicity in every input probability;
- homogeneous inputs match small hand-computed formulas;
- source scan rejects Monte Carlo, random sampling, and subset enumeration.

### Stage 4.2: PBFT Three-Phase Reliability Record

Implement the contract record and formula using declared matrices.

Implementation status:

- complete in `src/marl_topology/protocol/pbft_reliability.py`;
- active protocol variant: `stage4_pbft_three_phase_closed_form_v0`;
- documented in `docs/STAGE4_2_PBFT_THREE_PHASE_RELIABILITY.md`;
- consumes declared pre-prepare, prepare, and commit matrices only;
- still not connected to Stage 3 communication records, reward, latency, or
  energy.

Required tests:

- invalid `n < 3f + 1` is rejected;
- increasing any message probability cannot reduce consensus probability;
- removing edges or setting a phase matrix to zero lowers or preserves
  consensus probability;
- three phases are all required;
- byzantine filtering is conservative relative to no filtering;
- probability remains separate from latency, energy, timeout, and reward names.

### Stage 4.3: Stage 3 Adapter

Adapt Stage 3 network communication records into phase message matrices.

Implementation status:

- complete in `src/marl_topology/protocol/message_matrix_adapter.py`;
- active adapter id: `stage4_stage3_network_to_pbft_matrix_v1`;
- documented in `docs/STAGE4_3_STAGE3_MESSAGE_MATRIX_ADAPTER.md`;
- converts Stage 3 network records into declared PBFT phase matrices;
- still does not implement reward, protocol latency, protocol energy, training,
  or topology oracle evaluation.

Required tests:

- late messages past phase budget contribute zero delivery probability;
- disconnected topology lowers the matrix entries;
- same-resource interference can lower consensus probability through the
  message matrix;
- orthogonal resources can improve it relative to same-resource interference;
- failed communication is not reported as zero application latency.

### Stage 4.4: Expected-Initiator PBFT Reliability

Implementation status:

- complete in `src/marl_topology/protocol/pbft_reliability.py`;
- active model id: `pbft_expected_initiator_mean_field_v1`;
- computes per-primary `Psi_p` and uniform average
  `consensus_success_probability`;
- keeps the fixed-primary path as a helper.

Required tests:

- symmetric complete graph gives equal `Psi_p` for all primaries;
- weak primary has lower `Psi_p`, while topology reliability is the average;
- improving one primary's outgoing links improves that `Psi_p` and the average;
- uniform average equals the manual average of per-primary values;
- fixed-primary path remains a helper;
- conservative filtering lowers or preserves reliability;
- no Monte Carlo, random sampling, or subset enumeration.

### Stage 4.5: Baseline And Oracle-Candidate Review

Evaluate empty, full, sparse, and oracle-candidate topologies under the Stage 4
contract.

Implementation status:

- complete in `src/marl_topology/evaluation/stage4_baseline_oracle_review.py`;
- active stage id: `stage_4_5_baseline_and_oracle_review`;
- evaluates `empty`, `sparse_star`, `sparse_chain`, and `full` baselines;
- uses Stage 3 communication records, Stage 4.3 message matrices, and Stage
  4.4 expected-initiator PBFT reliability;
- reports registered metric concepts only.

Required tests:

- full graph remains a baseline, not an oracle;
- policy failure does not imply infeasible;
- reliability threshold satisfaction is separated from latency and energy
  minimization.
- oracle-candidate labels are not deployment actor inputs.

### Stage 4.6: Protocol Latency And Energy Accounting Review

Implementation status:

- complete in `src/marl_topology/protocol/pbft_accounting.py`;
- active accounting model id: `stage4_pbft_protocol_accounting_v1`;
- computes protocol `latency` and `energy` from Stage 3 network records grouped
  by PBFT phase;
- uses phase-max latency clipped to phase budget;
- uses scheduled attempt energy summed across phase records;
- integrated into Stage 4.5 baseline and oracle-candidate review.

Required tests:

- late records consume clipped phase budget and keep scheduled energy visible;
- accounting outputs are not consensus reliability;
- accounting diagnostics use `topology_diagnostics` and do not add metric
  names;
- oracle-labelled records and self-message records are rejected;
- no reward, training, actor, critic, COMA, MAPPO, v5 code, or old metric alias
  route is introduced.

### Stage 4.7: PBFT Application Evaluation Report

Implementation status:

- complete in `src/marl_topology/evaluation/stage4_application_report.py`;
- active report id: `stage_4_7_pbft_application_evaluation_report`;
- creates a single deterministic evaluation sensor from Stage 4.5 topology rows
  and Stage 4.6 protocol accounting diagnostics;
- flattens registered metric rows into a metric table;
- reports feasible topology names, infeasible topology names, lowest feasible
  latency view, and lowest feasible energy view;
- keeps the full graph as a baseline and the oracle-candidate as a review-only
  diagnostic.

Required tests:

- report uses Stage 4.5 as its source stage;
- all metric rows are registered and no new metric names are introduced;
- reliability feasibility depends only on the declared reliability threshold;
- Stage 4.6 protocol accounting diagnostics are present;
- full graph is not labelled oracle;
- oracle-candidate labels are not deployment actor inputs;
- replay script is print-only;
- no reward, training, actor, critic, COMA, MAPPO, v5 code, or old metric alias
  route is introduced.

### Stage 4.8: Communication/Consensus Boundary Audit

Implementation status:

- complete in `src/marl_topology/evaluation/stage4_boundary_audit.py`;
- active stage id: `stage_4_8_communication_consensus_boundary_audit`;
- evaluates boundary and failure cases before reward/objective contract freeze;
- confirms Stage 3.6 remains finite-blocklength and the old active SINR-only
  success surrogate is absent from active paths;
- confirms inverse reliability caps are visible diagnostics;
- confirms failed scheduled messages keep scheduled latency and energy visible;
- confirms expected-initiator PBFT distinguishes weak primary and central
  primary cases;
- keeps full graph as a baseline and not an oracle.

Required tests:

- all required Stage 4.8 cases are present;
- at least one consensus reliability value is non-saturated;
- failed scheduled messages have positive scheduled latency and energy but zero
  successful-delivery latency;
- unreachable inverse reliability target exposes capped diagnostics;
- weak and central primary cases are distinguishable;
- metric rows use only registered names;
- replay helper is print-only;
- no reward, training, actor, critic, COMA, MAPPO, v5 code, old metric alias,
  Monte Carlo, random sampling, or subset enumeration route is introduced.

## Negative Checks

Stage 4 implementation tasks must fail if they:

- use Monte Carlo, random sampling, or learned prediction for consensus
  probability;
- enumerate all success subsets;
- introduce reward terms;
- introduce legacy effective-success aliases;
- report one fixed global primary as topology-level reliability;
- use Stage 3 `network_delivery_probability` as
  `consensus_success_probability` directly;
- report timeout or quorum count as a standalone metric;
- label full graph as oracle;
- import v5 code.

## Acceptance For Stage 4.0

- Stage 3 coupling review exists.
- Stage 4 PBFT/application consensus plan exists.
- Protocol contract records the Stage 4 variant and defers implementation.
- Harness has a Stage 4.0 review task.
- Tests verify the plan, metric boundary, and forbidden inheritance.
- No implementation, reward, training, actor/critic, COMA, or v5 migration is
  performed.

## Residual Risks

- The three-phase formula is mean-field and does not fully model shared-message
  correlations.
- View change, leader failure, equivocation, and adversarial scheduling are
  deferred.
- Deadline-conditioned retransmission is implemented at Stage 3.6 link layer
  and still needs larger fixture coverage.
- Expected energy semantics require a separate contract before reward design.
