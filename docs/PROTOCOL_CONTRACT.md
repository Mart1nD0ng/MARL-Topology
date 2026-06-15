# Protocol Contract

## Stage 4.0 PBFT / Application Consensus Planning

Stage 4.0 introduces a contract plan only. It does not implement PBFT code,
does not add reward, does not train models, and does not migrate v5 code.

Planned protocol variant:

`stage4_pbft_three_phase_closed_form_v0`

This variant will consume Stage 3 communication records through declared
message-delivery matrices and will map its analytic round-success probability
to the existing metric-governance concept:

`consensus_success_probability`

The Stage 4.0 plan is documented in:

`docs/STAGE4_PBFT_APPLICATION_CONSENSUS_PLAN.md`

Key rules:

- PBFT is modeled as three communication phases: pre-prepare, prepare, commit.
- The default PBFT assumption is `n >= 3f + 1`.
- Total certificate quorum is `2f + 1`.
- External message quorum is `2f` when local self-message is counted.
- Global round success requires at least `2f + 1` replicas to reach committed
  local state by the deadline.
- Heterogeneous quorum probability must use an analytic coefficient or
  equivalent stable closed-form evaluator, not subset enumeration, random
  sampling, learned prediction, or Monte Carlo.
- In short: not subset enumeration.
- Phase deadline budgets transform Stage 3 communication delivery into
  phase-specific message-delivery probabilities.
- Communication delivery, timeout, quorum count, latency, energy, reward, and
  consensus probability remain separate names.

Stage 4.0 accepts v5 only as read-only learning evidence. The useful v5 lesson
is the need for a three-stage cascade over heterogeneous link probabilities.
The clean project must not inherit v5 metric aliases, reward shaping, phase
scripts, or training interfaces.

## Stage 4.1 Heterogeneous Quorum-Tail Utility

Stage 4.1 implements only the standalone heterogeneous quorum-tail utility:

`src/marl_topology/protocol/quorum_tail.py`

Active evaluator id:

`stage4_heterogeneous_quorum_tail_v1`

The utility computes the probability that at least `k` heterogeneous Bernoulli
inputs succeed. It is a coefficient evaluator over the generating polynomial
described in the Stage 4.0 plan. It does not enumerate success subsets, does
not use random sampling, does not use Monte Carlo simulation, and does not
import v5 code.

Public interfaces:

- `heterogeneous_quorum_tail`
- `evaluate_quorum_tail`
- `remove_largest_probabilities`
- `conservative_quorum_tail`
- `QuorumTailResult`

Boundary:

- Stage 4.1 is not a PBFT three-phase implementation.
- Stage 4.1 does not consume Stage 3 communication records.
- Stage 4.1 does not emit `consensus_success_probability` as a project metric.
- Stage 4.1 does not define latency, energy, timeout, reward, or training
  behavior.

## Stage 4.2 PBFT Three-Phase Reliability Record

Stage 4.2 implements the PBFT three-phase reliability record over declared
message-delivery matrices:

`src/marl_topology/protocol/pbft_reliability.py`

Active protocol variant:

`stage4_pbft_three_phase_closed_form_v0`

Implemented public interfaces:

- `PBFTThreePhaseConfig`
- `PBFTThreePhaseReliabilityRecord`
- `evaluate_pbft_three_phase_reliability`
- `evaluate_pbft_given_primary`

Required assumptions:

- `n >= 3f + 1`;
- total quorum is `2f + 1`;
- external quorum is `2f` when local self-message is counted;
- phases are exactly `pre_prepare`, `prepare`, and `commit`;
- missing directed matrix entries are zero delivery;
- self-message matrix entries are rejected;
- conservative unknown-fault filtering may remove the largest probabilities
  before quorum-tail evaluation.

Output mapping:

- In Stage 4.2, `consensus_success_probability` is primary-specific analytic
  PBFT reliability for the configured `primary_id`.
- In Stage 4.4, topology-level consensus reliability must use expected
  initiator averaging, not a single fixed primary.

Boundary:

- Stage 4.2 does not consume Stage 3 communication records.
- Stage 4.2 does not implement phase deadline gating.
- Stage 4.2 does not compute protocol latency or energy.
- Stage 4.2 does not implement reward, training, actor, critic, COMA, GNN, or
  LSTM behavior.
- Stage 4.2 does not migrate v5 code.

## Stage 4.4 Expected-Initiator PBFT Reliability

Stage 4.4 redefines topology-level PBFT reliability as the expected reliability
over all nodes acting as initiator and primary:

`pbft_expected_initiator_mean_field_v1`

Implemented public interfaces:

- `PBFTExpectedInitiatorConfig`
- `PBFTExpectedInitiatorReliabilityRecord`
- `evaluate_pbft_given_primary`
- `evaluate_expected_initiator_pbft_reliability`

Core formula:

```text
R_consensus(G, s) = (1 / |V|) * sum_p Psi_p(G, s)
```

`Psi_p` is computed by the primary-specific three-phase helper:

```text
a_i^(1,p) = 1 if i = p else Q_pre[p, i]
a_i^(2,p) = a_i^(1,p) * H_{2f}^f({a_j^(1,p) * Q_prepare[j, i] : j != i})
a_i^(3,p) = a_i^(2,p) * H_{2f}^f({a_j^(2,p) * Q_commit[j, i] : j != i})
Psi_p = H_{2f+1}^f({a_i^(3,p)})
```

Configuration:

- `self_vote_counted = true`
- `fault_filter = none / remove_largest`
- `primary_distribution = uniform`
- `model_id = pbft_expected_initiator_mean_field_v1`
- `mean_field_assumption = true`
- `view_change_mode = deferred / none`

Boundary:

- A fixed primary is only an internal helper or diagnostic, not topology-level
  reliability.
- `fault_filter_remove_largest` is a conservative engineering lower-bound
  approximation, not a strict Byzantine adversary model.
- Stage 4.4 does not implement view-change.
- Stage 4.4 does not use Monte Carlo, random sampling, learned prediction, or
  subset enumeration.
- Stage 4.4 does not implement reward, training, actor, critic, COMA, GNN, or
  LSTM behavior.
- Stage 4.4 does not migrate v5 code.

## Stage 4.5 Baseline And Oracle-Candidate Review

Stage 4.5 implements a review sensor for topology baselines and bounded
oracle-candidate feasibility checks:

`src/marl_topology/evaluation/stage4_baseline_oracle_review.py`

Active stage id:

`stage_4_5_baseline_and_oracle_review`

The review evaluates Stage 3 communication records through Stage 4.3 message
matrices and Stage 4.4 expected-initiator PBFT reliability.

Boundary:

- full graph is a baseline and not an oracle-candidate;
- failed baselines do not prove infeasibility;
- oracle-candidate labels are not deployment actor inputs;
- `network_delivery_probability` is not exported as
  `consensus_success_probability`;
- latency and energy remain separate registered concepts;
- no reward, training, actor, critic, COMA, GNN, LSTM, or v5 migration is
  introduced.

## Stage 4.6 Protocol Latency And Energy Accounting Review

Stage 4.6 implements the protocol-layer accounting review:

`src/marl_topology/protocol/pbft_accounting.py`

Active accounting model id:

`stage4_pbft_protocol_accounting_v1`

Implemented public interfaces:

- `PBFTPhaseAccountingRecord`
- `PBFTProtocolAccountingRecord`
- `account_pbft_protocol_latency_energy`

Boundary:

- protocol latency and energy are accounting outputs;
- reliability remains produced by Stage 4.2/4.4 PBFT reliability evaluators;
- Stage 4.6 does not export `consensus_success_probability`;
- Stage 4.6 does not implement reward;
- Stage 4.6 does not alter Stage 3 communication records;
- Stage 4.6 does not migrate v5 code.

Latency accounting:

```text
phase_latency_s = min(max scheduled network_scheduled_latency_s, phase_budget_s)
latency = sum_phase phase_latency_s
```

Energy accounting:

```text
phase_energy_j = sum scheduled network_energy_j
energy = sum_phase phase_energy_j
```

Late messages are deadline diagnostics for reliability and consume the clipped
phase budget in this review sensor. Scheduled attempt energy remains visible
even when delivery probability is zero.

Stage 4.8 makes the Stage 3 latency split explicit. Protocol accounting uses
`network_scheduled_latency_s`, while `network_latency_s` remains a compatibility
alias for success-conditioned latency.

## Stage 4.7 PBFT Application Evaluation Report

Stage 4.7 implements a deterministic application evaluation report:

`src/marl_topology/evaluation/stage4_application_report.py`

Active report id:

`stage_4_7_pbft_application_evaluation_report`

Implemented public interfaces:

- `Stage47EvaluationReportConfig`
- `Stage47TopologySummary`
- `build_stage4_7_pbft_application_evaluation_report`

Boundary:

- Stage 4.7 combines Stage 4.5 topology rows and Stage 4.6 accounting
  diagnostics;
- Stage 4.7 does not compute a new PBFT reliability formula;
- Stage 4.7 does not alter quorum, deadline, or fault-filter semantics;
- Stage 4.7 does not rename network delivery as consensus reliability;
- Stage 4.7 does not implement reward or training.

## Stage 4.8 Communication/Consensus Boundary Audit

Stage 4.8 implements a boundary calibration audit:

`src/marl_topology/evaluation/stage4_boundary_audit.py`

Active stage id:

`stage_4_8_communication_consensus_boundary_audit`

The audit checks non-saturated, boundary, failure, and resource trade-off
scenarios before reward/objective contract freeze.

Implemented boundary cases:

- `near_threshold_link`
- `deadline_tight_retransmission`
- `unreachable_reliability_target`
- `weak_edge_primary`
- `center_vs_edge_primary`
- `interference_full_graph_penalty`
- `sparse_resource_efficient`
- `failed_scheduled_message`

Protocol boundary:

- Stage 3.6 remains finite-blocklength URLLC link reliability.
- The old active SINR-only success surrogate remains removed.
- Inverse reliability caps are explicit diagnostics.
- Failed scheduled messages preserve scheduled latency and energy.
- Expected-initiator PBFT remains the consensus reliability model.
- Full graph is a baseline, not an oracle.
- No reward, training, actor, critic, COMA, GNN, LSTM, Monte Carlo, sampling,
  subset enumeration, or v5 code migration is introduced.

## Stage 4.3 Stage 3 Message-Matrix Adapter

Stage 4.3 implements the narrow adapter from Stage 3 communication records to
PBFT phase message matrices:

`src/marl_topology/protocol/message_matrix_adapter.py`

Active adapter id:

`stage4_stage3_network_to_pbft_matrix_v1`

Implemented public interfaces:

- `PBFTPhaseBudgets`
- `PBFTMessageMatrices`
- `build_pbft_message_matrices_from_network_records`
- `evaluate_pbft_reliability_from_network_records`

Message-matrix rule:

```text
M_phase[source, target] =
    network_delivery_probability
    if network_scheduled_latency_s <= phase_budget_s
    else 0
```

The previous shorthand `network_latency_s <= phase_budget_s` is superseded by
the scheduled-latency gate above because `network_latency_s` is now a
successful-delivery latency alias.

Boundary:

- Stage 4.3 consumes Stage 3 network records as communication evidence.
- Stage 4.3 uses scheduled network latency for phase deadline gating.
- Stage 4.3 does not export a consensus metric.
- Stage 4.3 does not rename `network_delivery_probability` as
  `consensus_success_probability`.
- Stage 4.3 does not define protocol latency, protocol energy, timeout reward,
  consensus reward, training, actor, critic, COMA, GNN, or LSTM behavior.
- Stage 4.3 does not migrate v5 code.

## Stage 2.8 Minimal Quorum / Timeout Review

Stage 2.8 reviews the active protocol boundary only. It does not implement a
full PBFT message simulator, does not add a reward term, does not add a new
metric, does not train a model, and does not migrate v5 protocol code.

Active protocol variant:

`stage2_minimal_quorum_graph`

This variant is a named Stage 2 abstraction. It is not the default PBFT model
below. It uses:

- `quorum_size`: declared directly as the minimum number of leader-reachable
  nodes needed for consensus feasibility;
- `success_probability_threshold`: declared threshold on
  `consensus_success_probability`;
- `deadline_s`: optional inclusive deadline in seconds.

Success condition for the active variant:

- leader-reachable node count is at least `quorum_size`;
- `consensus_success_probability` is at least
  `success_probability_threshold`;
- `latency <= deadline_s` when `deadline_s` is not `None`.

Timeout semantics:

- `deadline_s` is the configured maximum acceptable latency.
- `deadline_s = None` means no deadline gate is applied.
- `latency <= deadline_s` passes; `latency > deadline_s` is a timeout event.
- Timeout affects binary `consensus_success`.
- Timeout does not create a standalone metric, reward term, or v5-style
  effective-success alias.
- In Stage 2, `consensus_success_probability` is not timeout-gated. It remains
  the topology/link probability under the minimal quorum abstraction. If a
  future timeout-gated probability is needed, it must be registered as a new
  metric first.

Failure-reason priority is diagnostic only:

1. `quorum_unreachable`
2. `deadline_exceeded`
3. `probability_below_threshold`

If multiple failure causes are true, the current evaluator reports the first
diagnostic reason above. This diagnostic does not replace registered metrics.

Stage 2.8 acceptance:

- Quorum, deadline, timeout, consensus event/probability, latency, energy, and
  reward remain separate names.
- No `P_eff`, hard/soft/legacy mode taxonomy, timeout reward, or quorum reward
  is introduced.
- Boundary tests cover quorum validation, inclusive deadline behavior,
  deadline failure, and metric-name separation.

## PBFT Quorum

For a PBFT committee with `n` replicas and fault tolerance `f`, the quorum rule must be explicitly declared. The default PBFT assumption is:

- `n >= 3f + 1`
- commit success requires at least `2f + 1` valid commit messages

Any deviation must be named as a protocol variant.

## Consensus Success Condition

A consensus round succeeds only when all declared hard conditions hold:

- Required quorum is met.
- Required messages are valid.
- The round completes before the deadline/timeout.
- The topology and link schedule provide enough successful communication events under the selected evaluation mode.

## Deadline / Timeout

- `deadline` is the maximum allowed round completion time.
- `timeout` is a protocol event caused by exceeding the deadline.
- Timeout is not itself a reward term.
- Timeout may affect `consensus_success`, `consensus_success_probability`, and latency metrics according to the metric governance contract.

## Metric Mapping

`consensus_success`:

- Binary protocol success event under declared quorum and deadline rules.

`consensus_success_probability`:

- Estimated or analytic probability of protocol success under the current topology and link assumptions.
- Must declare whether it is analytic, sampled, replay-estimated, or a future registered surrogate.

Future derived metrics:

- Any effective-success, softened, hardened, aggregate, or risk-sensitive protocol metric must first be registered in `docs/METRIC_CONTRACT.md`.

## Hard and Soft Usage

- Registered metrics may guide training, ablation, or evaluation only for their declared `used_for` field.
- Training surrogates must not be reported as final task success unless the metric registry explicitly permits that use.

## Acceptance

- No protocol code may call a timeout, quorum count, or reward value a consensus metric.
- All protocol outputs must identify mode and horizon.
- Contract tests must include quorum boundary cases and deadline boundary cases.
