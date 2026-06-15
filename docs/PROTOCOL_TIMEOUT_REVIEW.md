# Stage 2.8 Protocol Timeout Review

This review hardens the minimal quorum, deadline, and timeout boundary before
future reward, replay writing, or training. It does not implement full PBFT and
does not change protocol code.

## Controlled Object

The controlled object is the Stage 2 protocol abstraction:

```text
selected topology + link records + ConsensusConfig
-> TopologyEvaluator
-> ConsensusResult and registered metrics
```

## Desired State

The project can reason about consensus success without inheriting v5
effective-success ambiguity. Quorum, deadline, timeout, consensus probability,
binary success, latency, energy, diagnostics, and reward remain separate.

## Active Protocol Variant

`stage2_minimal_quorum_graph`

This variant is deliberately smaller than PBFT:

- no pre-prepare / prepare / commit message simulator;
- no explicit `n`, `f`, or `2f + 1` derivation;
- no Byzantine adversary model;
- no view-change or retransmission model;
- no stochastic timeout sampling.

It directly declares `quorum_size`, `success_probability_threshold`, and
`deadline_s` for the current deterministic topology evaluator.

## Success Boundary

For the active variant, `consensus_success` is true only when:

- the leader component reaches at least `quorum_size` nodes;
- `consensus_success_probability >= success_probability_threshold`;
- `latency <= deadline_s` when a deadline is configured.

`deadline_s = None` disables the deadline gate. A configured deadline is
inclusive: equality passes, exceeding the deadline fails.

Explicit boundary: deadline_s = None disables the deadline gate.

## Metric Boundary

`consensus_success` is the binary event after quorum, probability threshold, and
deadline gates.

`consensus_success_probability` is the Stage 2 link/topology probability under
the minimal quorum abstraction. It is not multiplied by timeout or quorum
factors and is not a v5-style effective-success metric.

Explicit boundary: consensus_success_probability is the Stage 2 link/topology
probability and is not multiplied by timeout or quorum factors.

Explicit boundary: consensus_success_probability is not timeout-gated in Stage
2.

Exact review statement: consensus_success_probability is the Stage 2 link/topology probability.

Exact review statement: consensus_success_probability is not timeout-gated in Stage 2.

`latency` remains a separate registered metric. Timeout is a protocol event
derived from latency and deadline, not a standalone metric.

`topology_diagnostics` may include fields such as quorum size, reachable node
count, physics regime, and protocol variant. These are diagnostics, not
objectives or success metrics.

## Failure Reason Boundary

The current evaluator reports one diagnostic failure reason:

1. `quorum_unreachable`
2. `deadline_exceeded`
3. `probability_below_threshold`

The priority is a reporting convention only. If future analysis needs multiple
simultaneous failure causes, that must be introduced as a diagnostic contract,
not as a consensus metric.

## Reward Boundary

Timeout and quorum are not reward terms. A future reward proposal may reference
registered `consensus_success`, `consensus_success_probability`, `latency`, and
`energy`, but it must not optimize timeout count or quorum count as standalone
reward.

## V5 Anti-Inheritance

v5 remains a read-only experience library. This review does not copy v5
`P_succ`, `P_eff`, hard/soft/legacy effective-success modes, deadline factors,
quorum factors, reward terms, or phase scripts.

## Required Tests

- `ConsensusConfig` rejects invalid quorum, threshold, and negative deadline.
- `TopologyEvaluator` rejects quorum larger than node count.
- Deadline equality passes.
- Deadline exceedance sets `consensus_success` to false and reports
  `deadline_exceeded`.
- `consensus_success_probability` is not timeout-gated in Stage 2.
- Metric rows do not introduce `timeout`, `quorum`, `P_eff`, or derived legacy
  names.

## Cybernetic Review Map

### Controlled Object Identified

Controlled object: the minimal protocol boundary between topology evaluation
inputs and registered consensus metrics. Scope excludes reward, training,
full-PBFT simulation, and v5 code.

### Desired State Defined

Desired state: protocol semantics are testable and low-entropy. Success
criteria are explicit quorum, probability, and deadline gates with registered
metric evidence.

### State Variables Defined

State variables: protocol variant, quorum size, success probability threshold,
deadline, latency, timeout event, consensus success, consensus probability,
failure reason, and metric rows. Inputs are selected topology and link records;
outputs are `ConsensusResult` and registered metric rows.

### Sensors Defined

Sensors: unit tests, contract tests, static metric-name scans, harness
validation, baseline report checks, and project-state review.

### Actuators Defined

Safe actuators: protocol contract documentation, harness task metadata,
project-state update, and boundary tests. Unsafe actuators: full PBFT code,
reward code, training loops, v5 code writes, and new unregistered metric names.

### Feedback Loop Present

Feedback loop: baseline protocol inspection -> contract review -> boundary
tests -> harness validation -> negative scans -> owner decision.

### Verification Plan Present

Verification commands:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python scripts\replay\baseline_evaluation_report.py
python harness\scripts\score_rubric.py docs\PROTOCOL_TIMEOUT_REVIEW.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\protocol_timeout_review_stage2_8.score.json
```

Expected evidence: tests pass, task validation passes, baseline report remains
no-training/no-v5, and metric rows stay registered.

### Stability Risk Checked

Stability risk: timeout could be smuggled into a derived reliability metric or
reward term. Regression risk is controlled by metric-name separation and
reward-boundary tests.

### Observability Gap Checked

Observability gap: the current abstraction does not model PBFT message timing,
view changes, packet sizes, queueing, or stochastic timeout distributions.
Missing sensor: richer protocol simulator tests.

### Controllability Checked

Controllability: owner approval is required before future protocol expansion.
Safe changes remain limited to contracts, harness, and tests.

### Disturbance Checked

Disturbances: v5 effective-success aliases, deadline/quorum factor mixing,
reward pressure, deterministic link-regime limits, and future training noise.

### Delay Or Async Risk Checked

Delay risk: protocol docs and evaluator code can drift in later stages. The
post-task self-review gate and boundary tests provide refresh signals.

### Noise Or Flakiness Checked

Noise risk: no stochastic protocol sampling is active. Random baseline evidence
is seeded and not used to prove protocol generality.

### Decoupling Checked

Coupling map: protocol output feeds metrics, topology evaluator, oracle,
baseline reports, reward review, and future replay. Non-target behavior:
existing reward-free evaluation reports remain unchanged.

### Reliability Or Error Control Checked

Error control: independent sensors include unit tests, contract tests, harness
validation, static scans, and baseline report checks. False success risk is
treating timeout or quorum diagnostics as registered success metrics.

### Persistent Learning Update Suggested

Persistent follow-up: keep `protocol_timeout_review_stage2_8` as the gate for
future protocol changes, and require a new named protocol variant before full
PBFT timing logic.

## Residual Risks

- Stage 2 does not prove PBFT timing realism.
- Deadline behavior uses aggregated link latency from the current evaluator.
- Failure reason reports only one diagnostic cause.
- Future timeout-gated reliability metrics remain unregistered and blocked.
