# Stage 3.0 Communication Simulation Plan

Stage 3 builds the 3D communication simulation environment below MARL training
and below application-layer consensus. It replaces the Stage 2 deterministic
distance skeleton with contract-first physical, link, and network-layer
interfaces.

Stage 3.0 is design and interface freeze only. It does not implement full 3D
physics, reward, actor, critic, COMA, training, PBFT reliability metrics, or v5
code migration.

## Controlled Object

The controlled object is the Stage 3 communication simulation stack:

```text
3D Environment / Scenario
-> Geometry / Visibility
-> Channel Simulation
-> Link Layer Transmission
-> Network Layer Communication
```

Application-layer PBFT consensus and reward remain Stage 4 or later.

## Desired State

The desired state is a low-entropy, testable, explainable, and extensible 3D
urban communication simulator. It should produce communication-layer records
for link reliability, point-to-point latency, point-to-point energy, and network
delivery behavior without claiming PBFT consensus success.

## Stage Boundary

Stage 3 may define:

- building, road, lane, vehicle, RSU, and base-station geometry;
- 3D distance, ray-box visibility, LoS/NLoS, and blocker records;
- path loss, shadowing, tx/rx power, noise, interference, SINR, and packet
  success probability;
- point-to-point transmission latency and energy;
- selected communication topology, active transmissions, resources,
  interference groups, reachability, simple route or broadcast primitives, and
  network delivery records.

Stage 3 must not define:

- application consensus success metrics;
- PBFT quorum success;
- consensus reliability;
- consensus reward;
- application-level deadline objective;
- actor, critic, COMA, GNN, LSTM, training, or checkpoints.

## Stage 3 Layer Plan

| Stage | Layer | Goal | Main output |
| --- | --- | --- | --- |
| `3.0` | design and interface freeze | Lock responsibilities, units, records, tests, and exclusions | contracts and harness gates |
| `3.1` | 3D scenario and geometry visibility | Implement city objects, ray-box intersection, LoS/NLoS, blockers | geometry records |
| `3.2` | channel simulation v1 | Implement path loss, shadowing, tx/rx power, noise, interference, SINR, packet success | channel records |
| `3.3` | link transmission | Implement payload, bandwidth, propagation, transmission, processing, queue placeholder, energy | point-to-point link records |
| `3.4` | network communication | Implement selected-edge transmissions, resource groups, simple route/broadcast delivery | network communication records |
| `3.5` | micro-fixture suite | Add deterministic 3D communication fixtures | fixture reports |

## Contract Files

- `docs/GEOMETRY3D_CONTRACT.md`
- `docs/CHANNEL_MODEL_CONTRACT.md`
- `docs/LINK_TRANSMISSION_CONTRACT.md`
- `docs/NETWORK_LAYER_CONTRACT.md`

## Record And Metric Governance

Stage 3.0 registers no new metrics in code. Communication-layer outputs are
structured records unless a later task promotes a field into a registered metric.

Rules:

- Every summary or CSV metric field must map to the metric registry before use.
- Record fields such as `packet_success_probability`, `p2p_latency_s`, and
  `network_delivery_probability` are not application consensus metrics.
- Link/channel/network records must not be renamed as application reliability.
- Reward code must not use communication records until a future reward contract
  admits the field and its tests.

## Selected Edge Semantics

A selected edge in Stage 3 means a communication-layer transmission opportunity
or active communication relation under the current network-layer primitive.

- Selected edge does not mean consensus success.
- Selected edge does not mean packet success.
- Selected edge may generate transmission records when scheduled active.
- Unselected edge must not generate transmit energy.
- Full graph remains a baseline topology, not an oracle, upper bound, or
  resource optimum.

## Required Micro-Fixtures

| Fixture | Purpose |
| --- | --- |
| `free_space_close` | Close LoS link with high received power and low path loss |
| `free_space_far` | Far LoS link with larger path loss and lower packet success |
| `blocked_by_building` | Building blocker produces NLoS and blocker record |
| `urban_gap_los` | Gap or corridor preserves LoS through buildings |
| `urban_canyon_nlos` | Tall buildings produce NLoS or degraded channel |
| `rsu_high_los` | Higher RSU position can restore LoS |
| `two_transmitters_interference` | Same resource creates interference and lowers SINR |
| `orthogonal_channel_no_interference` | Orthogonal resources remove interference contribution |
| `multi_hop_delivery` | Network primitive aggregates multiple hops |
| `resource_redundant_topology` | Extra selected edges can increase resource use without being oracle |

## Feedback Loop

1. Freeze interface contracts.
2. Implement one layer at a time.
3. Add deterministic micro-fixtures before stochastic behavior.
4. Run unit tests, contract tests, and harness validation.
5. Update records and contracts only after evidence shows a missing field.

## Implementation Progress

- `Stage 3.0` complete: communication simulation contracts and harness gates.
- `Stage 3.1 - 3D city scenario and geometry visibility` complete: ray-box
  visibility, LoS/NLoS, blocker records, and geometry fixtures.
- `Stage 3.2 - channel simulation v1` complete: channel records for path loss,
  seeded shadowing, tx/rx power, noise, interference, SINR, and packet success
  probability.
- `Stage 3.3 - link transmission v1` complete: point-to-point payload,
  propagation delay, transmission delay, processing delay, explicit queueing
  placeholder, tx/rx/processing energy, latency, and energy records.
- `Stage 3.4 - network layer communication v1` complete: selected-edge
  semantics, explicit resource assignments, same-resource interference,
  orthogonal-resource exclusion, reachability, route/broadcast primitives,
  network delivery probability, network latency, and network energy records.
- `Stage 3.5 - micro-fixture suite review` complete: deterministic geometry,
  channel, link transmission, and network communication fixtures are inventoried
  and reviewed as communication-layer sensors.

## Acceptance

- Stage 3 contracts name responsibility, inputs, outputs, units, assumptions,
  omitted components, tests, failure modes, GOAL_SKELETON coupling, and Stage 4
  deferrals.
- Harness tasks exist for Stage 3 plan, geometry, channel, link transmission,
  and network-layer reviews.
- Project state marks Stage 3.0 complete and waits for owner decision.
- No reward, PBFT metric, actor/critic/COMA, training, or v5 code migration is
  introduced.

## Cybernetic Review Map

### Controlled Object Identified

Controlled object: the Stage 3 communication simulation contract stack. Scope
is physical, link, and network communication only.

### Desired State Defined

Desired state: a testable target state with explicit success criteria for
geometry, channel, link transmission, and network communication records before
implementation.

### State Variables Defined

State variables: scenario geometry, visibility state, channel record fields,
packet transmission fields, selected edge semantics, resource assignment,
network delivery record fields, and Stage 4 deferral state. Inputs and outputs
are named in each contract.

### Sensors Defined

Sensors: contract tests, harness validation, static check scans, fixture
reports, rubric score, and post-task self-review evidence.

### Actuators Defined

Safe actuators: documentation, harness task metadata, project state, tests, and
future config interfaces. Unsafe actuators: PBFT metrics, reward code, training,
actor/critic code, v5 writes, and unreviewed full simulator code.

### Feedback Loop Present

Feedback loop: baseline project state -> contract freeze -> tests -> harness
validation -> negative scans -> owner decision -> next layer implementation.

### Verification Plan Present

Verification commands:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python scripts\replay\baseline_evaluation_report.py
python harness\scripts\score_rubric.py docs\STAGE3_COMMUNICATION_SIMULATION_PLAN.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\stage3_communication_simulation_plan.score.json
```

Expected evidence: all contract tests pass, harness tasks validate, Stage 2
baseline report remains no-training/no-v5, and Stage 3 contracts contain no
application consensus metric definitions.

### Stability Risk Checked

Stability risk: communication-layer records could be mistaken for application
success or reward. Regression checks keep communication records separate from
consensus and preserve full graph as baseline only.

### Observability Gap Checked

Observability gap: no implemented Stage 3 simulator exists yet, so physical
monotonicity and network delivery behavior remain missing sensors until Stage
3.1-3.4.

### Controllability Checked

Controllability: owner approval is required before implementation. Safe
actuators are limited to contract and harness edits in Stage 3.0.

### Disturbance Checked

Disturbances: v5 channel semantics, old metric aliases, reward leakage,
unseeded stochasticity, unit confusion, hidden resource assignment, and future
training pressure.

### Delay Or Async Risk Checked

Delay risk: contracts and implementation can drift across stages. Project state
and post-task self-review provide the refresh signal.

### Noise Or Flakiness Checked

Noise risk: stochastic shadowing and random scheduling are deferred or seeded.
Stage 3.0 relies on deterministic contract checks only.

### Decoupling Checked

Coupling map: geometry feeds channel records; channel records feed link
transmission; link records feed network communication; Stage 4 consumes network
communication only after a separate application contract. Non-target behavior:
Stage 2 baseline reports remain unchanged.

### Reliability Or Error Control Checked

Error control: independent sensors include pytest, harness validation, rubric
scoring, static negative scans, and baseline reports. False success risk is
renaming packet or network delivery as consensus reliability.

### Persistent Learning Update Suggested

Persistent follow-up: keep the Stage 3 harness tasks as gates and add
micro-fixture tests before implementing each layer.

## Source-Awareness

These software-engineering controls are derived by analogy from the local
Engineering Cybernetics workflow and read-only v5 learning audits; they are not
direct physical-layer doctrine.

## Recommended Next Task

`Stage 4.0 - PBFT/application consensus contract planning`.

Reason: Stage 3 now has reviewed communication-layer records and micro-fixture
sensors. Stage 4 should start with a contract-only PBFT/application reliability
plan before implementing consensus metrics or reward.

## Residual Risks

- Stage 3.2 uses a simple v1 path-loss and packet-success surrogate, not an
  empirical V2X calibration.
- Network delivery aggregation is deterministic v1 behavior, not an empirical
  network protocol model.
- Stage 4 must still define PBFT/application reliability separately.
