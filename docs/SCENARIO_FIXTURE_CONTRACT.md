# Stage 2.5 Scenario Fixture Contract

This contract defines small deterministic scenario fixtures for clean-core evaluation. It broadens Stage 2 sensors beyond the single demo scene without adding complex 3D physics, stochastic channels, reward, training, or v5 code migration.

## Controlled Object

The controlled object is the scenario-fixture layer feeding:

```text
Scene3D -> CandidateGraph -> LinkModel -> TopologyEvaluator
-> TopologyOracle -> MinimalDecPOMDPEnv -> baseline reports
```

## Desired State

The project has a small suite of named, reproducible fixtures that expose different topology-control regimes before any learning work:

- dense feasible smoke scene;
- sparse feasible quorum path;
- quorum-unreachable infeasible scene.

## Required Fixture Fields

Every fixture must declare:

- `fixture_id`
- `description`
- `scene`
- `max_candidate_distance_m`
- `quorum_size`
- `success_probability_threshold`
- `deadline_s`
- `link_reference_distance_m`
- `expected_oracle_status`
- `expected_full_graph_success`

`fixture_id` must match `scene.scenario_id`.

## Active Fixtures

`demo_stage2`

- Purpose: dense four-node smoke scene used by Stage 2.
- Expected oracle status: `feasible`.
- Expected full graph success: `true`.

`sparse_chain_stage2`

- Purpose: sparse quorum path where success does not require full graph.
- Expected oracle status: `feasible`.
- Expected full graph success: `true`.

`quorum_blocked_stage2`

- Purpose: quorum-unreachable scene where infeasibility comes from exhaustive oracle evidence, not policy failure.
- Expected oracle status: `infeasible`.
- Expected full graph success: `false`.

## Physics Boundary

Stage 2.5 fixtures use only `stage2_deterministic_distance`.

They do not implement:

- LoS/NLoS ray tracing;
- path loss models;
- SINR or interference;
- shadowing;
- mobility;
- stochastic channel sampling.

Those features remain behind `docs/PHYSICS_CONTRACT.md`.

## Oracle And Baseline Boundary

- Full graph is a baseline, not an oracle.
- Oracle labels are report-side evidence only and must not enter `ActorObservation`.
- `feasible`, `infeasible`, and `unresolved` remain distinct states.
- Policy failure does not prove environment infeasibility.

## Metric Boundary

Scenario fixture reports may only use registered metric names:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

No fixture may introduce a new metric, reward field, CSV field, or legacy effective-success alias without metric registration.

## Verification

Required checks:

- fixture ids are unique;
- fixture ids match scenario ids;
- candidate edge ids are deterministic;
- expected oracle status matches exhaustive result for small fixtures;
- expected full graph success matches baseline result;
- all report metric rows use registered names;
- full graph is not labelled oracle;
- no training, reward, model, or v5 migration is introduced.

Run:

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python scripts\replay\scenario_fixture_report.py
```

## Residual Risks

- The fixtures are still small deterministic smoke tests.
- They do not cover city-scale geometry, buildings, LoS/NLoS, or mobility.
- They are sensors for clean-core stability, not evidence of MARL policy quality.
