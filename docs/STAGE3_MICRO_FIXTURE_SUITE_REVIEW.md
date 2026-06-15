# Stage 3.5 Micro-Fixture Suite Review

## Controlled Object

The controlled object is the deterministic Stage 3 communication micro-fixture
suite spanning geometry, channel, link transmission, and network communication.

## Desired State

The suite provides small, explainable, deterministic sensors for Stage 3
communication records before Stage 4 introduces PBFT/application semantics.

The desired state is:

- every required Stage 3 micro-fixture is represented by an implemented fixture
  or a declared layer-specific variant;
- every fixture is evaluable without writing result files;
- fixture records stay communication-layer records;
- fixture reports do not introduce metrics, reward, PBFT consensus semantics,
  training data, or v5 migration.

## Fixture Inventory

| Layer | Fixture ids |
| --- | --- |
| Geometry / Visibility | `free_space_close`, `free_space_far`, `blocked_by_building`, `urban_gap_los`, `urban_canyon_nlos`, `rsu_high_los`, `rsu_low_nlos` |
| Channel Simulation | `free_space_close`, `free_space_far`, `blocked_by_building`, `two_transmitters_interference`, `orthogonal_channel_no_interference` |
| Link Transmission | `payload_small`, `payload_large`, `bandwidth_narrow`, `bandwidth_wide`, `unselected_edge` |
| Network Communication | `multi_hop_delivery`, `disconnected_reachability`, `two_transmitters_interference_network`, `orthogonal_channel_no_interference_network`, `resource_redundant_topology` |

## Required Fixture Coverage

The Stage 3 plan requires these micro-fixture concepts:

- `free_space_close`
- `free_space_far`
- `blocked_by_building`
- `urban_gap_los`
- `urban_canyon_nlos`
- `rsu_high_los`
- `two_transmitters_interference`
- `orthogonal_channel_no_interference`
- `multi_hop_delivery`
- `resource_redundant_topology`

Stage 3.5 review status: all required fixture concepts are covered.

Layer-specific variants:

- `rsu_low_nlos` supports the RSU height effect comparison.
- `two_transmitters_interference_network` validates same-resource interference at
  the network layer.
- `orthogonal_channel_no_interference_network` validates orthogonal resource
  exclusion at the network layer.
- `disconnected_reachability` validates network reachability failure.
- `payload_small`, `payload_large`, `bandwidth_narrow`, `bandwidth_wide`, and
  `unselected_edge` validate link transmission latency and energy behavior.

## Review Sensor

`build_stage3_micro_fixture_suite_review()` returns an in-memory review payload.

Replay command:

```powershell
python scripts\replay\stage3_micro_fixture_suite_report.py
```

The script prints JSON and writes no result files.

Required checks:

- `required_micro_fixtures_covered`
- `geometry_expectations_match`
- `channel_expectations_match`
- `link_records_nonnegative`
- `network_records_nonnegative`
- `network_records_not_oracle`
- `training_run: false`
- `v5_code_migrated: false`
- `reward_implemented: false`
- `pbft_application_semantics_defined: false`
- `new_stage3_registry_entries: []`

## Boundary

The Stage 3.5 fixture report is not a metric report, not reward, not a PBFT
consensus report, not training data, and not an oracle result.

Communication record fields such as `packet_success_probability`,
`p2p_latency_s`, `p2p_energy_j`, `network_delivery_probability`,
`network_latency_s`, and `network_energy_j` remain Stage 3 record fields only.

Full graph remains a baseline topology, not an oracle or resource optimum.

## Negative Checks

The fixture suite must not:

- introduce PBFT quorum success, consensus reliability, or application-level
  deadline satisfaction;
- introduce reward terms or objective weights;
- write CSV, checkpoint, replay, or result files;
- read or write `D:\PhD_works\v5`;
- hide random behavior without a seed;
- label network delivery as consensus success;
- label full graph as oracle.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python scripts\replay\stage3_micro_fixture_suite_report.py
```

## Residual Risks

- The fixtures are deterministic sanity sensors, not empirical V2X calibration.
- Broadcast aggregation is still a v1 primitive and may overcount failure on
  redundant selected edges.
- Stage 4 must define PBFT/application reliability separately before any
  consensus metric or reward work.
