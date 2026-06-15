# Stage 2.7 Link Model Regime Review

This review locks the current Stage 2 link abstraction before richer physics, reward pressure, replay writing, or training are introduced.

## Controlled Object

The controlled object is the active link model regime:

```text
CandidateEdge.distance_3d_m -> SimpleLinkModel -> LinkRecord
```

## Desired State

The project has a named, testable, low-entropy link regime with explicit assumptions, outputs, omitted physics, and nonclaims.

## Active Regime

`stage2_deterministic_distance`

Purpose:

- keep the clean goal skeleton executable;
- provide deterministic link records for topology evaluator, oracle, baselines, and fixture reports;
- expose monotonic sanity checks before complex physics exists.

## Inputs

- `distance_3d_m`

Unit: meters.

## Outputs

- `link_success_probability`: range `[0, 1]`
- `latency_s`: seconds
- `energy_j`: joules
- `physics_regime`: `stage2_deterministic_distance`

## Formula Assumptions

- `link_success_probability = exp(-distance_3d_m / reference_distance_m)`
- `latency_s = base_latency_s + distance_3d_m / speed_of_light_mps`
- `energy_j = fixed_tx_energy_j + energy_per_m_j * distance_3d_m`

The regime is deterministic and has no random seed because it has no random sampling.

## Omitted Components

The regime does not model:

- building LoS/NLoS;
- path loss;
- shadowing;
- SINR;
- interference;
- bandwidth and packet size;
- queueing delay;
- processing delay;
- receive, idle, or processing energy;
- mobility.

## Nonclaims

Results under this regime are not:

- a full 3D V2X physical simulation;
- cross-regime policy quality evidence;
- reward definitions;
- PBFT timing evidence;
- measurement-calibrated link predictions.

Explicit negative claims:

- not a full 3D V2X physical simulation;
- not a reward definition.

## Couplings

- `CandidateGraph` supplies candidate edge distances.
- `TopologyEvaluator` consumes `link_success_probability`, `latency_s`, and `energy_j`.
- `ConsensusSuccess` uses link success probability only inside the current minimal quorum graph abstraction.
- `BaselineEvaluationReport` and `ScenarioFixtureReport` must report the physics regime.
- Future reward work must treat latency and energy as registered metrics, not hidden reward terms.

## Required Tests

- `distance_3d_m` is nonnegative.
- `link_success_probability` is in `[0, 1]`.
- `link_success_probability` is nonincreasing with distance.
- `latency_s` is nondecreasing and nonnegative with distance.
- `energy_j` is nondecreasing and nonnegative with distance.
- every fixture link record carries `stage2_deterministic_distance`.
- omitted components stay omitted until a future physics contract admits them.

## V5 Anti-Inheritance

This review does not migrate v5 channel, fading, SINR, BLER, resource-mask, reward, or phase-script logic.

v5 remains a read-only experience library. Any future richer link model must be introduced as a new named regime with its own contract tests.

## Acceptance

- Regime metadata exists in `src/marl_topology/link/regime.py`.
- `PHYSICS_CONTRACT.md` records the Stage 2.7 boundary.
- Harness task `link_model_regime_review` exists.
- Unit and contract tests pass without training or v5 writes.

## Residual Risks

- The model is intentionally simple and may overstate or understate feasibility.
- It cannot validate policy behavior under interference, LoS/NLoS, or mobility.
- Cross-regime generalization remains untested.
