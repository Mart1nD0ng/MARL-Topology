# Stage 5.0e Tau Consensus Fixture Family Design

## Scope

Stage 5.0e designs the scenario-family coverage required before a future
tau-consensus calibration run.

It does not implement fixtures, does not run calibration, does not select
`tau_consensus`, does not implement reward, does not train models, does not add
actor/critic code, and does not migrate v5 code.

Boundary shorthand: Stage 5.0e is fixture-family design only.

## Controlled Object

The controlled object is the calibration evidence surface that will feed the
Stage 5.0d tau-consensus report sensor after fixture implementations exist.

This surface is not the final threshold, not a reward, not an actor observation
source, and not an oracle label provider for deployment.

## Desired State

The project should have a low-entropy fixture-family specification that makes a
future calibration run observable and hard to overfit:

- scenario families cover clear, near-boundary, blocked, interference, deadline,
  topology, and PBFT-primary regimes;
- every metric-valued field maps to the existing metric governance concepts;
- every family declares topology variants and negative controls;
- Stage 4.8 remains a smoke-test seed, not the final calibration set;
- full graph remains a baseline, not an oracle;
- oracle candidates are review-only diagnostics and not actor inputs;
- no default numeric tau candidates are introduced.

## State Variables

- Coverage of geometry and visibility regimes.
- Coverage of URLLC finite-blocklength link regimes.
- Coverage of deadline and retransmission boundary cases.
- Coverage of network interference and resource assignment regimes.
- Coverage of sparse, dense, disconnected, redundant, and multi-hop topology
  variants.
- Coverage of expected-initiator PBFT primary asymmetry.
- Metric-governance consistency.
- Oracle-label leakage risk.
- Whether Stage 4.8 smoke-test rows are being mistaken for calibration evidence.

## Sensors

- Contract tests for required family ids and forbidden shortcuts.
- Harness task validation.
- Future fixture manifest validation.
- Future report row schema validation.
- Future calibration report checks for sparse alternatives, full-graph baseline
  status, and weak-primary diagnostics.

## Actuators

- Fixture-family design documentation.
- Harness task gates and negative checks.
- Contract tests.
- Future fixture manifest schemas.
- Future deterministic scenario builders under `src/marl_topology/`, only after
  owner approval.

## Disturbances

- Treating Stage 4.8 boundary rows as a final calibration set.
- Picking tau from a small or saturated scenario set.
- Full graph being treated as oracle or resource optimum.
- Reward, timeout, quorum, or edge-count terms entering calibration evidence.
- Actor inputs receiving oracle or future evaluation labels.
- V5 metric aliases, reward logic, phase scripts, or default thresholds being
  reintroduced.

## Coupling Map

| Layer | Fixture pressure | Required decoupling |
| --- | --- | --- |
| Geometry / visibility | LoS/NLoS can dominate link reliability | declare geometry family and blocker expectations |
| Channel / link | SINR, duration, bandwidth, deadline, and retransmission are coupled | use Stage 3.6 finite-blocklength records only |
| Network | full graph may increase interference and energy | include sparse, dense, disconnected, and redundant topology variants |
| PBFT | primary location can change consensus reliability | include weak-primary and center-vs-edge families |
| Objective | tau feasibility can hide latency/energy trade-offs | report feasible counts separately from latency/energy views |
| Dec-POMDP | oracle labels can leak into actor inputs | mark oracle candidates as diagnostics only |

## Required Fixture Families

These are design requirements for future fixture builders and calibration
manifests. Stage 5.0e does not implement them.

| family_id | Purpose | Required topology variants | Required signals | Negative checks |
| --- | --- | --- | --- | --- |
| `clear_free_space_reference` | Sanity reference for high-quality LoS communication | empty/disconnected, sparse feasible, dense full baseline | high link reliability, low latency, low energy | full graph is not oracle |
| `near_threshold_link_budget` | Expose non-saturated reliability near the feasibility boundary | sparse weak link, sparse improved link, dense baseline | consensus probabilities spanning below and above candidate tau values | no hidden default tau |
| `blocked_or_nlos_urban` | Test building blockage and NLoS penalty effect | blocked sparse path, alternate path, dense baseline | lower reliability or higher latency/energy than clear reference | NLoS is not renamed consensus failure |
| `urban_gap_or_corridor_los` | Verify that gaps/corridors can restore visibility | blocked path, corridor path, dense baseline | blocker records and improved link reliability through gap | geometry labels are diagnostics only |
| `rsu_height_recovery` | Check RSU/base-station height impact | low RSU, high RSU, dense baseline | visibility recovery, reliability change, latency/energy accounting | height is not a reward term |
| `same_resource_interference` | Show interference penalty under active simultaneous links | sparse no-interference, dense same-resource, redundant same-resource | SINR drop, reliability drop, energy/latency cost | full graph cannot be upper bound |
| `orthogonal_resource_mitigation` | Show resource separation reducing interference | same-resource dense, orthogonal dense, sparse feasible | improved network delivery under orthogonal resources | resource id is not consensus metric |
| `deadline_tight_retransmission` | Exercise retransmission count under tight deadline | one-attempt edge, multi-attempt edge, disconnected | deadline delivery probability and expected attempts | failed messages still carry scheduled cost |
| `unreachable_reliability_target` | Detect inverse reliability cap and infeasible links | impossible link, feasible alternative, dense baseline | required-reliability-met flag and capped required time | cap is diagnostic, not success |
| `sparse_vs_dense_tradeoff` | Compare feasible sparse candidates against dense/full baseline | sparse feasible, dense full baseline, redundant feasible | feasible count, latency, energy, edge count diagnostics | density is not a primary objective |
| `multi_hop_route` | Cover route aggregation beyond one hop | direct weak, multi-hop feasible, disconnected | route delivery probability, latency, energy accumulation | path existence is not PBFT success |
| `resource_redundant_topology` | Detect redundant topology with extra resource cost | minimal feasible, redundant feasible, dense baseline | reliability plateau with latency/energy increase | redundant edges do not get reliability bonus above tau |
| `failed_scheduled_message` | Preserve accounting for scheduled failures | selected failed edge, alternate feasible edge, disconnected | scheduled latency, successful-delivery latency, energy | failed scheduled message is not zero-cost |
| `weak_primary_distribution` | Probe expected-initiator PBFT asymmetry | weak-edge primary, improved weak primary, balanced baseline | per-primary reliability spread and mean reliability | fixed-primary result is not topology reliability |
| `center_vs_edge_primary` | Compare central RSU primary and edge vehicle primary | center-strong, edge-weak, balanced baseline | per-primary reliability ordering and average | no global-primary shortcut |
| `symmetric_pbft_reference` | Provide symmetric PBFT sanity case | symmetric sparse, symmetric dense, disconnected | equal per-primary reliability in symmetric graph | symmetry is not assumed for all families |

## Minimum Coverage For Future Calibration Runs

A future calibration run should not claim calibration coverage unless
the fixture manifest contains:

- at least one family from each axis: geometry, channel/link, network resource,
  topology trade-off, and PBFT primary asymmetry;
- at least one non-saturated case where `0 < consensus_success_probability < 1`;
- at least one failed scheduled-message case with positive scheduled latency or
  energy;
- at least one sparse feasible candidate and one dense full baseline;
- at least one case where full graph is worse than a sparse candidate on
  latency or energy;
- at least one weak-primary diagnostic family;
- explicit protocol settings for committee size, fault tolerance, primary
  distribution, fault filter, and view-change mode;
- explicit Stage 3 communication regime fields including
  `urlcc_finite_blocklength_v1`.

## Fixture Manifest Fields

Future fixture manifests should declare:

- `scenario_set_id`
- `family_id`
- `scenario_id`
- `fixture_id`
- `coverage_axis`
- `physics_regime`
- `geometry_visibility_regime`
- `channel_regime`
- `link_transmission_regime`
- `network_regime`
- `pbft_model_id`
- `accounting_model_id`
- `committee_size`
- `fault_tolerance`
- `primary_distribution`
- `fault_filter_mode`
- `view_change_mode`
- `deterministic_seed_policy`
- `topology_variant_ids`
- `expected_diagnostic_flags`
- `owner_scope_note`

These are manifest and diagnostic fields. They are not new core metrics.

## Topology Variant Requirements

Each implemented family should provide at least three topology variants unless
a future owner-approved exception is recorded:

- disconnected or weak baseline;
- sparse candidate;
- dense full-graph baseline.

Families may add review-only oracle candidates, but those labels must remain
diagnostics and must not enter deployment actor observations, policy inputs, or
action decisions.

## Metric Governance

Stage 5.0e adds no metric names.

Future fixture outputs must map metric-valued fields to:

- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The metric-valued list above is intentionally de-duplicated; `energy` appears
once and remains the registered energy concept.

Family ids, coverage axes, feasibility booleans, per-primary spread, route
labels, and owner-decision fields are diagnostics or identifiers unless a later
metric registration explicitly changes that status.

## Stage 5.0f Implementation Follow-Up

Stage 5.0f implements a minimal executable alpha suite from this design in:

- `docs/STAGE5_0F_TAU_CONSENSUS_FIXTURE_IMPLEMENTATION.md`
- `src/marl_topology/evaluation/calibration_fixture_suite.py`

The implementation still does not run calibration or select `tau_consensus`.

## Acceptance For Stage 5.0e

- The fixture-family design document exists.
- Required families cover geometry, channel/link, network resource, topology,
  deadline, and PBFT primary asymmetry.
- The document states that no fixture implementation, calibration run, tau
  selection, reward implementation, training, model work, or v5 migration occurs
  in Stage 5.0e.
- Stage 4.8 is explicitly smoke-test evidence only.
- Full graph remains baseline, not oracle.
- Oracle candidates are diagnostics only and not actor inputs.
- Metric-valued outputs map to existing metric governance concepts.

## Verification Commands

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json
```

## Deferred Work

- Implementing fixture builders.
- Running tau calibration.
- Selecting final `tau_consensus`.
- Extending the Stage 5.0d report to accept a full calibration manifest.
- Reward implementation and reward weight calibration.
- Training or model architecture work.

## Residual Risks

- Static fixture-family design cannot prove future scenario realism.
- Coverage axes may need revision after the first implemented calibration
  manifest exposes missing regimes.
- Owner risk tolerance and candidate tau values remain outside this stage.
