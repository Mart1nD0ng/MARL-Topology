# Stage 4.7 PBFT Application Evaluation Report

## Controlled Object

The controlled object is the Stage 4.7 application evaluation report:

`src/marl_topology/evaluation/stage4_application_report.py`

It combines Stage 4.5 baseline/oracle-candidate review rows with Stage 4.6
protocol latency and energy accounting diagnostics.

## Desired State

Stage 4.7 should provide a single deterministic evaluation sensor before any
objective or training work:

- compare topology candidates with registered metric concepts only;
- show `consensus_success_probability`, `latency`, `energy`, and
  `topology_diagnostics` together;
- keep reliability feasibility separate from latency and energy ranking;
- preserve full graph as baseline, not oracle;
- preserve oracle-candidate labels as review diagnostics, not deployment actor
  inputs;
- confirm Stage 4.6 accounting is present for every topology row.

## Implemented Interfaces

Active report id:

`stage_4_7_pbft_application_evaluation_report`

Public interfaces:

- `STAGE4_7_REPORT_STAGE_ID`
- `Stage47EvaluationReportConfig`
- `Stage47TopologySummary`
- `build_stage4_7_pbft_application_evaluation_report`

Replay helper:

`scripts/replay/stage4_7_pbft_application_evaluation_report.py`

The replay helper prints JSON only and does not write result files.

## Report Flow

1. Build the Stage 4.5 baseline/oracle-candidate report.
2. Flatten registered metric rows into a metric table.
3. Summarize topology rows with:
   - topology name and family;
   - selected edge count;
   - full-graph-baseline flag;
   - oracle-candidate flag;
   - deployment actor input flag;
   - reliability feasibility;
   - `consensus_success_probability`;
   - `latency`;
   - `energy`.
4. Select feasible topology views:
   - lowest feasible latency;
   - lowest feasible energy.
5. Emit checks for metric governance, protocol accounting, full-graph boundary,
   oracle boundary, and no training or v5 migration.

## Metric Governance

Stage 4.7 does not add metric names.

The report uses only:

- `consensus_success`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`

The lowest-latency and lowest-energy feasible topology summaries are report
views, not new metrics.

## Protocol Boundary

Reliability comes from Stage 4.4 expected-initiator PBFT reliability through
the Stage 4.5 report.

Latency and energy come from Stage 4.6 protocol accounting through the Stage
4.5 report.

Stage 4.7 does not:

- compute a new PBFT reliability formula;
- alter quorum or deadline semantics;
- rename network delivery as consensus reliability;
- implement view change;
- implement strict Byzantine adversary modeling.

## Boundary

Stage 4.7 does not:

- implement reward;
- train a model;
- implement actor, critic, COMA, GNN, or LSTM behavior;
- create replay datasets;
- export oracle labels as deployment actor inputs;
- migrate or run v5 code.

## Tests

Implemented tests cover:

- report stage id and source stage;
- registered metric rows and no new metric names;
- reliability feasibility depends only on the declared reliability threshold;
- lowest feasible latency and energy views are selected from feasible rows;
- full graph remains baseline and not oracle;
- oracle-candidate is not deployment actor input;
- Stage 4.6 protocol accounting diagnostics are present;
- replay script is print-only;
- source scan rejects v5, old metric aliases, training, actor, critic, COMA,
  MAPPO, and reward implementation routes.

## Residual Risks

- The report currently uses the deterministic Stage 4.5 reference scenario.
- Feasible topology views are descriptive selections, not final objectives.
- A future reward contract must decide how reliability plateaus and
  latency/energy tradeoffs are used for learning.
- Larger scenarios may require pagination, file output contracts, or stable
  scenario fixture identifiers.
