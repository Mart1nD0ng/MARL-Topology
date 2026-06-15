# Stage 7 Learning Evidence Dataset Contract

## Purpose

Stage 7 converts the existing simulator, topology evaluator, oracle diagnostics,
objective metrics, reward-surrogate components, and actor-safe schema into
reproducible learning evidence. It does not train a model, implement actor or
critic networks, create checkpoints, or migrate legacy code.

Controlled object: the learning evidence dataset boundary between evaluation
records and future learning systems.

Desired state: evidence rows are reproducible, actor-safe, and separated into
views so future learner code can consume the correct fields without leaking
oracle, objective, or future-outcome information into deployment actor inputs.

## Required Data Views

### actor_safe_view

Responsibility: carry only deployment-visible actor observations already
validated by the Dec-POMDP schema and Stage 6.1 actor-safe batch projection.

Allowed source: `ActorObservation` payloads and `ActorSafeBatch` rows.

Forbidden:

- oracle labels in `actor_safe_view`
- objective metrics in `actor_safe_view`
- future outcomes in `actor_safe_view`
- reward surrogate components in `actor_safe_view`
- global topology in `actor_safe_view`
- selected topology, full candidate graph, consensus result, latency, energy,
  topology diagnostics, feasibility labels, edge-delta labels, or centralized
  critic features in `actor_safe_view`

### critic_centralized_view

Responsibility: carry training-only centralized context for future CTDE work.

Allowed fields include scenario id, node ids, candidate edge ids, selected edge
ids, registered metric names, and view role markers. This view is never a
deployment actor input.

### learning_target_view

Responsibility: carry training-only targets derived from topology evaluation,
counterfactual edge deltas, objective metrics, and feasibility labels.

Minimum Stage 7.0 target rows use:

- `edge_id`
- `action_type`
- `delta_consensus_success_probability`
- `delta_latency`
- `delta_energy`
- `delta_feasibility`
- `target_role = learning_target_only`

### diagnostics_view

Responsibility: carry audit-only evidence, fixture names, topology family,
oracle-candidate diagnostic markers, tau requirement markers, and report
metadata. Diagnostics may explain rows but must not enter deployment actor
inputs.

## Row Contract

Each in-memory evidence row must include:

- `scenario_id`
- `topology_id`
- `topology_name`
- `selected_edges`
- `consensus_success_probability`
- `latency`
- `energy`
- `topology_diagnostics`
- `feasible_under_tau_requirement`
- `actor_safe_rows`
- `critic_view`
- `learning_targets`
- `diagnostics`

Registered objective metrics remain row-level evidence and training targets.
They are explicitly excluded from `actor_safe_rows`.

## Artifact Boundary

Evidence artifact export is allowed only after owner approval and run-manifest
validation. The artifact scope must be `evidence_dataset_only`.

Allowed outputs:

- manifest file
- learning evidence dataset file

Forbidden outputs:

- checkpoint files
- model files
- training logs
- optimizer state
- legacy path outputs
- any artifact path outside `result_save/evidence_dataset_only`

## Tests

Required tests:

- actor view leakage negative cases
- oracle labels excluded from actor view
- objective metrics excluded from actor view
- learning targets separated from actor-safe rows
- edge-delta targets exist for add, remove, and keep actions where applicable
- artifact writer rejects missing owner approval
- artifact writer rejects invalid manifest or wrong artifact scope
- evidence export writes only evidence files under a validated temporary
  artifact root during tests
- no model, checkpoint, training, final tau selection, or legacy migration code
  is introduced

## Deferred

Stage 8 may define an actor policy interface after a Stage 7 data quality
report. Training execution, actor/critic/GNN/LSTM implementation, reward weight
calibration, and final tau selection remain blocked.

## Stage 7.1 Quality Sensor

Stage 7.1 audits evidence quality before Stage 8.

Blocking checks:

- view coverage;
- actor-safe leakage;
- weak/sparse/dense topology family coverage;
- metric range sanity;
- edge-delta action coverage;
- nonzero target deltas.

Warnings such as no feasible rows at the declared tau or low feasibility class
diversity must be reported as training-coverage limitations. They do not by
themselves authorize training, and they do not block Stage 8 interface-contract
design when structural gates pass.
