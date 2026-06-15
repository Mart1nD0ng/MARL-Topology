# Stage 33 Production GNN Artifact Policy

## Controlled Object

The controlled object is artifact creation: manifests, reports, failure attribution, dataset split metadata, and any potential model checkpoint.

## Policy

Stage33 permits manifest-validated reports under:

`result_save/stage33_gnn_stability_repair/stage33_fixed_protocol/`

Stage33 does not permit claiming a production checkpoint unless the production actor gate passes.

## Actual Artifacts

Written artifacts:

- `manifest.json`
- `training_report.json`
- `failure_attribution_report.json`

Checkpoint status: no checkpoint written.

Reason: Stage33 failed. The active v3 production candidate did not pass the collapse gate, and the selected metric winner was archived v2. Creating or claiming a production GNN checkpoint would be misleading.

## Manifest Status

The Stage33 manifest validates through the dry-run manifest validator and records:

- active model id: `local_message_passing_gnn_edge_scorer_v3_residual_norm`
- critic id: `centralized_message_passing_graph_value_critic_v1`
- sampler id: `physical_plackett_luce_top_k_sampler`
- checkpoint_creation_allowed: `false`
- artifact_write_allowed: `manifest_validated_reports_only`

## Residual Risk

There is no production actor artifact after Stage33. Owner decision is required before any Stage34-style artifact validation or checkpoint policy change.
