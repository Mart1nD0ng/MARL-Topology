# Stage 17 Actor Observability and Label Disambiguation

## Controlled Object

The controlled object is the consistency between actor-safe local observation
signatures and edge learning labels. Stage 17 is a diagnostic/evidence stage.
It does not train, rerun PPO/MAPPO, rerun Stage 11-15, implement a model,
introduce COMA or Transformer, tune reward weights, modify v5, create
checkpoints, or write training artifacts.

## Desired State

The desired state is a report that answers whether actor-safe local
observations can support the current hard add/remove/keep edge labels. The
report must identify contradiction clusters, classify suspected causes, and
recommend whether to add actor-safe features, change target semantics, add
local history/messages, or keep the ambiguity critic-only.

## Executable Deliverables

- `src/marl_topology/data/actor_label_disambiguation.py`
- `scripts/replay/stage17_actor_label_disambiguation_report.py`
- `tests/unit/test_stage17_actor_label_disambiguation.py`
- `tests/contract/test_stage17_actor_observability_contract.py`

## Actor Observation Signature

Signature schema: `actor_safe_edge_observation_signature_v1`.

Fields used:

- `agent_id`
- `agent_kind`
- `time_step`
- `local_position_m`
- one incident entry from `local_neighbor_observations`
- `local_messages`
- `local_history`

Forbidden fields are rejected. The signature does not include global topology,
selected topology, oracle labels, consensus metrics, reward surrogate fields,
critic features, future outcome, or centralized training state.

## Detection Method

For each actor-safe row and each incident neighbor edge, Stage 17 joins the
actor observation signature with the row's edge-delta targets. It then groups
samples by identical rounded signature and checks conflicts in:

- add/remove/keep recommended action;
- raw counterfactual action;
- delta consensus_success_probability direction;
- delta latency direction;
- delta energy direction;
- feasibility-changing direction.

Each contradiction cluster reports:

- observation signature summary;
- affected agent and edge;
- number of samples;
- label distribution;
- metric delta distribution;
- scenario/topology ids;
- suspected cause;
- suggested fixes.

## Stage 16 Analysis Result

Stage 17 analyzed the Stage 16 in-memory evidence set:

- samples analyzed: `850`;
- actor observation signatures: `142`;
- contradiction clusters: `142`;
- contradictory samples: `850`;
- contradiction rate by signature: `1.0`;
- actor local observations sufficient for current hard labels: `false`.

Dominant causes:

- `missing_previous_topology_or_history`: `142` clusters.
- `missing_local_resource_context`: `142` clusters.
- `hard_label_should_be_soft_or_ranked`: `142` clusters.
- `oracle_label_not_actor_observable`: `140` clusters.
- `target_depends_on_global_context`: `120` clusters.
- `true_dec_pomdp_ambiguity`: `120` clusters.

The main finding is that current hard labels are not functions of actor-local
edge observations alone. The same local edge observation can appear under
different topology variants, where the target changes because the edge is
already selected, because removing or adding it changes global quorum
connectivity, or because dense/full topology resource effects are not visible
in the local edge feature.

## Cause Classification

### `missing_previous_topology_or_history`

The same local edge signature can map to `add_edge` when the edge is absent and
`remove_edge` when the edge is present. Current actor observations do not
include previous/current incident selected-edge state or projection history.

Suggested fix:

Add actor-safe local history of previous incident selected edges or previous
assembler projection outcome, or transform add/remove labels into
topology-state independent edge utility/ranking targets.

### `target_depends_on_global_context`

Consensus and feasibility deltas depend on global leader component and quorum
connectivity. The actor sees a local link, not whether that edge is globally
critical.

Suggested fix:

Do not force a hard actor label from a global counterfactual. Move this signal
to critic-only heads or use local ranking/soft targets guided by centralized
critic training.

### `missing_local_resource_context`

Latency and energy directions conflict because the effect of adding or removing
an edge depends on local budget, conflict, and current topology/resource state
that is not in the signature.

Suggested fix:

Add actor-safe local resource summaries such as local outgoing budget, local
rx/tx capacity use, local conflict group occupancy, or previous local
projection diagnostics. If those cannot be made deployment-local, keep the
effect critic-only.

### `hard_label_should_be_soft_or_ranked`

Many clusters mix keep/add/remove recommendations or have deltas that are
directionally sensitive to topology context. A single hard label discards useful
ordering information.

Suggested fix:

Replace hard add/remove/keep labels with soft delta targets, pairwise ranking,
or advantage-weighted edge scores.

### `oracle_label_not_actor_observable`

Oracle candidate rows are useful diagnostics but are not actor observations.
They should not become hard local actor labels without a transformation that
removes oracle-only context.

Suggested fix:

Keep oracle outputs diagnostic or target-only. Prefer critic-only handling or
distill only local, actor-observable ranking signals.

### `true_dec_pomdp_ambiguity`

Some ambiguity is inherent under the current Dec-POMDP boundary: the local
actor cannot infer global quorum role or counterfactual topology value from
one incident edge observation.

Suggested fix:

Owner decision is required: either allow additional local messages/history,
accept critic-only credit assignment for this ambiguity, or change the target
to a local ranking/soft objective.

## Completion Gate

Passed. Stage 17 can build actor-safe signatures, detect contradiction
clusters, classify suspected causes, and report fixes without training or
artifact writes.

## Next-Stage Readiness Gate

Failed for training reruns. Stage 11-15 rerun is not allowed because the
current hard labels are not fully actor-observable.

The next safe stage is:

`stage_18_evidence_rebuild_with_disambiguated_features_or_targets`

Alternative owner decision:

`owner_decision_on_dec_pomdp_feature_boundary`

## Stage 17 Answers

Actor local observations are not sufficient to support the current hard labels.

The main contradiction source is that the current labels encode topology-state
and global counterfactual effects, especially previous/current edge selection,
quorum-criticality, and resource/context effects not visible to the local edge
signature.

The next step should not be generic evidence expansion. It should rebuild
evidence with disambiguated features/targets:

- add actor-safe local history/resource/message summaries where allowed;
- change hard labels into soft/ranking targets;
- keep global counterfactual value in critic-only targets where needed;
- explicitly decide whether the Dec-POMDP feature boundary should include more
  local history/messages.

Rerunning Stage 11-15 is not allowed yet.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
```
