# Stage 19 - Supervised Actor Stack Rerun

Stage type: implementation-bearing model-training stage, bounded to
supervised actor-only reruns on Stage 18 disambiguated evidence.

Controlled object: local actor edge-scoring models trained from actor-safe
Stage 18 feature rows and soft/ranking actor targets.

Desired state: rerun the actor stack in the approved order
`MLP -> GNN -> GRU real multi-step subset -> LSTM after GRU sanity`, without
policy-gradient training, critic training, checkpoints, training artifact
writes, COMA, Transformer, reward-weight tuning, final tau selection, scale-up
training, or `v5` modification.

Source-awareness: this stage is derived from the Stage 18 disambiguated
evidence gate and EC-RULE project controls, not from direct migration of `v5`.

## Cybernetic Control Summary

Controlled object identified: the supervised local actor edge scorer subsystem
bounded to Stage 18 actor-safe evidence.

Desired state defined: finite non-collapsed supervised actor updates for MLP,
GNN, GRU, and conditional LSTM, with no RL, checkpoint, artifact, reward, tau,
COMA, Transformer, or `v5` side effects.

State variables defined: actor-safe feature tensor, soft utility target,
confidence weight, ranking-pair target, model loss, validation loss, pairwise
accuracy, temporal loss, output variance, update count, and boundary flags.

Sensors defined: unit tests, contract tests, Stage 19 JSON script report,
rubric score, pytest, harness validation, and no-forbidden-field source scans.

Actuators defined: editable training module, Stage 19 script, local model
config interfaces, docs, harness task, PROJECT_STATE, and tests.

Feedback loop present: Stage 18 before/after evidence rebuild is the baseline;
Stage 19 observes post-change supervised losses and boundary flags before
allowing only the next assembler-evaluation gate.

Verification plan present: run `python scripts\train\stage19_supervised_actor_stack.py`,
`python -m pytest -q`, `python harness\scripts\validate_tasks.py`, and the
rubric scoring command recorded in `harness/reports`.

Stability risk checked: the GNN improvement is small, temporal evidence is
limited, and Stage 20 must protect against over-interpreting supervised loss.

Observability gap checked: Stage 19 does not yet observe environment-side
projection behavior, consensus feasibility after assembler projection, or
deployment score stability.

Controllability checked: the only safe actuator was supervised actor-only
parameter update in memory; policy-gradient, critic rerun, checkpoint, and
artifact-writing actuators stayed disabled.

Disturbance checked: small evidence size, random initialization, train/validation
split sensitivity, and real multi-step scarcity remain active disturbances.

Delay or async risk checked: no async queue, service, or delayed artifact writer
is involved; the only timing risk is repeated script runtime.

Noise or flakiness checked: fixed seed `19` is used and metrics are reported as
single-seed gate evidence, not final statistical proof.

Decoupling checked: actor-safe features, actor targets, critic-only targets,
and assembler topology activation remain separate consumers.

Reliability or error control checked: negative tests and source scans guard
against false success from checkpoint writes, forbidden fields, policy-gradient
paths, COMA, Transformer, and `v5`.

Persistent learning update suggested: Stage 20 should add a harness-backed
follow-up sensor for environment-side assembler policy evaluation before any
PPO/MAPPO discussion resumes.

## Inputs

- Source dataset: `stage18_disambiguated_actor_evidence_v1`.
- Actor samples: `850`.
- Actor-safe feature schema: `stage19_stage18_actor_feature_tensor_v1`.
- Feature count: `31`.
- Pairwise ranking pairs: `412`.
- Real multi-step temporal subset: `72` sequences, `216` active sequence
  entries, time steps `0, 1, 2`.

Actor inputs are built from Stage 18 `actor_safe_view` only. The actor tensor
contains local identity/type, local position, local neighbor/link estimates,
local topology/projection history, local resource/conflict estimates, and local
message/history summaries.

Forbidden actor inputs remain excluded:

- oracle labels;
- global topology;
- consensus success probability;
- latency and energy objective outputs;
- reward surrogate;
- future outcomes;
- critic-only edge-delta targets.

## Targets

The supervised actor target is no longer hard add/remove/keep by default.
Stage 19 trains against:

- `actor_edge_utility_target`;
- `actor_edge_utility_confidence`;
- pairwise ranking targets over local candidate edges.

Global counterfactual deltas remain critic-only or diagnostic. They are not
used as actor input.

## Execution Order

1. Local MLP edge scorer.
2. Local GNN edge scorer.
3. Local GRU edge scorer on the real multi-step actor-safe subset.
4. Local LSTM edge scorer only after GRU sanity passes.

The GRU sanity condition is finite loss reduction with non-constant outputs.
LSTM is a comparison after GRU sanity, not a license to start recurrent
deployment or policy-gradient training.

## Results

Default Stage 19 run:

| Model | Initial loss | Final/train loss | Validation loss | Pairwise accuracy | Non-collapsed |
|---|---:|---:|---:|---:|---|
| MLP | `0.8884038925` | `0.6044725180` | `0.6004618406` | `0.9358974359` | yes |
| GNN | `0.8218849301` | `0.5975568295` | `0.5954169631` | `0.9358974359` | yes |
| GRU | `0.7573379874` | `0.5235753655` | n/a | n/a | yes |
| LSTM | `0.6419312954` | `0.5229729414` | n/a | n/a | yes |

GNN validation loss was lower than MLP by `0.0050448775`, so the recommended
starting actor for the next supervised evaluation gate is `GNN`. LSTM matched or slightly improved GRU on the real multi-step sanity subset after GRU passed,
but temporal models remain diagnostic until an environment-side assembler
policy evaluation says they are useful.

## Completion Gate

Passed:

- Stage 18 disambiguated evidence was consumed.
- MLP rerun completed.
- GNN rerun completed.
- GRU real multi-step sanity completed.
- LSTM was run only after GRU sanity passed.
- Actor parameters updated in the supervised actor-only path.
- Outputs were finite and non-collapsed.
- No checkpoint was written.
- No training artifact was written.
- No critic training, PPO/MAPPO, COMA, Transformer, scale-up training,
  reward-weight tuning, final tau selection, or `v5` modification occurred.

## Next-Stage Readiness Gate

Stage 19 does not authorize PPO/MAPPO or a full Stage 11-15 replay. It supports
a narrower Stage 20 gate:

`stage_20_supervised_actor_policy_evaluation_with_environment_assembler`

Stage 20 should evaluate the Stage 19 supervised actors through the
environment-side topology assembler and compare feasibility, projection
rejections, consensus reliability diagnostics, latency, energy, and actor-score
stability before any policy-gradient rerun is reconsidered.

## Residual Risks

- The dataset is still small: `850` edge samples and `412` ranking pairs.
- The GNN improvement over MLP is small and should be treated as a gate signal,
  not final architecture selection.
- GRU/LSTM were tested on a real multi-step subset, but temporal evidence is
  still limited.
- Stage 19 does not prove tau feasibility or deployment performance because
  the actor scores have not yet been evaluated through the assembler in a full
  environment loop.
