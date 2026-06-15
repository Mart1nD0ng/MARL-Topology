# Post-Task Self-Review - Stage 18 Evidence Rebuild With Disambiguated Actor Features And Targets

## Completed Task

Stage 18 rebuilt learning evidence after Stage 17 actor-label
disambiguation. It added actor-safe local topology/resource/projection/message
and link-estimate features, rebuilt actor targets into soft utility and
pairwise ranking targets, moved global counterfactual effects into critic-only
targets, added before/after contradiction reporting, updated documentation,
registered a harness task, added tests, and updated PROJECT_STATE.

Stage 18 did not train, rerun Stage 11-15, run PPO/MAPPO, implement a model,
add COMA or Transformer, tune reward weights, select final tau, modify v5,
create checkpoints, or write training artifacts.

## Intended Desired State

The desired state was an implementation-bearing evidence rebuild where actor
inputs and actor targets have matching semantics: local edge scorer inputs are
local and actor-safe, actor targets are soft/ranking/confidence-weighted, and
global edge-delta/oracle effects are critic-only.

## Actual Achieved State

Stage 18 reproduced the Stage 17 baseline before rebuild:

- raw sample count: `850`
- signature count: `142`
- contradiction cluster count: `142`
- hard-label contradiction rate: `1.0`

After rebuild:

- rebuilt sample count: `850`
- actor-safe signature count: `548`
- hard-label allowed subset contradiction rate: `0.0`
- soft utility targets: `850`
- ranking pairs: `412`
- samples moved to critic-only: `850`
- high-ambiguity samples downweighted: `850`
- remaining hard-label contradiction clusters: `0`

## Evidence

- `scripts/replay/stage18_evidence_rebuild_report.py` prints a report and
  records `artifact_written: false`.
- Stage 18 unit tests cover feature schema, target rebuild, evidence views,
  before/after stats, and owner-gated Stage 19 recommendation.
- Stage 18 contract tests cover docs, harness task, PROJECT_STATE, report-only
  script behavior, and no actor leakage.

## Gates Passed

- Stage 16 raw contradiction reproduction gate.
- Actor-safe feature rebuild gate.
- Forbidden actor field rejection gate.
- Soft utility target gate.
- Pairwise ranking target gate.
- Hard-label diagnostic-only gate.
- Critic-only global target separation gate.
- Report-only no-artifact gate.
- Harness registration and project-state sync gate.

## Gates Deferred

- Owner approval for Stage 19.
- Actual supervised actor rerun.
- PPO/MAPPO and scale-up training.
- COMA and Transformer ablations.
- Reward-weight tuning and final tau selection.

## New Risks

Soft utility targets are distilled from local link estimates plus global
counterfactual diagnostics, so they are suitable for soft/ranking supervision
but not for hard classification claims. Real multi-step evidence exists, but
sequence volume is still modest, so recurrent reruns should remain bounded.

## Regressions Checked

Actor-safe views do not contain global topology, oracle labels, raw consensus
metrics, raw latency/energy metrics, reward surrogate fields, edge-delta
targets, future outcomes, or critic-only fields. Stage 18 source files do not
add model, PPO/MAPPO, COMA, Transformer, checkpoint, v5, or training-artifact
paths.

## Required Stage 18 Questions

Does Stage 18 lower hard-label contradiction training risk? Yes. Contradictory
hard labels are no longer primary actor targets, and the hard-label allowed
subset contradiction rate is `0.0`.

Are actor local observations more sufficient than Stage 17? Yes. The rebuilt
actor signature count increased from `142` to `548` because local
topology/history/resource/projection context now disambiguates many previously
identical signatures.

Which hard labels were downweighted, softened, or moved to critic-only? All
`850` actor-edge samples were converted to soft utility targets and all
global edge-delta effects were moved to critic-only targets. Hard add/remove
and keep labels remain diagnostic-only in this dataset.

Does true Dec-POMDP ambiguity remain? Yes. Ambiguity from global quorum
context and oracle/nonlocal effects still exists, but it is now represented by
confidence and ambiguity metadata instead of hidden hard-label conflicts.

Is rerun of Stage 11-15 allowed? A full Stage 11-15 rerun, especially the
policy-gradient pilot, is not allowed. An owner-approved Stage 19 supervised
actor stack rerun on Stage 18 evidence is allowed by the data gate.

If allowed, which models should rerun? Rerun MLP first, then GNN. GRU can be
rerun on the real multi-step subset, and LSTM only after GRU sanity. PPO/MAPPO
is still blocked.

If not allowed, next step? Not applicable for the data gate; Stage 19 is the
recommended owner-gated next task. If the owner rejects the feature boundary,
the fallback is `stage_18b_actor_feature_boundary_owner_decision_or_target_rebuild`.

Are PPO/MAPPO, COMA, Transformer, and scale-up training still forbidden? Yes.
They remain blocked.

## Recommended Next Task

`stage_19_rerun_supervised_actor_stack_on_disambiguated_evidence`

## Owner Decision Required

Yes. Stage 18 does not self-authorize Stage 19 or any training execution.
