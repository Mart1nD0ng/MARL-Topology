# Stage 32 Decision Packet

## Owner decision (requested)

- recommended option: `option_b_execute_production_training_with_gnn_actor_and_repaired_critic`
- recommended next task: `stage32_production_training_execution_with_gnn_actor_repaired_critic_and_scaled_data`
- design contract: `docs/STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md`
- scale-up approved: `False` (this packet requests it)
- owner decision required: `True`
- rationale: Stage 31 resolved all six Stage 26-30 blockers and verified
  `ready_for_production_training_scale_up`; the only remaining gap is model
  capacity and scale, which production training provides.

## Carry-over readiness (Stage 31, verified)

| Blocker | Status | Evidence |
| --- | --- | --- |
| B0 tau=0.9 reachability | `RESOLVED` | measured feasibility gradient, tau not lowered/faked |
| B1 reward/objective alignment | `RESOLVED` | feasibility-first surrogate, inversion ~0.025 vs ~0.11 |
| B2 data scale | `RESOLVED` | 150 unique contexts, zero-leakage split (scalable) |
| B3 projection friction | `RESOLVED` | budget-aware sampler, 0.000 tx_budget rejection |
| B4 reliability margin | `RESOLVED` | violation moves down, not up, under training |
| learning signal | `RESOLVED` | held-out feasibility improves toward the teacher ceiling |

## Options

- `option_a_hold`: keep the Stage 31 readiness result, run no scale-up. Lowest
  value; the verified stack stays unused.
- `option_b_execute` (recommended): implement the GNN actor + repaired graph
  critic + variable proposal size, scale the generator to thousands of contexts,
  and run multi-seed production training with the frozen surrogate and sampler,
  gated on reliability/critic/projection.
- `option_c_partial`: GNN actor only, keep a simple baseline (no graph critic).
  Cheaper but lower value function quality.
- `option_d_data_first`: scale the data to thousands and re-verify readiness
  before training. Reasonable as a precursor to option B if the owner wants a
  fresh readiness check at scale first.

## What option B authorizes

- GNN actor `local_message_passing_gnn_edge_scorer_v2` as the active actor.
- Centralized graph value critic `centralized_message_passing_graph_value_critic_v1`
  as the policy-gradient baseline.
- A variable proposal-size head replacing the fixed `N-1` proposal.
- Scaled scenario generation, leakage-checked split, and warm-start teacher labels.
- Multi-seed training execution, checkpoints, and run-manifest-validated artifacts
  (pending the artifact-root decision below).

## What stays blocked

- Lowering or selecting a final tau (tau stays 0.9).
- v5 code migration.
- COMA, Transformer, recurrent PPO.
- A message-level PBFT simulator.
- Reward-weight tuning beyond the Stage 31 owner-approved recalibration.

## Open owner decisions

1. Approve `option_b` execution? (`yes` / `no`)
2. Artifact root for checkpoints and training artifacts:
   - (recommended) extend the `result_save` allowlist with one new scope
     `stage32_production_training` and validate manifests with the Stage 5.10
     validator; or
   - use a dedicated artifact root outside `result_save`.
3. Scale and compute budget:
   - scenario count (recommended 2,000-5,000),
   - seed count (recommended 5),
   - training step / wall-clock budget.

## Boundary

This packet recommends one next stage only. It does not self-authorize Stage 32
execution, scale-up training, checkpoint creation, the artifact-root change, or
any tau change. Execution begins only after the owner answers the open decisions.
