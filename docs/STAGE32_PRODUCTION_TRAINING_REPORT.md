# Stage 32 Production Training Report

- status: `execution_complete` (2000-scenario, 5-seed run)
- verdict: `infrastructure_validated_but_gnn_actor_did_not_beat_mlp` (honest negative result for the GNN upgrade)
- design contract: `docs/STAGE32_PRODUCTION_TRAINING_DESIGN_CONTRACT.md`
- decision: option B (GNN actor + repaired graph critic), owner-approved
- reproduce: `python scripts/train/stage32_production_training_run.py 2000 5`
- artifacts: `result_save/stage32_production_training/stage32_gnn_critic_production/`

## What Stage 32 executes

Stage 32 scales the Stage 31 readiness stack into a production training run with
the contract's capability upgrades, all other Stage 31 decisions inherited
unchanged (tau=0.9 hard gate, feasibility-first surrogate, budget-aware sampler,
mean-field PBFT, Dec-POMDP locality):

- Actor: the message-passing GNN edge scorer
  `local_message_passing_gnn_edge_scorer_v2`, used the way it was designed —
  each candidate physical edge contributes two directed ego-graph rows grouped by
  ego node; their scores are max-aggregated to a physical-edge score. It consumes
  the v2 constraint-aware actor-safe features.
- Critic: the Stage 27/28 repaired centralized graph value critic
  `centralized_message_passing_graph_value_critic_v1`, pretrained Stage-27-style
  on a diverse spread of topologies (held-out explained-variance metric), then
  used as the policy-gradient baseline (`advantage = standardized_signal - V`).
- Action: a variable proposal size (a learned per-edge inclusion threshold,
  budget-feasible by construction) replaces the fixed `N-1` proposal.
- Data: 2000 unique procedural scenario contexts, context-keyed leakage-checked
  train/eval/test split.
- Protocol: supervised warm start on teacher labels, then clipped policy-gradient
  fine-tune with the graph-critic baseline and keep-best on the eval signal, over
  5 seeds (3201-3205); MLP baseline on the same data for the head-to-head.
- Artifacts: a Stage 5.9-compliant run manifest (validated by the Stage 5.10
  dry-run validator) plus the training report; `result_save` allowlists extended
  with the `stage32_production_training` scope. Model-weight checkpoints are
  intentionally **not** persisted: the project's no-checkpoint discipline forbids
  `torch.save` in `src`, and the GNN underperformed the MLP so its weights are
  not the production model. Persisting weights would require a separate
  owner-approved unfreeze of the checkpoint gate.

## Results (2000 scenarios, 5 seeds)

Data: 2000 unique scenario contexts (0 duplicates), zero cross-split leakage,
split 1400/300/300, feasible fraction 0.69, full graph feasible 0.0.

| Metric | GNN actor (5-seed mean) | MLP baseline (3-seed) | Teacher ceiling |
| --- | ---: | ---: | ---: |
| held-out test tau-feasible | **0.239** | **0.447** | 0.623 |
| per-seed GNN test tau-feasible | 0.447, 0.103, 0.060, 0.447, 0.140 | (stable ~0.447) | |
| eval tau-feasible (mean) | 0.202 | 0.360 | |
| violation rate (mean) | 0.798 | 0.640 | |
| critic explained variance (held-out, mean) | **0.900** | n/a | |
| projection rejection rate | **0.000** | 0.000 | |
| mean selected edges (variable size) | 4.43 | 4.43 | |
| run manifest valid (Stage 5.10) | True | — | |

### Honest verdict

The Stage 32 capability bet — that the message-passing GNN actor would outperform
the Stage 31 MLP — **did not hold**. The GNN is **training-unstable**: 2 of 5
seeds reached MLP-parity (test 0.447, ~72% of the teacher ceiling) but 3 seeds
collapsed (0.06-0.14), so the 5-seed mean (0.239) sits **below** the stable MLP
(0.447). The simpler MLP remains the better, more reliable deployment actor.

What the run **did** validate:

- The repaired centralized graph value critic works: held-out explained variance
  ~0.90 (matching the Stage 27/28 ~0.95 capability).
- The budget-aware sampler keeps projection rejection at exactly 0.0.
- The procedural generator scales to 2000 unique, zero-leakage contexts.
- The run-manifest + Stage 5.10 validator + result_save artifact discipline hold
  end to end (a validated manifest + training report persisted; no model-weight
  checkpoints, preserving the project's no-checkpoint discipline).

### Recommendation

Keep the Stage 31 MLP actor as the production deployment actor (it reaches ~72%
of the achievable ceiling and is stable). The GNN is **not a blocker**; it is an
unsuccessful upgrade that needs stabilization before adoption: learning-rate
scheduling/warmup, seed ensembling or best-of-N seed selection, a larger
warm-start, or weight averaging across the converged seeds. The repaired critic
and the scaled, leakage-checked data are reusable regardless of the actor choice.

## Boundaries preserved

tau fixed at 0.9; mean-field PBFT kept; metric governance unchanged; Dec-POMDP
locality preserved (teacher/oracle/critic targets never enter actor inputs); no
v5 migration; COMA/Transformer/recurrent PPO out of scope.
