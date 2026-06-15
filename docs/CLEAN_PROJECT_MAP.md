# Clean Project Map — Production Main Body

Map of the consolidated MARL-Topology main body (2026-06-15). Everything below is the canonical
runnable surface; retired alternatives are recoverable from git tag `v0-full-import`.

Design thesis being served: **MARL (MAPPO/CTDE) plans a decentralized communication topology for
urban 3D V2X networks that satisfies a PBFT consensus reliability constraint (per-scene success
≥ τ = 0.9) while minimizing latency and energy, under variable node counts N.**

---

## 1. Simulation environment

Two tiers — do not confuse them.

**Production tier (the real physics).** `data/stage21_objective_stack_evidence.py`
(`Stage21ObjectiveStackEvaluator`) chains:

| Stage | Module | Role |
|---|---|---|
| Scene | `scenario/scene.py`, `scenario/urban_grid.py` | 3D Manhattan grid, buildings/roads/lanes/RSUs, mobility |
| Visibility | `geometry3d/primitives.py`, `geometry3d/visibility.py` | 3D ray-box LoS/NLoS blockage |
| Channel | `channel/model.py` | FSPL default; opt-in TR 37.885 V2V + TR 38.901 UMi, NLOSv, shadowing, SINR |
| Link | `link/transmission.py` | URLLC finite-blocklength PER, deadline retransmission, latency/energy |
| Network | `network/communication.py` | multi-hop routing, per-hop SINR, product delivery |
| MAC | `protocol/stdma_scheduler.py` | SINR-validated spatial-reuse TDMA |
| Matrices | `protocol/message_matrix_adapter.py` | per-phase delivery + relay reach + wired RSU backhaul |
| Consensus | `protocol/pbft_reliability.py`, `protocol/quorum_tail.py` | analytic 3-phase PBFT, τ = 0.9 feasibility |
| Accounting | `protocol/pbft_accounting.py` | latency/energy, reliability-free (metric separation) |
| Graph | `topology/candidate_graph.py` | candidate edge set from a scene |

**Skeleton tier (Stage-2, distance-only — baselines & leakage tests only):**
`env/dec_pomdp_env.py` (`MinimalDecPOMDPEnv`), `link/simple_link_model.py`, `topology/evaluator.py`,
`topology/oracle.py`, `protocol/consensus.py`, `objectives/latency_energy.py`.

Dec-POMDP boundary contract: `env/dec_pomdp_schema.py` (whitelist/blacklist validators).

## 2. Configuration

There is **no `configs/*.yaml`** — configuration is code-level:

- Physics regime / scenario: `data/stage31_scenario_generator.py` (`ProductionScenarioConfig`,
  `PhysicsRegime`); endpoint budgets in `budgets.py`.
- Training/run: `Stage33GNNStabilityConfig` in `training/production_mappo_adapter.py`.
- Reward surrogate: `objectives/surrogate_signal.py` (`feasibility_first_barrier_v2`),
  normalization refs in `objectives/normalization.py`.
- Metric governance: `metrics/registry.py`. Model gating: `models/model_registry.py`.
- τ = 0.9 is pinned in-source and audited against `docs/TAU_DECISION_RECORD.md`.

## 3. Model architecture

**Production actor (one):** `LocalMessagePassingGNNV3ResidualNorm`
(`models/local_gnn_edge_scorer.py`, id `local_message_passing_gnn_edge_scorer_v3_residual_norm`) —
hidden 48, 3 message-passing layers, residual + LayerNorm. The only entry gated
`active_for_stage33_production`.

**Diagnostic baselines / inactive (kept, registered):** local MLP edge scorer
(`models/local_mlp_edge_scorer.py`), GNN v2 base and role-resource v3 (same file as production),
centralized MLP critic (`models/centralized_mlp_critic.py`).

**Critics (training-only, CTDE):** active graph value critic
(`models/centralized_message_passing_graph_critic.py`, with opt-in quorum-tail pooling in
`models/quorum_tail_pool.py`); enriched MLP value critic candidate
(`models/enriched_centralized_mlp_critic.py`) compared against it in Stage-27 selection.

**Assembly / decode:** `policies/physical_link_assembler.py`, `policies/topology_assembler.py`
(env-side projection), `training/policy_gradient/samplers.py` (active = Plackett-Luce top-k).
Genuinely-local mutual-acceptance decode (`policies/decentralized_baselines.py` +
`env/dec_pomdp_schema.py`) is currently wired to non-learning baselines only — see the open
decentralization gap in the README.

## 4. MARL training flow (one)

`training/production_mappo_adapter.py` (Stage 33, `run_fixed_protocol`):
1. behaviour-cloning warm start from the search-teacher edges;
2. Stage-27 graph value-critic pretraining (`training/critic_repair_trainer.py`,
   `training/critic_dataset.py`, `training/return_normalization.py`);
3. clipped on-policy actor-critic fine-tune using `training/mappo/{advantages,losses,rollout}.py`
   and the rollout/loss helpers in `training/mappo/{trainer,stage25_pilot,stage28_repaired_critic_pilot}.py`
   and `training/policy_gradient/pilot_runner.py`;
4. keep-best validation gating; run-manifest validation (`training/run_manifest_validator.py`).

CLI entry: `scripts/train/` (Stage 33 driver). Fair evaluation/report drivers live in `scripts/replay/`.

## 5. Retired in the consolidation (recover from tag `v0-full-import`)

- **Deployment-actor variants:** recurrent GRU/LSTM scorers; temporal GNN and attention GNN variants.
- **Pre-MARL supervised lineage:** Stage 11/12/14/19/21 supervised actor/critic trainers, the
  supervised batching/losses helpers, and the Stage 21→23 supervised fair-evaluation gate.
- **Parallel/archived flows:** the Stage 32 custom training loop, the Stage 31 readiness probe,
  the Stage 15 PPO toy pilot, the Stage 31 surrogate-recalibration helper.
- **Superseded diagnostics:** Stage 26 health, Stage 29 pre-scale decision, Stage 30 repair
  diagnostics, graph-necessity metrics.
- Each was removed together with its tests and `scripts/` drivers. Root literature-survey scratch was
  also removed. `logs/` and `result_save/` provenance were left untouched.
