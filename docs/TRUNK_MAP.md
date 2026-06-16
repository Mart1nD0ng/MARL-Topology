# TRUNK MAP — what is the model trunk vs scratch/history

Authoritative classification of every component, so a reviewer can tell at a glance what is
load-bearing production trunk vs diagnostic/skeleton/scratch/history. Companion to
`CLEAN_PROJECT_MAP.md` (which describes the trunk); this file *classifies the whole repo* and
proposes the semantic rename that removes the stage1-35 naming.

Verified 2026-06-15: 912/914 tests pass; all core physics/channel/link/network/protocol/
evaluator/model/Stage-33-flow contracts pass. The 2 failures are non-structural (a moved-path
`v5` validator test; a Stage-23 policy-gradient micro-gate that returns owner_decision_required
by design). **The consolidation did not damage the trunk.**

## 2026-06-16 UPDATE — decentralized MARL production trunk (supersedes the Stage-33 ego/global path)

The production trunk is now the **decentralized CTDE multi-agent flow**, authoritatively
declared in `src/marl_topology/training/production_trunk.py`. It replaces the Stage-33
ego-graph + global-Plackett-Luce path as the production model; that path is **retained only
as the diagnostic baseline / ablation**.

- **Production actor:** `models/local_khop_gnn_edge_scorer.py`
  (`LocalKHopGNNEdgeScorer`, `local_khop_message_passing_gnn_edge_scorer_v1`) — a genuinely
  multi-hop yet Dec-POMDP-local message-passing edge scorer (receptive field = K hops via
  local neighbour signalling; verified by a receptive-field test). Fixes the v3 1-hop
  ego-graph ceiling. v3 is now a registered diagnostic baseline.
- **Production decoder (true end-to-end decentralization):**
  `training/policy_gradient/decentralized_sampler.py`
  (`DecentralizedPerNodeMutualSampler`) — each node ranks only its own incident edges within
  its radio budget; an edge activates iff both endpoints accept (mutual acceptance). The joint
  log-prob factorizes per agent (no double counting). The global Plackett-Luce + assembler is
  kept only as the centralized-decode ablation.
- **Production flow (genuine MARL, not a bandit):** `training/decentralized_marl.py`
  (`DecentralizedCTDEFlow`, `decentralized_ctde_mappo_v1`) — per-agent clipped PPO objective
  from the factorized per-owner log-probs + a training-only centralized graph critic (CTDE);
  genuinely sequential via a per-node energy battery (an action depletes it, shrinking the
  next-step feasible action set). Runnable driver + figures/tables/report:
  `scripts/train/decentralized_marl_production_training.py`.
- **Legacy designation:** `ACTIVE_STAGE33_GNN_MODEL_ID` (v3) and
  `ACTIVE_POLICY_GRADIENT_SAMPLER_ID` (global PL) are intentionally **left in place** as the
  diagnostic-baseline lineage; flipping them would break the legacy diagnostic flow for a pure
  rename. `production_trunk.py` is the authoritative production designation.

## Legend
- **TRUNK** — load-bearing production code (imported by the evaluator / actor / Stage-33 flow).
- **BASELINE** — registered diagnostic models/policies the trunk is measured against (inactive).
- **SKELETON** — Stage-2 distance-only tier, kept only for baselines & leakage tests.
- **SCRATCH** — research/experiment code & artifacts, NOT imported by trunk (safe to ignore for analysis).
- **HISTORY** — append-only development record (stage docs, per-stage result dumps).

---

## 1. TRUNK — `src/marl_topology/` (keep; stage-named ones proposed for semantic rename)

### Simulation / physics chain (all semantic already)
`scenario/{scene,urban_grid}.py`, `geometry3d/{primitives,visibility}.py`, `channel/model.py`,
`link/transmission.py`, `network/communication.py`, `protocol/{stdma_scheduler,
message_matrix_adapter,pbft_reliability,quorum_tail,pbft_accounting}.py`,
`topology/candidate_graph.py`, `budgets.py`, `metrics/registry.py`.

### Stage-named TRUNK modules → proposed semantic name (importers)
| current (stage-named) | role | → proposed | importers |
|---|---|---|---|
| `data/stage21_objective_stack_evidence.py` | **the production evaluator** (physics→PBFT) | `data/objective_stack_evaluator.py` | 17 |
| `data/stage31_scenario_generator.py` | procedural urban scenarios + τ-gradient + SA teacher | `data/scenario_generator.py` | 8 |
| `data/stage33_graph_structure_dataset.py` | graph-structure dataset for the MARL flow | `data/graph_structure_dataset.py` | 6 |
| `data/stage31_production_dataset.py` | production-scale dataset + leakage-checked splits | `data/production_dataset.py` | 3 |
| `data/stage22_action_semantics_evidence.py` | physical-link (undirected) evidence over the stack | `data/physical_link_evidence.py` | 4 |
| `data/stage21_assembler_aware_targets.py` | assembler-aware edge-priority targets | `data/assembler_aware_targets.py` | 4 |
| `data/learning_evidence_stage18.py` | evidence rebuild w/ disambiguated actor features | `data/learning_evidence_disambiguated.py` | 4 |
| `data/learning_evidence_stage16.py` | evidence-fixture quality audit | `data/learning_evidence_quality_audit.py` | 3 |
| `training/mappo/stage25_pilot.py` | base MAPPO rollout/GAE/loss protocol | `training/mappo/base_protocol.py` | 6 |
| `training/mappo/stage28_repaired_critic_pilot.py` | repaired-critic protocol (graph payload) | `training/mappo/repaired_critic_protocol.py` | 4 |

### Actor / critic / decode
- **Production actor (active, registry-gated):** `models/local_gnn_edge_scorer.py`
  (`LocalMessagePassingGNNV3ResidualNorm`). ⚠️ *direction*, not a passed artifact — the Stage-33
  GNN training gate failed (all 5 seeds collapsed, no checkpoint). The campaign's feasibility
  numbers come from the SCRATCH artifact `logs/_artifacts_step3.pt` (`GlobalMessagePassingActor`),
  not this registry model. Do not claim a successfully MAPPO-trained production GNN.
- **Critics (training-only, CTDE):** `models/centralized_message_passing_graph_critic.py`
  (+ `models/quorum_tail_pool.py`), candidate `models/enriched_centralized_mlp_critic.py`.
- **Decode/assembly:** `policies/{physical_link_assembler,topology_assembler}.py`,
  `training/policy_gradient/samplers.py` (active = global Plackett-Luce top-k). ⚠️ the learned
  actor's deployed decode is scene-**global** (the hidden-centralization gap); genuinely-local
  `policies/decentralized_baselines.py` is wired to non-learning baselines only.
- **MARL flow (one):** `training/production_mappo_adapter.py` (Stage-33 CTDE) + the
  `training/mappo/` and `training/policy_gradient/` helpers + `training/critic_*`.

## 2. BASELINE (registered, inactive — keep)
`models/{local_mlp_edge_scorer,centralized_mlp_critic}.py`, GNN v2 / role-resource v3 (in
`local_gnn_edge_scorer.py`), `policies/{baselines,decentralized_baselines}.py`.

## 3. SKELETON (Stage-2 distance-only — keep for baselines & leakage tests)
`env/{dec_pomdp_env,dec_pomdp_schema}.py`, `link/simple_link_model.py`,
`topology/{evaluator,oracle}.py`, `protocol/consensus.py`, `objectives/latency_energy.py`.

## 4. SCRATCH — NOT imported by trunk (ignore for trunk analysis)
- `logs/` — 91 research `.py` (phase*, step*, the density-axis campaign `campaign_common.py`,
  `fast_stage21.py`, `recovery_scaling.py`, viz scripts) + 53 `.pt`/`.pkl`/`.json` artifacts
  (~9.8 MB), incl. `_artifacts_step3.pt`. **Confirmed: 0 imports from `src/` or `tests/`.**
- `result_save/` (~11 MB) — per-stage run dumps (manifests/metrics/CSVs/PNGs).
- `scripts/replay/` — 23 contract/report runner drivers (verification, not production entry).

## 5. HISTORY (append-only record — keep, but it is not the trunk)
- `docs/STAGE*.md` — 140 per-stage development records (STAGE3–STAGE34).
- `docs/*_CONTRACT.md` + decision records — **43 DURABLE contracts** (these ARE part of the
  spec; not history). Keep at `docs/` top level.
- `scripts/train/stage*.py` — per-stage training drivers (Stage-33 is the production driver).

## Retired (recover from git tag `v0-full-import`)
Recurrent/temporal/attention scorers; pre-MARL supervised lineage; Stage-32 custom loop; Stage-26/
29/30 diagnostics. Removed with their tests & drivers.
