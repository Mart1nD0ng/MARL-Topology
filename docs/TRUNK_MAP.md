# TRUNK MAP — what is the model trunk vs scratch/history

Authoritative classification of every component, so a reviewer can tell at a glance what is
load-bearing production trunk vs diagnostic/skeleton/scratch/history. Companion to
`CLEAN_PROJECT_MAP.md` (which describes the trunk); this file *classifies the whole repo* and
proposes the semantic rename that removes the stage1-35 naming.

Verified 2026-06-15: 912/914 tests pass; all core physics/channel/link/network/protocol/
evaluator/model/Stage-33-flow contracts pass. The 2 failures are non-structural (a moved-path
`v5` validator test; a Stage-23 policy-gradient micro-gate that returns owner_decision_required
by design). **The consolidation did not damage the trunk.**

## 2026-06-17 UPDATE — RECOVERED production trunk (the validated 0.82 result, restored into src)

The production trunk is the **recovered, validated** decentralized pipeline that produced the
project's best result (held-out decentralized feasibility **0.82**, N=8 **0.957**, under the full
TR 37.885 stochastic stack at the 4-RSU / 20 dBm operating point — `docs/URBAN_V2X_RESEARCH_LOG.md`
Step-3). It was recovered from the deleted `logs/` research path into src and is reproducible
from a frozen artifact.

- **Production actor:** `models/message_passing_graph_edge_scorer.py`
  (`MessagePassingGraphEdgeScorer`, `kround_message_passing_graph_edge_scorer_v1`) — a K-round
  bidirectional message-passing GNN edge scorer (decentralized-with-communication: K hops of
  local neighbour signalling, no global-state shortcut). This is the actor that produced the
  validated result.
- **Production decoder (true end-to-end decentralization):**
  `policies/decentralized_mutual_acceptance.py` (`local_mutual_assemble`) — each node ranks only
  its own incident edges within its radio budget (logit >= 0); an edge activates iff both
  endpoints accept. Per-node computable, zero global state. `global_argsort_assemble` is kept
  only as the centralized-decode ABLATION (its cost is ~0 at the multi-RSU operating point).
- **Recipe / eval (BC distillation + CTDE critic-planner):** `training/decentralized_distillation.py`
  — BC-distil the (budget-aware SA or critic-planner) teacher into the actor with weight decay +
  early stopping + keep-best; the centralized graph critic (`models/
  centralized_message_passing_graph_critic.py`) is training-only (CTDE) and never reaches the
  deployed actor. Dataset-shard I/O lives in the drivers/tests (src stays I/O-free).
- **Reproduction:** `scripts/train/reproduce_recovered_step3.py` loads the frozen
  `_artifacts_step3.pt` (3 actors + critic + norm; under the gitignored `recovered_artifacts/`)
  and reproduces 0.82; pinned by `tests/unit/test_recovered_step3_reproduction.py` (skips if the
  artifacts are absent).
- **Load-bearing Stage-33 infrastructure (RETAINED, not a baseline):** `training/
  production_mappo_adapter.py` (`build_row_contexts`), `data/stage33_graph_structure_dataset.py`,
  `training/mappo/stage28_repaired_critic_pilot.py` (`_graph_payload`, the 8/8-dim actor-safe
  feature schema). The recovered trunk depends on these.

### DELETED 2026-06-17 — the unvalidated 2026-06-16 "new trunk"
The 2026-06-16 rebuild (`training/decentralized_marl.py` `DecentralizedCTDEFlow`,
`models/local_khop_gnn_edge_scorer.py`, `training/policy_gradient/decentralized_sampler.py`,
`training/production_trunk.py`, `scripts/train/decentralized_marl_production_training.py`) was a
fresh re-implementation that **never reproduced the validated result** (it ran at ~0 under
default config). It was deleted so the repo has ONE trunk (the recovered, validated one). Lesson
recorded: the working pipeline + its trained artifact had been left in `logs/` scratch and never
wired into src — the production default must equal the validated best, not a re-derivation.

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
  GNN training gate failed (all 5 seeds collapsed, no checkpoint). NOTE: this whole "Actor/
  critic/decode" subsection describes the SUPERSEDED Stage-33 ego-graph + global-Plackett-Luce
  path — see the 2026-06-16 update at the top of this file for the decentralized CTDE production
  trunk that replaced it. The old density-report numbers came from `_artifacts_step3.pt`
  (`GlobalMessagePassingActor`), which was removed with `logs/` (2026-06-16).
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
- `logs/` — **REMOVED 2026-06-16.** It held ~97 research scripts + gitignored data artifacts
  (incl. `_artifacts_step3.pt`, the `GlobalMessagePassingActor` behind the old density-report
  numbers). The one load-bearing piece, the vectorized evaluator `fast_stage21.py`, was
  productionized into `data/vectorized_objective_stack_evaluator.py` (equivalence-verified) and
  its checker into `tests/unit/test_vectorized_objective_stack_evaluator.py`; everything else
  was one-off research superseded by the decentralized trunk and deleted (owner-approved).
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
