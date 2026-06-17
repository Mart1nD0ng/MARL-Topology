# MARL-Topology

Learning distributed communication-topology control for urban 3D V2X networks with a
multi-agent reinforcement-learning (MARL / Dec-POMDP, CTDE) policy. The design objective is to
plan topologies that satisfy a **PBFT consensus reliability constraint** (per-scene success
probability ≥ τ = 0.9) while **minimizing latency and energy**, under variable node counts N.

> This is a PhD research repository. Results are reported conservatively; see
> `docs/URBAN_V2X_RESEARCH_LOG.md` and `docs/DENSITY_AXIS_CAMPAIGN_REPORT.md` for the honest
> state of the evidence, and `docs/PROJECT_STATE.md` for the staged ledger.

## What the project is

- **Simulation environment (production tier).** A standards-grounded urban 3D V2X stack:
  Manhattan grid with buildings/roads/vehicles/RSUs → 3D ray-box LoS/NLoS visibility →
  channel (FSPL by default; opt-in 3GPP TR 37.885 V2V + TR 38.901 UMi, stochastic NLOSv, spatially
  correlated shadowing) → URLLC finite-blocklength links → multi-hop routing with SINR/interference
  → scheduled-MAC (STDMA) → relay + wired RSU backhaul → analytic three-phase PBFT reliability →
  separate latency/energy accounting. Orchestrated by `Stage21ObjectiveStackEvaluator`
  (`src/marl_topology/data/stage21_objective_stack_evidence.py`).
  A distance-only `MinimalDecPOMDPEnv` skeleton is kept for baselines and leakage tests only.
- **Policy (production actor).** One deployment actor:
  `local_message_passing_gnn_edge_scorer_v3_residual_norm` — a local K-round message-passing GNN
  that scores edges from **local/neighbor observations only** (the Dec-POMDP boundary is enforced
  at the env, model, assembler, and sampler layers). A centralized graph value-critic is used at
  **training time only** (CTDE).
- **Training flow.** One MARL pipeline: `training/production_mappo_adapter.py` — behaviour-cloning
  warm start → Stage 27 graph value-critic pretraining → clipped on-policy actor-critic fine-tune,
  with keep-best validation gating.
- **Objective framing.** Reliability is a *constraint* (feasibility-first barrier surrogate, no
  reward above τ); latency and energy are the *objectives*, minimized only inside the feasible
  region. Metric governance is enforced by `src/marl_topology/metrics/registry.py`.

## Layout

```
src/marl_topology/      production package (env, channel, link, network, protocol,
                        geometry3d, scenario, topology, objectives, metrics, policies,
                        models, data, training, evaluation)
tests/                  unit + contract/boundary tests (decentralization, metric/reward
                        governance, the production-actor gate)
scripts/                runnable train/ (incl. the decentralized-MARL production driver) and
                        replay/ drivers for the production stages
harness/                cybernetic task specs, rubrics, templates, governance tooling
docs/                   contracts + staged research record (provenance)
result_save/            gitignored datasets and run artifacts (the logs/ research scratch was
                        removed 2026-06-16; its vectorized evaluator was productionized into src/)
```

See `docs/CLEAN_PROJECT_MAP.md` for the full env / configuration / architecture map and what was
retired in the production-main-body consolidation.

## Production trunk (2026-06-17): recovered, validated decentralized pipeline

The production model is the **recovered, validated** decentralized pipeline that produced the
project's best result — held-out **decentralized feasibility 0.82 (N=8: 0.957)** under the full
TR 37.885 stochastic stack (v2x_37885 + shadowing + NLOSv + scheduled MAC + relay-3 + wired RSU
backhaul + coverage-gated membership) at the **4-RSU / 20 dBm** operating point. See
`docs/URBAN_V2X_RESEARCH_LOG.md` (Step-3) and `docs/DENSITY_AXIS_CAMPAIGN_REPORT.md`.

- **Actor** `MessagePassingGraphEdgeScorer` (`models/message_passing_graph_edge_scorer.py`) — a
  K-round bidirectional message-passing GNN edge scorer (decentralized-with-communication: K hops
  of local neighbour signalling, no global-state shortcut).
- **Decoder** `local_mutual_assemble` (`policies/decentralized_mutual_acceptance.py`) — each node
  ranks only its own incident edges within its radio budget; an edge activates iff **both endpoints
  accept**. Per-node computable, zero global state. The global-argsort decode is kept only as the
  centralized-decode ablation (cost ≈ 0 at the multi-RSU operating point).
- **Recipe** `training/decentralized_distillation.py` — BC-distil the (budget-aware SA or
  critic-planner) teacher into the actor (weight decay + early stopping + keep-best); the
  centralized graph critic is training-only (CTDE). Dataset-shard I/O lives in the drivers/tests.
- **Reproduce** `python scripts/train/reproduce_recovered_step3.py` loads the frozen
  `_artifacts_step3.pt` and reproduces 0.82 (pinned by `tests/unit/test_recovered_step3_reproduction.py`).

> The 2026-06-16 "new trunk" (`decentralized_marl.py` / `LocalKHopGNNEdgeScorer` /
> `DecentralizedPerNodeMutualSampler` / `production_trunk.py` / its driver) was an unvalidated
> re-implementation that never reproduced the result (~0 under default config); it was **deleted
> 2026-06-17** so the repo has one trunk. The Stage-33 dataset/adapter/critic infrastructure it now
> reuses is load-bearing (not a baseline).

## Open status (honest)

- The recovered trunk **reproduces 0.82 held-out from a frozen artifact** (decentralization cost 0).
  Reported grades are *fractions of scenes* clearing the per-scene τ = 0.9 bar, not reliability
  itself; the < 1.0 rate is a dataset-composition statement (deliberately mixed hard/infeasible
  families + all-nodes-validator structure), not a method/physics wall — per-scene consensus
  routinely reaches 1.0 and the operating-point envelope is 1.00 τ-achievable with realistic placement.
- **Retraining to 0.82 (vs reproducing the frozen actor)** needs the critic-planner arm of the recipe
  ported into a driver (the BC-on-SA-teacher arm in `decentralized_distillation.train_actor` reaches
  the ~0.5 SA-teacher ceiling). The operating-point dataset build (4 RSU / 20 dBm / N {8,12,16} /
  TR 37.885 stochastic) is in `docs/URBAN_V2X_RESEARCH_LOG.md` Step-3.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json
```

The full suite passes with one tolerated `xfail`: the Stage 23 REINFORCE sampler micro-gate is a
superseded scaffold (the production decode path is decentralized local mutual acceptance on the
K-round message-passing actor, not the Plackett-Luce/Bernoulli samplers it ranks; under the
owner-approved feasibility-first barrier reward its 3-scene tie-break shifted), kept as a documented
`xfail` pending removal of that dead path. The earlier legacy-reference manifest gap is now fixed —
the validator flags any external `v5` tree as a legacy-reference artifact root.

## Boundary rules

- Deployment actors use local observations/history only; centralized critics are training-only.
- New physics/metrics are opt-in and leave default behaviour byte-identical.
- `D:\PhD_works\v5` is read-only legacy reference; the project inherits the goal, not the structure.
