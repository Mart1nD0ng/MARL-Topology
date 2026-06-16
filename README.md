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
scripts/                runnable train/ and replay/ drivers for the production stages
harness/                cybernetic task specs, rubrics, templates, governance tooling
docs/                   contracts + staged research record (provenance)
logs/, result_save/     gitignored experiment scripts, datasets, and run artifacts
```

See `docs/CLEAN_PROJECT_MAP.md` for the full env / configuration / architecture map and what was
retired in the production-main-body consolidation.

## Production trunk (2026-06-16): decentralized CTDE MARL

The production model is now the **decentralized CTDE multi-agent flow** (authoritative
designation in `src/marl_topology/training/production_trunk.py`):

- **Actor** `LocalKHopGNNEdgeScorer` — a genuinely multi-hop yet Dec-POMDP-local message-passing
  edge scorer (receptive field = K hops via local neighbour signalling; no global-state shortcut).
  It removes the v3 1-hop ego-graph ceiling (v3 is now a registered diagnostic baseline).
- **Decoder** `DecentralizedPerNodeMutualSampler` — each node ranks only its own incident edges
  within its radio budget; an edge activates iff **both endpoints accept** (mutual acceptance). The
  joint log-prob factorizes per agent (no double counting). This **closes the prior decentralization
  gap**: the scene-global top-k decode is kept only as the centralized-decode ablation.
- **Flow** `DecentralizedCTDEFlow` — per-agent clipped PPO from the factorized per-owner log-probs +
  a training-only centralized graph critic (CTDE); genuinely **sequential** (a per-node energy
  battery makes an action shrink the next-step feasible set), so it is **not** a static-frame bandit.
- **Driver** `scripts/train/decentralized_marl_production_training.py` runs it at configurable scale
  (K-ablation, variable N) on CPU or GPU and emits checkpoints, figures, tables, and a report.

## Open status (honest)

- The decentralized trunk is **verified by tests** (multi-hop receptive field, strict locality,
  mutual-acceptance + log-prob factorization, non-bandit energy dynamics, end-to-end training on real
  scenes) but the **large-scale training campaign on a GPU is still pending** — run the driver above to
  produce the headline feasibility-vs-N / K-ablation results. Reported grades are *fractions of scenes*
  clearing the per-scene τ = 0.9 bar, not reliability itself, and are operating-point / propagation
  specific.
- At fixed small N (≤ 16) a decentralized actor matches its search-teacher feasibility ceiling
  (≈ 0.69–0.82). At N ≥ 24 a frozen small-N actor's cross-scene grade collapses; this is consistent
  with a **learnability/search gap**, not a proven physics wall — the multi-hop actor + scale-up
  training target exactly this regime.
- The legacy Stage-33 ego-graph + global-Plackett-Luce path is retained as the diagnostic baseline;
  its `ACTIVE_STAGE33_*` constants are intentionally not flipped (a pure rename that would break the
  diagnostic flow). `production_trunk.py` is the authoritative production designation.

## Verification

```powershell
python -m pytest -q
python harness\scripts\validate_tasks.py
python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json
```

Two unit tests are known, tolerated pre-existing failures (a legacy-reference manifest check and the
Stage 23 micro-gate after the reward switched to the feasibility-first barrier); everything else passes.

## Boundary rules

- Deployment actors use local observations/history only; centralized critics are training-only.
- New physics/metrics are opt-in and leave default behaviour byte-identical.
- `D:\PhD_works\v5` is read-only legacy reference; the project inherits the goal, not the structure.
