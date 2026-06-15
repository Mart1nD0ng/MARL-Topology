# V5 Learning Ledger

Legacy path: `D:\PhD_works\v5`

Policy: this is no longer a code-asset migration ledger. MARL-Topology inherits the goal, not the structure. `v5` is an experience library for goals, design attempts, and failure lessons. Do not treat it as a code template.

## Ledger Columns

| v5 theme | clean skeleton layer | what to learn | useful lesson | failure / avoid lesson | when to consult v5 | required evidence before adoption | status |
|---|---|---|---|---|---|---|---|
| Project goal | Whole system | Original target of V2X consensus-aware topology control | Goal remains relevant: topology should support consensus reliability while reducing latency and energy | Do not inherit high-entropy structure | When validating top-level objective wording | Goal statement and acceptance criteria in new docs | active reference |
| Simulation environment structure | `Scene3D`, `CandidateGraph`, `LinkModel` | How v5 represented scenarios, nodes, links, and topology candidates | Useful abstractions may inform inputs and outputs | Do not copy scenario code before clean interfaces exist | When a skeleton layer needs design alternatives | Interface comparison and contract tests | pending learning audit |
| Consensus reliability idea | `ConsensusSuccess`, `TopologyEvaluator` | How consensus success was estimated or replayed | PBFT/quorum reasoning may provide edge cases | Do not import old names or formulas without registration | When implementing consensus evaluator | Metric registration, protocol tests, known cases | pending learning audit |
| Effective success naming | Metric governance | Why previous effective success definitions existed | May reveal useful reliability composition concerns | Avoid `P_eff` soft/hard complexity until needed | Only if minimal metrics cannot express a needed evaluation | Registered metric with tests and formula source | deferred |
| Replay and counterfactual evaluation | `TopologyOracle`, `PolicyBaselines` | How fixed-state comparisons were attempted | Counterfactual replay can support oracle checks | Avoid actor information leakage and phase-script coupling | When designing oracle or replay baselines | Leakage tests and held-state replay contract | pending learning audit |
| Topology feasibility audit | `TopologyOracle` | Which topology failures were important | Feasible/infeasible case design may be reusable as tests | Do not copy phase34 script as a library | When building oracle test cases | Rewritten cases under new contracts | pending learning audit |
| Reward design | Future objective module | What reward attempts failed or succeeded | Failure records can guide exclusions | Old reward is not the new reward | Only during reward contract review | Reward-hacking analysis and metric-governance mapping | reject as code, learn as failure record |
| Actor/critic architecture | `PolicyBaselines`, later MARL training | Which architectures were tried and why | Prior failures may guide baselines and leakage tests | Do not default to old actor/critic/COMA structure | Only after Dec-POMDP env and baselines exist | Architecture review and leakage tests | deferred |
| Configs and phase scripts | Harness and scripts | Operational lessons from experiments | Some run parameters may expose missing sensors | Do not use phase scripts as new mainline | When designing harness tasks or reports | Distilled lesson, not copied script | legacy-only |
| Old skills | `.agents/skills` | Useful project rules and failure modes | Banlists and review templates can be distilled | Do not carry v5-specific assumptions blindly | When a new skill needs failure examples | Scope review against new contracts | distill only |

## Adoption Rule

A v5 idea may influence the new project only when a clean skeleton layer has a concrete design question. Adoption requires:

1. The question being answered.
2. The v5 lesson or counterexample.
3. The new module boundary affected.
4. The metric or contract registration affected.
5. Required tests before implementation.

## Explicit Non-Migration Defaults

- No default migration of v5 code.
- No default migration of phase scripts.
- No default migration of old reward.
- No default migration of metric names.
- No default migration of actor/critic architecture.
- No training or simulator implementation during learning audit.

## Stage 1 Learning Items

| item type | GOAL_SKELETON layer | learning item | v5 evidence | clean-project action | status |
|---|---|---|---|---|---|
| learn goal | Whole system | V2X topology planning remains the target: consensus reliability constraint, latency/energy objectives, MARL/Dec-POMDP discipline | `doc/DESIGN_v6.md`, `README.md` | Keep target, reset structure | confirmed |
| learn simulation idea | `Scene3D`, `LinkModel` | Buildings, LOS/NLOS, V2V/V2I, subchannels, interference, and resource constraints can change conclusions | `doc/v6_phase11_results.md`, configs, physics tests | Start with named deterministic regimes and sanity baselines | confirmed |
| learn simulation idea | `CandidateGraph` | Action masks and candidate graphs need stability audits | Phase30b summary in `doc/GPT_PRO_UPLOAD_VALIDATION_SUMMARY_2026-05-12.md` | Add candidate graph tests for node coverage, edge identity, mask retention, and repair | confirmed |
| learn failure | `ConsensusSuccess` | Effective-success naming grew too complex and had legacy aliasing | `doc/phase35_effective_success_semantics.md`, `core/effective_success.py`, `verify/test_effective_success.py` | Keep minimal consensus metrics until a derived metric is registered | confirmed |
| learn failure | `TopologyEvaluator`, `PolicyBaselines` | Full-mask can be reproduced by actor and still be resource redundant or physically dominated | Phase32 and Phase36 summaries | Treat full-mask as baseline only; require sparse and oracle comparisons | confirmed |
| learn failure | `TopologyOracle` | Policy failure is not environment infeasibility | `doc/phase36_topology_oracle_v3.md`, Phase36 summary | Preserve `unresolved` label and require proof before infeasible claims | confirmed |
| learn failure | `LatencyEnergy`, `MARLTraining later` | Reward economics can favor empty graph, redundant graph, or threshold artifacts | Phase9 failure record, Phase38/38.1 docs, reward summaries | Require reward plateau, resource tradeoff, and reward-hacking gates | confirmed |
| learn failure | `PolicyBaselines` | Fixed 0.5 deterministic threshold can cause full-mask or empty-graph deployment collapse | `models/actor_lifecycle.py`, Phase9 failure record, Phase32/39 summaries | Add deployment calibration sweep before policy claims | confirmed |
| learn failure | `MARLTraining later` | Q/COMA diagnostics can disagree: sign accuracy, rank, calibration, and rare-safety recall must be separated | Phase32 and Phase37.1 summaries, `doc/phase39_target_design.md` | Gate credit assignment on oracle-labeled fidelity and calibration | confirmed |
| avoid route | Whole system | Phase-script stacking raised entropy | read-only inventory: 28 phase scripts, 37 phase tests, 762 phase references | Durable algorithms belong in `src/marl_topology/` | confirmed |
| avoid route | Future objective module | Old reward is not reusable | Phase38 no reward passed all hard gates; Phase38.1 split old labels | Learn failure modes only | confirmed |
| avoid route | Metric governance | Old metric names are not reusable defaults | Phase35 effective-success semantics | Register all future metrics before use | confirmed |
| unresolved question | `DecPOMDPEnv` | Whether v5 actor violated Dec-POMDP locality is not proven, but global-matrix actor inputs make it a risk | `models/actor_lifecycle.py`, `training/rollout.py`, `core/environment.py` | Define actor-local schema and leakage tests before model work | unresolved |
| unresolved question | `LinkModel`, `TopologyOracle` | Which minimal physics regime is sufficient for learnable topology is not settled | Phase11 showed realism changed absolute success | Use staged physics regimes and oracle feasibility before training | unresolved |
