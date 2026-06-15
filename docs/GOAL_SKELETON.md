# Goal Skeleton

This is the shortest path to express the project goal before adding complex physics, reward shaping, or neural architectures.

```text
Scene3D
-> CandidateGraph
-> LinkModel
-> TopologyEvaluator
-> ConsensusSuccess
-> LatencyEnergy
-> TopologyOracle
-> DecPOMDPEnv
-> PolicyBaselines
-> MARLTraining later
```

## Layers

| Layer | Responsibility | Input / Output | What must be tested | What should not be added yet | How v5 may inform this layer |
|---|---|---|---|---|---|
| `Scene3D` | Represent vehicles, RSUs, roads, and buildings at the level needed to create communication candidates | Input: scenario config. Output: typed scene objects and positions | Deterministic scene loading, units, node identity, basic geometry validity | Full ray tracing, mobility realism, rendering | Learn what scenario facts v5 needed, not its file structure |
| `CandidateGraph` | Build possible communication edges among vehicle and RSU nodes | Input: `Scene3D`. Output: candidate graph with node and edge ids | Node coverage, edge identity, no duplicate edges, configurable radius or candidate rule | Learned topology policy, protocol scoring | Learn candidate construction pitfalls from v5 |
| `LinkModel` | Give each candidate edge minimal communication estimates | Input: candidate edge and scene facts. Output: success probability, latency, energy estimates | Value ranges, units, monotonic sanity where applicable | Full 3D physical simulator, stochastic channel complexity | Learn which link features v5 used and which caused confusion |
| `TopologyEvaluator` | Evaluate a selected topology using link estimates and protocol requirements | Input: candidate graph, selected edges, link records. Output: evaluation record | Handles empty, disconnected, and simple connected graphs | MARL reward, neural policy, full PBFT timing | Learn v5 evaluation cases without copying phase logic |
| `ConsensusSuccess` | Decide or estimate whether consensus succeeds under current protocol abstraction | Input: topology evaluation and protocol config. Output: `consensus_success` or `consensus_success_probability` | Quorum/deadline boundary cases, deterministic simple cases | Effective-success variants, soft/hard metric families | Learn protocol edge cases from v5 only after registering metrics |
| `LatencyEnergy` | Compute objective quantities for evaluated topology | Input: selected topology and link records. Output: latency and energy | Units, aggregation declaration, nonnegative values | Reward weights, tail-risk metrics | Learn old latency/energy assumptions as candidates, not defaults |
| `TopologyOracle` | Provide sanity baselines and feasibility labels for small scenes | Input: scene, candidate graph, evaluator. Output: feasible choices or diagnostic labels | Known feasible/infeasible cases, no actor leakage | Large optimizer, hidden deployment input | Learn v5 topology audits as test-case inspiration |
| `DecPOMDPEnv` | Expose local actor observations and joint topology actions | Input: scene/evaluator/oracle diagnostics as allowed. Output: local observations, actions, global training info when declared | Actor-local schema, forbidden global fields, reset/step shape | LSTM/GNN policy, training loop | Learn v5 leakage risks and observation mistakes |
| `PolicyBaselines` | Provide simple non-learning policies for regression and comparison | Input: env observation or allowed graph facts. Output: topology action | Deterministic output, no forbidden actor info, baseline metrics | PPO/MAPPO/COMA, complex neural policies | Learn which simple v5 baselines were useful |
| `MARLTraining later` | Train policies only after the previous layers have contracts and baselines | Input: tested env and reward contract. Output: trained policy and diagnostics | Seeds, baselines, metric governance, leakage checks | Stage 0 implementation or GPU dependencies | Learn v5 training failures as risk controls |

## Current Rule

Build the skeleton that can express the target. Add physical detail, reward detail, and model complexity only when a lower layer produces stable evidence that the next layer needs them.

## Stage 2 Implementation Status

Stage 2 implements the goal skeleton as a minimal deterministic pipeline. It does not implement full 3D physics, actor/critic models, COMA, reward training, or v5 code migration.

| Layer | Stage 2 status | Implementation | Evidence |
| --- | --- | --- | --- |
| `Scene3D` | implemented minimal container and Stage 2.5 deterministic fixtures | `src/marl_topology/scenario/scene.py`, `src/marl_topology/scenario/fixtures.py` | node identity, deterministic scene tests, fixture ids, expected oracle statuses |
| `CandidateGraph` | implemented distance-filtered graph | `src/marl_topology/topology/candidate_graph.py` | stable edge identity and duplicate prevention tests |
| `LinkModel` | implemented simple deterministic model | `src/marl_topology/link/simple_link_model.py` | distance monotonic reliability, nonnegative latency and energy tests |
| `TopologyEvaluator` | implemented minimal evaluator | `src/marl_topology/topology/evaluator.py` | empty/full topology response and registered metric output tests |
| `ConsensusSuccess` | implemented minimal quorum graph abstraction | `src/marl_topology/protocol/consensus.py` | consensus responds to selected topology tests |
| `LatencyEnergy` | implemented simple aggregation | `src/marl_topology/objectives/latency_energy.py` | nonnegative objective tests |
| `TopologyOracle` | implemented small-graph exhaustive oracle | `src/marl_topology/topology/oracle.py` | full graph remains baseline, unresolved is distinct from infeasible |
| `DecPOMDPEnv` | minimal reset/step wrapper implemented, dynamics/training deferred | `src/marl_topology/env/dec_pomdp_schema.py`, `src/marl_topology/env/dec_pomdp_env.py` | actor-local schema, reset/step wrapper, centralized-training view separation, leakage-negative tests, and Stage 2.4 baseline report path |
| `PolicyBaselines` | implemented global regression baselines, decentralized non-learning baselines, and Stage 2.4 report sensor | `src/marl_topology/policies/baselines.py`, `src/marl_topology/policies/decentralized_baselines.py`, `src/marl_topology/evaluation/baseline_report.py` | empty/full/greedy/random, local no-edges/all-edges/top-k/threshold/random baseline tests, registered-metric report tests |
| `MARLTraining later` | not implemented | deferred | training precondition gate blocks Stage 2 training |

Stage 2 demo: `scripts/replay/demo_goal_skeleton.py` constructs a small scene, candidate graph, link records, evaluator, empty/full/greedy/random baselines, and an exhaustive small-graph oracle. It writes no result files.

Stage 2.4 report: `scripts/replay/baseline_evaluation_report.py` prints a deterministic JSON baseline report over the same demo scenario. It compares global non-learning baselines and decentralized `ActorObservation` to `EdgeActionDecision` baselines, keeps full graph as a baseline only, and keeps the exhaustive oracle as a separate reference.

Stage 2.5 fixtures: `scripts/replay/scenario_fixture_report.py` prints the same baseline/oracle evidence across `demo_stage2`, `sparse_chain_stage2`, and `quorum_blocked_stage2`. These fixtures broaden smoke-test coverage without adding complex 3D physics, reward, model code, training, or v5 migration.

## Stage 1 Caution Notes From V5

- `Scene3D`: name the physics regime before comparing policies. v5 showed easier physics can make full-mask reproduction look like learning.
- `CandidateGraph`: action masks are part of the problem definition. Test mask stability and candidate coverage before training.
- `LinkModel`: link realism can be staged; do not start with stochastic channel complexity before deterministic sanity scenes pass.
- `TopologyEvaluator`: full-mask is a baseline only. It is not an oracle, upper bound, or resource optimum by default.
- `ConsensusSuccess`: do not introduce effective-success variants until a concrete module needs one and the metric is registered.
- `LatencyEnergy`: keep units and aggregation explicit; do not let reward weights hide latency or energy semantics.
- `TopologyOracle`: unresolved means unresolved, not infeasible. Store replayable topology evidence from the start.
- `DecPOMDPEnv`: deployment actor observation schema must be local-first before any GNN/LSTM actor is implemented.
- `PolicyBaselines`: fixed 0.5 threshold is a baseline to test, not a deployment rule.
- `MARLTraining later`: training waits for oracle, baselines, metric registry, reward contract, leakage tests, and credit-fidelity gates.
