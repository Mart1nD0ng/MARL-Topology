# V5 Learning Audit

This audit reads `D:\PhD_works\v5` as a read-only experience library. It is organized by the MARL-Topology `GOAL_SKELETON`, not by v5 phase numbers. No code, reward, metric name, phase script, actor/critic architecture, or COMA implementation is approved for migration.

## V5 Areas Read

- `README.md`
- `doc/DESIGN_v6.md`
- `doc/phase35_effective_success_semantics.md`
- `doc/phase36_topology_oracle_v3.md`
- `doc/phase37_edge_edit_dataset_audit.md`
- `doc/phase371_actor_q_diagnostic_join.md`
- `doc/phase38_reward_contract_ablation.md`
- `doc/phase381_reward_label_semantic_split.md`
- `doc/phase39_target_design.md`
- `doc/GPT_PRO_UPLOAD_VALIDATION_SUMMARY_2026-05-12.md`
- `doc/audit_2026-05-03_phase8_FINAL_diagnosis.md`
- `doc/v6_phase10_rca_report.md`
- `doc/v6_phase11_results.md`
- `doc/v6_phase9_failure_records/9_1_4_eval_det_empty_graph.md`
- selected summaries under `result_save/phase32*`, `phase36*`, `phase37*`, `phase371*`, `phase38*`, `phase39*`
- targeted code comments and interfaces in `core/`, `models/`, `training/`, `scripts/`, and `verify/`

Debug records and result summaries are treated as noisy evidence unless supported by tests, code comments, or later audits.

## Layer Audit

| GOAL_SKELETON layer | What v5 did | Experience | Failure | New project design | Do not copy |
|---|---|---|---|---|---|
| `Scene3D` | Used configurable mobility, buildings, RSUs, V2V/V2I, LOS decay, fading, interference, subchannels, and seed controls. | Scenario facts matter: building/LOS/interference regime changed whether full-mask appeared strong. | Cross-regime comparisons became misleading; easier physics masked learning limits. | Start with deterministic small scenes and explicit regime names before adding realism. | v5 config sprawl and physics ramps as default complexity. |
| `CandidateGraph` | Built action masks from physical feasibility and later resource-aware masks. | Candidate space itself can make or break feasibility; masks need stability audits. | Some masks were too wide, too narrow, or produced oracle-needle cases. | Candidate graph must have node/edge identity tests, mask-retention diagnostics, and simple repair rules. | Resource-aware mask policies before minimal graph tests. |
| `LinkModel` | Used path loss, LOS/NLOS, fading, interference, resource conflict, and link reliability. | Link features can support actor/Q diagnostics and oracle labels. | Bimodal link reliability left few learnable middle-band edges in some regimes. | Begin with range/unit tests and monotonic sanity before stochastic channel complexity. | Direct import of v5 channel details or BLER thresholds. |
| `TopologyEvaluator` | Evaluated active topology against link stats, latency, energy, quorum, deadline, and full-mask baseline. | Evaluation must separate topology diagnostics from objective metrics. | Full-mask normalization and reward scoring could make redundant topology look optimal. | Evaluate empty, full, sparse heuristic, and oracle candidates under the same registered metrics. | Full-mask as upper bound or success proof. |
| `ConsensusSuccess` | Implemented PBFT-style cascading success, then later introduced effective success modes. | Quorum/deadline edge cases are valuable as tests. | `P_succ`/`P_eff`/hard/soft/legacy semantics became too complex and required later repair. | Keep `consensus_success` and `consensus_success_probability` minimal until a registered derived metric is necessary. | Old `P_eff` names, formulas, or mode taxonomy. |
| `LatencyEnergy` | Computed latency ratios, energy ratios, full-mask energy baseline, timeout penalties, and resource scores. | Latency/energy must be real objectives, not hidden proxies for topology density. | Penalty magnitudes and sentinel values distorted learning; reward could prefer disconnect or redundant full mask. | Keep units explicit and test nonnegative latency/energy before reward weights. | Old latency ratio, sentinel behavior, or energy normalization as defaults. |
| `TopologyOracle` | Built audit-only oracle searches and edge-edit datasets. | Oracle is essential: policy failure does not prove infeasibility. | Earlier candidate carryover lacked replayable topology adjacency, limiting regression claims. | Store replayable topology ids/edge lists from the start; distinguish feasible, infeasible, and unresolved. | Phase34/36 scripts as implementation. |
| `DecPOMDPEnv` | Exposed full graph tensors to actor/training interfaces and used edge memory features. | Edge memory can be useful if it is local and schema-governed. | Actual Dec-POMDP locality is not proven by v5 interfaces. | Define local actor observation schema before model work; central info is critic-only. | Global graph tensors as deployment actor default. |
| `PolicyBaselines` | Compared deterministic actor, stochastic actor, full-mask, sparse/oracle candidates, and calibration variants. | Baselines exposed threshold and full-mask artifacts. | Fixed 0.5 threshold could produce full-mask or empty-graph collapse. | Baselines must include fixed threshold, calibrated threshold, stochastic, empty, full, sparse heuristic, and oracle. | Fixed 0.5 threshold as deployment rule. |
| `MARLTraining later` | Ran PPO/MAPPO/COMA/PopArt experiments with many phase gates and post-hoc audits. | Training diagnostics found real issues in critic calibration, reward scaling, stochastic/deterministic gaps, and physics realism. | Training often preceded stable oracle, metric, reward, and leakage gates; phase scripts accumulated entropy. | Training is explicitly later; require oracle, baselines, metric registry, reward contract, leakage tests, and credit-fidelity gates first. | Actor/critic/COMA architecture, reward, or phase scripts. |

## Cross-Layer Findings

- Goal inheritance is valid: v5 consistently targeted V2X topology planning under consensus reliability, latency, and energy.
- Structure inheritance is unsafe: v5 accumulated many phase scripts, metrics, reward variants, and architecture fixes.
- The clean project should use v5 mainly for failure tests: metric aliases, full-mask artifacts, threshold calibration, oracle unresolved handling, reward plateau, credit fidelity, and phase-script entropy.

## New Design Rules

1. Register metrics before writing CSV fields or reward terms.
2. Separate consensus success, deadline, quorum, latency, energy, diagnostics, and reward.
3. Treat full-mask as a diagnostic baseline, not an oracle.
4. Require topology oracle evidence before declaring environment infeasible.
5. Do not deploy actor with fixed 0.5 threshold without calibration evidence.
6. Do not train COMA/Q-based credit before oracle fidelity and calibration gates.
7. Keep phase scripts out of durable implementation.
8. Define Dec-POMDP actor-local observation schema before model architecture.
9. Do not compare policy results across physics regimes without named regime and baseline set.
10. Mark every failure lesson as seed, confirmed, contradicted, or unresolved.

