# V5 Failure Lessons

Scope: read-only learning audit of `D:\PhD_works\v5`. This document records design lessons, rules, and test gates only. It does not approve migration of v5 code, reward, phase scripts, metrics, actor/critic architecture, or COMA implementation.

Confidence labels:

- `seed`: hypothesis retained as a starting point, not proven by reviewed evidence.
- `confirmed`: supported by v5 docs, result summaries, tests, or code comments.
- `contradicted`: v5 evidence argues against the hypothesis.
- `unresolved`: evidence is insufficient or noisy.

## Lessons

### L001

- lesson_id: `L001_metric_governance_first`
- title: Metric names proliferated faster than governance.
- affected GOAL_SKELETON layer: `ConsensusSuccess`, `TopologyEvaluator`, `LatencyEnergy`
- symptom in v5: `P_succ_base`, `P_eff`, `P_eff_new`, hard/soft/legacy modes, log-space diagnostics, arithmetic summaries, and deadline/quorum factors coexisted.
- evidence: `doc/phase35_effective_success_semantics.md` separates `P_succ_base` from `P_eff_new` and defines `legacy`, `hard_eval`, and `soft_train`; `core/effective_success.py` implements `legacy:P_eff=P_succ_base`, `hard`, and `soft` definitions; `core/reward_contracts.py` reads both `P_eff_new` and `P_succ_base`.
- confidence: confirmed
- clean-project design rule: Keep Stage 1 metrics to `consensus_success`, `consensus_success_probability`, `latency`, `energy`, and `topology_diagnostics`; register every new metric before use.
- required test or harness gate: Metric registry contract test that rejects unregistered CSV/summary metric names and rejects reward/metric name reuse.

### L002

- lesson_id: `L002_effective_success_alias_risk`
- title: Effective success aliasing hides timeout and quorum semantics.
- affected GOAL_SKELETON layer: `ConsensusSuccess`
- symptom in v5: Legacy effective success could equal base success, while later hard/soft modes multiplied or gated by deadline and quorum factors.
- evidence: `core/effective_success.py` documents `legacy: P_eff = P_succ_base`, `hard: P_eff = P_succ_base iff deadline/quorum gates pass else 0`, and `soft: P_eff = P_succ_base * deadline_factor * quorum_factor`; `verify/test_effective_success.py` has separate hard timeout, hard quorum, soft deadline, and soft quorum tests.
- confidence: confirmed
- clean-project design rule: Do not introduce `P_eff` until a specific module needs it; consensus event, consensus probability, deadline, quorum, and reward remain separate registered quantities.
- required test or harness gate: Protocol metric test for quorum boundary, deadline boundary, and name separation.

### L003

- lesson_id: `L003_full_mask_not_oracle`
- title: Full mask is not a reliable upper bound or a resource optimum.
- affected GOAL_SKELETON layer: `TopologyOracle`, `PolicyBaselines`, `TopologyEvaluator`
- symptom in v5: Full-mask behavior could look optimal under one reward/objective, while sparse feasible topologies existed and physical/resource Pareto checks challenged full-mask assumptions.
- evidence: `result_save/phase32_credit_deployment_audit_p31_ep100/summary_all.json` classified the actor as `OBJECTIVE_FULLMASK_OPTIMAL`, with `sparse_feasible_state_frac=1.0` and `eval_det_fullmask_edge_equal_frac=1.0`; `result_save/phase36_topology_oracle_v3/topology_feasibility_summary_v3.json` reports `strict_physical_pareto_dominates_full_count=142` and `constrained_resource_redundant_full_count=238`.
- confidence: confirmed
- clean-project design rule: Treat full mask as one diagnostic baseline only. Do not call it an upper bound or optimum without oracle evidence under registered metrics.
- required test or harness gate: Baseline comparison gate with empty, full-mask, sparse heuristic, and oracle candidates; report consensus, latency, energy, and diagnostics separately.

### L004

- lesson_id: `L004_reliability_plateau_before_resource_objective`
- title: Reward can keep favoring redundant topology after reliability is satisfied.
- affected GOAL_SKELETON layer: `LatencyEnergy`, `TopologyOracle`, `MARLTraining later`
- symptom in v5: Redundant-edge labels and full-mask objective behavior showed that reliability and resource tradeoffs were not cleanly separated.
- evidence: `doc/phase38_reward_contract_ablation.md` reports no reward contract passed all hard gates; `doc/phase381_reward_label_semantic_split.md` states that drops far above `tau` are not equivalent to drops that cross feasibility; `core/reward_contracts.py` includes plateau and margin-guard variants and comments about not continually rewarding redundant reliability above `tau`; Phase37 summary reports 4,747 redundant-edge rows.
- confidence: confirmed
- clean-project design rule: Reliability is a constraint gate; after the registered consensus reliability threshold is satisfied, optimize latency and energy unless a registered safety-margin metric says otherwise.
- required test or harness gate: Reward review gate requiring above-threshold plateau behavior and redundant-edge resource checks before training.

### L005

- lesson_id: `L005_oracle_before_infeasible_claims`
- title: A failed policy is not evidence that the environment is infeasible.
- affected GOAL_SKELETON layer: `TopologyOracle`, `PolicyBaselines`, `MARLTraining later`
- symptom in v5: Later audits explicitly avoided claiming infeasibility when oracle search was incomplete.
- evidence: `doc/phase35_effective_success_semantics.md` says not to describe unresolved oracle cases as infeasible without exhaustive search or credible bound; `doc/phase36_topology_oracle_v3.md` says `certified_infeasible` is not emitted without proof; Phase36 summary reports `oracle_feasible_count=254`, `unresolved_hard_candidate_count=66`, and `certified_infeasible_count=0`.
- confidence: confirmed
- clean-project design rule: Every policy failure report must compare against topology oracle or explicit baseline coverage before changing reward/model/training.
- required test or harness gate: `topology_oracle_design` gate with known feasible/infeasible cases and an `unresolved` state distinct from infeasible.

### L006

- lesson_id: `L006_fixed_threshold_deployment_risk`
- title: Fixed 0.5 thresholding can collapse deployment behavior.
- affected GOAL_SKELETON layer: `PolicyBaselines`, `DecPOMDPEnv`, `MARLTraining later`
- symptom in v5: Deterministic actor evaluation used `>= 0.5` thresholds and could reproduce full mask or empty graph depending on probability calibration.
- evidence: `models/actor_lifecycle.py` thresholds probe/drop probabilities at `>= 0.5`; Phase9 failure record reports deterministic empty-graph collapse with no pair above 0.5; Phase32 summary reports actor threshold sign accuracy about 0.538 and eval deterministic topology equal to full mask in all audited states; `doc/phase39_target_design.md` says actor imitation should use calibrated deployment, not fixed 0.5 thresholding.
- confidence: confirmed
- clean-project design rule: Deployment calibration is a separate policy layer. Fixed 0.5 threshold is a baseline, not the default deployment rule.
- required test or harness gate: Calibration sweep gate comparing stochastic, fixed-threshold, calibrated-threshold, full-mask, and oracle baselines.

### L007

- lesson_id: `L007_credit_calibration_not_impossibility`
- title: Q/COMA failure is a credit-calibration problem until oracle evidence says otherwise.
- affected GOAL_SKELETON layer: `TopologyOracle`, `PolicyBaselines`, `MARLTraining later`
- symptom in v5: Q credit could show sign accuracy but poor ranking/calibration, and frozen actor/Q diagnostics were not accepted as teachers.
- evidence: Phase32 summary shows `q_delta_sign_accuracy=0.965` but `q_delta_spearman=-0.475` and small predicted magnitude; Phase37.1 summary shows `q_pred_delta_vs_true_delta_P_eff_spearman=-0.021` and actor 0.5 threshold failure rate `0.640`; `doc/phase39_target_design.md` gates COMA until a direct edge-delta critic passes fidelity, rare-safety, ranking, and calibration gates.
- confidence: confirmed
- clean-project design rule: Do not interpret a failed COMA/Q run as proof that edge values are unlearnable. Require oracle-labeled fidelity and calibration tests.
- required test or harness gate: Credit-assignment gate with sign accuracy, rank correlation, calibration error, rare safety recall, and oracle edge-hit rate.

### L008

- lesson_id: `L008_phase_script_entropy`
- title: Phase script stacking increased project entropy.
- affected GOAL_SKELETON layer: Whole system, `TopologyOracle`, `PolicyBaselines`, `MARLTraining later`
- symptom in v5: Long-lived logic accumulated in phase-numbered scripts and tests.
- evidence: read-only inventory found 28 `scripts/phase*.py` files, 37 `verify/test_phase*.py` files, and 762 phase references across `scripts`, `verify`, and `doc`.
- confidence: confirmed
- clean-project design rule: Phase scripts are disposable orchestration only; reusable behavior must enter `src/marl_topology/` with contracts and tests.
- required test or harness gate: Code entropy audit that flags durable algorithms inside `scripts/phase*`.

### L009

- lesson_id: `L009_dec_pomdp_boundary_before_actor`
- title: Actor information boundaries must be designed before actor architecture.
- affected GOAL_SKELETON layer: `DecPOMDPEnv`, `PolicyBaselines`, `MARLTraining later`
- symptom in v5: Actor and rollout interfaces carried full matrices such as `edge_feat`, `phys_adj`, `topo_prev`, `link_r`, and topology history; whether that is valid depends on an explicit observation contract.
- evidence: `models/actor_lifecycle.py` takes full `edge_feat`, `r`, `phys_adj`, and `topo_prev` tensors; `training/rollout.py` stores full `edge_feat`, `phys_adj`, `topo_prev`, `link_r_post`, and topology history; `core/environment.py` comments distinguish observation-only edge memory from masks but do not establish the new project's Dec-POMDP boundary.
- confidence: unresolved
- clean-project design rule: The new actor schema must be local-first and audited before any LSTM, GNN, GNN+LSTM, COMA, or direct edge-delta implementation.
- required test or harness gate: Negative leakage test that rejects global topology, oracle labels, future trajectory fields, and critic-only features from deployment actor batches.

### L010

- lesson_id: `L010_training_after_oracle_baseline_contracts`
- title: Training should wait for oracle, baseline, metric, and reward stability.
- affected GOAL_SKELETON layer: `MARLTraining later`
- symptom in v5: Later stages repeatedly paused training and prohibited reward/model mutation until audit artifacts passed.
- evidence: Phase36/37/37.1/38 docs repeatedly state no training or checkpoint mutation; Phase38 summary says no reward contract passed all hard gates and production reward replacement remained prohibited; Phase39 summary says direct edge-delta training was next but PPO/COMA production training remained prohibited.
- confidence: confirmed
- clean-project design rule: No MARL training until `Scene3D` through `PolicyBaselines` have contract tests, oracle comparisons, metric registry, reward contract, and leakage tests.
- required test or harness gate: Training-review gate that fails if oracle/baseline/metric/reward/Dec-POMDP evidence is missing.

### L011

- lesson_id: `L011_debug_hypotheses_need_status`
- title: Debug records contain useful but noisy hypotheses.
- affected GOAL_SKELETON layer: Whole system
- symptom in v5: Some confident root-cause claims were later refuted or caveated.
- evidence: `doc/audit_2026-05-03_phase8_FINAL_diagnosis.md` self-corrects earlier claims about gradient starvation and wrong statistics; `doc/v6_phase10_rca_report.md` rejects the Phase10 asinh-gradient-dead diagnosis and refutes a `Phi` unbounded claim.
- confidence: confirmed
- clean-project design rule: Every failure lesson must carry status: seed, confirmed, contradicted, or unresolved.
- required test or harness gate: Learning-audit template must include confidence and evidence fields; debug logs alone cannot confirm a lesson.

### L012

- lesson_id: `L012_physics_can_mask_learning_claims`
- title: Physics realism can change whether a policy appears to work.
- affected GOAL_SKELETON layer: `Scene3D`, `LinkModel`, `TopologyEvaluator`, `MARLTraining later`
- symptom in v5: Easier physics made full-mask reproduction appear successful; more realistic interference reduced absolute success and exposed policy limits.
- evidence: `doc/v6_phase11_results.md` states Phase10 used easier physics where full mask gave high success, while Phase11's realistic `n_subchannels=8`, TX direction sampling, and interference made full-mask success around 0.23-0.30 and actor matching full-mask around 0.20; configs include building density/size, LOS decay, fading, interference, and oracle comments.
- confidence: confirmed
- clean-project design rule: Add physics realism incrementally only after simple sanity scenes and baselines exist; never compare policies across physics regimes without declaring the regime.
- required test or harness gate: Physics scenario gate with named regime, seed, units, baseline full-mask/empty/sparse/oracle evaluations, and no cross-regime success claims.

### L013

- lesson_id: `L013_asinh_gradient_dead_refuted`
- title: Do not preserve a root-cause hypothesis after later audits refute it.
- affected GOAL_SKELETON layer: `MARLTraining later`, future objective module
- symptom in v5: An earlier Phase10 plan treated asinh consensus shaping as gradient-dead.
- evidence: `doc/v6_phase10_rca_report.md` says the asinh-gradient-dead diagnosis was empirically wrong and that the observed gradient was small but non-zero.
- confidence: contradicted
- clean-project design rule: Debug hypotheses must be retired when later evidence refutes them; do not build new reward design on refuted diagnoses.
- required test or harness gate: Learning audit must include a contradicted-hypothesis section and prevent contradicted lessons from becoming design requirements.

### L014

- lesson_id: `L014_phi_unbounded_refuted`
- title: Do not treat disputed shaping quantities as unsafe without code evidence.
- affected GOAL_SKELETON layer: future objective module
- symptom in v5: A `Phi` unbounded claim appeared in the debugging record.
- evidence: `doc/v6_phase10_rca_report.md` reports that `Phi` was strictly bounded `[0,1]` in the code path it reviewed.
- confidence: contradicted
- clean-project design rule: Before banning or adopting a shaping term, verify its range from implementation or contract evidence.
- required test or harness gate: Reward-design review must require range/unit tests for any potential or shaping quantity.

### L015

- lesson_id: `L015_policy_frozen_stat_refuted`
- title: Single-window diagnostics can produce false root causes.
- affected GOAL_SKELETON layer: `MARLTraining later`
- symptom in v5: An earlier audit claimed the policy was essentially frozen from KL/clip statistics.
- evidence: `doc/audit_2026-05-03_phase8_FINAL_diagnosis.md` self-corrects that claim and reports that the policy was actively learning across much of training.
- confidence: contradicted
- clean-project design rule: Training diagnostics must report window definitions, multiple windows, seeds, and raw columns before causal claims.
- required test or harness gate: Training-review gate must reject root-cause claims based on a single aggregate window.
