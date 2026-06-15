# V5 Lesson To Gate Map

This map converts Stage 1 v5 learning into gates for Stage 2 and later work. It is not an implementation plan for v5 code. `D:\PhD_works\v5` remains read-only, and every gate below must be satisfied inside the clean MARL-Topology scaffold before the affected layer becomes relied upon.

## Gate Activation Policy

- Stage 2 skeleton implementation may add minimal contracts, typed data, and tests for `Scene3D`, `CandidateGraph`, `LinkModel`, and `TopologyEvaluator`.
- A gate becomes active when a task touches its `GOAL_SKELETON` layer or introduces the named risk.
- Active gates must produce contract tests, harness evidence, or an explicit deferral note.
- Deferred gates do not permit v5 inheritance; they only postpone implementation.

## Lesson Gate Table

| gate id | lessons | GOAL_SKELETON layer | design rule | required contract | required test | required harness gate | active stage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `metric_governance_gate` | `L001_metric_governance_first` | `ConsensusSuccess`, `TopologyEvaluator`, `LatencyEnergy` | Keep metrics governance-first; only `consensus_success`, `consensus_success_probability`, `latency`, `energy`, and `topology_diagnostics` are active defaults. | `docs/METRIC_CONTRACT.md` registration section. | Reject unregistered CSV/summary metrics and reward/metric name reuse. | `design_metric_contract`, `implement_goal_skeleton_v0` | Stage 2, before any metric field is emitted. |
| `consensus_protocol_naming_gate` | `L002_effective_success_alias_risk` | `ConsensusSuccess` | Consensus event/probability, deadline, quorum, and reward must stay separate. | `docs/PROTOCOL_CONTRACT.md` plus metric registration for any derived reliability metric. | Quorum boundary, deadline boundary, and name-separation tests. | `review_topology_evaluator_contract`, `design_metric_contract` | Stage 2 when evaluator or consensus success is introduced. |
| `full_mask_not_oracle_gate` | `L003_full_mask_not_oracle` | `TopologyEvaluator`, `TopologyOracle`, `PolicyBaselines` | Full-mask is a baseline, not an oracle, upper bound, or resource optimum. | Baseline/evaluator contract declaring empty, full, sparse, and oracle roles. | Baseline report separates consensus, latency, energy, and diagnostics. | `review_topology_evaluator_contract`, `topology_oracle_design` | Stage 2 evaluator review; required before baseline claims. |
| `reward_plateau_resource_gate` | `L004_reliability_plateau_before_resource_objective` | `LatencyEnergy`, future objective module, `MARLTraining later` | Reliability is a constraint; after threshold satisfaction, optimize latency and energy unless a registered safety-margin metric exists. | `docs/REWARD_CONTRACT.md` and `docs/METRIC_CONTRACT.md`. | Above-threshold plateau and redundant-edge resource tests before training. | `review_reward_contract` | Stage 3+ reward design; blocked during Stage 2 implementation unless objectives are touched. |
| `oracle_before_infeasible_gate` | `L005_oracle_before_infeasible_claims` | `TopologyOracle`, `PolicyBaselines`, `MARLTraining later` | Policy failure or evaluator failure is not infeasibility without oracle or proof. | Oracle outcome contract with `feasible`, `infeasible`, and `unresolved`. | Known feasible/infeasible cases and unresolved preservation. | `topology_oracle_design`, `review_topology_evaluator_contract` | Stage 2 when oracle labels or infeasible claims appear. |
| `fixed_threshold_is_baseline_gate` | `L006_fixed_threshold_deployment_risk` | `PolicyBaselines`, `DecPOMDPEnv`, `MARLTraining later` | Fixed `0.5` thresholding is a baseline to test, not deployment policy. | Policy baseline/deployment contract. | Calibration sweep or explicit deferral test before policy claims. | `actor_architecture_review` | Stage 3+ policy work; Stage 2 may record as deferred. |
| `credit_calibration_gate` | `L007_credit_calibration_not_impossibility` | `TopologyOracle`, `PolicyBaselines`, `MARLTraining later` | COMA/Q/direct edge-delta claims require oracle-labeled calibration, ranking, and rare-safety checks. | Credit-assignment review contract. | Sign, rank, calibration, rare-safety recall, oracle edge-hit, and magnitude tests. | `actor_architecture_review` | Stage 3+ architecture work; not active for Stage 2 skeleton unless credit claims appear. |
| `phase_script_entropy_gate` | `L008_phase_script_entropy` | Whole system | Durable behavior belongs in `src/marl_topology/`; phase scripts are orchestration-only. | Code organization contract in `AGENTS.md` and `docs/CODEX_WORKFLOW.md`. | Hygiene/entropy test flags `scripts/phase*` and durable logic outside `src`. | `project_entropy_audit`, `skill_calibration_audit` | Always active. |
| `dec_pomdp_leakage_gate` | `L009_dec_pomdp_boundary_before_actor` | `DecPOMDPEnv`, `PolicyBaselines`, `MARLTraining later` | Actor schema is local-first; global topology, oracle labels, future fields, and critic-only tensors are forbidden at deployment. | `docs/DEC_POMDP_CONTRACT.md`. | Negative leakage test for actor batches and checkpoint-loading boundaries. | `design_dec_pomdp_contract`, `actor_architecture_review` | Stage 2 only if env/observation schema is touched; mandatory before policy work. |
| `training_precondition_gate` | `L010_training_after_oracle_baseline_contracts` | `MARLTraining later` | No MARL training before oracle, baselines, metric registry, reward contract, and leakage tests exist. | Training/evaluation contract. | Training-review test fails if prerequisite evidence is missing. | `actor_architecture_review`, `review_reward_contract` | Stage 3+; always blocks training in Stage 2. |
| `debug_status_gate` | `L011_debug_hypotheses_need_status`, `L013_asinh_gradient_dead_refuted`, `L014_phi_unbounded_refuted`, `L015_policy_frozen_stat_refuted` | Whole system, future objective/training modules | Debug lessons must carry status and contradicted hypotheses cannot become design rules. | Learning-audit report contract. | Lesson status test requires `seed`, `confirmed`, `contradicted`, or `unresolved`. | `v5_learning_audit`, `skill_calibration_audit` | Always active when citing v5 debug evidence. |
| `physics_regime_declaration_gate` | `L012_physics_can_mask_learning_claims` | `Scene3D`, `LinkModel`, `TopologyEvaluator`, `MARLTraining later` | Declare physics regime, units, seed behavior, and baseline scope before comparing policies or link results. | `docs/PHYSICS_CONTRACT.md` and link-model contract. | Named-regime, units, deterministic sanity, and no cross-regime-claim tests. | `review_scene3d_candidate_graph`, `review_link_model_contract`, `design_3d_physics_contract` | Stage 2 for Scene3D/LinkModel contracts; no full physics implementation. |

## Stage 2 Minimum Gate Set

Before Stage 2 implementation of the minimal skeleton starts, these gates must be active:

- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `phase_script_entropy_gate`
- `physics_regime_declaration_gate`

Before Stage 2 claims evaluator, oracle, or policy-baseline behavior, also activate:

- `full_mask_not_oracle_gate`
- `oracle_before_infeasible_gate`
- `dec_pomdp_leakage_gate` if actor observations or env schema are touched.

## Forbidden V5 Inheritance

The gates above do not authorize copying v5 code. They block:

- Old reward formulas or weights.
- Old metric names and effective-success mode taxonomy.
- Old phase scripts as mainline implementation.
- Full-mask optimality claims without oracle evidence.
- Fixed `0.5` threshold deployment defaults.
- COMA/Q/direct edge-delta architecture defaults.
- Global actor inputs and oracle label leakage.
