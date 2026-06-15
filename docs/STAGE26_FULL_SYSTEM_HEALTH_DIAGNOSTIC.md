# Stage 26 Full-System Health Diagnostic

Stage 26 is a diagnostic stage, not a training stage. It reads frozen Stage 25 artifacts and Stage 21/22 evaluation contexts, then produces a component health scorecard and root-cause matrix.

## Controlled System

- Controlled object: active pre-scale policy-gradient stack, including project state, evidence, communication, consensus, reward surrogate, assembler, sampler, actor, critic, rollout loop, visualization, and scale-readiness gates.
- Desired state: identify which components most plausibly explain Stage 25 weak improvement without changing the active stack.
- Feedback loop: frozen evidence -> component metrics -> health scorecard -> root-cause matrix -> owner decision packet.
- Forbidden actuators avoided: training updates, scale-up, reward tuning, sampler switching, COMA, Transformer, recurrent actor work, final tau selection, checkpoints, v5 writes, and unmanifested artifacts.

## Summary

- verdict: `stage26_pass_full_system_health_diagnostic_complete`
- pass gate: `True`
- recommended option: `option_b_repair_critic_before_more_training`
- recommended next task: `stage_27_critic_baseline_repair_before_more_training`
- scale-up approved: `False`

## Component Scorecard

| Component | Status | Score | Blocks Scale-Up | Labels |
| --- | --- | ---: | --- | --- |
| `harness_state_health` | `PASS` | 100.0 | `False` | `state_consistency` |
| `data_health` | `WARN` | 60.0 | `True` | `sufficient_for_small_pilot_only,data_too_small,data_too_duplicate` |
| `communication_health` | `PASS` | 85.0 | `False` | `communication_signal_usable` |
| `consensus_health` | `PASS` | 85.0 | `False` | `consensus_signal_healthy` |
| `reward_objective_health` | `WARN` | 65.0 | `True` | `reward_objective_mismatch` |
| `assembler_health` | `WARN` | 68.0 | `True` | `tx_budget_bottleneck` |
| `sampler_health` | `WARN` | 68.0 | `True` | `sampler_projection_mismatch` |
| `actor_health` | `WARN` | 70.0 | `True` | `actor_assembler_mismatch` |
| `critic_health` | `FAIL` | 35.0 | `True` | `critic_unused_or_weak,critic_high_bias,critic_value_scale_mismatch,critic_underfit` |
| `mappo_loop_health` | `WARN` | 62.0 | `True` | `unstable_seed,critic_not_helpful` |
| `visualization_health` | `PASS` | 100.0 | `False` | `visualization_healthy` |
| `scale_readiness` | `FAIL` | 35.0 | `True` | `scale_up_blocked_pending_owner_decision,critic_repair_required_before_scale,data_scale_readiness_risk` |

## Root Cause Summary

The strongest Stage 25 limiter is critic health: value prediction explained variance was weak and value scale mismatch was large. Data size/duplication, reward/objective tension, projection friction, and one unstable seed are secondary blockers for scale-up.

## Verification

- `python scripts\replay\stage26_full_system_health_report.py`
- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
