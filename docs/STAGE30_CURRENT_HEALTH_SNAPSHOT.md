# Stage 30 Current Health Snapshot

- verdict: `stage30_repair_loop_blocked_awaiting_owner_decision`
- iteration count: `4`
- dominant blocker: `surrogate_objective`
- recommended next task: `stage31_owner_decision_on_data_expansion_and_active_alignment_repair`

| Component | Status | Evidence | Confidence | Repair Class |
| --- | --- | --- | --- | --- |
| `harness_state` | `PASS` | Stage 29 and Stage 30 control-plane reports are present. | `high` | `hard_correctness` |
| `surrogate_objective` | `FAIL` | inversion rate 0.0 -> 0.0; active training surrogate unchanged | `medium-high` | `surrogate_objective_alignment` |
| `reliability_margin` | `FAIL` | tau-feasible delta -0.0234375; violation delta 0.0234375 | `medium` | `reliability_margin` |
| `projection_alignment` | `WARN` | Top proposal rejection changed by 0.009549; the repaired-critic policy did not reduce projection friction. | `medium` | `projection_alignment` |
| `data_scale` | `WARN` | medium-high because the formal pilot reused 10 unique contexts into 24 slots | `medium` | `data_scale` |
| `critic_health` | `PASS` | EV 0.9465631796667974; correlation 0.9733077206959327 | `high` | `critic_regression` |
| `sampler_health` | `WARN` | medium-low; sampler remains valid but projected proposals are not fully aligned | `medium` | `sampler_health` |
| `actor_health` | `WARN` | medium; actor moved but did not reduce projection friction | `medium` | `actor_score_calibration` |
| `policy_gradient_loop` | `WARN` | Stage 28 completed all seeds but did not improve all readiness dimensions. | `medium` | `policy_gradient_loop` |
| `communication` | `PASS` | medium if link probabilities are saturated or flat | `medium` | `hard_correctness` |
| `consensus` | `PASS` | medium when expected-initiator averaging exposes weak primary spread | `medium` | `reliability_margin` |
