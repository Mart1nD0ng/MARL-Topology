# Stage 29 Root Cause and Decision Packet

## Owner Decision

- recommended option: `option_b_repair_reward_objective_and_projection_before_larger_pilot`
- recommended next task: `stage_30_reward_objective_projection_alignment_repair`
- larger pilot approved: `False`
- scale-up approved: `False`
- owner decision required: `True`
- rationale: Stage 28 fixed the value-baseline failure, but the repaired-critic policy update still worsened surrogate reward and projection rejection while data health remains small-pilot-only.

## Root Cause Matrix

| Component | Status | Evidence | Contribution | Confidence | Recommended Repair | Blocks Larger Pilot | Blocks Scale-Up | Blocks Recurrent Policy | Blocks Reward Tuning | Blocks Sampler Change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `critic_baseline` | `PASS` | Stage 28 critic EV 0.946563, value-return correlation 0.973308, selected graph value critic reused. | low after Stage 27/28 repair | `high` | keep selected graph value critic active for future owner-approved runs | `False` | `False` | `True` | `False` | `False` |
| `reward_objective_alignment` | `FAIL` | Stage 28 latency delta -0.00004688, energy delta -0.00013750, surrogate reward delta -1.865734. | high because the policy improved latency/energy while the surrogate worsened | `medium-high` | run a reward/objective and projection-alignment repair stage without selecting final tau or tuning weights by default | `True` | `True` | `True` | `True` | `True` |
| `assembler_projection_alignment` | `WARN` | Top proposal rejection delta versus supervised 0.009549; Stage 28 minus Stage 25 top rejection 0.005208. | medium because repaired-critic updates did not reduce projection friction | `medium` | diagnose accepted/rejected score separation and projection constraints before larger pilot | `True` | `True` | `True` | `False` | `True` |
| `reliability_constraint` | `WARN` | Tau-feasible delta -0.023438; violation delta 0.023438; allowed absolute bound 0.05. | medium because reliability stayed within bound but moved in the wrong direction | `medium` | keep reliability as a hard gate in any future pilot | `False` | `True` | `True` | `True` | `False` |
| `data_scale_readiness` | `WARN` | medium-high because the formal pilot reused 10 unique contexts into 24 slots | medium because Stage 26 still labels data as small-pilot-only | `medium` | expand or diversify scenario evidence before any scale-up claim | `False` | `True` | `True` | `False` | `False` |
| `small_scale_policy_update` | `WARN` | All three seeds completed, latency/energy improved versus supervised, but reward and projection did not improve together. | medium because the update signal is usable but not clean enough for larger training | `medium` | do not run a larger pilot until reward/projection blockers are resolved or owner accepts the risk | `True` | `True` | `True` | `True` | `True` |
| `boundary_and_harness` | `PASS` | Stage 28 reported reward weights unchanged, sampler fixed, no checkpoint, and no scale-up. | low; control-plane evidence is healthy | `high` | keep owner gate and manifest/report discipline | `False` | `False` | `True` | `False` | `False` |

## Options

- `option_a_larger_pilot_now`: blocked because reward/objective and projection evidence remain mixed
- `option_b_reward_objective_projection_repair`: recommended because the critic is healthy but surrogate reward and projection rejection worsened in the repaired-critic rerun
- `option_c_data_expansion_before_training`: reasonable secondary option because Stage 26 data health remains small-pilot-only with duplicate contexts
- `option_d_hold_training`: reasonable if owner wants no further training work until the active objective contract is reviewed

## Boundary

This packet recommends one next stage only. It does not self-authorize the next stage or any training run.
