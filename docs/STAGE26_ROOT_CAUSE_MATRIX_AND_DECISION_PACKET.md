# Stage 26 Root Cause Matrix and Decision Packet

## Decision

- recommended option: `option_b_repair_critic_before_more_training`
- recommended next task: `stage_27_critic_baseline_repair_before_more_training`
- scale-up approved: `False`
- owner decision required: `True`
- rationale: critic health is the strongest blocker: explained variance is weak, value scale mismatch is large, and the loop labels critic_not_helpful

## Root Cause Matrix

| Component | Status | Evidence | Contribution | Confidence | Repair | Blocks Scale-Up | Blocks LSTM | Blocks Reward Tuning | Blocks Sampler Change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `data_health` | `WARN` | Stage 25 train/eval split; Stage 21/22 actor and critic target views | medium-high because the formal pilot reused 10 unique contexts into 24 slots | `high` | expand and diversify scenario/evidence set before larger training | `True` | `True` | `False` | `False` |
| `communication_health` | `PASS` | Stage 3 finite-blocklength link recomputation; Stage 21 network records | medium if link probabilities are saturated or flat | `medium` | monitor | `False` | `True` | `False` | `False` |
| `consensus_health` | `PASS` | Stage 4 expected-initiator PBFT evaluations | medium when expected-initiator averaging exposes weak primary spread | `medium` | monitor | `False` | `True` | `False` | `False` |
| `reward_objective_health` | `WARN` | Stage 25 reward surface analysis; Stage 25 eval aggregate deltas | medium because reward worsened while latency and energy improved | `high` | run reward-objective diagnostic repair without weight tuning | `True` | `True` | `True` | `False` |
| `assembler_health` | `WARN` | Stage 25 projection diagnostics | medium because top proposal rejection worsened slightly and tx budget rejections are present | `high` | diagnose tx budget and projection mismatch without changing active assembler | `True` | `True` | `False` | `True` |
| `sampler_health` | `WARN` | Stage 25 sampler logprob and entropy metrics | medium-low; sampler remains valid but projected proposals are not fully aligned | `medium` | diagnose proposal diversity and projection conversion without switching sampler | `True` | `True` | `False` | `True` |
| `actor_health` | `WARN` | Stage 25 actor score summaries; Stage 21/22 actor targets | medium; actor moved but did not reduce projection friction | `medium` | inspect actor score calibration and target alignment without training | `True` | `True` | `False` | `False` |
| `critic_health` | `FAIL` | Stage 25 value-loss/explained-variance metrics; critic prediction vs return pairs | high because explained variance is near zero and value scale mismatch is large | `high` | repair value target scale and critic baseline before more training | `True` | `True` | `False` | `False` |
| `mappo_loop_health` | `WARN` | Stage 25 update and eval metrics | medium-high because one seed stopped and critic signal was weak despite safe KL/entropy | `high` | fix critic/loop diagnostics before larger pilot | `True` | `True` | `False` | `False` |
| `harness_state_health` | `PASS` | docs/PROJECT_STATE.md; result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config/training_report.json; source scan for dynamic torch and forbidden architecture paths | low unless manifest or state drift appears | `high` | monitor | `False` | `True` | `False` | `False` |

## Options

- `option_a`: Proceed only if no critical fail and critic/loop are acceptable.
- `option_b`: Repair critic before more training.
- `option_c`: Expand scenario/data before more training.
- `option_d`: Reward-objective diagnostic repair without weight tuning.
- `option_e`: Assembler/sampler diagnostic repair without sampler switch.
- `option_f`: Actor feature/target repair.
- `option_g`: Hold training and request owner decision.

## Owner Gate

Stage 26 recommends the next task only. It does not authorize scale-up, LSTM, reward tuning, sampler switching, checkpoint creation, or any training run.
