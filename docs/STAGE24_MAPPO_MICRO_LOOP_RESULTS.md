# Stage 24 MAPPO Micro-Loop Results

## Verdict

```text
stage24_pass_complete_critic_integrated_micro_loop
```

Stage 24 implemented and ran a complete small-scale MAPPO-style loop. The active sampler remains:

```text
physical_plackett_luce_top_k_sampler
```

## Smoke Mode

Command:

```powershell
python scripts\train\stage24_mappo_micro_loop.py --mode smoke
```

Result: passed loop completeness with 16 transitions, 2 minibatches per epoch, one policy update, centralized critic values used in advantage computation, finite losses, finite KL, and actor/critic parameter changes.

## Micro Mode

Command:

```powershell
python scripts\train\stage24_mappo_micro_loop.py --mode micro
```

Result: passed closeout decision mode with 64 transitions, minibatch size 16, three update epochs, and five policy updates.

## Key Evidence

| Check | Result |
| --- | --- |
| Centralized critic used in advantage computation | passed |
| GAE implemented | passed |
| PPO/MAPPO-style clipped actor loss | passed |
| Critic value loss optimized | passed |
| Actor parameters changed | passed |
| Critic parameters changed | passed |
| Reward config unchanged | passed |
| Actor leakage | none detected |
| Checkpoint/artifact write | none |
| Scale-up training | not performed |
| Active sampler count | exactly one |

## Validation

```powershell
python -m pytest -q
# 787 passed

python harness\scripts\validate_tasks.py
# Task validation passed: 82 tasks

python scripts\train\stage24_mappo_micro_loop.py --mode smoke
# stage24_smoke_pass_complete_loop_check

python scripts\train\stage24_mappo_micro_loop.py --mode micro
# stage24_pass_complete_critic_integrated_micro_loop
```

## Micro Diagnostics

| Sampler | Tau feasible before/after | Projection rejection | KL max | Actor delta | Critic delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `endpoint_budgeted_physical_proposal_sampler` | 0.46875 -> 0.46875 | 0.0854166667 | 0.0000609718 | 0.1647816824 | 0.1019107377 |
| `physical_plackett_luce_top_k_sampler` | 0.234375 -> 0.234375 | 0.0572916667 | 0.0000463538 | 0.1938287550 | 0.2630051463 |

No sampler collapsed to empty or full graph. Entropy remained above the configured floor. Approximate KL stayed far below `0.03`.

## Residual Risk

The critic value loss is finite and optimized, but still high because this is a short micro-loop over a small evidence set with uncalibrated value predictions. Stage 24 therefore does not authorize scale-up training. Stage 25 should plan a small formal pilot with stronger repeated-seed evidence and value calibration diagnostics.
