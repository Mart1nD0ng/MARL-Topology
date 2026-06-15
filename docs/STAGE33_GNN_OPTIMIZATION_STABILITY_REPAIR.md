# Stage 33 GNN Optimization Stability Repair

## Controlled Object

The controlled object is GNN actor optimization inside the official MAPPO loop: actor LR, critic LR, warmup, scheduler, gradient clipping, PPO diagnostics, collapse criteria, and seed-level evaluation.

## Fixed Protocol

Stage33 ran exactly three fixed GNN stability configs:

- `low_lr_no_warmup`: actor_lr `1e-3`, max_grad_norm `0.5`, no warmup.
- `low_lr_with_warmup`: actor_lr `5e-4`, max_grad_norm `0.5`, linear warmup plus cosine decay.
- `low_lr_with_warmup_stronger_grad_clip`: actor_lr `5e-4`, max_grad_norm `0.35`, linear warmup plus cosine decay.

Shared controls:

- critic_lr `1e-4`
- weight_decay `1e-5`
- update_epochs `2`
- entropy_coef unchanged at `0.01`
- tau_requirement_min unchanged at `0.9`
- five seeds: 3301..3305
- official Stage25-compatible micro-loop dimensions for train/eval/test

## Run Evidence

Training report: `result_save/stage33_gnn_stability_repair/stage33_fixed_protocol/training_report.json`.

Summary from the run:

| model | config | collapse_rate | stopped seeds | stop reason | eval tau mean |
|---|---:|---:|---:|---|---:|
| active v3 residual norm | low_lr_no_warmup | 1.0 | 5/5 | kl_explosion | 0.0 |
| active v3 residual norm | low_lr_with_warmup | 1.0 | 5/5 | kl_explosion | 0.0 |
| active v3 residual norm | low_lr_with_warmup_stronger_grad_clip | 1.0 | 5/5 | kl_explosion | 0.0 |
| archived v2 GNN | low_lr_with_warmup | 0.8 | 4/5 | kl_explosion | 0.0 |
| role/resource v3 | low_lr_with_warmup | 1.0 | 5/5 | kl_explosion | 0.0 |
| MLP diagnostic | low_lr_with_warmup | 0.8 | 4/5 | kl_explosion | 0.0 |

## Interpretation

Lower LR, warmup, and stronger gradient clipping did not repair the active v3 GNN. The collapse signature is dominated by KL explosion, so Stage33 is classified as an optimization failure. Because the selected metric winner was archived v2, it is also an architecture-selection failure for the active production candidate.

## Acceptance

Stage33 stability gate did not pass. Collapse rate exceeded the 20 percent gate for every GNN candidate.

## Residual Risk

The protocol is intentionally bounded. More optimization configs would violate Stage33 owner constraints unless approved as a new owner decision.
