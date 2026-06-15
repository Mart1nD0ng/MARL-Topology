# Stage 34 Logit/KL/Stability Diagnostics

Implementation: `src/marl_topology/evaluation/stage34_gnn_diagnostics.py`.

Diagnostics include:

- logit_mean/std/min/max
- probability_mean/std
- entropy
- selected_edge_count
- proposal_count
- approx_kl_mean
- approx_kl_max
- clip_fraction
- actor_grad_norm
- critic_grad_norm
- logprob_ratio_mean/max
- score_shift_mean/max
- parameter_delta
- value_loss
- policy_loss
- advantage_mean/std
- collapse_seed_count and collapse reasons

The Stage34 diagnostic script emits finite pre-update diagnostics for all seven ablations and the full per-update diagnostic schema. It does not claim training stability because the full protocol did not run.
