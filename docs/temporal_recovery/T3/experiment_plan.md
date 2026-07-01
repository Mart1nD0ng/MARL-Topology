# T3 — experiment plan (belief correction target; task 3.1/3.2)

## Hypothesis (one variable: the belief parametrization)
R2's belief predicted the ABSOLUTE current psucc logit and echoed the stale input (never beat the stale-echo
floor). Reparametrize so the head predicts a CORRECTION on the stale value:
`belief_logit = stale_logit + head_output` (echo = head_output 0 = zero-baseline). Then the head learns only the
SIGNED stale→current delta (a smaller, structured target that also carries direction, task 3.2), and echo is no
longer a global optimum. Hypothesis: the correction belief beats the stale-echo floor and/or improves the
recovered-psucc anchor feasibility.

## Controlled variable
`belief_logit = stale_logit + head` (correction) vs `belief_logit = head` (absolute, R2). Identical actor init,
GRU, features (leak-free velocity extras), loss fn (weighted Huber), weights, epochs, `residual_leak=0.1` (T2).
In-policy belief (shares the actor GRU). True current psucc = training-only label.

## Metrics (5 seeds × {random,urban}, delay-1, held; 95% CI)
- `cor_beats_floor` = stale_echo_MSE − correction_MSE (the R2 failure metric).
- `correction_vs_abs_mse` = absolute_MSE − correction_MSE (does the parametrization reduce error?).
- `dir_acc` = fraction of MOVED edges with correct correction sign (task 3.2; chance 0.5).
- `feas_gain` = feas(recovered psucc → anchor) − feas(stale → anchor) (T1 ranking/decision metric).
- Sub-pilot: move-weighting the loss by |correction_target| (magnitude emphasis).

## Load-bearing tests
`test_temporal_recovery_T3_correction.py`: correction target nonzero on moved edges (echo not optimal) / echo
zero-baseline reproduces stale / directional accuracy rewards correct sign / the generator's parametrization is
a single variable (absolute=head vs correction=stale_logit+head, same head).

## Decision rule
- correction beats floor (CI>0, non-negligible) AND/OR improves the anchor → the correction recovers CSI → KEEP+build.
- correction ≈ floor and no anchor gain → NEGATIVE (magnitude not recovered); keep the parametrization, diagnose
  the limit, proceed to T4 (edge recurrence) / T5 (uncertainty).

## Definition of done
Failing-first tests; 5-seed CIs + raw artifact; scope explicit; decision.md; adversarial Workflow.
