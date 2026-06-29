# Q14 (post-campaign addendum) — stale-CSI + residual JOINT experiment (closing the one open loop)

> Status: a POST-COMPLETION addendum to the Q0–Q13 POMDP-QP-FAR campaign. It does NOT reopen the campaign
> headline; it closes the single open loop the campaign left: the residual policy was never trained/evaluated
> UNDER stale CSI (Q9–Q12 all ran current CSI; Q2 only validated CSI *prediction* at the perception layer
> and deferred the policy question to "Q12", which then ran current CSI).

## Hypothesis (the single thing this closes)
Stale/partial CSI (Axis A) was validated only as a perception precondition (Q2). This addendum asks the
control-level question directly: **under stale CSI, (1) does the deployable anchor actually lose feasibility,
and (2) can the feasible-anchored residual policy — with cross-frame recurrence to exploit temporal
structure — recover that loss?**

## Single change (one variable: the CSI mode is now active in the residual loop)
A new eval/diagnostic `scripts/diagnostics/stale_csi_residual_joint.py` reuses the verified residual+PBRS
machinery (`residual_pbrs_train.py`) but builds the train/held scenes with a `CsiObservationModel`
(`current | delay-1 | delay-2 | partial`), and adds a recurrent rollout/eval that **carries the GRU hidden
state across frames** (vs the memoryless `hidden=None`). Hard constraints preserved: deployed actor sees
ONLY the stale observed channel ĝ_t + age + mask + motion + prev topo; the final metric is the TRUE
closed-form PBFT C/E/L on the current channel; PBRS is training-only; eval uses NO shaping; the anchor is
the deployable reference.

## Two sections
1. **Anchor sensitivity (training-free):** roll the deployable `local_hysteresis` anchor under each CSI mode,
   score with the true channel. Does stale CSI hurt the anchor? (If not, there is nothing to recover.)
2. **Residual recovery:** under stale CSI, train the residual policy {memoryless, recurrent} with a FREE
   config (residual_prior −1.0, anchor_reg 0.0 so it CAN deviate). Does the residual recover the loss, and
   does recurrence beat memoryless? Report residual vs anchor feasibility, paired diff + CI, retention,
   divergence.

## Success / failure criterion
- Recovery succeeds iff the residual (esp. recurrent) feasibility is significantly ABOVE the stale anchor.
- Recovery fails iff residual ≤ anchor (clamped, or deviates-and-worsens), or recurrence == memoryless.

## Scope
5 seeds × {urban, random} × held-N{8,12,16}, 6 frames; N≤16 only. This is a diagnostic addendum, NOT a new
headline; it sharpens the campaign's Axis-A conclusion with control-level evidence.
