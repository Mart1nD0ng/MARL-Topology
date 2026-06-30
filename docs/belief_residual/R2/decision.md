# R2 — decision (CSI belief prediction auxiliary) — REVISE: belief is a NO-OP CSI predictor (HONEST NEGATIVE)

**Result: REVISE / HONEST NEGATIVE.** The belief auxiliary is correctly built, ACTIVE in a training loss on
the policy actor, and leak-free — BUT it does **NOT recover the current CSI**: its held belief MSE sits
**at/above the trivial stale-echo floor** (predict the stale observed psucc), even **with leak-free velocity
features** and 80–400 epochs; recurrence doesn't help either. So the belief head learns to **echo its stale
input**, recovering ~none of the staleness — a no-op CSI *predictor*. This was caught by the R2 verification
(Workflow `wozljm9uf`, MAJOR) and I reproduced + extended it (velocity, more epochs). Path-specific negative
for TechSpec chain 1 (belief recovery). The belief machinery (correct, leak-free) is kept but **not claimed
to recover CSI**. **5-seed `floor − belief` CI is ENTIRELY NEGATIVE** (delay1 −0.00095 [−0.0015,−0.0004];
delay2 −0.00098 [−0.0015,−0.0004]; partial −0.00079 [−0.0013,−0.0002]; **beats_floor 0/5** all modes) — the
belief head is significantly *worse* than the trivial echo (`belief_floor_multiseed.json`).

## What is verified TRUE (the mechanism is built honestly)
- **Belief loss enters a real training loss on the policy actor** — gradients reach `belief_head` AND the
  shared GRU (spy test); not a Q2-style standalone diagnostic. (Workflow lens 1 PASS.)
- **Leak-free** — the belief target reads `scene.context(t)` (true current psucc, training-only label); the
  actor input `ef[:,0]` is the STALE observed value; an exhaustive per-column scan found no true-CSI leak in
  any actor input. (Workflow lens 1 PASS.)
- **Held belief MSE decreases over epochs** — but only to the stale-echo floor (it learns the echo).

## What is FALSE (the corrected claim) — the no-op finding
- The belief head does **NOT beat the stale-echo floor**: delay1 belief 0.116 vs floor 0.116; delay2 0.112
  vs 0.111; partial 0.052 vs 0.051 — at/above the floor. **Adding leak-free velocity does not help** (delay2
  belief 0.1118 vs floor 0.1115); **more epochs do not help** (200ep 0.1138 / 400ep 0.1175 — overfits, never
  beats floor 0.1136). The trained head moves only ~2.6% toward the true value and its correction direction
  is uncorrelated with the truth (corr ≈ −0.035), despite ~15% of links being materially stale.
- Therefore the earlier R2 framing ("belief improves prediction → works") was **misleading**: the improvement
  is the head learning the trivial echo, not recovering CSI. Corrected here.
- Recurrence: also a null (5-seed paired mem−rec CI spans 0) — but this is now secondary: BOTH arms fail to
  beat the echo floor, so neither is a functioning belief predictor.

## Why (diagnosis) — leak entanglement + per-frame ceiling
- The recovery signal Q2 used (rel_vel / distance_delta) is, in the env, bundled in `_motion_edge_block`
  with a `csi_delta = cur − prev` term computed from the TRUE current channel — a true-CSI **leak**. So
  `motion_features` is kept off in stale-CSI runs, leaving the leak-free feature set = stale CSI + age only.
- R1 added a leak-free velocity helper (`csi_belief.leak_free_motion_features`, rel_vel + distance_delta, no
  `csi_delta`); even with it the head cannot beat the echo floor. So at N≤16 on the policy actor, the
  per-frame velocity does not let the head recover the current channel beyond the stale value (Q2's clean-
  predictor recovery does not transfer to the graph actor / held scenes here).

## Path-specific scope (Contract v4 §5, §14)
> "On the policy actor (`BeliefResidualActor`) with leak-free features (stale CSI + age + rel_vel + distance_
> delta), the supervised belief head does NOT recover the current link psucc beyond echoing the stale
> observation (held MSE at/above the stale-echo floor; 5 seeds; 80–400 epochs; velocity on/off; recurrent
> and memoryless alike). The belief loss is genuinely active and leak-free; it is a no-op CSI predictor."
This does NOT claim temporal modeling / belief is impossible in general (Q2 recovered on a clean predictor),
only that this policy-actor belief head does not, here.

## Disposition / next
- **REVISE**: keep the belief machinery (it is correct + leak-free + a valid component), but **do NOT claim
  it recovers CSI**. The Mechanism-Path Matrix records belief as ACTIVE_IN_LOSS but **no-op for recovery**.
- For R3: treat `L_CSI` as an OPTIONAL auxiliary (ablatable), NOT a load-bearing recovery mechanism. The
  campaign's recoverable value, if any, must come from the other chains (residual PPO / CTDE critic;
  beneficial-edit supervision; evidence-gating) — R3+ tests those. The "belief-guided" premise is weakened.
- Honest test pins the finding: `test_belief_does_not_beat_stale_echo_floor` (asserts the no-op).

## Acceptance (Contract v4 §15)
- Claim Cards ✓ (corrected) / Mechanism-Path Matrix ✓ / tests pass ✓ (7 R2 incl. the honest-negative pin;
  suite green) / activation artifact ✓ / pilots + multi-seed raw ✓ / decision ✓ / scope explicit ✓.
- Exit condition (Workflow R2): "belief loss enters training loss" MET ✓; "recurrent significantly <
  memoryless" NOT met; **"belief recovers CSI (beats stale-echo floor)" NOT met → HONEST NEGATIVE.**
- Decision: **REVISE** — belief kept as a verified-but-no-op component; proceed to R3 with `L_CSI` ablatable.
- Next: **R3** — residual PPO + CTDE value critic (the training-stability chain), with raw-L2 active; `L_CSI`
  included only as an ablatable auxiliary. The R0 PPO-spy tripwire flips at R3.

## Adversarial verification
- Workflow `wozljm9uf` (4 lenses): lens 1 (belief-active+leak-free) PASS; **lens 2 (belief improves) MAJOR —
  the head only reaches the stale-echo floor (no-op)**; lens 3 (recurrence-null) PASS; lens 4 (additive)
  MINOR. **This decision ADOPTS the MAJOR finding** (the earlier "belief works" framing is corrected to the
  no-op negative), reproduced + extended (velocity, epochs) above. 5-seed `floor − belief` CI entirely
  negative (beats_floor 0/5) — decisively confirms the belief head is a no-op (slightly worse than the echo).
