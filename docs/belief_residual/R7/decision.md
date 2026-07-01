# R7 — decision (adaptive anchor-KL / safety constraint) — CONFIRMATORY NEGATIVE (== anchor); KEEP mechanism

**Result: the adaptive anchor-KL trust region lands EXACTLY at the anchor — a CONFIRMATORY negative that closes
the "fixed vs adaptive anchor protection" objection.** Replacing R3's FIXED anchor pull (`residual_prior=−1.0`)
and Q14's fixed flip penalty with an ADAPTIVE anchor-KL coefficient (tighten on retention-drop / loosen on
val-improve), and removing the fixed pull entirely (`residual_prior=0.0`), the trained residual policy still
converges to the anchor (edit_rate 0.0, retention 1.0) on all 5 seeds, both regimes. The adaptive controller is
load-bearing (beta_anchor genuinely tightens AND loosens per seed) and never diverges — but there is no
beneficial-direction gradient for it to move toward (R3's finding), so loosening finds nothing and the policy
stays at the anchor. **KEEP the mechanism (correct + load-bearing + stable); the fixed flip penalty was NOT the
reason the residual == anchor — R6's precision-limit diagnosis stands.**

## 5-seed A/B (`adaptive_kl_metrics.json`; deployed MAP decode, true PBFT feasibility; 20 epochs, 6 scenes)
| metric | random | urban |
|---|---|---|
| anchor_feas | 0.300 | 0.783 |
| fixed (R3, residual_prior=−1.0) resid_feas | 0.300 | 0.783 |
| adaptive (R7, residual_prior=0.0 + adaptive KL) resid_feas | 0.300 | 0.783 |
| **adaptive − anchor feasibility** | **0.000 [0.0, 0.0]** | **0.000 [0.0, 0.0]** |
| **adaptive − fixed feasibility** | **0.000 [0.0, 0.0]** | **0.000 [0.0, 0.0]** |
| adaptive edit_rate | 0.0 [0,0] | 0.0 [0,0] |
| adaptive retention | 1.0 [1,1] | 1.0 [1,1] |
| beta_anchor_final (mean) | 0.131 | 0.127 |
| diverged | 0/5 | 0/5 |

**Read:** adaptive == fixed == anchor, exactly, on every seed (adaptive−anchor CI is a degenerate [0.0, 0.0]).
The `beta_anchor` trajectories are non-trivial and per-seed distinct — e.g. random seed 0 climbs 0.10→0.80 then
settles to 0.27 (tightens under retention pressure, loosens when stable); random seed 4 loosens monotonically
0.15→0.018 (retention stays 1.0, so it keeps loosening) — so the controller is genuinely adaptive and
load-bearing, NOT a constant. Despite that, the MAP-decode residual never leaves the anchor (edit_rate 0). No
divergence, no retention collapse.

## Why (mechanism-level diagnosis — Contract v4 §13)
Same DATA/precision limit as R6, now confirmed from the TRAINER side:
- The adaptive anchor-KL correctly controls the trust-region SIZE (deviation from the anchor), and when it
  LOOSENS (beta small, no fixed pull) the policy is free to deviate — but it does NOT, because the PPO advantage
  provides no gradient toward a beneficial flip (R3: the residual policy has no direction signal; R4/R5 put the
  direction in the frozen HEADS, not in this policy). So loosening the trust region finds nothing to move to.
- This CLOSES the objection "R3/R6 == anchor only because the anchor protection (fixed prior / fixed thresholds)
  was too rigid": even a learned, retention-driven, self-loosening trust region lands at the anchor. The binding
  limit is the deployable PRECISION / availability of the direction signal in the deployed policy, NOT the
  anchor-protection scheme.

## Effect-on-Decision (Contract v4 §3)
The adaptive anchor-KL changes the ACTOR LOSS (anchor-KL term in the loss; grad pulls flip logits down) and the
TRAINING dynamics (beta_anchor adapts per epoch) — but it does NOT change the final deployed topology vs R3
(edit_rate 0.0 == anchor on all seeds). So the mechanism is load-bearing on the loss/training but the deployed
action is unchanged (== anchor) — reported honestly as a null effect on the topology, not a forward-only change.

## Path-specific scope (Contract v4 §5)
> "Replacing the FIXED anchor pull (residual_prior=−1.0 / fixed flip penalty) with an ADAPTIVE anchor-KL
> (retention-driven tighten/loosen, residual_prior=0.0), the residual PPO policy still decodes to the anchor
> EXACTLY (adaptive−anchor feasibility 0.000 CI[0,0], edit_rate 0.0, retention 1.0; 5 seeds; random + urban;
> N≤16; current-frame channel, stale overlay off). The adaptive controller is load-bearing (beta_anchor
> tightens AND loosens per seed) and never diverges, but finds no beneficial deviation. A CONFIRMATORY negative:
> the fixed anchor protection was NOT why R3/R6 == anchor — the binding limit is the deployable direction signal,
> not the trust-region scheme. NOT a controller/divergence failure."

## Tests (failing-first → pass)
5 in `tests/unit/test_belief_residual_R7_adaptive_kl.py` (fail on HEAD: helpers/adaptive path absent → pass):
anchor_kl_penalty = mean flip-prob over candidates + its gradient pulls toward the anchor; update_beta controller
law (tighten/loosen/hold + clamps); the adaptive path still calls `ppo_clip_actor_loss` and trains the critic
(one variable vs R3); deployable 0-eval MAP decode, no fixed flip-penalty as the sole protection. R3+R7 12/12;
full suite 815/0.

## Acceptance (Contract v4 §15) / Decision
- adaptive residual feasibility CI > anchor → would be a positive (revise R6). **NOT observed** (CI [0,0]).
- adaptive == anchor (edit 0, retention 1, CI spans/at 0) → CONFIRMATORY negative. **Observed.** KEEP the
  mechanism; R6's precision-limit binding remains. → R8 (stale-CSI premise) or close-out.
- adaptive unstable → REVISE. **NOT observed** (0/5 diverged, retention 1.0).
- **Decision: KEEP (mechanism correct + load-bearing + stable); CONFIRMATORY NEGATIVE on the deployed gain.**

## Adversarial verification (Workflow) — see below
