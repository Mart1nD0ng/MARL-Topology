# Q9 PART 2 — decision (residual + PBRS end-to-end training) — Q9 COMPLETE

**Result: KEEP. The residual+PBRS arm is correctly wired and trains end-to-end (eval no shaping); the
deployable A/B is an HONEST NEGATIVE (no improvement over the anchor), and on URBAN the residual RL is
UNSTABLE without an anchor trust-region (the Q5 instability recurs) — with the trust-region it is stable
and converges to the anchor.** This is the campaign's recurring pattern, now for the residual+PBRS arm:
the learned policy at best MATCHES the strong deployable anchor; the feasibility-region learning
bottleneck is not broken.

## What changed (the residual arm's trainer; eval no shaping)
`scripts/diagnostics/residual_pbrs_train.py` — a self-contained residual+PBRS REINFORCE trainer:
- Rollout: per frame compute the `local_hysteresis` anchor; `sample_residual` (Q9-PART1) around it →
  topology + log-prob; reward `r_t = reward_of(true)`; PBRS `F_t` from `dquorum_potential` (Φ = −D_quorum,
  terminal Φ_T = 0, the bridge, training-only); shaped `s_t = r_t + F_t`.
- REINFORCE on the shaped return-to-go + a running baseline; an optional **anchor trust-region**
  `--anchor-reg` (penalize the expected number of flips → keep the policy near the anchor).
- Eval (NO shaping): MAP residual decode (flip iff `z > 0`, consistent with the sampler) → true C/E/L,
  retention, switches, vs the anchor's own deployed metrics. PBRS appears nowhere in eval.

## Result (pilot: train 8, held 8, 6 frames, REINFORCE, single seed)
| run | data | diverged | resid feasibility | anchor feasibility | retention | resid energy | anchor energy | switches/frame |
|---|---|---|---|---|---|---|---|---|
| PBRS, no trust-region, 12 upd | random | no | 0.556 | 0.556 | **1.0** | = anchor | = anchor | 1.17 |
| PBRS, no trust-region, 25 upd | random | no | 0.521 | 0.521 | **1.0** | = anchor | = anchor | 1.13 |
| PBRS, **no trust-region, 25 upd** | urban | no | **0.042** | 0.646 | **0.0** | 0.250 | 0.104 | **11.08** |
| PBRS, **anchor-reg 0.5, 25 upd** | urban | no | 0.646 | 0.646 | **1.0** | = anchor | = anchor | 1.04 |

## The findings (honest)
1. **Mechanism correct + stable training**: the residual+PBRS arm trains end-to-end, the PBRS telescopes
   (Q9 PART 1), and the eval uses NO shaping (true C/E/L). No NaN divergence (the Q5 NaN mode is avoided).
2. **Q5 RL-instability RECURS on urban without a trust-region**: at 25 updates the urban residual policy
   BREAKS the anchor (retention 0.0, feasibility collapse 0.65→0.04, energy/switches explode) — it walks
   off the anchor manifold, exactly the Q5 urban failure. **The contract's retention metric flagged it
   (urban retention = 0.0).**
3. **The anchor trust-region fixes it**: `--anchor-reg 0.5` keeps the urban policy at the anchor
   (retention 1.0, matches anchor) — the residual learning is safe.
4. **No improvement over the anchor**: in every stable run the residual+PBRS policy CONVERGES TO the
   anchor (retention 1.0, identical feasibility/energy). The PBRS shaping (optimum-preserving) does not
   make the residual RL BEAT the strong deployable anchor at this scale — consistent with the Q7 (~22%)
   / Q8 (urban-47%) central ceilings and the campaign's "learned ≈ deployable at best" pattern.

## Honesty / scope
- Single-seed pilot, REINFORCE, short training — NOT a multi-seed headline (that is Q12). The
  result is reported as-is: the residual+PBRS deployable policy matches (does not beat) the anchor, and
  needs the trust-region to stay safe on urban.
- This trainer is the residual arm's training entrypoint (the BCSP `--baseline` trunk uses a different
  action space). Folding it into the trunk `--baseline` arms is deferred (no win to justify it).
- Final metric is ALWAYS the true closed-form PBFT C/E/L; D_quorum/PBRS are training-only.

## Acceptance table (Contract v3 §15)
- Phase: **Q9 — full residual + PBRS (PART 1 mechanism + PART 2 end-to-end training) — COMPLETE**
- Status: **mechanism VALIDATED_POSITIVE** (PBRS telescopes + optimum-preserving + trains stably);
  **deployable A/B VALIDATED_NEGATIVE** (residual+PBRS matches, does not beat, the anchor; urban needs the
  trust-region to avoid the Q5 instability).
- Implemented ✓ / Wired (residual trainer entrypoint) ✓ / Active in this run ✓ (training + eval-no-shaping)
- Test scale: 9 unit (3 PBRS + 3 sampler + 3 trainer) + suite 763/0; pilot single-seed
- Positive: PBRS correct + optimum-preserving; residual arm trains; anchor trust-region prevents the Q5
  instability; retention reported (the urban-0.0 red flag caught + fixed).
- Negative: residual+PBRS gives NO improvement over the deployable anchor (matches it at best); urban
  unstable without the trust-region.
- Conclusion scope: the residual+PBRS machinery is correct and safe (with the trust-region) but does NOT
  beat the anchor at this scale — the feasibility-region learning bottleneck persists.
- Next action: **Q10 — local edge handshake** (two-round endpoint score exchange + shared edge score to
  reduce mutual-acceptance mismatch; control-communication cost recorded; deployment-decentralized).
