# Q9 — decision (full residual + PBRS) — PART 1: mechanism + RL-readiness

**Result: KEEP. The PBRS mechanism + the residual RL-readiness are implemented and VERIFIED.** The
end-to-end deployable residual-policy training A/B (wiring the residual+PBRS arm into the trunk rollout/
PPO) is the remaining Q9 step (PART 2, next fire) — it is bounded by the Q7/Q8 central ceilings and its
multi-seed headline belongs to Q12.

## What changed (mechanism; opt-in, training-only; default path unchanged)
1. `src/marl_topology/training/potential_shaping.py` — the Ng-Harada PBRS primitive:
   - `dquorum_potential(ev, topology, eta) = −eta·D_quorum(topology)` (training-only, via the bridge).
   - `episode_pbrs(potentials, gamma, lam)` → `F_t = lam(γΦ_{t+1} − Φ_t)` with TERMINAL `Φ_T = 0`.
   - `discounted_shaping_sum` → which telescopes to `−lam·Φ_0`.
2. `training/residual_action.py` — the residual RL sampler (makes the Q6 action space PPO-trainable):
   - `sample_residual` (per-candidate-edge Bernoulli flip ~ sigmoid(logit) + tractable log-prob),
     `residual_logp`, `residual_candidate_mask`, `residual_decode_from_flips` (zero flips → anchor).

## Verification (math proof → real-shard)
- **6 new unit tests, all pass; full suite 760/0:**
  - PBRS TELESCOPES to `−lam·Φ_0` (terminal Φ=0); PBRS PRESERVES the small-MDP optimum under 3 distinct
    potentials (Ng-Harada — a sign error in the shaping form would flip an argmax and fail this test).
  - residual sampler log-prob is self-consistent (recomputed == sampled), budget-respecting; zero flips
    decode to the anchor exactly; add-mode candidate mask = non-anchor edges only.
- **Real-shard PBRS smoke (random + urban, 5-frame episodes):** the D_quorum-potential shaping telescopes
  EXACTLY to `−lam·Φ_0` (random 1.295 == 1.295; urban 0 == 0, the feasible anchor having Φ_0 = 0). The
  mechanism is correct on real data.

## Why this matters (the campaign's principled lever)
PBRS with `Φ = −D_quorum` is the Ng-Harada potential the campaign's central diagnosis identified: it
adds a dense gradient on the feasibility plateau (where the true reward is flat) WITHOUT changing the
optimal policy — so it can only help learning, never optimize a fake objective. D_quorum (Q4-authorized)
enters the TRAINING reward ONLY as this optimum-preserving shaping; the EVALUATION uses NO shaping (true
C/E/L). This is the rigorous answer to "reshape the reward without violating the real metric".

## Honesty / scope
- This delivers the MECHANISM (PBRS + residual sampler), verified. It is NOT yet wired into the trunk's
  end-to-end training, and there is NO deployable residual-policy A/B yet — that is PART 2.
- The end-to-end residual policy is bounded by the central-reference ceilings (Q7 add-repair ~22% random;
  Q8 prune urban-47%) and must heed the Q5 RL-instability (urban BC diverged). A deployable win is NOT
  assumed; the multi-seed A/B is Q12.
- PBRS is only used in the T>1 dynamic task with terminal Φ=0; never claimed in the static T=1 bandit.

## Acceptance table (Contract v3 §15)
- Phase: **Q9 (PART 1) — PBRS mechanism + residual RL-readiness**
- Status: **VALIDATED_POSITIVE (mechanism correct + optimum-preserving + RL-ready)** — not a policy result.
- Implemented ✓ / Wired into trunk training ✗ (PART 2) / Active in this run ✓ (unit + real-shard smoke)
- Test scale: 6 new unit + suite 760/0 + real-shard PBRS smoke (random + urban)
- Mechanisms active: PBRS primitive + residual sampler. Not tested: end-to-end residual-policy training (PART 2 / Q12).
- Positive: PBRS telescopes + preserves the optimum (proven); D_quorum potential telescopes on real data;
  residual sampler is consistent + budget-safe.
- Negative: none in scope (no policy claim).
- Conclusion scope: the PBRS + residual-RL machinery is correct and optimum-preserving; NO deployable
  performance claim yet.
- Next action: **Q9 PART 2** — wire the residual+PBRS arm into the dynamic trunk (rollout uses
  `sample_residual` around the anchor; PPO ratio uses `residual_logp`; reward adds `episode_pbrs` from
  `dquorum_potential`, training-only, eval no shaping; activation artifact), real-shard smoke, single-seed
  pilot A/B (residual+PBRS vs anchor) reporting retention/feasibility/E/L. Then Q10/Q11/Q12.
