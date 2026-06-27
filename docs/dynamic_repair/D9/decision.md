# D9 — decision

**Result: KEEP (mechanism wired + verified + active; A/B headline deferred to D13).** Per-agent COMA
counterfactual credit is now available in the dynamic episode arm, opt-in, budget-neutral, and
adversarially verified.

## What changed (one variable: per-agent counterfactual credit, opt-in)
- `episode_rollout` (D9 path): when `--counterfactual`, the action-conditioned Q critic
  (`critic_sees_action=True`) produces per-agent COMA advantages
  `A_{i,t}=Q(s_t,S_t)−E_{S̃_i}Q(s_t,S̃_i,S_{-i})` via the reused static `counterfactual_advantages`
  (the counterfactuals re-decode through the local mutual decoder and re-forward the Q critic — FREE: no
  evaluator call → the evaluator budget is unchanged at 1/frame). `FrameRecord.cf_adv` stores them.
- `_critic_value` dispatches V (`critic_scene_value`) vs action-conditioned Q (`critic_q_value` over the
  recorded active-edge one-hot); the critic regresses `Q(s_t,S_t)→G_t`.
- `run_dynamic_training`: builds the Q critic when counterfactual; the PPO advantage per (frame,agent) is
  the per-agent `A_{i,t}` (not the shared `G_t−V`); activation records `critic_sees_action`,
  `counterfactual`, `k_cf`, `counterfactual_budget_neutral`.
- `train_decentralized_rl.py`: the `--counterfactual requires graph-mappo` guard now exempts `--dynamic`
  (the dynamic arm has its own COMA wiring).
- Default off → V critic + shared advantage (byte-identical to pre-D9).

## Tests (failing-first; the 2 wiring tests fail on `df9ad2e` with TypeError, pass after)
- `test_dynamic_counterfactual_advantages_not_all_equal` (per-agent A_{i,t} within a frame have nonzero
  variance — real per-agent credit, unlike the shared advantage),
  `test_dynamic_counterfactual_budget_neutral` (n_cf == n_v == n_frames evaluator calls — budget-neutral),
  plus `test_dynamic_q_critic_sees_action` and `test_dynamic_counterfactual_fixes_s_minus_i` (Q
  action-conditioning + COMA S_{-i}-fixed unbiasedness on a dynamic frame).
- Dynamic-RL file **21/21**; unit suite **684/0**; contract **63/0**; `--dynamic --counterfactual` smoke
  exit 0 (activation: `critic_sees_action=true, k_cf=4, counterfactual_budget_neutral=true`).

## Adversarial verification (focused single agent, 4 lenses, PASS, no concerns)
1. **Budget neutral**: counterfactual calls ONLY the critic (`critic_q_value`) + the mutual decoder,
   never `reward_of`/the evaluator/a solver; per-frame evaluator count unchanged.
2. **Byte-identity off**: default builds a V critic, `_critic_value`→`critic_scene_value`, `cf_adv=None`,
   shared advantage — pre-D9 path; no existing dynamic test broke.
3. **COMA correctness**: Q action-conditioned, regresses to G_t; per-agent A_i aligned with `per_agent`
   (same filter); per-agent A_i fed to the per-agent PPO ratio; S̃_i drawn from (θ_i,b_i) only
   (independent of the realized S_i — unbiasedness).
4. **Decentralization (D4 lesson)**: the Q critic is training-only (`dynamic_eval` takes no critic; deploy
   uses `local_mutual_assemble`); D9 is purely critic-side → the vectorized-evaluator blind spot does NOT
   apply.

## Scope / next
- D9 delivers the MECHANISM (correct, active, budget-neutral). It does NOT claim COMA improves
  feasibility/credit at scale — consistent with the static-campaign pattern (Phase 8 COMA was
  verified-correct but did not beat the shared advantage at N≤16) and the D8 finding (the binding limit is
  RL feasibility-region learning, not credit assignment). The dynamic-COMA-vs-shared-advantage A/B
  (≥5 seeds, per-seed + CI) is part of the D13 campaign — NOT claimed here.
- Next: **D10** (dynamic SCQ — closed-form counterfactual supervision of the Q critic; requires
  `--counterfactual`).
