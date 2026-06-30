# R3 — decision (residual PPO + CTDE value critic) — KEEP: PPO stabilizes the trainer; == anchor (missing direction signal)

**Result: KEEP.** The residual trainer now runs GENUINE residual PPO (it calls `ppo_clip_actor_loss` with
per-edge/per-agent ratios, logs approx_kl/clip_fraction, target_kl early-stops) + a CTDE value critic
(explained variance ~0.56 after a LayerNorm fix) + entropy + the R1 raw-L2. **PPO is STABLE — no
clamp/collapse (retention 1.0, diverged False) — fixing the free-REINFORCE bimodal collapse.** BUT the PPO
residual policy **== the anchor** (edit_rate 0.0): it makes no beneficial deviation. Per Contract v4 §5 this
is a **missing-direction-signal** outcome (R4–R5), NOT a PPO/trainer failure. **5-seed A/B (`ppo_vs_reinforce.json`):
PPO 0/5 collapse on random AND urban (retention 1.0 all seeds, edit 0.0 all) vs free-REINFORCE 1/5 collapse
on each (random seed3 retention 0.30; urban seed1 retention 0.0, feas→0) — PPO eliminates the collapse.**
Verification: Workflow `wj8pzo538` (verdict below).

## What changed (one variable: the residual TRAINER, REINFORCE → PPO+critic; new trainer, REINFORCE retained)
`scripts/diagnostics/residual_ppo_train.py`:
- rollout stores per-edge `logp_old`, decisions, candidate_mask, reward, a global state for the critic.
- PPO inner epochs recompute per-edge `logp_new`; `graph_mappo.ppo_clip_actor_loss(logp_new, logp_old, adv)`
  with advantage repeated per candidate edge; **target_kl early-stops**; logs ratio/approx_kl/clip_fraction.
- `ResidualValueCritic` (LayerNorm-bounded global summary incl. a TRUE-CSI summary, training-only): the
  critic target is the **standardized** return (raw returns are large + low-variance on infeasible-heavy
  random → ill-conditioned EV); Huber + grad-clip; `A_t = G_norm − V`.
- entropy bonus; raw-L2 (R1) active; `L_CSI` ablatable (default off — R2 no-op).
- `residual_action.residual_logp_per_edge` (additive) — the per-edge logp (per-agent ratio; the joint ratio's
  variance explodes per the PPO docstring).

## Effect-on-Decision (PPO pilot, urban current-CSI, single seed)
| metric | value | reading |
|---|---|---|
| ppo_calls | 80 | PPO genuinely called |
| approx_kl / clip_fraction | ~0.0 / ~0.0 | small, KL-bounded updates (stable) |
| critic EV / value_loss | **0.56 / 0.18** | the critic is a real baseline — but EV is a SINGLE-run value; across seeds it ranges **0.08–0.56** (positive, not robust). The −1e5 was an online-undertraining artifact (a converged critic on raw returns gives EV ~0.37); LayerNorm + standardized target is a conditioning fix |
| entropy | 0.57 | stochastic policy (not collapsed to deterministic) |
| residual_feas / anchor_feas | **0.875 / 0.875** | residual == anchor |
| edit_rate / retention / diverged | **0.0 / 1.0 / False** | no edits, no collapse — STABLE clamp at the anchor |

The clamp-vs-collapse dichotomy is resolved by PPO toward **stable clamp**: it never collapses, but with no
beneficial-edit direction signal and the `residual_prior` keeping it near the anchor, the stable policy stays
AT the anchor.

**5-seed REINFORCE-vs-PPO A/B (same actor / rollout / scenes; only the optimizer differs):**
| trainer | random collapse | urban collapse | retention | edit_rate |
|---|---|---|---|---|
| free REINFORCE | **1/5** (seed3 ret 0.30, feas 0.083) | **1/5** (seed1 ret 0.0, feas 0.0) | bimodal | variable (0–0.60) |
| **residual PPO+critic** | **0/5** | **0/5** | **1.0 all** | **0.0 all (== anchor)** |

PPO removes the collapse on both regimes (0/5 vs 1/5 each) and is uniformly stable; it also never deviates
(edit 0.0), so residual feasibility == anchor feasibility on every seed.

## Path-specific scope (Contract v4 §5)
> "Replacing the residual trainer's REINFORCE optimizer with residual PPO + a CTDE critic REMOVES the
> free-REINFORCE bimodal collapse (5-seed A/B; PPO retention 1.0 / 0 collapse). It does NOT make the residual
> beat the anchor (edit_rate 0, residual == anchor) — because there is no beneficial-edit DIRECTION signal
> (R4–R5) and the policy has no gradient pointing off the anchor. This is a trainer-stability win + a
> direction-signal gap, NOT a PPO failure."

## Tests (failing-first → pass; load-bearing + effect-on-decision)
7 in `tests/unit/test_belief_residual_R3_ppo.py` (fail on HEAD: trainer/helper absent → pass). Full suite
**794/0**. The spy test (`test_residual_ppo_calls_ppo_clip_actor_loss`) is the realized R0 tripwire flip on
the residual PPO path; the REINFORCE baseline's R0 spy still passes (retained for the A/B).

## Acceptance (Contract v4 §15)
- Claim Cards ✓ / Mechanism-Path Matrix ✓ / load-bearing + effect-on-decision tests pass ✓ (7/7, suite
  794/0) / activation artifacts ✓ / pilot + 5-seed A/B raw ✓ / decision ✓ / scope explicit ✓.
- Exit condition (Workflow R3): bimodal clamp/collapse reduced ✓ (PPO 0 collapse vs REINFORCE bimodal) AND
  KL/clip/critic healthy ✓ (EV 0.56). PPO == anchor reported as a missing-direction-signal gap (→ R4–R5).
- Decision: **KEEP** — residual PPO + critic is the new canonical trainer (stable). The remaining lever is
  the DIRECTION signal (beneficial-edit supervision), not the trainer.
- Next: **R4** — beneficial oracle-edit dataset (per anchor, enumerate local add/remove/swap; compute
  ΔC/ΔD_quorum/ΔE/ΔL/ΔJ; label only positive-gain LOCAL edits — never full oracle topology). R4–R5 supply the
  direction signal R3 shows is missing.

## Adversarial verification (Workflow `wj8pzo538`, 4 lenses) — overall PASS
- **PPO GENUINELY CALLED PER-EDGE — PASS.** Spy fired; the PPO inputs are 84 elements = total candidate edges
  (one per (frame,edge), advantage repeated per edge), with **144 distinct per-edge ratios at inner epochs >0**
  — conclusively per-agent, NOT a joint per-frame scalar. Epoch-0 ratio==1 (approx_kl 0);
  `residual_logp_per_edge.sum()==residual_logp`; target_kl=1e-6 → 2 inner epochs vs 1e9 → 8. Residual path
  calls PPO (Contract v4 §7), not the trunk.
- **CTDE CRITIC REAL — PASS.** `critic_parameter_delta>0` across seeds; EV is the textbook
  `1−Var(G_norm−V)/Var(G_norm)` on the standardized target (constant predictor → EV 0; trained → positive),
  matching `graph_mappo.explained_variance`. The critic reads a true-CSI training-only global state and is
  ABSENT from the deployed-style eval. **Nuance (folded in): EV is single-run; across seeds 0.08–0.56 (not
  robust); the −1e5 was an online-undertraining artifact, not intrinsic to raw returns.**
- **PPO STABLE vs REINFORCE COLLAPSE — PASS.** A/B re-ran byte-identical on 3 seed-pairs; fair (same actor/
  rollout/scenes, only the optimizer differs). PPO 0/5 collapse both regimes; REINFORCE 1/5 each.
- **== ANCHOR HONESTLY SCOPED — PASS.** edit_rate 0.0 reported openly; framed as a missing-direction-signal
  gap (R4–R5), not a PPO win or failure; additive (protected files byte-identical; R0 spy still passes; 794/0).

**Verdict: KEEP — R3 stands (PPO+critic genuinely active + stable; == anchor = direction-signal gap). The EV
single-run/robustness nuance is recorded; proceed to R4.**
