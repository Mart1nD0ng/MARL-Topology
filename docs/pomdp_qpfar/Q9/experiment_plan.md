# Q9 — Full residual + PBRS (potential-based shaping with Φ = −D_quorum)

## Hypothesis (the single thing this stage establishes)
A potential-based shaping reward `F_t = λ_Φ(γΦ_{t+1} − Φ_t)` with `Φ = −D_quorum` and terminal `Φ_T = 0`
(Spec §9) is correctly implemented: it TELESCOPES (Σγ^t F_t = −λΦ_0 + λγ^T Φ_T = −λΦ_0) and therefore
PROVABLY PRESERVES the optimal policy (Ng–Harada–Russell), so it can only HELP learning, never change the
true objective. D_quorum (Q4-authorized) is the only place a proxy enters the reward, and ONLY as
optimum-preserving shaping; the final evaluation uses NO shaping (true C/E/L).

## Single change (one variable, delivered incrementally)
THIS stage: the PBRS primitive `src/marl_topology/training/potential_shaping.py` (Φ=−D_quorum, telescope,
terminal Φ=0) + its proofs, AND the residual policy stochastic sampler + per-edge log-prob (so the Q6
residual action space is RL-trainable), + training-only wiring of the PBRS shaping into the dynamic
reward (eval no shaping). The multi-seed deployable A/B (residual+PBRS vs anchor/baseline) is the Q12
campaign; here a single-seed pilot confirms the mechanism fires and is optimum-preserving.

## Design (code-grounded; Spec §9)
- `dquorum_potential(evaluator, topology, eta) = −eta · D_quorum(topology)` (training-only, via the
  bridge). Higher Φ = lower deficit = closer to feasibility.
- `episode_pbrs(potentials, gamma, lam_phi)` → `[F_0..F_{T-1}]`, `F_t = λ(γΦ_{t+1} − Φ_t)`, with the
  TERMINAL `Φ_T = 0` (Spec §9.1 η_T=0). The shaped reward `r'_t = r_t + F_t`.
- The residual sampler: per relevant edge, a Bernoulli flip (add a non-anchor edge / remove an anchor
  edge) from the actor's per-edge logit; the per-edge log-prob sums to the residual log-prob for PPO.
  Decoded by the Q6 `residual_decode` (full mode); zero residual = anchor.
- Eval/deploy: NO shaping (the reward F_t is added only in training); final metrics = true C/E/L.

## Failing-first tests (fail on HEAD)
- `test_pbrs_telescopes_with_terminal_zero` — `Σ γ^t F_t == λ(−Φ_0)` for arbitrary potentials, Φ_T=0.
- `test_pbrs_preserves_small_mdp_optimum` — on a tiny constructed MDP, value-iteration gives the SAME
  optimal policy under the shaped and unshaped reward (Ng–Harada).
- `test_pbrs_terminal_potential_zero` — the last `F_{T-1} = λ(γ·0 − Φ_{T-1})` (terminal Φ=0).
- `test_residual_sampler_logp_matches` — the residual sampler's summed per-edge log-prob equals the
  recomputed log-prob of the sampled residual (consistency for PPO).
- `test_zero_logit_residual_is_anchor` — a residual sampled at logit→ the inert region decodes to the anchor.

## Success criterion (Q9 passes iff)
1. Tests pass; PBRS telescopes + preserves the small-MDP optimum (the math guarantee holds).
2. A real-shard smoke runs the residual policy + PBRS shaping (training-only) on real `--dyn-data`,
   exits 0, records a `mechanism_activation` (Φ=−D_quorum, terminal Φ=0, eval-no-shaping), and the
   eval uses true C/E/L (no shaping).
3. A single-seed pilot A/B (residual+PBRS vs anchor) reports retention / feasibility / E/L honestly
   (no headline claim; multi-seed is Q12).

## Failure criterion
PBRS does not telescope / changes the small-MDP optimum (implementation bug); the eval uses shaping;
the residual sampler logp is inconsistent; D_quorum potential leaks the true CSI to the deployed actor.

## Out of scope
The multi-seed deployable headline (Q12); the local edge handshake (Q10); PNA in residual (Q11). Heed
the Q5 RL-instability + the Q7/Q8 central ceilings — a deployable residual win is NOT assumed.
