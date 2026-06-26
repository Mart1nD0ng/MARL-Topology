# D6 — decision

**Result: KEEP.** The warm-start is now decoder-aware (BCSP-subset likelihood), an annealed teacher-BC
anchor protects it during PPO, and an optional critic warm-start is available — all opt-in, with full
teacher-source/leakage/drift reporting. Adversarial verification PASS (no gaps).

## What changed (one variable: the warm-start/anchor mechanism, opt-in)
- `_bcsp_teacher_trajectory`: per-frame decoder-aware teacher — per-agent BCSP proposals `S_i^teacher` =
  the myopic teacher's incident edges at node i, CAPPED at the node budget `b_i` (so `|S_i| ≤ b_i` is in
  the BCSP support; `subset_logp` is −inf otherwise). The reconstructed teacher = the local mutual
  decode of those proposals (decoder-reachable); it threads as the teacher prev.
- `warmstart_actor_bcsp`: minimize `Σ_i −log π_i(S_i^teacher)` (the BCSP subset NLL) — decoder-aware
  imitation, replacing per-edge BCE. Opt-in `--dyn-warmstart-mode {bce,bcsp}` (default `bce` → byte-identical).
- `bcsp_teacher_anchor_loss` / `_teacher_subset_nll`: the same teacher NLL, reused as an annealed BC
  anchor `λ_BC(u)·L_BCSP` added to the PPO actor loss, `λ_BC(u)=λ_BC0·(1−u/updates)`. Opt-in
  `--dyn-bc-anchor λ` (default 0 → byte-identical).
- `warmstart_critic`: pretrain the critic `V(s_t)→G_t^teacher` (training-only). Opt-in
  `--dyn-critic-warmstart K` (default 0).
- **Reporting (Contract §10.3):** result/activation record teacher source (myopic-greedy over canonical
  variants), `teacher_uses_evaluator=true`, `teacher_uses_held=false`, `warmstart_alone_return`, and
  `post_rl_drift_held_return = held_return − warmstart_alone_return`.

## Tests (failing-first; fail on `fb01645` with ImportError, pass after)
- `test_decoder_aware_teacher_reconstructs_topology` (proposals mutual-decode to recon; budget-respected;
  recon ⊆ teacher), `test_bcsp_warmstart_increases_teacher_subset_logp` (more epochs → lower teacher NLL),
  `test_ppo_kl_anchor_limits_drift_from_teacher` (anchor keeps the actor closer to the teacher under a
  drift-inducing update), `test_critic_pretrain_tracks_teacher_return` (critic warm-start lowers MSE).
- Unit suite **675/0**; contract **63/0**; D6 smoke (`--dyn-warmstart-mode bcsp --dyn-bc-anchor 1.0
  --dyn-critic-warmstart 5`) **exit 0** (decoder-aware warm-start NLL 4.93, critic-return MSE 4.54,
  warm-start-alone return + post-RL drift + teacher fields recorded). Defaults (bce / anchor 0 / critic 0)
  byte-identical.

## Adversarial verification (focused single agent, 4 lenses, PASS, no gaps)
1. Decoder-aware correctness: budget-capped proposals, mutual-decode recon, BCSP-NLL (not BCE).
2. Teacher leakage: teacher from train only; held measurement-only (never feeds checkpoint); honest
   `uses_held=false` + drift reporting.
3. Anchor + byte-identity off: defaults preserve the legacy path; linear anneal; anchor pulls toward the
   warm-start; no existing dynamic test broke.
4. No bypass / honesty: no new evaluator path; deployed actor/decoder untouched; no over-claim (the
   warm-start-vs-RL effect remains a deferred pilot, not a result).

## Scope / next
- D6 delivers the MECHANISM (decoder-aware warm-start + anti-drift anchor + critic warm-start), genuinely
  active and recorded. The multi-arm A/B (warm-start-only / +PPO / +PPO+BC / +PPO+KL) and whether the
  anchor actually reduces post-RL drift at scale is a **pilot** to run with D8 — NOT claimed here (the
  tiny smoke showed drift 0.0, but that is smoke params, not a headline).
- Next: **D7** (fair deployable baselines: local link-threshold / top-budget / hysteresis / static-actor-
  per-frame; group the central evaluator-greedy references separately — Contract §10.1, gap #8).
