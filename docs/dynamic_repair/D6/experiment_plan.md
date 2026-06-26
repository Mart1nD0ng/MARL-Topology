# D6 — experiment_plan

**Stage:** D6 — protect the warm-start: decoder-aware BCSP-subset imitation + an anti-drift anchor in
PPO (Plan §8; gap #9).

**Hypothesis:** The warm-start is a per-edge BCE toward the teacher's 0/1 edge indicator
(`warmstart_actor`), which is NOT decoder-aware (it ignores the BCSP budget cap + mutual-acceptance
semantics the deployed decoder uses), and PPO then runs FREE afterwards (no anchor), so it drifts off
the feasible warm-start (report §6.4: "RL DEGRADES the imitation warm-start"). Replacing the warm-start
with the decoder-aware BCSP-subset likelihood and adding an annealed teacher anchor to the PPO loss
should (a) align the warm-start with the deployed decoder and (b) limit post-RL drift.

**Single change (one variable = the warm-start/anchor mechanism), opt-in:**
- `_bcsp_teacher_trajectory`: per frame, the per-agent BCSP proposal subsets `S_i^teacher` = the myopic
  teacher's incident edges at node i, CAPPED at the node budget `b_i` (so `|S_i| ≤ b_i` is in the BCSP
  support — `subset_logp` is `-inf` otherwise). The "reconstructed teacher" topology = the local mutual
  decode of those capped proposals (decoder-reachable by construction); it threads as the teacher prev.
- `warmstart_actor_bcsp`: minimize `Σ_i −log π_i(S_i^teacher)` (the BCSP subset NLL) — the
  decoder-aware imitation (replaces per-edge BCE). Opt-in `--dyn-warmstart-mode {bce,bcsp}` (default
  `bce` → byte-identical).
- `bcsp_teacher_anchor_loss`: the same teacher NLL, reused as an annealed BC anchor added to the PPO
  actor loss: `L = L_PPO − c_ent·H + λ_BC(u)·L_BCSP`, `λ_BC(u) = λ_BC0·(1 − u/updates)` (linear anneal
  to 0). Opt-in `--dyn-bc-anchor λ` (default 0 → byte-identical).
- `warmstart_critic` (optional): pretrain the critic `V(s_t) → G_t^teacher` (the teacher's discounted
  return). Opt-in `--dyn-critic-warmstart K` (default 0).
- **Reporting (Contract §10.3):** the result records teacher source (`myopic-greedy over canonical
  variants, uses the candidate evaluator — training-only`), `teacher_uses_evaluator=true`,
  `teacher_uses_held=false`, warm-start-alone held return (already measure warm-start-alone feasibility;
  add the RETURN), and post-RL drift (`held_return − warmstart_alone_return`).

**Controlled variables:** reward/objective (D2), split (D3), evaluator (D4), observation (D5),
architecture, decoder, sampler. Only the warm-start/anchor mechanism changes, and only when its flags
are set. Teacher stays training-only; the deployed actor unchanged.

**Required failing tests (fail on `fb01645`, pass after):**
- `test_bcsp_warmstart_increases_teacher_subset_logp` — more `warmstart_actor_bcsp` epochs → higher
  teacher subset logp (lower NLL): the decoder-aware warm-start converges.
- `test_decoder_aware_teacher_reconstructs_topology` — for a budget-respecting teacher, the local mutual
  decode of `S_i^teacher` reconstructs the teacher topology exactly.
- `test_ppo_kl_anchor_limits_drift_from_teacher` — under a drift-inducing update, the BC anchor (λ>0)
  keeps the teacher subset logp higher than no anchor (λ=0): the anchor limits drift.
- `test_critic_pretrain_tracks_teacher_return` — `warmstart_critic` reduces the critic's MSE to the
  teacher returns (it tracks `G_t^teacher`).

**Expected activation:** `mechanism_activation.json` records `warmstart_mode`, `bc_anchor_lambda`,
`critic_warmstart_epochs`, and the teacher-source / leakage fields.

**Success criterion:** the 4 tests pass; full unit + contract suites green; a `--dynamic
--dyn-warmstart-mode bcsp --dyn-bc-anchor 1.0` smoke exits 0 with the activation fields and reports
warm-start-alone return + post-RL drift; defaults (bce / anchor 0) byte-identical.

**Failure criterion:** if the decoder-aware warm-start + anchor does NOT reduce post-RL drift vs the BCE
baseline in a pilot, report it honestly (the anchor mechanism is correct but may not help at this
scale) — do not over-claim; the multi-arm A/B (warm-start-only / +PPO / +BC / +KL) is the D6 pilot.
