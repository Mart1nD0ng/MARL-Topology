# D10 — decision

**Result: KEEP (mechanism wired + verified + active; A/B deferred to D13).** Dynamic SCQ — closed-form
one-step counterfactual supervision of the Q critic — is now available in the dynamic arm, opt-in, with
its (non-budget-neutral) evaluator cost honestly reported.

## What changed (one variable: dynamic SCQ critic supervision, opt-in `--scq`, requires `--counterfactual`)
- `dynamic_scq_targets`: builds the DYNAMIC one-step SCQ targets
  `Δy_i = ΔR_i + γ(V(s_{t+1}) − V(s̃_{t+1}))` (Plan §12). The immediate `ΔR_i = r_t − r̃_t` is the EXACT
  evaluator difference per UNIQUE counterfactual (reusing the static `scq_counterfactual_targets`; this is
  the SCQ budget — NOT budget-neutral). The bootstrap term re-decodes the counterfactual into the NEXT
  frame's prev-topology, rebuilds the next obs (`scene.observation(t+1, cf_topo)`), and re-forwards the Q
  critic — a critic forward, FREE (no evaluator). At the terminal frame Δy reduces to ΔR. This one-step
  RETURN target is what makes it a DYNAMIC SCQ, distinct from the static single-step ΔR primitive.
- `dynamic_scq_loss`: `L_SCQ = mean[(Q(s_t,S_t) − Q(s_t,S̃_i,S_{-i})) − Δy_i]²`, recomputed grad-on each
  critic epoch (Δy detached); enters ONLY the critic loss.
- `episode_rollout`: stores `scq_ctx={per_agent, logits}` when `--scq`. `run_dynamic_training`: builds the
  SCQ targets ONCE per update (pays the evaluator budget once), adds `scq_coef·L_SCQ` to the critic loss
  each epoch, and records `scq`, `scq_m`, `scq_coef`, `scq_select`, `scq_budget_neutral=False`,
  `scq_evaluator_calls_per_update` (+ per-update `scq_loss`/`scq_evaluator_calls` in the history).
- Default off → no scq_ctx, no extra evaluator calls, critic loss unchanged (byte-identical).

## Tests (failing-first; fail on `3c9cdc4` with ImportError, pass after)
- `test_dynamic_scq_exact_delta_nonzero` (SCQ spends budget; Δy nonzero), `test_dynamic_scq_state_fork_isolation`
  (a fork toggles only one agent's incident edges), `test_dynamic_scq_cache_duplicate_topologies` (one
  evaluator call per unique cf per frame), `test_dynamic_scq_loss_enters_critic` (grad reaches the critic).
- Dynamic-RL **25/25**; unit suite **688/0**; contract **63/0**; `--dynamic --counterfactual --scq` smoke
  exit 0 (`scq_budget_neutral=False`, `scq_evaluator_calls_per_update=5`).

## Adversarial verification (focused single agent, 4 claims, PASS, no blockers)
1. **Dynamic, not static**: Δy = ΔR + γ(V(s_{t+1}) − V(s̃_{t+1})); bootstrap free; terminal reduces to ΔR.
2. **Budget honest**: `scq.counterfactual_calls` counts ONLY the ΔR evaluations (reused `r_actual`, one per
   unique cf); the bootstrap is a critic forward + obs rebuild (no evaluator); `scq_budget_neutral=False`
   + `scq_evaluator_calls_per_update` reported.
3. **SCQ enters critic only**: added to `v_loss`, never the actor; Δy detached, ΔQ grad-on; gradient reaches
   the critic.
4. **Byte-identity off + caching + no bypass**: scq=False → unchanged critic loss, no extra evals; duplicate
   cf topologies deduped (one eval per unique); the Q critic is training-only (`dynamic_eval` takes no
   critic). Full suite no regression.

## Scope / next
- D10 delivers the MECHANISM (correct, active, budget-reported). It does NOT claim SCQ improves Q fidelity
  or feasibility — consistent with the static Phase-9 pattern (SCQ verified-correct but no gain; ~2× budget,
  held Q fidelity unchanged) and the D8 finding (RL feasibility-learning is the binding limit). The
  SCQ-vs-no-SCQ A/B (held Q fidelity + per-seed/CI) is part of D13 — NOT claimed here.
- Next: **D11** (chance / CVaR / Pareto into the dynamic task — episode-level constraints/objectives).
