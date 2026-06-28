# Q9 PART 2 — Residual + PBRS end-to-end training (the residual arm's trainer)

## Hypothesis (the single thing this stage tests)
A residual policy (Q6 action space, sampled by the Q9-PART1 sampler) trained end-to-end with the PBRS
shaping `F_t = λ(γΦ_{t+1} − Φ_t)`, `Φ = −D_quorum`, terminal Φ=0 can be trained without diverging, and —
evaluated WITHOUT shaping (true C/E/L) — its deployed residual topology stays near the anchor (high
retention) and does not fall below it. A deployable WIN is NOT assumed (bounded by the Q7 ~22% / Q8
urban-47% central ceilings; heed the Q5 RL-instability).

## Single change (one variable)
A self-contained residual+PBRS REINFORCE trainer `scripts/diagnostics/residual_pbrs_train.py` (the
residual arm's training entrypoint; the BCSP trunk uses a different action space, so the residual policy
gets its own trainer — folding it into the `--baseline` arms is deferred unless it shows a win). Training
uses the shaped reward; EVALUATION uses NO shaping (true closed-form C/E/L).

## Design (code-grounded; Q6 + Q9-PART1)
- Actor: `DynamicRecurrentActor` (memoryless) → per-edge logits. Residual logits `z = logits +
  residual_prior` with `residual_prior = −3` (the residual inductive bias: a fresh policy starts AT the
  anchor — σ(z)≈0.05, few flips — and learns small edits).
- Rollout (training): per frame compute the anchor `local_hysteresis(obs)`; `sample_residual(z, anchor)`
  → topology + log-prob; reward `r_t = reward_of(true topo)`; PBRS `F_t` from `dquorum_potential` (the
  bridge, training-only); shaped reward `s_t = r_t + F_t`.
- REINFORCE: shaped return-to-go `G_t = Σ γ^{k−t} s_k`; advantage `A_t = G_t − baseline` (running mean);
  loss `−Σ_t logp_t · A_t`. Critic-free (the trunk's REINFORCE-EMA lineage).
- Eval (NO shaping): MAP residual decode `residual_decode(z, add/remove thresholds at 0)` (deterministic,
  deployed); report true feasibility (C ≥ τ) / energy / latency / RETENTION (anchor edges kept) /
  switches / mutual acceptance, vs the anchor's own metrics. PBRS appears NOWHERE in eval.

## Controlled variables
Data random + urban; current CSI (one variable = residual+PBRS); add/full residual mode; same scenes/
seeds; the anchor (deployable) and the residual policy both deployed-style at eval.

## Failing-first tests (fail on HEAD)
- `test_eval_uses_no_shaping` — the eval reward path equals `reward_of` with NO `F_t` term (the PBRS
  shaping is added only in training).
- `test_residual_rollout_retains_anchor_in_add_mode` — under add mode the sampled topology ⊇ the anchor.
- `test_pbrs_shaping_added_only_in_training` — the training shaped reward = `r_t + F_t`; the eval = `r_t`.
- `test_activation_records_pbrs_and_eval_no_shaping` — the activation dict records Φ=−D_quorum, terminal
  Φ=0, eval_uses_shaping=False, residual mode.

## Success criterion (Q9 PART 2 passes iff)
1. Tests pass; a real-shard smoke trains the residual+PBRS policy (exits 0, activation recorded).
2. Training does NOT diverge (the loss/return is finite; no NaN — the Q5 failure mode is avoided).
3. A single-seed pilot A/B (residual+PBRS vs anchor) is reported honestly: retention, feasibility, E/L —
   whether the residual policy matches/beats/loses to the anchor, stated as-is (no headline; multi-seed
   is Q12).

## Failure criterion / honest negative
The residual+PBRS training diverges (Q5 mode) OR the residual policy falls well below the anchor on
feasibility while retention is low (it broke the anchor) → honest negative; the residual lever does not
help the deployable policy at this scale.

## Out of scope
Multi-seed headline (Q12); local edge handshake (Q10); PNA in the residual frame (Q11); deep wiring into
the BCSP `--baseline` trunk (deferred unless the residual arm shows a win).
