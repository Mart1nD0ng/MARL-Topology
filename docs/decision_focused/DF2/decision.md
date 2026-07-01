# DF2 — decision (goal 2 GATE) — v2, hardened by adversarial verification

**Disposition: HONEST NEGATIVE on conversion — with a rigorous, non-tautological mechanism.** Raising the shadow
decorrelation distance (the DF1 knob) does NOT make the stale-CSI drop convertible into recovered feasibility. The
`d_corr` knob is KEPT (opt-in, byte-identical, physically real, DF1-useful) but does not solve goal 2. This
REFINES the campaign's binding limit; the gate was pre-registered and fails on the evidence.

> **Provenance.** The v1 conclusion was adversarially verified (3-critic Workflow `wf_4242d978-612`): verdict
> **REVISE, conclusion_robust = TRUE** — the negative stands, but the evidence framing over-claimed. This v2 folds
> in every required caveat. The load-bearing changes: (a) the negative is anchored on a TIGHT point-identity, not
> the wide RGF CI; (b) the ρ actually achieved is scoped honestly (deployment ρ≈0; DF2b ρ≤0.42; high-ρ
> feasibility UNTESTED); (c) the geometry-vs-temporal attribution is now shown NON-tautologically via a
> pure-temporal arm.

## Result 1 — oracle re-gate (`oracle_regate_metrics.json`, 5 seeds, fast R8 mobility)
**Load-bearing evidence (tight):** `feas(realizable) == feas(stale)` to rounding at every `d_corr` (diffs
`{+0.000, −0.009, +0.000, +0.003, +0.000}`) WHILE `MSE(realizable) < MSE(echo)` with **non-overlapping** CIs
(e.g. `d_corr=10`: echo `[0.098,0.120]` vs real `[0.073,0.087]`). So there is a **real per-link prediction
improvement that produces ZERO feasibility movement** — the campaign's MSE⊥decisions signature, under the env
knob. Valid-not-inconclusive: `RGF(true_csi)=1`, headroom `Feas(true)−Feas(stale)` CI-positive 5/5 (the anchor
CAN use perfect CSI). 
**Caveat (verification hole 1):** the `RGF(realizable)` interval itself is too WIDE to independently carry the
null — its CIs (widths 0.40–1.07) contain both 0 AND the pre-registered +0.15 target at every `d_corr`. So the
negative rests on the point-identity + MSE-non-overlap, NOT on `RGF≈0`.
**Caveat (verification hole 5):** the pre-registered primary endpoint `RGF(realizable)−RGF(geometry_only)` is
VOID — `geometry_only` is degenerate (feas ~0.29 ≪ stale 0.62; distance uninformative at the saturated tx=20
operating point), which mechanically inflates that difference (JSON shows it positive, CI-excludes-0). We
substitute the stale echo as the baseline (correct), and label this a post-registration change.

## Result 2 — attribution probe, NON-tautological (`df2b_persistent_recovery.json` v2, idealized boundary links)
Nested feature-subset arms on persistent links where ρ sweeps: `echo` (stale only), **`temporal2`** (two stale
psucc lags, NO distance — the pure-temporal recoverer), `geometry` (echo + current distance), `full`. Across
`d_corr {10,25,50,100,200}` (measured `ρ ≈ {−0.00, 0.29, 0.36, 0.42, 0.41}`; 200 links × 5 field seeds):
- **PURE-TEMPORAL recovery is decision-irrelevant at every ρ.** `temporal2` side-accuracy advantage over echo =
  `{−0.001, +0.000, +0.005, +0.006, +0.003}` — ≤ 0.006 everywhere. Its MSE advantage is real but SHRINKS with ρ
  (Spearman −0.9). So two lags of stale psucc reduce MSE yet do NOT recover the decision — MSE⊥decisions, now for
  temporal recovery specifically and NON-tautologically (this arm has no distance). (The side-adv Spearman is
  +0.9 but the magnitude never exceeds 0.006 — a monotone but decision-negligible rise.)
- **The only non-trivial decision advantage comes from GEOMETRY, and only at LOW ρ.** `geometry` side-advantage =
  `{+0.044, +0.005, +0.003, +0.002, +0.002}` — largest at the lowest ρ, shrinking as ρ rises (Spearman −0.9).
- `MSE(echo)` falls `{0.27, 0.14, 0.067, 0.045, 0.034}` — higher ρ makes the echo accurate on its own.
- `flip_recovery` is reported with the floor stated (0 = copies the wrong echo call; ~0.5 = independent guess), so
  the higher-ρ values (~0.08–0.10) read as near-zero recovery, not "weak recovery"; trends use Spearman over ρ,
  not the earlier misleading endpoint diff (verification hole 4).

## Mechanism (the refined, rigorous finding)
**Raising temporal autocorrelation is SELF-DEFEATING for stale-CSI recovery, and pure temporal recovery is
decision-irrelevant regardless.** (i) The correlation that makes the channel predictable (high ρ) also makes the
stale echo accurate (`MSE(echo)` collapses), so the recoverable advantage shrinks as ρ rises. (ii) Even given the
ingredients (two stale lags), a pure-temporal predictor gives ≤0.01 decision-side advantage at any ρ — it lowers
MSE but not decision error. (iii) What little decision recovery exists comes from current GEOMETRY (leak-free
distance) and only when the channel changes fast (low ρ). (iv) At the faithful, bimodal operating point the
per-link advantage does not convert to graph feasibility at all (Result 1's point-identity).

## Scope (verification hole 2 — mandatory, no over-claim)
The autocorrelation was NOT actually raised in the conversion experiment: DF2's in-scene `ρ_mean ≈ 0` (negative)
at every `d_corr` — fast mobility + bimodal saturation washed the knob out. So "no conversion at every `d_corr`"
is measured at `ρ≈0`. The positive-ρ evidence lives ONLY in DF2b (boundary tx=−8, NLOSv off, τ=0.5, idealized
links, ρ up to ~0.42, plateauing below the ~0.9 that `v·dt/d_corr` predicts). The conclusion transports to the
saturated deployment **by mechanism, not by measurement**; the convertibility of genuinely high ρ (>0.42) on the
feasibility metric remains **UNTESTED**. This is stated, not hidden.

## Disposition & next
- **KEEP** the `d_corr` knob (opt-in, byte-identical; DF1 proved it raises per-link ρ). Default-off; does not beat
  the anchor.
- **Goal 2 = honest negative (mechanistically explained, non-tautological).** No downgrade: the attribution hole
  the verifier flagged was CLOSED with a pure-temporal arm (DF2b v2), not softened away.
- **Proceed to DF3 (goal 1)** — decision-focused prediction, the owner's headline goal, independent of goal 2.
  DF2's negative is scoped to MSE-trained predictors; whether a DECISION-focused objective beats MSE at the
  boundary is exactly what DF3 tests.
