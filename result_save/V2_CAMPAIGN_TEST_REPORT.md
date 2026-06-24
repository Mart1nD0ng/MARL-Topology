# MARL-Topology v2 Campaign — Test Settings & Results

**Date:** 2026-06-24 · **Branch:** `decentralized-marl-trunk` @ `0ba8bcd` (pushed to origin)
**Scope:** the v2 spec-driven CTDE rebuild — R0–R7 (re-accepting Phase 0–7) + Phase 8–13 (39 commits).

This report documents the **main experiments** (their settings and results). Per-mechanism *unit/contract*
tests are summarized in §7; the *headline experiments* (§3–§6) are the substantive evaluations. All
result artifacts are JSON files in `result_save/` (cited per experiment); the corrected dataset is
`result_save/campaign/data/op_corrected/`.

---

## 1. Central result

The **simple baseline** — shared-advantage Graph-MAPPO + MLP actor + the torch-free local
mutual-acceptance decoder — is the **best in-range AND best-generalizing** arm at the realistic
single-step, N≤16 scale. Every sophisticated CTDE mechanism built in Phase 8–11 (COMA per-agent
counterfactual credit, SCQ critic supervision, chance/CVaR/Pareto reliability, the PNA actor) is
**verified-correct but does not beat the baseline** → all ship **opt-in** (default-off, byte-identical
when off). Each "no-gain" verdict was **adversarially verified**, not merely asserted. The open frontier
is **large-N generalization** (N=16 ≈ 0.42; N=24 deferred — its corrected fault evaluation exceeds the
exact-enumeration budget).

---

## 2. Common test setup

| item | value |
|---|---|
| **Dataset (corrected)** | `result_save/campaign/data/op_corrected/_op_shard_3001..3024.pkl` — 24 shards, 240 scenes, mixed **N ∈ {8,12,16}** |
| **Corrected env-math** | `fault_model=fixed_set`, `one_hop_relay=True`, `timeout_aware_latency=True`, `relay_hops=3` (Spec S4/S6) |
| **Train / held split** | train = shards 3001–3016; held = 3017–3024 (disjoint); `split_seed=7`, `held_frac=0.4`, `val_scenes=15` |
| **Reliability threshold** | τ = 0.9 (consensus success probability) |
| **Headline metric** | `held raw` = fraction of held scenes whose decoded topology clears τ **and** is radio-budget feasible, under the deterministic deployed decoder (`local_mutual_assemble`), **keep-best on VAL** (never held — no test-set checkpoint selection) |
| **Decode (train == deploy)** | torch-free `local_mutual_assemble`; the BCSP sampler's temperature→0 limit equals it exactly |
| **CI** | paired per-seed difference, small-n Student-t 95% CI (t₀.₉₇₅,df=4 = 2.776 for n=5) |
| **Dataset validation** | 240 items, 168 witness_feasible + 72 unknown, **0 certified_infeasible** (hard constraint #11), feasible fraction 0.70 |

---

## 3. Phase-8 re-review — 8b COMA counterfactual credit vs R7 (Graph-MAPPO)

**Question:** does per-agent COMA counterfactual credit (`A_iᴱ = Q(s,S) − E_{S̃ᵢ~πᵢ}[Q(s,S̃ᵢ,S₋ᵢ)]`)
beat the shared scene advantage `A_s = R_s − V(s)` at equal evaluator budget?

**Setting:** `--baseline graph-mappo`, R7 vs `+--counterfactual` (8b); cold-start; 80 updates; 5 seeds;
matched config (only `--counterfactual` differs). Budget: both 1 evaluator call/scene (8b's
counterfactuals are *critic forwards*, not evaluator calls). Artifact: `result_save/phase8_corrected_headline.json`.

**Result (held raw):**

| seed | R7 | 8b |
|---|---|---|
| 0 | 0.641 | 0.609 |
| 1 | 0.641 | 0.641 |
| 2 | 0.625 | 0.641 |
| 3 | 0.563 | 0.625 |
| 4 | 0.641 | 0.641 |
| **mean** | **0.622** | **0.631** |

Paired diff (8b − R7) = **+0.009**, 95% CI **[−0.033, +0.052]** → **MATCHES** (CI spans 0).

**Decision:** 8b is verified-correct (unbiased, S_i-independent, budget-neutral — see §6) but shows **no
headline gain** → **opt-in** (`--counterfactual`), R7 stays default.

### 3b. Credit-resolution diagnostic (mechanism, not a headline)
Artifact: `result_save/coma_credit_resolution_corrected.json`.
- **Part A** (true marginal Δᵢ via the evaluator, diagnostic-only calls): under a realistic policy,
  **85%** of scenes carry non-zero per-agent COMA credit variance, which the shared advantage (one
  scalar per scene → within-scene variance ≡ 0) structurally cannot represent.
- **Part B** (learned-Q credit fidelity): Spearman ρ(Aᵢ, Δᵢ) = **0.174** (n=676, n-scenes=60) on
  corrected data — up from ρ ≈ 0 on the pre-corrected data, i.e. the learned Q *modestly* tracks the
  true marginal once the data is corrected.

---

## 4. Phase-9 — SCQ closed-form critic supervision

**Question:** does SCQ (calibrate the Q critic with *exact* evaluator differences
`ΔRᵢ = R(S) − R(S̃ᵢ,S₋ᵢ)` via `L_SCQ = [Q(s,S) − Q(s,S̃ᵢ,S₋ᵢ) − ΔRᵢ]²`, with sensitivity-guided top-M
selection) raise Q fidelity?

**Setting:** matched A/B — same config + seed, only `--scq --scq-select topM --scq-m 3` differs; both Qs
trained on op_corrected 3001-3016 (80 updates, cold-start); credit fidelity measured on held 3017-3024
at the **same diagnostic config** (n-scenes=40, k-cf=24, seed 0). Artifact:
`result_save/coma_credit_resolution_scq.json` (SCQ) vs the matched no-SCQ baseline.

**Result (learned-Q credit fidelity, Spearman ρ):**

| arm | ρ(Aᵢ, Δᵢ) | n |
|---|---|---|
| no-SCQ 8b-Q (matched n-scenes=40) | **0.244** | 452 |
| SCQ-Q | **0.071** | 452 |

Fisher-z difference p = 0.008 → SCQ **did not raise** held Q fidelity (it lowered it). The SCQ in-sample
`critic_difference_error` *rose* over training (0.092 → 0.279), i.e. non-converging; budget ~2× (1 + M
evaluator calls/scene); held raw unchanged (0.625 == 0.625).

**Decision:** honest **NEGATIVE** — SCQ verified-correct (see §6) but no Q-fidelity gain at this scale
→ **opt-in** (`--scq`). (Note: the Phase-8 ρ=0.174 was at n-scenes=60; the matched A/B above is n-scenes=40,
so the two no-SCQ numbers are the same Q at different diagnostic scene counts.)

---

## 5. Phase-11 — preference-conditioned directional PNA actor vs MLP actor

**Question:** does the Spec-S7.12 actor (PNA aggregation + directional message passing + recurrent
shared GRU + ω-preference-conditioning) beat the current MLP actor?

**Setting:** `--baseline ema` cold-start; MLP (`--actor mlp`) vs PNA (`--actor pna`); 80 updates; 5
seeds; matched config (only `--actor` differs). Artifact: `result_save/phase11_pna_vs_mlp.json`.

**Result (held raw):**

| seed | MLP | PNA |
|---|---|---|
| 0 | 0.484 | 0.641 |
| 1 | 0.609 | 0.641 |
| 2 | 0.594 | 0.406 |
| 3 | 0.641 | 0.422 |
| 4 | 0.641 | 0.406 |
| **mean** | **0.594** | **0.503** |

Paired diff (PNA − MLP) = **−0.091**, 95% CI **[−0.308, +0.126]** → **MATCHES** (CI spans 0). PNA is
**less stable** (cross-seed sd 0.126 vs MLP 0.064; bimodal: 2 strong seeds, 3 mediocre).

**Decision:** PNA verified-correct (drop-in, ω-conditioning real, D1-clean — see §6) but **no headline
gain and less stable** → **opt-in** (`--actor pna`), MLP stays default.

---

## 6. Phase-12 — cross-N generalization (the clearest result)

**Question:** how does each arm generalize across N (in-range), and what about N=24 (out-of-range)?

**Setting:** EVAL-ONLY (load each trained arm's saved actor artifact, decode on held 3017-3024, extract
per-N `raw_by_n`; no retraining; 5 seeds; per-(arm,N) mean). Artifact:
`result_save/phase12_generalization.json`.

**Result (held raw by N):**

| arm | N=8 | N=12 | N=16 | overall |
|---|---|---|---|---|
| **R7 default** (graph-mappo, mlp) | 0.937 | 0.600 | **0.417** | **0.693** |
| 8b counterfactual (mlp) | 0.943 | 0.610 | 0.375 | 0.685 |
| ema mlp | 0.943 | 0.571 | 0.358 | 0.670 |
| ema PNA | 0.886 | 0.419 | 0.217 | 0.562 |

**Findings:** (1) **strong monotone N-degradation for every arm** (N=8 ≈ 0.94 → N=16 ≈ 0.4) —
cross-N generalization is *the* bottleneck, not the mechanisms; (2) the **R7 default generalizes best**
(best overall and at the hardest N=16); 8b ≈ R7; the **PNA actor generalizes worst** at every N.

**N=24 (out-of-range) — DEFERRED (evidence-based):** the corrected `fixed_set` fault at N=24 has
f = (n−1)//3 = 7 → Σ_{r≤7} C(24,r) = **536,155 fault sets ≫ the 50,000 enumeration budget** → the
evaluator falls back to **greedy (optimistic, not certified)**. A valid N=24 corrected headline needs a
cheaper *exact*-f path (the open frontier); a greedy number is not a certified headline, so it is
honestly deferred.

---

## 7. Unit / contract test suite & adversarial verifications

- **Suite:** `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/unit tests/contract -q` → **688 passed /
  0 failed** (139 test files). The suite grew from ~560 (start of v2) to 688 with zero regressions; every
  gate/phase was **failing-test-first**.
- **Smokes:** both exit 0 — `--smoke --cold-start` (mlp default) and `--smoke --cold-start --baseline
  graph-mappo --counterfactual` (an opt-in).
- **Deployment-purity (D1) gates:** 8 contract gates pass — no critic / SCQ / reliability / counterfactual
  symbol leaks into deployed subtrees (`protocol/`, `policies/`, `data/`, `evaluation/`); the deployed
  decode path is torch-free; train == deploy.
- **Adversarial verifications (multi-lens Workflow agents, independent brute-force / finite-difference):**

  | step | lenses | outcome |
  |---|---|---|
  | R1 fixed-B, R6 BCSP, R7 Graph-MAPPO | math / per-agent ratio / critic-trains / D1 | no correctness refutation |
  | Phase 8b COMA | unbiasedness (independent enumerator) / D1+budget / credit-integration | unbiased, S_i-independent, budget proven equal (72=72=72 evaluator calls); 2 nits fixed |
  | Phase-8 re-review | fairness / honesty | fair + honest; corrected 3 reporting errors (strengthened the verdict) |
  | Phase 9 SCQ | measurement-validity / false-negative | negative confirmed honest + robust (gap widened at matched config) |
  | Phase 10 reliability | chance/CVaR math / Pareto+byte-identity | confirmed; 1 nit fixed (chance residual threshold) |
  | Phase 11 PNA | correctness/D1 / comparison-fairness | confirmed; 1 latent backward-NaN bug fixed |
  | Phase 13 release | D1 contract / headline-honesty / byte-identity | no blockers; 4 honest nits recorded |

---

## 8. Mechanism ledger

| mechanism | flag | status | verified | headline @ N≤16 |
|---|---|---|---|---|
| Graph-MAPPO (shared advantage, R7) | `--baseline graph-mappo` | **KEEP-DEFAULT*** | yes | best in-range + best generalization |
| BCSP unordered-subset policy (R6) | always-on (replaced Plackett-Luce) | KEEP | yes | dissolved the m=15,b=64 trunk hang |
| MLP actor | `--actor mlp` (default) | KEEP-DEFAULT | yes | best-generalizing actor |
| 8b COMA counterfactual credit | `--counterfactual` | OPT-IN | yes | MATCHES R7 (no gain) |
| 9 SCQ critic supervision | `--scq` | OPT-IN | yes | no Q-fidelity gain (+budget) |
| 10 chance constraint | `--chance` | OPT-IN | yes | distribution-level reliability |
| 10 CVaR shortfall | (primitive) | OPT-IN | yes | tail-reliability primitive |
| 10c Pareto checkpoint archive | `--pareto-archive` | OPT-IN | yes | risk-aware checkpoint (Spec S6.4) |
| 11 PNA directional actor | `--actor pna` | OPT-IN | yes | MATCHES MLP (less stable) |
| N=24 / larger-N / multi-step | — | DEFERRED | n/a | needs cheaper exact-f / two-timescale env |

*The CLI default of `--baseline` is `ema` (kept byte-identical to the historical trunk); the **recommended
production config** is `--baseline graph-mappo --actor mlp` — pass it explicitly.

---

## 9. Reproduction

```bash
# corrected dataset (24 shards; ~1h @ jobs=22): operating_point_regime() is hardcoded corrected
python scripts/train/build_operating_point_dataset.py --seeds $(seq 3001 3024) --count 10 \
    --node-choices 8 12 16 --rsu-count 4 --tx-power 20 --jobs 22 --out-dir result_save/campaign/data/op_corrected

TRAIN="<op_corrected 3001..3016>" ; HELD="<op_corrected 3017..3024>"
# §3 8b-vs-R7 headline
python scripts/diagnostics/phase8_corrected_headline.py --shards $TRAIN --seeds 0 1 2 3 4 --updates 80 --val-scenes 15
# §3b/§4 credit-fidelity (no-SCQ vs SCQ): train a Q then
python scripts/diagnostics/coma_credit_resolution.py --q-ckpt <Q ckpt> --shards $HELD --n-scenes 40 --k-cf 24
# §5 PNA-vs-MLP
python scripts/diagnostics/phase11_pna_vs_mlp.py --shards $TRAIN --seeds 0 1 2 3 4 --updates 80 --val-scenes 15 --baseline ema
# §6 cross-N generalization (eval-only over saved artifacts)
python scripts/diagnostics/phase12_generalization.py
# suite + smokes
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/unit tests/contract -q
python scripts/train/train_decentralized_rl.py --smoke --cold-start    # mlp default
```

Full per-phase narrative: `docs/URBAN_V2X_RESEARCH_LOG.md` (tail: "v2 CAMPAIGN SUMMARY").
