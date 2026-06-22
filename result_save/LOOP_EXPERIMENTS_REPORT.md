# Loop Experiments Report — Decentralized MARL for Urban V2X Topology Planning

**Scope:** the autonomous research loop of 2026-06-21/22 (research-log iterations 17–22). Each experiment
followed a strict one-variable protocol: locate bottleneck → single hypothesis → minimal implementation →
multi-config training + held-out eval + **multi-seed** + baselines → decide (keep / rollback). All results are
on the held-out **N = 24** scale-extrapolation set (training families are N ∈ {8, 12, 16}; N = 24 is out of
range). The deployed policy is **keep-best** (selected on a validation split, never on the held set).

Hard invariants honored throughout: fully decentralized execution **and** learning (no critic, no global state);
closed-form global PBFT consensus reliability as the objective (Poisson-binomial quorum tail, never Monte-Carlo);
τ = 0.9 never relaxed; a single principled reward (potential shaping + Lagrangian duals, no weighted bag).

---

## 0. Baseline — the trunk under test

The "dense" trunk: critic-free REINFORCE + RLOO (leave-one-out baseline), Gaussian gate-boundary exploration on
the feasibility-by-construction `local_mutual_assemble` decoder, dense potential reward `r = (c − τ) − λ_b·g_b`
(c = closed-form consensus reliability), Lagrangian dual ascent on the budget constraint, solvable-only training
mask. Cold-start: random-init actor, **no behavior-cloning warm-start, no SA/critic oracle**.

**Baseline metrics (dense β = 0, N = 24 held, 5 seeds, keep-best deployed):**

| metric | value (95% CI) | meaning |
|---|---|---|
| raw (all held scenes) | **0.692** [0.603, 0.782] | fraction of held scenes given a feasible topology |
| SA-teacher ceiling | 0.515 [0.473, 0.558] | the *centralized* oracle's feasible rate |
| margin (raw − ceiling) | **+0.177** [+0.104, +0.249] | how far the decentralized policy beats the oracle |
| conditional (solvable scenes) | **0.894** [0.784, 1.0] | of provably-solvable scenes, fraction reaching c ≥ 0.9 |
| mean energy (feasible) | 0.725 J | measured, not optimized in baseline |
| mean latency (feasible) | 0.029 s | measured; near-constant (topology-invariant) |
| mean edges (feasible) | 29.5 | — |

**Headline:** the oracle-free, cold-start, fully-decentralized policy beats the centralized SA oracle at an
out-of-range scale, with a 95% CI on the margin that **excludes 0** across 5 independent seeds. This baseline is
the reference every experiment below was tested against.

---

## 1. Experiment A — per-node consensus-decomposition reward (research-log iter 17)

**Hypothesis (one variable):** replace the single per-scene scalar potential `(c − τ)`, shared across all of a
scene's edge log-probs, with a **per-node** potential `φ_j = per_primary[j] − τ` (validator j's closed-form
consensus reliability as initiator), crediting each edge (u,v) by `adv_u + adv_v`. Goal: sharper credit
assignment to lift/stabilize the result, with no critic and no local proxy. (No paper in the 40-source review
decomposes a closed-form Poisson-binomial consensus tail per-agent.)

**Invariant safety:** the global closed-form consensus `c` is unchanged and still drives the constraint + eval;
`per_primary` is read from the *same* evaluation. For uniform initiators `Σ_j weights[j]·φ_j = c − τ` exactly
(pinned by a unit test to 1e-12), so the per-agent signal is an exact decomposition, not a local proxy.

**Design / iteration:**
- **v1** (per-node potential + per-node RLOO): a `per_primary` accessor bug (read from `.metrics` where it is
  absent — it is a top-level attribute) zeroed all advantages → training frozen at N = 24. Fixed.
- **v1-fixed**: unfroze but *under-learned* — per-node RLOO dilutes the coherent global signal into tiny,
  partially-cancelling per-node advantages (feasibility ~0.05 / VAL 0.05 at update 80 vs dense VAL 0.70).
- **v1.1**: keep the **coherent global** RLOO advantage `A_k`, only **modulate** each edge by a per-node
  bottleneck weight `w_j` (deficit `τ − per_primary[j]`, mean-1 normalized). Competitive single-seed → ran the
  full A/B.

**Key data — full 4-seed A/B (v1.1 vs dense, same seed/split pairs):**

| seed/split | dense margin | per-node v1.1 margin |
|---|---|---|
| 0 / 7 | +0.192 | +0.231 |
| 1 / 11 | +0.077 | +0.038 |
| 2 / 17 | +0.192 | +0.192 |
| 3 / 23 | +0.231 | +0.308 |
| **mean** | **+0.173** [+0.067, +0.279] | **+0.192** [+0.012, +0.372] |

**Paired delta (v1.1 − dense): +0.019, 95% CI [−0.060, +0.098] → no significant difference**, and v1.1 has
*higher* variance (sd 0.113 vs 0.067).

**Verdict: NULL — rolled back.** Per-node credit does not beat the dense scalar and increases variance (the
opposite of the stabilize hypothesis). The dense scalar's coherent global advantage already captures the
per-scene consensus signal; decomposing it per-node is redundant for this objective.

---

## 2. Experiment C — energy / reliability-energy Pareto (research-log iter 18–20)

**Hypothesis:** the dense reward already has a feasible-gated energy term `β·er`; sweep β to trace the
reliability-energy Pareto front (the DoD low-energy target), and test margin-gated energy (shed only when
`c ≥ τ + δ`) to seek a *free* energy win by removing redundant edges.

**Key data — single-seed β sweep (seed 0/split 7, keep-best deployed):**

| config | margin | conditional | energy (J) | ΔE vs β=0 |
|---|---|---|---|---|
| β = 0 | +0.192 | 0.933 | 0.754 | — |
| β = 0.1 | +0.231 | 0.933 | 0.759 | +0.6% (term too weak to bite) |
| β = 0.3 | +0.038 | 0.800 | 0.651 | **−13.8%** (steep feasibility cost; final-update policy collapsed, keep-best caught an earlier checkpoint) |
| β = 0.3, margin = 0.05 | +0.154 | 0.933 | 0.890 | +18% (gate rarely fires at N=24's boundary-feasible regime) |

**Key data — multi-seed confirmation (β = 0.3, 4 seeds, keep-best):**

| | margin | conditional | energy (J) |
|---|---|---|---|
| β = 0 | +0.173 [+0.067, +0.279] | 0.906 | 0.732 [0.614, 0.849] |
| β = 0.3 | +0.135 [−0.011, +0.280] | 0.873 | 0.696 [0.602, 0.790] |

**Paired energy change β=0.3 vs β=0: −4.5%, 95% CI [−15.5%, +6.5%] → includes 0, not significant**, and β=0.3
also loses feasibility (its margin CI now grazes 0).

**Verdict: NULL — rolled back the `--energy-margin` knob.** The single-seed −14% was an optimistic draw.
Reliability and energy are **fundamentally coupled** (energy ∝ link count / power ∝ consensus); there is no
free-lunch low-energy point at N = 24's boundary-feasible regime. The policy *can* trace a Pareto tradeoff
(deployable points β=0 high-reliability and β=0.3 low-energy), but no significant energy reduction exists at
fixed feasibility. (Latency, separately, is near-constant ~0.029 s — topology-invariant, not optimizable.)

---

## 3. Experiment B — live consensus dual (research-log iter 21–22)

**Hypothesis:** in dense mode the consensus dual `λ_c` is computed and ascended but **inert** — it never enters
the reward (consensus is handled entirely by the potential). Add a **sparse binary** consensus-violation cost
(`−λ_c` on infeasible samples; MACPO dense/sparse split) so `λ_c` becomes live: one potential `(c − τ)` + two
distinct Lagrangian duals (consensus `λ_c`, budget `λ_b`), still a single principled objective. Opt-in flag
`--live-consensus-dual`, default off (default path byte-unchanged; unit suite 413 pass).

**Key data — N = 24, keep-best deployed, paired vs dense on the same seed/split:**

| seed/split | dense margin | B (live-dual) margin |
|---|---|---|
| 0 / 7 | +0.192 | +0.231 |
| 1 / 11 | +0.077 | +0.154 |
| 2 / 17 | +0.192 | +0.192 |
| 3 / 23 | +0.231 | +0.308 |
| 4 / 29 | +0.192 | +0.154 |
| **mean (n=5)** | **+0.177** [+0.104, +0.249] | **+0.208** [+0.128, +0.288] |

- Paired delta at **n = 4**: +0.048, 95% CI [−0.011, +0.107] (borderline, B ≥ dense on 4/4).
- Paired delta at **n = 5**: **+0.031, 95% CI [−0.031, +0.093] → non-inferior** (the 5th seed regressed B; the
  n=4 edge was an optimistic few-seed draw).

**Verdict: NON-INFERIOR — kept as an opt-in ablation.** B is never worse on average but not significantly
better; it is **kept for its rigor value** (it makes "dual ascent on consensus + budget" literally true, closing
a documented honesty gap) at no metric cost. The dense Ng-Harada potential alone remains the principled default.
Bonus: the 5th seed *strengthened the dense headline* (margin CI tightened from n=4 [+0.067, +0.279] to n=5
[+0.104, +0.249]).

---

## 4. Summary & conclusions

| experiment | one-variable change | paired vs dense (multi-seed) | verdict |
|---|---|---|---|
| A — per-node credit | scalar → per-node potential | +0.019, CI [−0.060, +0.098], ↑variance | **NULL** (rolled back) |
| C — energy / Pareto | β sweep + margin-gate | energy −4.5%, CI [−15.5%, +6.5%] | **NULL** (rolled back) |
| B — live consensus dual | inert → live `λ_c` (sparse cost) | +0.031, CI [−0.031, +0.093] | **NON-INFERIOR** (kept as ablation + honesty fix) |

**Net finding:** the dense β = 0 trunk is a **robust optimum** — oracle-beating cold-start at out-of-range
N = 24, n = 5 margin **+0.177, 95% CI [+0.104, +0.249], 5/5 seeds** — that credit-assignment, energy, and
constrained-dual tweaks do **not** significantly beat.

**Honest novelty positioning** (consistent with the 2026-06-21 evidence review): the defensible contributions
are the **object** (closed-form PBFT consensus reliability used directly as the RL reward), the
**feasibility-by-construction decentralized decoder + matched gate-boundary explorer**, and the **empirical
result** (an oracle-free cold-start decentralized learner beating the centralized oracle out-of-range) — **not**
a new optimization *method*.

**Methodological law established by this loop:** single-/few-seed results in this problem are systematically
**optimistic** (Experiments A, C, and B each looked promising single-seed and washed out under multi-seed). Every
quantitative claim requires **≥ 5 independent seeds** with a reported confidence interval.

**Remaining work (not attempted this loop; needs owner-gated heavy compute):** multi-config domain randomization
(density × power × scale breadth) and larger-N out-of-range builds (N = 32/48) — both require expensive dataset
construction, not cheap reward/optimization tweaks.

---

*Generated by the autonomous research loop. Full per-iteration detail: `docs/URBAN_V2X_RESEARCH_LOG.md`
(iterations 17–22). Evidence audit: `docs/REVIEW_2026-06-21_EVIDENCE_PASS.md`.*
