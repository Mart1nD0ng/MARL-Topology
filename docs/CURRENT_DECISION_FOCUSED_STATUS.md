# CURRENT — Decision-Focused CSI Recovery (DF) campaign

**Status: DF0 IN PROGRESS (framing + research + validation design).** No large experiments yet
(Contract v4: `experiment_plan` + validation design before any campaign run).

- **Base HEAD:** `1bf773f` (Temporal-Recovery T7 close-out — CAMPAIGN COMPLETE, pushed to `origin/decentralized-marl-trunk`).
- **Branch:** `decentralized-marl-trunk`. **Contract:** v4 (Claim-Path evidence regime).
- **Successor to:** the Temporal-Recovery campaign (T0–T7). This campaign attacks the *binding limit* that
  campaign diagnosed, on the two axes the owner named.

---

## 1. The affirmed diagnosis (owner-confirmed, the starting point)

The Temporal-Recovery campaign closed as an **honest negative with a refined binding limit**. The owner
affirmed the three-part diagnosis, which we now treat as established (not to be re-litigated):

1. **The "reversal headroom" is real.** Stale (delay-1) CSI drops the deployable `local_hysteresis` anchor a
   lot (urban ≈ 0.815 → 0.650). The T1 oracle showed that injecting **true current** psucc into the same anchor
   *converts* that headroom back (+0.165). The current channel is also **partially recoverable** from leak-free
   features (T6: stale adds recoverable R² beyond geometry).
2. **The core mathematical reason for the failure: MSE / R² ≠ topology-decision precision.** Predicting that the
   *average* channel got better does not help the *decision-critical edge* decisions. A predictor can win on MSE
   and still lose at the anchor's keep/add thresholds.
3. **Direction learned, magnitude not.** T3 found the *sign* of the stale→current change is learnable
   (`dir_acc ≈ 0.92`) but the *magnitude* is not — the model is qualitatively right, quantitatively wrong, so it
   cannot make the boundary-critical decision.

**Binding limit (Temporal-Recovery, refined):** deployable **precision at the anchor's decision boundary** —
realizable predictions beat the stale echo *on average* but not at the decision-critical edges near the
keep/add thresholds. This campaign attacks that limit head-on.

---

## 2. The three goals (owner, this campaign) — NO downgrade permitted

> 严格遵守开发契约且不得对目标降级逃避困难问题。仔细思考前两个目标在开发过程中应该如何验证，确保验证方案的正确性才能得到有效的答案。

**Goal 1 — Decision-focused CSI prediction.** Optimizing MSE is ineffective. Solve *accurate, decision-relevant*
per-edge CSI prediction — align the predictor with the downstream anchor decision, not average error. The
concrete method requires a thorough survey (SPO+/decision-focused learning/boundary-weighted/threshold
classification) — **in flight (DF0 research).**

**Goal 2 — Env with real temporal correlation in the decision-critical component.** Give the production channel's
decision-critical component (`psucc`) genuine, physically-real, *stronger* temporal autocorrelation, so that a
nowcaster has recoverable structure to exploit. (This is the "task-1 env modification" flagged as the open
frontier at the T-campaign close-out.)

**Goal 3 — Activation: is leaky-tanh best? Why not softmax?** Re-examine the activation choice; answer the
softmax question precisely (preliminary answer in §6, to be confirmed empirically at DF4).

**The owner's explicit demand:** for goals 1 and 2, the **validation design must be provably correct** — an
invalid validation gives an invalid answer. §5 is therefore the crux of DF0 and is being adversarially
red-teamed before any experiment runs.

---

## 3. Gap from HEAD (`1bf773f`) — what exists vs what is missing

### Goal 2 (env temporal correlation) — the physics is already ON; the *correlation is too fast*, and the knob is hardcoded
- **`channel/model.py` has a standards-based, temporally-correlated shadow-fading field** —
  `_shadow_field_value(seed, x, y)`: a unit-variance lattice Gaussian field, cell size =
  `SHADOW_DECORRELATION_DISTANCE_M` (10 m, per 3GPP TR 36.885 urban), bilinearly interpolated from seeded corner
  normals, **keyed by a HIDDEN per-scene seed and by POSITION only**. A moving vehicle therefore sees
  **temporally-correlated shadowing** across trajectory frames (Gudmundson-type). Gated by `shadowing_37885`
  (requires `path_loss_model = v2x_37885`).
- **VERIFIED: it is already ON in the campaign channel.** `operating_point_regime()`
  ([build_operating_point_dataset.py:57](../scripts/train/build_operating_point_dataset.py)) sets
  `path_loss_model="v2x_37885", shadowing_37885=True, nlosv_37885=True`, and `run_dynamic_training` builds its
  regime from exactly that ([dynamic_rl.py:562](../src/marl_topology/training/dynamic_rl.py)). The
  decision-critical psucc *does* carry a stochastic shadow component — the earlier "deterministic FSPL" reading
  was wrong.
- **The real reason T6 found weak recoverability: the correlation is TOO FAST to nowcast.** With `dt_s = 1.0`
  ([dynamic_frames.py:106](../src/marl_topology/training/dynamic_frames.py)) and urban speeds ~5–15 m/s the
  vehicle moves 5–15 m per frame — **≳ the 10 m decorrelation cell** — so the frame-to-frame shadow
  autocorrelation is `ρ_frame ≈ exp(−v·dt / d_corr) ≈ 0.37`: nearly i.i.d. Stale psucc's shadow component barely
  predicts the current one, leaving mostly the (geometry-recoverable) LOS/distance part. **The decorrelation
  distance is a hardcoded module constant** (`SHADOW_DECORRELATION_DISTANCE_M`), not a config field.
- **Missing (the modification):** (a) **expose the decorrelation distance as a config-threaded, swept knob** —
  the physical analogue of T6's synthetic `rho` sweep, but in the real standards-based channel; the governing
  ratio is `v·dt/d_corr` (finer `dt` or larger `d_corr` both raise `ρ`); (b) the **leak-free / fair-baseline /
  controlled-knob / right-metric validation** proving the modification is honest (§5). Enabling shadowing is NOT
  the diff and must not be re-reported as new.

### Goal 1 (decision-focused prediction) — the objective is the gap
- `training/csi_belief.py` has `belief_target_logits` (logit of true current psucc), `belief_correction_target`
  (stale→current delta), `directional_accuracy`, and NLL/uncertainty heads. **All are trained with MSE / NLL.**
- **Missing:** a **decision-aligned objective** whose gradient sharpens precision *at the anchor decision
  boundary* — plus a validation that scores on the DEPLOYED decision (feasibility), not MSE.

### Goal 3 (activation) — leaky-tanh present
- `models/belief_residual_actor.py`: residual head = `z_max*tanh(raw/z_max) + leak*raw` (T2). Sound for the
  *per-edge* logit. The softmax question is a *mechanism* question (per-edge vs edit-selection) — §6.

---

## 4. Campaign structure (DF0–DF6) — one variable per stage

| Stage | Goal | What changes (single variable) | Gate / disposition |
|------|------|-------------------------------|--------------------|
| **DF0** | framing | research + validation design + status | this doc; no experiment |
| **DF1** | 2 | enable+parametrize temporally-correlated shadowing in dynamic channel | leak-free + byte-identical-off + knob monotonicity PROVEN |
| **DF2** | 2 | oracle re-gate across decorrelation sweep | GATE: headroom becomes *convertible*? |
| **DF3** | 1 | decision-focused objective vs MSE vs echo | feasibility (not MSE); fraction of oracle gap closed |
| **DF4** | 3 | activation A/B (leaky-tanh vs edit-selection softmax) + integration | feasibility A/B |
| **DF5** | — | 5-seed campaign on winning config | per-seed + CI |
| **DF6** | — | docs close-out + 中文 report | — |

Ordering rationale: **change one variable at a time** (Contract). DF1/DF2 isolate the *env* change (does
realistic temporal autocorrelation create *convertible* headroom?); DF3 isolates the *objective* change (does
decision-focus convert what MSE could not?) — run on BOTH the original and modified channel so the two effects
are separately attributable. A negative at any gate is a valid, scoped result — **no downgrade to force a
positive.**

---

## 5. Validation-design correctness (THE CRUX — adversarially red-teamed, hardened)

> The finalized, test-backed design is `docs/decision_focused/DF0/validation_design.md` **(v2)**. It was attacked
> by two independent adversarial critics (both verified against code); **both returned REVISE and found real
> defects that would have produced invalid answers.** v2 folds in every fix. The two *blocking* controls the
> red-team added — read v2 for the full set:
> - **Goal 2 — MARGINAL-INVARIANCE gate (the most dangerous hole):** the `d_corr` sweep must move ONLY the
>   autocorrelation `ρ`, with the psucc **marginal distribution held fixed** (KS-tested; field renormalized if
>   not). Otherwise a "recovery converts" result is a *marginal-shift / difficulty* artifact. Plus the
>   nested-subset unification: `geometry_only` = best predictor from distance alone, so the leak test becomes
>   **incremental** R² over distance and the primary endpoint `RGF(recovered) − RGF(geometry_only)` is well-posed;
>   RGF is a **pooled-mean ratio bootstrapped over scenes**, never a per-frame fraction.
> - **Goal 1 — RECALIBRATION CONTROL arm:** BWAR's asymmetric hinge can win by a *global bias shift*, not boundary
>   precision, so the headline is `Feas(BWAR) − Feas(recalibrated-MSE)` (a 2-param monotone shift of the MSE
>   predictor). The aleatoric-negative is gated on the marginal-slot/mutual audit + a val hyperparameter sweep.
>
> The principles below are the v1 sketch, superseded in detail by v2.

### Goal 2 (env modification) — five ways it could be WRONG, and the guard for each
1. **Leak-free (the #1 risk).** Enabling shadowing must NOT hand the actor the *current* shadow value. The shadow
   field is position + hidden-seed keyed; the actor's leak-free features are current *distance*/velocity, not
   absolute endpoint positions and not the field seed → the actor **cannot** reconstruct current shadowing; it
   must nowcast it from stale. **Guard:** a load-bearing test that FAILS if any current-shadowing-derived value
   reaches the actor observation tensor; the evaluator stays on the true current channel. Watch subtle leaks
   (current-distance → deterministic-shadow mapping; actor/evaluator sharing a tensor).
2. **Fair baseline.** The anchor baseline must be re-scored under the SAME modified channel and the SAME per-frame
   true realization; only the psucc vector fed to the anchor differs across arms (stale echo vs recovered vs
   true). **Guard:** paired comparison, fixed seeds.
3. **Controlled knob.** Prove the knob does what we claim: **measure empirical `Corr(psucc_t, psucc_{t-1})`** of
   the decision-critical component vs decorrelation distance, confirm it rises monotonically. Otherwise
   "stronger temporal correlation" is unproven.
4. **Right metric.** Shadowing makes the channel HARDER (more stochastic) → raw feasibility FALLS. The claim is
   NOT "shadowing raises feasibility"; it is "shadowing creates recoverable temporal structure that a nowcaster
   can exploit to close MORE of the oracle gap." **Metric = recovered FRACTION of the oracle gap
   `(true-CSI − stale-echo)`, per knob setting** — not raw feasibility.
5. **No goalpost-moving.** Report as a NAMED, physically-justified channel regime, with results on BOTH the
   original and modified channel, clearly scoped. Byte-identical to HEAD when off / knob at baseline.

### Goal 1 (decision-focused prediction) — five ways it could be FAKE, and the guard for each
1. **Right metric:** score on anchor feasibility (deployed decision), NOT MSE. Decision-focus may *worsen* MSE
   while improving feasibility — that divergence is the expected *signature*, not a bug.
2. **Same decision, train and deploy:** the training-time decision oracle must be the SAME `local_hysteresis`
   anchor (same thresholds) used at deployment, else the alignment is to the wrong target.
3. **Leak-free:** true CSI only as a training label; predictor inputs are stale + leak-free only.
4. **Effect-on-decision:** must show the objective changes WHICH edges flip near the boundary, with a feasibility
   consequence — not just a different forward value.
5. **Honest ceiling + held split + CIs:** the true-CSI oracle is the ceiling; report the fraction of the oracle
   gap closed on HELD scenes, ≥5 seeds + CI, no single-seed headline.

---

## 6. Goal 3 — activation: preliminary answer (to be confirmed empirically at DF4)

**Anchor structure (VERIFIED, [dynamic_baselines.py:48](../src/marl_topology/training/dynamic_baselines.py)):**
`local_hysteresis_proposals` is NOT an independent per-edge threshold — per node it takes the **budget-`b_i`
top-k** of {kept edges with psucc ≥ 0.4} ∪ {new edges with psucc ≥ 0.6} (keep-priority), then an edge survives
only if **both** endpoints accept it (mutual-accept AND). So the decision has a real **per-node budget** and a
**mutual coupling**. The residual head adds a per-edge logit offset to this.

**Why NOT softmax over the per-edge residual logits:** the residual head emits a **per-edge scalar logit**. Softmax
normalizes a vector to a distribution summing to 1 — imposing **global competition** among all incident edges and
a scale that **depends on N** (edge count) — which breaks cross-N generalization and the ability to keep/add many
edges at once, and injects wrong off-diagonal `−p_i p_j` gradients. That is a *category error* for what is a set
of per-edge offsets. Softmax's per-edge degenerate (sigmoid) is just bounded squashing — which tanh/leaky-tanh
already provide with a controllable range.

**Where softmax/Gumbel IS right — and why it is the highest-value next mechanism:** an **edit-selection** head that
ranks *which* candidate edit(s) to apply under the node budget (a categorical / top-k / Gumbel-softmax policy over
a candidate edit set). The anchor's real per-node budget makes this competition *correct*, not artificial. Crucially
(activation lens): edit-selection consumes only the **ordering** of edits — the rankable **direction** signal the
T-campaign found learnable (`dir_acc ≈ 0.92`) — while **sidestepping the per-edge magnitude** the campaign found
does not convert. It is the one reformulation that routes around the magnitude wall. **DF4 will A/B** per-edge
leaky-tanh vs an edit-selection (top-k / Gumbel) head on the feasibility metric; a selector CI that clears the
anchor where leaky-tanh does not would overturn "keep per-edge."

**Is leaky-tanh optimal for the per-edge logit?** It satisfies the two properties that matter — (a) **bounded**
output (trust-region/PPO stability) and (b) a **gradient bounded away from zero** (so the recurrent temporal
signal always reaches the *acted* logit, the T2 fix). Plain tanh/hardtanh saturate (the original defect);
identity is unbounded (instability); relu/softplus are one-sided. DF4 will confirm empirically and consider
softsign / small-scale-linear-with-norm-penalty as alternatives.

---

## 7. Hard constraints (unchanged, non-negotiable)

1. Deployment fully decentralized; 0 evaluator calls at decision time.
2. Deployed actor never sees true current CSI, critic, evaluator, central solver, or global decoder.
3. Training MAY use true current CSI as a supervised label / critic input — marked **training-only**.
4. Real PBFT reliability/energy/latency always evaluated on the **true current channel** + closed-form
   whole-network quorum-tail.
5. Any proxy / D_quorum / PBRS / CSI-auxiliary is a training signal only; never replaces the final evaluation.
6. Imitate only anchor-positive LOCAL edits; never the full oracle topology.
7. Every claim carries a Claim Card + Mechanism-Path Matrix; failing test first; report effect on the final
   ACTION/topology (not just forward); held/test never used for teacher/checkpoint/training.
8. ≥5 seeds + CI for any headline; no single-seed/pilot headline.
9. Commit per stage to `decentralized-marl-trunk`; **do NOT push** (owner's decision).
10. Ultracode ON → use Workflow for adversarial verification; no backticks in Workflow/agent prompts.

---

## 8. DF0 status ledger

- [x] Campaign framed; tasks DF0–DF6 created; base HEAD `1bf773f` confirmed clean.
- [x] Key finding: the temporally-correlated shadowing physics **already exists** (`_shadow_field_value` /
      `shadowing_37885`), unused in the campaign channel → goal 2 is *enable + parametrize + validate*, not *invent*.
- [ ] Research synthesis (decision-focused methods; env-validation red-team; activation) — **in flight** (3 agents).
- [ ] `docs/decision_focused/DF0/{experiment_plan,validation_design,claim_card,mechanism_path_matrix,decision}.md`.
- [ ] Adversarial Workflow verification of the DF0 validation design (before any experiment).
