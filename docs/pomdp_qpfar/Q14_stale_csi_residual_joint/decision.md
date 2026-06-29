# Q14 — decision (stale-CSI + residual JOINT experiment) — HONEST NEGATIVE, with mechanism

**Result: the open loop is closed with a clean honest negative.** Under stale CSI the deployable anchor
DOES lose feasibility (urban 0.80→0.66 at delay-1, significant), but the feasible-anchored residual policy
does NOT recover it — it is bimodal (stay-at-anchor or collapse), and cross-frame recurrence is behaviorally
inert. Temporal modeling brings no control gain even when finally wired into the control loop; the loss is
recoverable only by fresher CSI, not by learning. Adversarial verification: Workflow `wmr40pusa` (4 lenses,
overall MINOR; the two load-bearing claims PASS and were independently reproduced).

## Section 1 — anchor sensitivity (5 seeds, training-free; anchor on stale ĝ_t, metric on true channel)
| data | current | delay-1 | delay-2 | partial(ρ=0.5) |
|---|---|---|---|---|
| **urban** | **0.804** [0.748,0.861] | **0.662** [0.601,0.724] | 0.583 [0.458,0.709] | 0.700 [0.599,0.801] |
| random | 0.342 [0.160,0.523] | 0.296 [0.115,0.476] | 0.304 [0.156,0.452] | 0.308 [0.151,0.466] |

Urban current vs delay-1 CIs **do not overlap** → a significant −0.142 drop; energy and latency also rise
monotonically with delay (the anchor is genuinely misled by stale CSI into picking worse edges). Random is
~flat (its frames are mostly in the deeply-infeasible core). **Stale CSI is a real perception loss.**

## Section 2 — residual recovery (5 seeds, FREE config residual_prior −1.0, anchor_reg 0.0)
| data / mode | residual feas | anchor feas | paired (resid−anchor) | retention | diverged |
|---|---|---|---|---|---|
| urban delay-1 | **0.404** | 0.662 | **−0.258** [−0.698,+0.181] | 0.60 | 0/5 |
| urban delay-2 | 0.363 | 0.583 | −0.221 [−0.505,+0.063] | 0.61 | 0/5 |
| random delay-1/2 | 0.296/0.304 | =resid | +0.000 | 1.00/0.99 | 0/5 |
| memoryless vs recurrent | bit-identical on EVERY arm | | | | |

**The deviation is BIMODAL, not uniform.** retention 0.60 = (3 seeds at 1.0 + 2 seeds at 0.0)/5: on urban
delay-1, **3/5 seeds stay exactly at the anchor (feas≈0.66), 2/5 collapse to the empty topology (feas 0)**;
mean 0.40. The residual NEVER beats the anchor on any seed. The paired CI **spans 0** → this is a
strong-mean-negative, not a tight significant one — but the qualitative fact is unambiguous: free deviation
is either inert or catastrophic, never beneficial. (The leashed Q9-Q12 config would instead pin retention
at 1.0 = the stale anchor.) The best stale-CSI policy remains the (stale) anchor itself.

## Why recurrence is inert (verified mechanism — NOT a wiring bug)
Carrying the GRU hidden state is genuinely active (it changes the GRU output and the edge-head pre-activation
`raw`). But in the trained policy **~83% of the edge logits are saturated at the tanh ±10 rail** (edge_head
`raw` ≈ 3000; `logit = 10·tanh(raw/10)`), and the tanh derivative at the rail is ≈0, which **annihilates the
hidden-state contribution to the final logit** → memoryless and recurrent decode bit-identically. The
temporal signal is real but quantized away by (a) the saturated logit head and (b) the discrete flip decode.
This matches Q2's "modest magnitude (~13-23% MSE reduction)" — the temporal information is small and the
discrete topology action eats it.

---

# Codebase analysis (read-only; grounds the two follow-up questions)

## Q-A: is the stale-CSI drop caused by a MISSING temporal-awareness module? — NO.
1. **The anchor is a pure stateless heuristic** (`dynamic_baselines.py::local_hysteresis_action`) — no model,
   no temporal module at all. It drops because it applies fixed psucc thresholds to STALE values and cannot
   interpret the appended `csi_age`/`csi_observed_mask` columns. "Missing temporal module" is moot — there is
   no model.
2. **The learned actor HAS a temporal module and it IS wired/active.** `DynamicRecurrentActor` has a
   per-node `GRUCell` carrying hidden state across frames; under stale CSI the edge-feature tensor `ef`
   carries the age/mask columns and the full `ef` is fed to the actor (the features ARE in the input).
3. So the drop is NOT a missing module. It is **(a) input degradation** (stale psucc) **+ (b) the GRU's
   contribution being annihilated by logit saturation** (above) **+ (c) no explicit CSI-recovery
   supervision** — the GRU is implicit temporal modeling with no supervised target, and the Q2 CSI predictor
   is a pre-training gate diagnostic, never wired into the policy loss; the critic reads the same stale `ef`.
   **The module is present but ineffective, and was never taught to turn history into a current-CSI estimate.**

## Q-B: how to un-clamp the policy and get BENEFICIAL autonomous deviation (not clamp, not collapse)?
Two distinct failure mechanisms:
- **Clamp:** `residual_prior = −3.0` strongly biases logits negative → ~no flips → stay at anchor (a safety
  mechanism).
- **Collapse (bimodal):** when freed, plain REINFORCE with a scalar moving-average baseline, **no entropy
  bonus, no value critic**, has two basins (≈0 flip prob = clamp, ≈1 flip prob = collapse); random init picks
  the basin, and nothing pulls it back. Logit saturation then locks whichever basin it lands in.

**Levers (all already in the codebase, just unused/underused) — the recommended combination:**
1. **Imitate the central ORACLE repairs, not the anchor** (`residual_repair.py::greedy_dquorum_add_repair` /
   `repair_head_targets`, Q7 — recovers 22% of anchor failures). This is the single biggest lever: the policy
   deviates toward *worse* because nothing points it toward *better*; the oracle repairs provide a
   BETTER-THAN-ANCHOR supervised attractor (BCE/DAgger term).
2. **Anneal the trust-region** (curriculum `anchor_reg` high→low) instead of a fixed value → explore near the
   anchor first, widen as learning stabilizes (replaces clamp-vs-collapse with gradual relaxation).
3. **Add a CTDE value critic** (`CentralizedGraphCritic` exists in `dynamic_rl.py`) → advantage
   `A_t=G_t−V(s_t)` replaces the scalar baseline → variance reduction stops the stochastic basin collapse.
4. **Entropy bonus + raw-logit L2** (`residual_action.py` already yields logp; the actor logit head saturates)
   → keep the policy stochastic and unsaturated → prevents the bimodal collapse AND restores the GRU's
   influence (fixes the recurrence-inert finding).
5. **D_quorum as a dense supervised auxiliary head** (`potential_shaping.py`/`quorum_deficit.py`) → gradient
   even on the infeasible plateau where true C is flat (currently used only as weak PBRS).
6. **PPO clip / KL trust-region** (`graph_mappo.py::ppo_clip_actor_loss`/`approx_kl`) instead of the crude L1
   flip penalty → principled divergence bounding that naturally prevents collapse.
7. **(ties Q-A) supervised CSI-prediction auxiliary** → force the GRU hidden state to predict true current
   psucc → make the temporal module actually useful under stale CSI.

**Core insight:** beneficial autonomous deviation does not come from *removing* the anchor constraint (that
just collapses) — it comes from **giving the policy a better-than-anchor target (oracle repairs) + a
variance-reduced, unsaturated, entropy-regularized trainer + a gradually-annealed trust-region.** That turns
"deviate from the anchor toward worse" into "deviate toward the oracle-discovered better."

> These are UNTESTED design recommendations grounded in existing modules, not validated results. Any of them
> would be a new one-variable-per-round experiment under the Contract (failing test first, ≥5 seeds, CI).

## Acceptance (Contract v3 §15)
- Phase: **Q14 (addendum) — stale-CSI + residual joint experiment**
- Status: **VALIDATED_NEGATIVE** — stale CSI hurts the anchor (real perception loss); the residual does not
  recover it (bimodal: inert or collapse); recurrence inert (saturation). Temporal axis → no control gain.
- Test scale: 5 seeds × {urban, random} × N{8,12,16}; eval-only diagnostic; suite unchanged (no src/test change).
- Conclusion scope: N≤16, these stale modes, 5 seeds, this training budget. NOT a claim that temporal modeling
  is useless in general — only that, as architected/trained here, it buys no control gain under stale CSI.
- Artifacts: `scripts/diagnostics/stale_csi_residual_joint.py` (joint experiment),
  `scripts/diagnostics/success_case_analysis.py` (the per-frame C/E/L success-case analysis).
- Verification: Workflow `wmr40pusa` (overall MINOR; corrected a wrong mechanism claim [saturation, not
  subthreshold] and the bimodal framing). The success-case analysis was verified by Workflow `wsdvtxdv7`.
