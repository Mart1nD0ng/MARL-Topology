# Q5 — decision (local_hysteresis imitation actor)

**Result: KEEP the hysteresis-teacher machinery; HONEST MIXED imitation result + a decisive diagnosis
that shapes Q6.** BC-imitation of the deployable anchor REPRODUCES it on random (teacher-forced F1 0.89,
NLL converges) but FAILS on urban (NLL diverges, F1 0.24). Root cause diagnosed. The implication: Q6 must
use the explicit residual design where the anchor is the DIRECTLY-COMPUTED base (Spec §10.2:
`x = H ⊕ Δ`), so "zero residual = anchor" is EXACT by construction — NOT a from-scratch BC-imitation
actor. This satisfies the contract's "reproduce the anchor before residual RL" at the decoder level
(exact), while honestly recording that pure BC-imitation is unreliable on urban.

## What changed (correct, reusable; default byte-identical)
- `training/dynamic_baselines.py::local_hysteresis_proposals` — exposes the anchor's per-node accept
  subsets + mutual topology (refactor; `local_hysteresis_action` unchanged, byte-identical).
- `training/dynamic_rl.py::_hysteresis_teacher_trajectory` — the DEPLOYABLE anchor as a decoder-aware BC
  teacher (0 evaluator calls; NOT the central myopic teacher), same per-frame dict shape as the D6
  myopic teacher so it plugs into the existing warm-start / NLL machinery.
- `scripts/diagnostics/anchor_imitation.py` — warm-start an actor to the anchor + measure reproduction
  (teacher-forced F1, held feasibility actor-vs-anchor, switches, BCSP NLL). Eval-only.

## Result (pilot: train 14, held 10, 6 frames)
| data | teacher NLL init→final | decoded F1 (TF) | actor held feas | anchor held feas | actor sw/frame | anchor sw/frame |
|---|---|---|---|---|---|---|
| random (1-RSU) | 2.97 → **1.40** | **0.89** | 0.13 | 0.23 | 0.25 | 1.05 |
| urban (4-RSU) | 3.9 → **20.7** (diverges) | **0.24** | 0.33 | 0.79 | 17.1 | 1.98 |

## The diagnosis (why urban BC-imitation fails) — code-grounded
- Urban node budgets are HETEROGENEOUS: an RSU has budget **64** (≥ its degree 11 → unconstrained), a
  vehicle has budget **2**. The anchor's per-node accept proposals then impose CONFLICTING targets on a
  SHARED edge logit: an RSU-vehicle edge is "propose" for the unconstrained RSU but "not in the top-2"
  for the vehicle, so no single edge-logit field fits all nodes' subsets. The BCSP-subset NLL diverges
  (logits blow up to ~9 in 10 epochs; lr-independent at 0.05/0.01/0.005).
- A per-edge BCE toward the anchor's MUTUAL topology ALSO fails (F1 0.79→0.26): the deployed decoder
  (`local_mutual_assemble`, budget-top-k per node) selects differently from the hysteresis THRESHOLD
  rule, so matching the mutual topology via raw edge logits does not reproduce the anchor either.
- On random (single RSU, near-homogeneous budgets) none of this bites → F1 0.89.

## Why this does NOT block Q6 (and in fact motivates its design)
Spec §10.2 defines the residual policy as `x_t = H(ô_t, x_{t-1}) ⊕ Δ_θ(...)`: the anchor `H` is the
DIRECTLY-COMPUTED deployable heuristic (0 eval calls), and the actor produces only the residual `Δ`.
"Zero residual = anchor" is therefore EXACT — the residual base reproduces the anchor with no BC loss at
all. The contract's "不复现 local_hysteresis 就开始 residual RL" is satisfied at the decoder level (the
anchor IS the base, exactly), NOT via a lossy BC-imitation actor. The Q5 failure on urban is the evidence
that a from-scratch BC-imitation foundation is the WRONG choice on urban — confirming the explicit
residual-base design.

## Honesty / scope
- This is a single-seed pilot; the random F1 0.89 and the urban divergence are both robust to the lr
  sweep, but no multi-seed CI is claimed (it is a gate diagnostic, not a headline).
- The covariate-shift gap (random teacher-forced F1 0.89 but deployed actor_feas 0.13 < anchor 0.23) is
  itself a reason to prefer the residual base (the deployed actor IS the anchor at zero residual, no
  compounding rollout drift).
- I did NOT spin the urban failure as a pass; the imitation gate, read as "can the actor BC-reproduce
  the anchor", FAILS on urban. The path forward is the residual-base design, recorded transparently.

## Acceptance table (Contract v3 §15)
- Phase: **Q5 — local_hysteresis imitation actor**
- Status: **NEGATIVE_BUT_SCOPE_LIMITED** (BC-imitation: random PASS / urban FAIL-diagnosed) + the
  hysteresis-teacher machinery is VALIDATED_POSITIVE (correct, reusable, 0-eval, byte-identical default).
- Implemented ✓ / Wired (diagnostic + teacher trajectory) ✓ / In trunk warm-start ✗ (the `--dyn-warmstart-teacher`
  flag is added in Q6 where the residual base needs it) / Active in this run ✓ (imitation pilot)
- Test scale: 4 new unit + suite 739/0; pilot single-seed train 14 / held 10
- Positive: hysteresis teacher correct (recon == anchor, 0 eval); BC-imitation reproduces anchor on random.
- Negative: BC-imitation FAILS on urban (diagnosed: budget-64 RSU vs budget-2 vehicle conflicting
  shared-edge targets; decoder budget-top-k ≠ threshold rule).
- Conclusion scope: pure BC-imitation of the anchor is unreliable on urban → use the explicit residual
  base (Q6), which reproduces the anchor exactly by construction. NOT a claim that the actor cannot learn.
- Next action: **Q6 — residual action space** (`x = H ⊕ Δ`, H = directly-computed local_hysteresis,
  zero residual = anchor EXACT, add/remove/swap, local mutual decoder, 0 action-eval calls). The anchor
  is the base (exact reproduction); the actor learns the residual.
