# AGENTS.md — working agreement for this repo

This is a decentralized constrained-RL codebase for V2X topology control, **under
active reconstruction** toward a CTDE architecture. Read this before changing anything.

## Authority of record

Two documents are the binding technical spec. When old code, comments, results, or this
file conflict with them, **they win** (defer to mathematical correctness + current code
facts + the specs):

- `docs/MARL-Topology-Technical-Spec.md` — the target architecture and the environment
  math (PBFT quorum, route/relay, energy, latency, solvability).
- `docs/MARL-Topology-Engineering-Plan.md` — the phased reconstruction plan (Phase 0–13).
- `docs/CURRENT_HEAD_STATUS.md` — the frozen Phase-0 snapshot of where HEAD actually is.

Target architecture: **CTDE Graph-Counterfactual PPO + SCQ exact counterfactual
supervision + recurrent directional PNA actor + vector graph-temporal critic +
chance/CVaR reliability constraint + preference-conditioned Pareto policy.**

## The core boundary: train centralized, deploy fully decentralized

\[ \text{Centralized Training with Fully Decentralized Execution (CTDE)} \]

- **Training MAY use** a centralized graph critic, a global PBFT evaluator, joint
  observation history, joint actions, and training-time counterfactual simulation.
- **Deployment MAY use only** a node's own local observation + history, real exchangeable
  neighbor messages, public protocol parameters (n, f, q, τ), and the deployment
  preference ω.
- **Deployment MUST NOT use** a critic, global state, a centralized/global-argsort
  decoder, a central solver/searcher, or evaluator verification of candidates.

Therefore: **a central critic is legal; a central deployment decoder is not.** This
replaces the retired "critic-free is a hard invariant" framing — see
`docs/CURRENT_HEAD_STATUS.md` §4. "critic-free" is no longer the project identity; the
identity is *efficient global credit in training + strict local execution in deployment*.

## Hard invariants (do not downgrade, proxy, or substitute)

- **D1 — deployment decentralization.** The deployed actor uses local/neighbor info
  only; no central critic, no global state, no global decoder/argsort, no
  solver/searcher/evaluator at inference. The main rollout uses the **same** local
  mutual-acceptance action semantics from training through deployment.
- **D2 — reliability is closed-form, whole-network.** Final reliability is the fixed
  closed-form PBFT quorum-tail (Poisson-binomial), never Monte Carlo or a local proxy;
  PBFT n/f/q satisfy the quorum-intersection + liveness conditions; the Byzantine fault
  set is a single fixed `B` held across all protocol phases.
- **D3 — `τ ≥ 0.9`**, the same constant in training and evaluation.
- **D4 — generalization.** Domain randomization + held-out + varying N + **≥5 seeds with
  CI**; never report success on the train config or a single operating point. smoke/pilot
  params never produce research conclusions.
- **D5 — honest solvability.** A finite-search miss is `unknown`, never
  `certified_infeasible`; unknown scenes are not silently deleted; no teacher topology /
  teacher energy normalization props up the main method.
- **D6 — mechanism activation.** Every mechanism a run claims (RLOO M≥2, temporal,
  counterfactual, SCQ, dual up/down, Pareto) must have a runtime activation assertion +
  logged evidence, or it must fail fast.

## The trunk (transitional)

`scripts/train/train_decentralized_rl.py` — the **single-step critic-free REINFORCE +
RLOO** trunk over `MessagePassingGraphEdgeScorer` edge logits, decoded by
`local_mutual_assemble` (`policies/decentralized_mutual_acceptance.py`), reward = the
feasibility-margin `(c − τ)` on the closed-form quorum-tail `c` (`protocol/quorum_tail.py`) + a
Lagrangian budget dual, `τ = 0.9`. (The `−τ` is a constant offset on a single-step `T=1` bandit —
a fixed baseline that preserves the policy-gradient direction; it is **not** Ng–Harada potential-
based shaping, which requires an MDP state-potential difference `γΦ(s′)−Φ(s)` that does not exist
here. The earlier "Ng–Harada potential" wording was an unproven misdescription, corrected in v2 R0.)

During reconstruction this remains the production entry **and a required baseline**
(Spec §15: REINFORCE-EMA / RLOO-M≥2 are mandatory comparison baselines). The new CTDE
entry (`scripts/train/train_ctde_scq.py`, Engineering-Plan §3.1) becomes the sole
production trunk only at Phase 13. Keep exactly **one** production trunk; do not build a
second long-lived parallel one.

## Gate discipline (this repo is gate-driven)

- The deployment-purity gates scan `src/marl_topology/**.py` for banned literals (`MAPPO`,
  `COMA`, `Transformer`, `optimizer`, `class Critic(`, `class Actor(`, `def reward`,
  `train_loop`, `torch.save`, …). As of 2026-06-23 these are **scoped to the deployed
  paths**: the `models/` and `training/` subtrees are EXEMPT, so a centralized graph
  critic + Graph-MAPPO/COMA/PPO + optimizers/checkpoints live there legally (CTDE,
  Phases 7–13), while `protocol/`, `policies/`, `data/`, `evaluation/` stay scanned. This
  enforces D1 (centralized training, fully decentralized execution) at the layer level.
  The three canonical scans are `test_stage8_*`, `test_stage9_0_*`, `test_stage8_0_*`;
  the 287 stale stage-contract *process* gates (asserting on deleted lineage docs) were
  retired the same day. Phases 7–9 are **unblocked**. Do not silently weaken a *correct*
  physics/protocol/deployment-decentralization test to go green; if you add CTDE code,
  it belongs under `models/`/`training/`, never in a deployed path.
- Write diagnostics to `logs/` (gitignored, ungoverned). `result_save/` is allowlist-
  gated and gitignored (only `*.md` reports are tracked).

## Workflow per change (one hypothesis per round)

1. Bottleneck → single hypothesis (one variable) → write the controlled variables.
2. Failing test/diagnostic first → minimal implementation.
3. targeted tests + smoke + mechanism-activation check + affected unit suite.
4. minimal pilot (confirm the mechanism truly fires) → paired multi-seed (≥5) A/B.
5. Report per-seed values, mean, CI, failure cases, OOD, and cost (evaluator-calls /
   env-steps / wall-clock). Record in `docs/URBAN_V2X_RESEARCH_LOG.md` + a `decision.md`.
6. Keep / revise / rollback. If stuck 3 rounds, re-check evaluator, labels, action
   reachability, critic error, train/deploy mismatch, checkpoint, mechanism activation —
   do not stack another module.

## Verification gate

```bash
python scripts/train/train_decentralized_rl.py --smoke --cold-start   # must exit 0
python scripts/train/evaluate_actor_on_dataset.py --help              # must import
python -m pytest tests/unit -q
```

The smoke must exit 0, print a `[data] pool ...` line, and run a few updates.

## Headline result (under the corrected environment math)

> **CURRENT STATE (2026-06-28) — three campaigns complete; see `docs/CURRENT_HEAD_STATUS.md` for the live
> ledger.** The in-range-parity snapshot below is a HISTORICAL milestone (pre-CTDE recalibration); it has
> since been superseded by FOUR completed, adversarially-verified campaigns that all reach the same honest
> finding — at N ≤ 16, every sophisticated mechanism is verified-correct but gives no headline gain, and the
> binding limit is **RL feasibility-region learning** (sharpened by the latest campaign to the deployable
> PRECISION of the beneficial-edit direction signal):
> - **v2 static** (R0–R7 + Phase 8–13): Graph-MAPPO + MLP + closed-form decoder is best in-range and
>   best-generalizing.
> - **Dynamic-repair** (D0–D14): the T>1 task built + corrected (10 gaps) on real 4-RSU urban data; all
>   mechanism A/B CIs span 0; learned arms below the zero-eval deployable heuristics.
> - **POMDP-QP-FAR** (Q0–Q13): stale/partial CSI (`--csi-mode`) + quorum-deficit potential `D_quorum`/PBRS
>   + feasible-anchored residual learning + edge handshake + PNA-in-residual — **all opt-in, default-off,
>   byte-identical when off, each verified-correct**; the deployable `local_hysteresis` anchor (== the
>   residual policy) is the best deployable arm; the central myopic reference is a grouped ceiling (NOT a
>   deployable baseline), and it beats the anchor only modestly and not significantly (Q12 paired CI spans
>   0). No learned arm beats the deployable anchor. Ledger: `docs/URBAN_V2X_RESEARCH_LOG.md` +
>   `docs/pomdp_qpfar/`.
> - **Belief-Guided Evidence-Gated Residual PPO** (R0–R8 + R10; Contract v4 Claim-Path evidence): the full
>   method vs the Q14 residual failure — CSI-belief aux, residual PPO + CTDE critic, beneficial-edit
>   supervision (repair/safety/edit heads), an evidence gate, an adaptive anchor-KL. **The beneficial-edit
>   direction signal genuinely EXISTS (R4) and is locally RANKABLE (R5, held top-k CI>0 — the campaign's first
>   deployable-learning positive), but it does NOT CONVERT into a deployed gain: no `tau_edit` beats the anchor
>   on the current channel (R6) or the stale channel (R8), and an adaptive anchor-KL also lands at the anchor
>   (R7).** The stale-CSI premise is CONFIRMED (stale degrades the anchor, urban −0.165 ≈ Q14) but the method
>   does not repair it. Binding limit = **the deployable PRECISION of the direction signal** (not its existence,
>   learnability, the trainer, the gate, the temporal/belief chain, or a leak). All mechanisms opt-in /
>   default-off, adversarially verified per stage. Ledger: `docs/CURRENT_BELIEF_RESIDUAL_STATUS.md` (4-chain
>   diagnosis §5) + the `BELIEF-GUIDED RESIDUAL PPO CAMPAIGN SUMMARY` in `docs/URBAN_V2X_RESEARCH_LOG.md` +
>   `docs/belief_residual/R*/`.
>
> Open frontier (deferred): large-N (N ≥ 24) generalization (needs a cheaper exact-fault evaluator); a
> higher-precision local direction signal (the only lever the Belief-Residual campaign leaves open).

The legacy "+0.177 beat at out-of-range N = 24" is **retired** — it was produced under the
pre-recalibration (incorrect) env-math. Under the **corrected** math (P0–P4 + recalibration)
**and the corrected sampler** (the NaN-gumbel no-exploration bug fixed 2026-06-23; the trunk
samples through the verified Plackett-Luce action API), an oracle-free cold-start decentralized
learner **approximately matches the SA oracle in-range** (N ∈ {8,12,16}): 5-seed RL keep-best
**0.610** vs SA ceiling 0.655, margin **−0.045, CI95 [−0.083, −0.006]** (final-update margin
−0.045 [−0.101, +0.011]) — at parity-to-marginally-below. The sampler fix is a train/deploy-
alignment correctness fix but is **neutral on the headline** (the pre-fix 0.624/−0.031 is
statistically indistinguishable; a single-shard pilot gain did not generalize). The out-of-range
N = 24 comparison under correct math is the open question (gated on a `fixed_set` cost
optimization + a heavy build). This in-range parity is the baseline the CTDE method (Graph-MAPPO,
in progress) must match or beat under an equal evaluator-call budget. The ceiling is a finite SA
search — per D5 a baseline, not an infeasibility proof.
