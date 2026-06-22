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
`local_mutual_assemble` (`policies/decentralized_mutual_acceptance.py`), reward = Ng–Harada
potential `(c − τ)` on the closed-form quorum-tail `c` (`protocol/quorum_tail.py`) + a
Lagrangian budget dual, `τ = 0.9`.

During reconstruction this remains the production entry **and a required baseline**
(Spec §15: REINFORCE-EMA / RLOO-M≥2 are mandatory comparison baselines). The new CTDE
entry (`scripts/train/train_ctde_scq.py`, Engineering-Plan §3.1) becomes the sole
production trunk only at Phase 13. Keep exactly **one** production trunk; do not build a
second long-lived parallel one.

## Gate discipline (this repo is gate-driven)

- Frozen contract gates scan `src/marl_topology/**.py` for banned literals (`MAPPO`,
  `COMA`, `Transformer`, `optimizer`, `class Critic(`, `class Actor(`, `def reward`, …).
  These encode the retired critic-free identity and **block Phases 7–9**; they are
  retired in a staged way as each subsystem is re-implemented (see
  `docs/CURRENT_HEAD_STATUS.md` §5). Do not silently weaken a *correct* physics/protocol
  test to go green.
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

## Headline result (current baseline to beat, not regress)

Cold-start, oracle-free decentralized learning beats the SA oracle at out-of-range scale
N = 24 (5-seed margin **+0.177, CI95 [+0.104, +0.249]**, on `raw_mean − ceiling`). This
is the **critic-free baseline** the CTDE method must match or beat under an equal
evaluator-call budget. If a change moves it, say so honestly in the research log. Note
the ceiling here is a finite SA search, which per D5 is a baseline, not an
infeasibility proof.
