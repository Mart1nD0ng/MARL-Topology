# CURRENT HEAD STATUS — Phase 0 freeze (spec-driven reconstruction)

> Produced by Phase 0 of the `MARL-Topology-Engineering-Plan.md` reconstruction loop.
> Authority of record: `docs/MARL-Topology-Technical-Spec.md` +
> `docs/MARL-Topology-Engineering-Plan.md` (2026-06-22). Where old code/comments/results
> conflict with those two specs, the specs win.
> **Iron rule of this document: every claim cites an artifact (file:line, test id, or
> command output). Anything not so grounded is marked `UNVERIFIED`.**

This file answers one Phase-0 question: **has the current HEAD changed the "known
status" the specs assume?** It is a frozen snapshot, not an architecture re-analysis.

---

## 1. Snapshot

| field | value | source |
|---|---|---|
| git HEAD | `15fcb8a78d4b90d787df21f110e139781e07939c` | `git rev-parse HEAD` |
| HEAD subject | "Add the literature review corpus (40 papers across 6 themes)" | `git log -1` |
| consolidation commit | `4e7aa62` "Consolidate to a single decentralized trunk; remove the centralized-critic MAPPO lineage" | `git log` |
| branch | `decentralized-marl-trunk` | session git status |
| Python | 3.11.5 | `python --version` |
| Torch | 2.5.1+cu118 | `python -c "import torch"` |
| platform | Windows-10-10.0.26200-SP0 | `platform.platform()` |
| working tree | clean except the 2 new spec docs (untracked) | `git status --short` |

Dataset / artifact manifest: trunk run artifacts live under the **gitignored**
`result_save/` (per `.gitignore:22`, only `*.md` + `.gitkeep` tracked). No committed
dataset shard manifest exists at HEAD → recorded here as a Phase-0 gap (the plan's
`data_manifest.json` deliverable, §22, is not yet present). `UNVERIFIED`: exact shard
inventory (artifacts are local-only).

---

## 2. Test baseline at HEAD (`python -m pytest -q`)

```
289 failed, 503 passed in 123.44s
```
Artifact: `logs/phase0_full_test_run.txt` (gitignored).

**Breakdown:** 287 of 289 failures are in `tests/contract/`; 2 are in `tests/unit/`
(`test_run_manifest_validator_stage5_10.py`, `test_minimal_training_stack_stage6_0.py`).
Failures span stage2–stage9, stage16–32 contract files (reward/objective/plan/design/
report/exit-gate/harness suites).

**Root cause (verified, not a physics regression):** the consolidation `4e7aa62`
deleted the governance/lineage artifacts these contracts assert on, but left the
contracts. Confirmed removed:
- `docs/PROJECT_STATE.md` — absent (`Glob **/PROJECT_STATE.md` → none). The
  gate-memory invariant "`PROJECT_STATE.md` must keep `owner_decision_required: true`"
  is now moot because the file is gone; every stage contract that greps it fails.
- the tau decision record, CODEX_WORKFLOW, stage docs (STAGE*.md), harness-task
  fixtures, stage-closeout templates — none remain under `docs/` (only 7 docs survive,
  `Glob docs/*.md`).
- the whole MAPPO/critic/planner lineage (`training/mappo/`, `training/critic_*`,
  `models/centralized_*`, planner drivers) — removed per `AGENTS.md` "What was removed".

**Interpretation:** the **physics / protocol / channel / link / quorum / model unit
suite is green** (it is the bulk of the 503 pass). The red 287 are *stale lineage
governance gates*, not correctness failures. They are protocol/metric/lineage
**incompatible** in the sense of Engineering-Plan §3.1 / Phase-13 (`retired_due_to_
protocol_metric_change`). Resolution is **staged**, not done in Phase 0: each later
phase retires its own slice as it re-implements the corresponding subsystem; the
banned-literal `src/` gates (see §5) must be lifted before Phase 7. **No correct
physics/protocol test is to be weakened to go green** (forbidden shortcut).

`UNVERIFIED` until per-phase triage: that *all* 287 are purely lineage-stale vs. a few
encoding a still-valid contract. Phase 1+ triages each slice before retiring it.

---

## 3. Known-status confirmation (the 11 facts the specs assume)

Audited against current code (`scripts/train/train_decentralized_rl.py`,
`src/marl_topology/`). Status = CONFIRMED (unchanged) / CHANGED / PARTIAL.

| # | Known-status claim | Status | Evidence |
|---|---|---|---|
| 1 | Trunk is single-step contextual-bandit, not multi-step MDP | **CONFIRMED** | `train_decentralized_rl.py:31` header "single-step contextual-bandit MARL"; no episode loop / `s_{t+1}` / transition anywhere. |
| 2 | Critic-free REINFORCE; RLOO only if `samples_per_scene>1`; default M=1 → EMA baseline | **CONFIRMED** | argparse `--samples-per-scene default=1` (~:290); EMA baseline path when M=1, leave-one-out only when M>1 (~:422). |
| 3 | Actor = parameter-shared K-round message-passing edge scorer | **CONFIRMED** | `models/message_passing_graph_edge_scorer.py` (`MessagePassingGraphEdgeScorer`), K rounds, shared module across rounds; default K=4. |
| 4 | Production decoder = `local_mutual_assemble`; `global_argsort_assemble` also present | **CONFIRMED** | both in `policies/decentralized_mutual_acceptance.py`; trunk main rollout uses local mutual; global argsort kept as centralized-decode ablation only. |
| 5 | No centralized critic in the learning loop | **CONFIRMED** | no critic class instantiated in trunk; `models/centralized_*` removed (`AGENTS.md`). |
| 6 | PBFT n/f/q fixed at `f=1,q=3` for arbitrary N | **CHANGED → still high-risk** | `f` now scales: `fault_tolerance=min(config, ⌊(n−1)/3⌋)` at `data/stage21_objective_stack_evidence.py:377`; `<4` validators → probability 0 (`:386–389`). **But** fault model = `FAULT_FILTER_REMOVE_LARGEST` (remove-largest **per evaluation**), which violates Spec §4.7 (single fixed Byzantine set `B` across all phases). No `classic_exact` / `safe_generalized` switch; quorum-safety conditions (`2q−n>f`, `q≤n−f`) **`UNVERIFIED`** — they live in `protocol/pbft_reliability.py` + `protocol/quorum_tail.py`, to be checked in Phase 1. |
| 7 | Binary `feasible_exists` solvability label from finite search (no tri-state) | **CONFIRMED** | `data/stage31_scenario_generator.py:561,575` set `feasible_exists = True/False`; a search miss is labeled `False` (the spec §5.1 forbids: a finite-search miss must be `unknown`, not `certified_infeasible`). |
| 8 | Latency is max-over-selected with deadline/phase clipping (degenerate) | **CONFIRMED** | `objectives/latency_energy.py:28` `max(record.latency_s …)`; `protocol/pbft_accounting.py:237` `min(max_latency_s, phase_budget_s)` clips to phase budget. Not quorum-completion / timeout-aware (Spec §4.10). |
| 9 | Route/relay risks double-counted multi-hop | **PARTIAL** | `protocol/stdma_scheduler.py:113–121` per-route latency looks single-layer; relay default `relay_hops=1`; no double-count found in phase-record filtering, but the one-hop-matrix vs relay-DP layering (Spec §4.2 regression `A–B–C: H=1→0, H=2→>0`) is **`UNVERIFIED`** — needs the Phase-2 regression test. |
| 10 | Energy is repeated all-pairs, not phase-specific | **CHANGED** | `protocol/pbft_accounting.py:226,248` sums `record.network_energy_j` over phase records (per-phase message accounting), not 3× all-pairs. Still missing relay/MAC-control/policy-comm/reconfig/view-change terms (Spec §4.9). |
| 11 | No Temporal Value Test / myopic-vs-horizon comparison | **CONFIRMED** | grep across `src/`+`scripts/` for "temporal value"/"myopic"/"finite-horizon" → 0 hits. |

**Net:** the specs' picture of HEAD holds. The two genuine deltas — (6) `f` already
scales with N (but the fault model and q-safety remain wrong/unproven), and (10) energy
is already per-phase (but incomplete) — make those phases *smaller* than the specs
assumed, not absent.

Trunk knobs at HEAD (for reproducibility): `--reward-mode barrier` (default), `--beta
0.1`, `--entropy-coef 0.005`, `--samples-per-scene 1`, `--live-consensus-dual` flag
exists (default off). Working research recipe is the dense β=0 cold-start recipe in
`docs/URBAN_V2X_RESEARCH_LOG.md` / `result_save/LOOP_EXPERIMENTS_REPORT.md`.

---

## 4. Governance conflict resolved this phase: critic-free → CTDE

The single most important Phase-0 finding is a **direct contradiction** between the
existing repo invariants and the new specs:

- `AGENTS.md` / `README.md` **I1** asserts "no central critic" is a *hard, never-
  downgrade invariant* and "critic-free … the strongest form of decentralized learning".
- `MARL-Topology-Technical-Spec.md` §2.1 + §19 require **CTDE** (Centralized Training,
  Decentralized Execution): a centralized graph critic is **legal in training**; only
  the **deployment decoder / global state / evaluator** must stay decentralized. §19
  explicitly says the project must **stop using "critic-free" as its identity**.
- Engineering-Plan Phase 0 work-item 3 directs: delete "no critic is a hard invariant"
  from AGENTS.md; replace with "deployment is strictly decentralized"; state plainly
  "central critic legal, central decoder illegal".

**Resolution (applied this iteration):** `AGENTS.md` rewritten to the CTDE boundary.
The binding invariant is now **deployment-decentralization**, not critic-freeness. The
critic-free REINFORCE trunk is **retained as a registered baseline** (Spec §15 line 1–2:
REINFORCE-EMA / RLOO M≥2 are required baselines), not as the project identity. README
I1 to be aligned in Phase 13 (migration/release) to avoid churn now.

This is owner-authorized: the loop instruction's hard-constraint #4 ("training may use a
centralized graph critic") and "defer to the two specs on conflict".

---

## 5. Frozen-gate trap inventory (forward blocker — must be lifted before Phase 7)

The repo is gate-driven; several frozen contract gates scan `src/marl_topology/**.py`
for **banned literals** and will reject the very modules the plan requires:

- Banned src-wide substrings include `MAPPO` (uppercase), `COMA`, `Transformer`,
  `optimizer`, `class Critic(`, `class Actor(`, `def reward`, `train_loop`,
  `torch.save`/`checkpoint_path`. (`backward(` was unfrozen earlier for training.)
- `result_save/` top-level entries are constrained by ~15 allowlist gates → write
  diagnostics to `logs/` (gitignored, ungoverned) instead. This phase followed that:
  test logs went to `logs/phase0_full_test_run.txt`.
- The (now-absent) `docs/PROJECT_STATE.md` invariant.

**Impact:** Phase 7 (`graph_mappo.py`), Phase 8 (`graph_counterfactual_ppo.py` / COMA),
Phase 9 (`scq_counterfactual.py`), Phase 7-critic (`centralized_graph_temporal_critic.py`)
**cannot be added** while these gates are frozen, because their names/contents trip the
bans. These gates encode the **retired** critic-free identity (§4). They must be retired
as part of doc-governance — **the same staged retirement as the 287 contract failures**.
Recorded here as the gating dependency for Phases 7–9; not actioned in Phase 0.

---

## 6. Config-tier status

`configs/` exists with `env/ model/ objective/ physics/ protocol/ scenario/ train/`
(operational config groups). It has **no** `smoke/ pilot/ research/` tiers, and
`configs/train/` is empty. Engineering-Plan Phase 0 work-items 4–5 (config tiers + a
unified run manifest) are **deferred to the next Phase-0 iteration** — they need to be
built on the existing (failing) `stage5_9`/`stage5_10` run-manifest contracts rather
than invented fresh, so they get their own focused iteration. No test gates `configs/`
structure (`git grep configs/ tests/**` → none), so adding tiers later is additive/safe.

---

## 7. Phase-0 actions

**Applied this iteration:**
1. Wrote this `docs/CURRENT_HEAD_STATUS.md` (the frozen status of record).
2. Rewrote `AGENTS.md` from critic-free-invariant to the CTDE / deployment-decentralized
   boundary (§4).
3. Logged Phase 0 in `docs/URBAN_V2X_RESEARCH_LOG.md`.

**Deferred (next Phase-0 iteration):** config tiers `configs/{smoke,pilot,research}/`
and the unified run manifest (§6).

**Not in Phase 0 (later phases):** retiring the 287 stale contract gates + the
banned-literal src gates — staged per phase (§2, §5); Phase 1 quorum/fault-set fixes.

---

## 8. Decision

- **Keep:** the green physics/protocol/model core; the critic-free trunk demoted to a
  required baseline.
- **Record:** HEAD matches the specs' assumed status except deltas (6) and (10), which
  shrink — not remove — Phases 1 and 4.
- **Gate:** P0–P4 must complete before any model-architecture effectiveness claim
  (Engineering-Plan §4). No model A/B this phase.
- **Next single hypothesis (Phase 1):** replace the per-evaluation `REMOVE_LARGEST`
  fault model with a single fixed Byzantine set `B` (`C_robust = min_{|B|≤f} C(B)`) and
  add the `classic_exact` / `safe_generalized` `PBFTQuorumSpec` with property tests
  (`2q−n>f`, `q≤n−f`, intersection + liveness), plus a Torch quorum-tail with
  reference-DP parity + gradcheck. Triage and retire the stage4 quorum contracts as that
  subsystem is re-implemented.
