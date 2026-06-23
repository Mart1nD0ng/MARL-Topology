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

---

## 9. Phase-1a update (2026-06-22) — safe quorum spec landed

The `PBFTQuorumSpec` half of fact (6) is **done** (`protocol/quorum_spec.py`,
`classic_exact` / `safe_generalized`, `q = ⌊(n+f)/2⌋+1`, asserts `2q−n>f`, `q≤n−f`,
`n≥3f+1`; wired into both PBFT configs with default `safe_generalized`). The unsafe
classic-`2f+1`-at-n>3f+1 bug is pinned by `tests/unit/test_pbft_quorum_spec.py`.
**Protocol-incompatible:** all N>3f+1 reliability numbers prior to commit shift down
(safe quorum is harder); old N>3f+1 headlines are retired. Suite: 289 failed / **512
passed** (was 289/503), **zero new failures**. One Stage-2.8 forbidden-code gate was
updated to whitelist the reviewed `quorum_spec.py` (a lineage gate — it also bans
`byzantine`/`view_change`, so it is **slated for full retirement in Phase 1b** when
`fault_set_robustness.py` forces it). **Still open in fact (6):** the `REMOVE_LARGEST`
per-phase fault filter (violates the single-fixed-`B` requirement, Spec §4.7) → Phase 1b.

**Phase-1b update:** the correct fixed-`B` robustness primitive is landed
(`protocol/fault_set_robustness.py` — `C_robust = min_{|B|≤f} C(x;B)`, exact enumeration,
hard-min for eval / softmin for training, budget guard, f=0 parity vs the validated
cascade; `tests/unit/test_fault_set_robustness.py`). It is **not yet wired** into the
production evaluator — exact enumeration is infeasible at the trunk's N=24,f=7 (C=346k),
so the wiring iteration must add a cost-managed worst-case (greedy fixed-`B` above a
budget). Until then `REMOVE_LARGEST` remains the production fault model and the reward
signal is unchanged. Suite 289 fail / 520 pass; zero new failures.

**Phase-1b-wire update (REVISE):** a wiring attempt was reverted on evidence. (1) A
modeling bug was found and **fixed**: the primitive zeroed faulty *primaries*, capping
`C_robust ≤ (n−f)/n` (< τ at small n → τ unreachable, which broke 10 scenario/feasibility
tests); corrected to average over the honest initiators only (deferred view-change),
verified perfect-links → 1.0. (2) The wiring is O(n⁴) (11× unit-suite cost even at f=1)
and shifts reliability enough to require re-calibrating the stage31 scenario τ-gradient —
both **deferred** to Phase 1b-wire-v2. Production still uses `REMOVE_LARGEST`; the
corrected primitive now has a greedy fallback + strategy metadata. Suite 289 fail / **524
pass**, zero new failures.

**Phase-1c update — Torch quorum-tail landed.** `protocol/torch_quorum_tail.py`
(`torch_quorum_tail`) — the reference DP as a batched, autograd-differentiable Torch op.
Verified: reference parity <1e-12, small-N exact vs brute force, `gradcheck`, and the
analytic Spec §4.5 sensitivity `∂Q/∂p_i = P(exactly q−1 of the others)` <1e-10
(`tests/unit/test_torch_quorum_tail.py`). Standalone submodule (not in `protocol/__init__`,
so the base package stays Torch-free); two stage8/stage9 torch-purity sub-tests were
updated to allow that one spec-mandated file. Suite 289 fail / **531 pass**, zero new
failures. This completes the Phase-1 *primitives*; the remaining Phase-1 item is the
deferred 1b-wire-v2 (fixed-B cost + scenario re-calibration).

**Phase-2 update — route/relay (opt-in correct semantics).** Confirmed the double
multi-hop layer: the evaluator BFS-routes every pair (so the record is already end-to-end
multi-hop), then `_multi_hop_reach` relays again at `relay_hops>1`; even at the production
`relay_hops=1` the A–B–C topology wrongly gives P(A→C)>0. `_multi_hop_reach` is itself the
correct one-hop→relay DP — only its input is wrong. Fix landed **opt-in**
(`build_pbft_message_matrices_from_network_records(one_hop_relay=…)`, default off): when on,
the matrix is built from direct-link records only, so the relay DP is the single multi-hop
layer (A–B–C regression passes; `tests/unit/test_route_relay_semantics.py`, 1 xfail pins the
default-mode bug). Remaining: latency-aware relay (deadline propagation). Suite 289 fail /
**538 pass** / 1 xfail, zero new failures.

> **⚠️ Strategic convergence (owner decision approaching).** The corrected environment math
> (1b-wire-v2 fixed-`B` fault model, 2 one-hop relay, and foreseeably 3/4) all break the
> **same** stage31 scenario τ-gradient calibration, so each is landed *verified + opt-in +
> inert*. Activating them requires **one heavy/owner-gated recalibration campaign** (rebuild
> the scenario dataset + re-tune the τ-gradient under the corrected math) — that is the real
> P0–P4 exit gate. Recommendation: land the Phase 3/4 primitives, then run **one** batched
> recalibration that flips all corrected-environment flags together, rather than piecemeal.

## 10. Recalibration (owner-approved 2026-06-22) — step 1: measured, low-impact

Owner chose **Recalibrate + activate**. Step 1 added configurable knobs to
`Stage21ObjectiveStackConfig` — `fault_model` (`remove_largest` | `fixed_set`) and
`one_hop_relay` — wired through both evaluators, **default off → byte-identical** (zero new
failures). **Impact measurement** (`logs/recalib_impact.txt`, 12 scenes N∈{8,12,16}): the
corrected math (`fixed_set` + `one_hop_relay`) at **`relay_hops≥2`** gives essentially the
**same feasibility as baseline** (0.731 vs 0.733); `relay_hops=2` suffices; only `relay_hops=1`
(direct-only) collapses. **The recalibration does not collapse feasibility** — the Phase-1b-wire
collapse was the τ-cap bug (since fixed) + relay_hops=1, not the model. Recommended production
config: `fault_model="fixed_set", one_hop_relay=True, relay_hops=2`. Next: flip the default,
wire the timeout latency + tri-state labels, migrate absolute-number tests, multi-seed headline.

**Recalibration COMPLETE (2026-06-23).** All corrected env-math is wired into the production
regime (2a reliability/relay, 2b timeout latency, 2c tri-state labels), the pipeline validated
end-to-end, and a corrected in-range 5-seed headline established on a 144-scene rebuilt dataset:
RL keep-best **0.624** vs SA ceiling **0.655**, **margin −0.031, CI95 [−0.086, +0.024]**
(final-update +0.007 [−0.021, +0.036]) — both CIs span 0 → the oracle-free cold-start
decentralized learner **matches the SA oracle in-range** under correct math (neither beats nor
trails). The legacy "+0.177 at N=24" is **retired** (`retired_due_to_protocol_metric_change`);
it was out-of-range under the incorrect math. The corrected **out-of-range N=24** comparison is
the open question (gated on a `fixed_set` O(n⁴) cost optimization + a heavy build). README +
AGENTS headlines updated to this honest result.

---

**Phase-3 update — tri-state solvability (opt-in).** Confirmed the binary `feasible_exists`
conflates a *finite-search miss* with *infeasible* (stage31 sets it `False` when no candidate
reaches τ — violates #11). New `src/marl_topology/solvability/` package:
`classify_solvability` (witness_feasible iff LB≥τ; certified_infeasible iff a **proven** UB<τ;
else unknown) + `solvability_from_finite_search` (a finite search yields only W or U, never I)
+ split-isolated `WitnessMemory` (`merge_from` refuses test→train, S5.3). 11 tests. Production
`feasible_exists` untouched; the relabel + the proven optimistic UB (S5.2) are part of the
batched recalibration. Suite 289 fail / **549 pass** / 1 xfail, zero new failures.
