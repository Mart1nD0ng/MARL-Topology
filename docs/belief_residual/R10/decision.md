# R10 — decision (honest close-out, pure docs) — CAMPAIGN COMPLETE

**Result: the Belief-Guided Evidence-Gated Residual PPO campaign is COMPLETE (R0–R8 + R10).** R10 is a pure-docs
close-out — no code, no new experiments; the suite is unchanged (817/0). It records the campaign's central
finding and 4-chain diagnosis across the repo's authority docs.

## What R10 wrote (docs only)
- `docs/CURRENT_BELIEF_RESIDUAL_STATUS.md` — a **🏁 CAMPAIGN COMPLETE** banner (top) + the **§5 FINAL 4-CHAIN
  DIAGNOSIS** ledger (where the method's value is, and where it stops).
- `docs/URBAN_V2X_RESEARCH_LOG.md` — the **BELIEF-GUIDED RESIDUAL PPO CAMPAIGN SUMMARY** (tail): the per-stage
  ledger R0–R8 with commit hashes, the central result, the 4-chain diagnosis, and the per-stage Workflow IDs.
- `docs/CURRENT_HEAD_STATUS.md` — a **🏁 Belief-Residual COMPLETE** banner above the POMDP-QP-FAR one (the
  latest campaign; the 4th honest negative, now precisely localized).
- `AGENTS.md` — added the Belief-Residual campaign as the 4th bullet in the superseding-campaigns ledger; the
  binding limit sharpened to "the deployable PRECISION of the beneficial-edit direction signal".

## Central result (the campaign conclusion)
**No deployable arm beats the `local_hysteresis` anchor on the CURRENT channel (R6/R7) OR the STALE channel
(R8) at N≤16.** The beneficial-edit direction signal genuinely EXISTS (R4) and is locally RANKABLE (R5 — the
campaign's first deployable-learning positive), but does NOT CONVERT into a deployed feasibility/return gain.
The stale-CSI premise is CONFIRMED (stale degrades the anchor, urban −0.165 ≈ Q14 0.80→0.66) but the method
does not repair it. **Binding limit = the deployable PRECISION of the direction signal** — not its existence,
learnability, the trainer (R3/R7), the gate (R6/R8), the temporal/belief chain (R1/R2), or a leak.

## 4-chain diagnosis (full table in `CURRENT_BELIEF_RESIDUAL_STATUS.md` §5)
1. temporal/belief ✗ non-load-bearing (R1 logit-only; R2 belief no-op vs stale-echo floor).
2. trainer ✓ correct but no direction (R3 residual PPO eliminates collapse but == anchor; R7 adaptive anchor-KL
   also == anchor — the fixed penalty was not the cause).
3. direction supervision EXISTS (R4) → RANKABLE (R5) → NOT CONVERTIBLE (R6 current + R8 stale).
4. evidence gate ✓ correct/safe/load-bearing but == anchor (no `tau_edit` beats it; firing is net-negative).

## Provenance / honesty
Every stage R1–R8 carries a Claim Card + Mechanism-Path Matrix + failing-first load-bearing test +
Effect-on-Decision test + a path-specific negative + a per-stage adversarial multi-lens Workflow (all confirmed
load-bearing + leak-free with independent reproductions). All 5-seed results have CIs; no headline rests on a
pilot or a span-0 mean. The 1-seed R8 pilot (+0.0625) is explicitly labeled noise. Commits R0–R8 + R10 on
`decentralized-marl-trunk`, **NOT pushed** — the push is the owner's decision (asked at close-out).

## Decision
**CAMPAIGN COMPLETE.** No R9 consolidation run (there is no positive to consolidate — the 5-seed A/Bs at each
stage already establish the negative with CIs). The push decision is deferred to the owner (AskUser). Open
frontier: N≥24 (cheaper exact-fault evaluator) and a higher-precision local direction signal (the only lever
the campaign leaves open).
