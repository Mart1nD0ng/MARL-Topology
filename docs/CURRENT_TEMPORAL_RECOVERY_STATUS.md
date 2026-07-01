# CURRENT_TEMPORAL_RECOVERY_STATUS — HEAD vs the Temporal-Recovery target

> Campaign: **Temporal Recovery** (the T-series). Owner `/loop`, dynamic mode. Governed by
> `docs/MARL-Topology-Development-Contract-v4.md` (Claim-Path evidence) + v3 (still binding) and the
> Belief-Guided Residual PPO TechSpec (`docs/MARL-Topology-Belief-Guided-Residual-PPO-TechSpec.md`, §2–§3
> define the CSI-belief formulation this campaign redesigns). Commit per stage to `decentralized-marl-trunk`;
> **push = owner's call (do NOT push unprompted).** Ultracode ON → Workflow for adversarial verification.
>
> HEAD at campaign start: `c74a71d` (Belief-Guided Residual PPO campaign COMPLETE + pushed). This campaign is a
> **direct successor** that reopens that campaign's chain-1 (temporal/belief), which R1/R2 declared
> non-load-bearing — but for two concrete, *architectural* reasons (below), not because recovery is impossible.

---

## 0. Why this round — the exact failure we attack

The prior campaign ended at an honest negative: **stale CSI degrades the deployable `local_hysteresis` anchor**
(R8: urban 0.815→0.650, drop **0.165 CI[+0.118,+0.212]**; random 0.055 CI[+0.009,+0.101]) — and no learned arm
recovered it. This drop is *headroom*: if a temporal module could recover the current channel from stale
history, the anchor's own decisions would improve toward the non-degraded ceiling, and MARL could finally beat
the degraded anchor. The prior chain-1 failed here for **two fixable architecture defects**, confirmed by the
T0 audit (file:line below), plus two structural mismatches:

1. **Direct-`p_t` belief target → stale-echo is the strong local optimum.** `belief_target_logits`
   ([csi_belief.py:58](src/marl_topology/training/csi_belief.py:58)) is the *absolute* true-current logit;
   `csi_belief_loss` ([:88](src/marl_topology/training/csi_belief.py:88)) is Huber(belief_logit, absolute_true).
   For any edge whose channel barely moved between t−1 and t (stale ≈ current), the loss-optimal output is
   `logit(stale_input)` — i.e. **echoing the observation is optimal**. R2 confirmed: the head learned the echo
   (correction-direction corr ≈ −0.035), never beating the stale-echo floor. → **T3: predict a correction
   (stale→current delta), not the absolute `p_t`.**
2. **`tanh` saturation on the decision path.** The residual logit is
   `z = z_max·tanh(raw/z_max)`, z_max∈[2,4] default 3 ([belief_residual_actor.py:94](src/marl_topology/models/belief_residual_actor.py:94)).
   The decode thresholds on `sign(residual_logit) = sign(raw)` ([dynamic_baselines.py:59](src/marl_topology/training/dynamic_baselines.py:59)),
   so the `tanh` magnitude-squash + the R1 raw-L2 pull flatten the GRU's contribution to the *acted* logit
   (R1: action_delta stayed **0** — recurrent vs memoryless topology bit-identical). → **T2: a non-saturating
   activation on the recurrent→logit path.**
3. **Per-node recurrence for an edge-attribute quantity.** The only recurrent cell is a **per-node**
   `GRUCell(hidden,hidden)`, hidden `[N,H]` ([belief_residual_actor.py:41](src/marl_topology/models/belief_residual_actor.py:41));
   edges read it only via endpoint states `hu,hv`. But CSI (psucc/latency/energy, cols 0–3) is an **edge**
   attribute ([graph_payload.py:61](src/marl_topology/data/graph_payload.py:61)). The dynamics we must track are
   per-edge; the state that tracks them is per-node. → **T4: edge-level recurrent state/hidden.**
4. **Point prediction only.** The belief head outputs a single mean logit — no uncertainty. Under partial/stale
   observation the *confidence* of a recovered value is decision-relevant (a low-confidence recovery should not
   flip a link). → **T5: predict uncertainty (variance), not just the mean; physical residual model.**

**Before building any of this, T1 must prove the ceiling is realizably reachable — the gate.**

---

## 1. The gate (T1) — nowcasting oracle, NOT the old anticipation oracle

There is a **prior temporal-oracle negative** in the repo that must NOT be conflated with this one (Contract
v4 §5/§14). `logs/diagnose_anticipation_value.py` (URBAN_V2X_RESEARCH_LOG "Thread 1 — Temporal actor") found a
**clairvoyant policy seeing the future frame t+h does +0.000 vs myopic** — because feasibility loss under motion
is *geometric* (a vehicle drives out of range; no topology fixes that). That is a **forecasting** result.

**This campaign asks a different question — nowcasting/filtering:** the actor is *denied the true current CSI*
(it sees stale t−1). R8 proved this denial alone costs the anchor 0.165 (urban). The open question is whether a
*realizable* predictor can recover the current channel from stale t−1 + leak-free geometry, or whether only a
clairvoyant oracle (that literally reads frame t) can. **T1 answers exactly this**, exploiting that the anchor
is a deterministic function of psucc (col 0, [dynamic_baselines.py:59](src/marl_topology/training/dynamic_baselines.py:59)):

| Arm | psucc the anchor sees | expected (from R8) | meaning |
|---|---|---|---|
| **A floor** | stale (t−1) | urban 0.650 / random ~0.28 | degraded baseline |
| **B current-belief** | R2 direct-`p_t` head prediction | ≈ A (echoes stale) | where the *current* model sits |
| **C physics-recovery** | regression on [stale, rel_vel, dist, Δdist] → current | **?** | realizable-info ceiling |
| **D oracle ceiling** | true current (t) | urban 0.815 / random ~0.34 | clairvoyant upper bound |

- **Ceiling = D − A** (= R8 stale_drop). Real and large (urban 0.165). Confirms recoverable info *exists in
  principle*.
- **Realizable recovery = C − A.** The decisive new probe. If **C − A > 0** and C < D → there is realizable
  headroom that a better belief model (T2–T5) can chase → **proceed**. If **C − A ≈ 0** → the stale→current
  change is stochastic/unpredictable-from-geometry → per your task-1 fallback, **improve the env's temporal
  hidden features** (add leak-safe predictable structure) rather than the model.
- **MSE landscape** (diagnostic): stale-echo MSE vs physics-regression MSE vs current-belief MSE vs 0. Separates
  the *information* question from the *architecture* question (R2 could not — it only measured the flawed model).

T1 is 5-seed × {random, urban}, true evaluator scores the produced topology, 0 eval at decision.

---

## 2. Mechanism-Path Matrix (Contract v4 §2) — HEAD snapshot (audited T0)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | belief_residual_actor / csi_belief | status @ HEAD | defect | target |
|---|---|---|---|---|
| CSI-belief target = **absolute current `p_t`** | `belief_target_logits` + `csi_belief_loss` | ACTIVE_IN_LOSS | stale-echo optimum (task 3.1) | **T3** correction target |
| **`tanh` residual saturation** on decision path | `forward` `z=z_max·tanh(raw/z_max)` | ACTIVE_IN_EVAL | squashes GRU→acted-logit; action_delta 0 (task 2/3.3) | **T2** non-saturating |
| **per-node** `GRUCell` recurrence | `self.gru`, hidden `[N,H]` | ACTIVE_IN_EVAL | edge dynamics tracked by node state (task 3.4) | **T4** edge-level state |
| edge-level recurrent state / edge hidden | — | **NOT_PRESENT** | — | **T4** |
| physical residual model (physics + neural) | — | **NOT_PRESENT** | pure-neural extrapolation | **T1 probe / T5** |
| uncertainty (variance) head | belief head outputs mean only | **NOT_PRESENT** | no confidence signal (task 4.3) | **T5** |
| CSI-recovery → anchor injection (oracle) | — | **NOT_PRESENT** | — | **T1** |
| leak-free motion features (`rel_vel`, Δdist) | `leak_free_motion_features` | CALLABLE (belief extra) | present but did not rescue R2 | reused T1/T5 |
| stale/partial CSI overlay | `CsiObservationModel` (delay/partial) | ACTIVE_IN_EVAL (opt-in) | — | the regime under test |

**Audit takeaway:** the belief/recurrence machinery is real, leak-free, and in-loss — but its *target*
(absolute `p_t`), its *activation* (`tanh` on the acted logit), and its *state locus* (per-node for per-edge
dynamics) each make recovery structurally hard. This campaign fixes those, gated by T1's proof that recovery is
realizable at all.

---

## 3. T-stage plan (one variable per stage; T1 result reshapes T2–T6)

| T | deliverable | exit condition |
|---|---|---|
| **T0** | freeze + audit + this status doc + failing-first audit tests pinning the 4 defects | 4 defects pinned in code as load-bearing tests ✓ |
| **T1** | **oracle upper-bound (GATE):** 4-arm anchor (stale/belief/physics/true) + MSE landscape, 5-seed | C−A>0 → recovery realizable → proceed; C−A≈0 → env-feature fallback |
| T2 | activation redesign (non-saturating recurrent→logit) | recurrent vs memoryless topology_delta moves (action_delta ≠ 0) |
| T3 | belief target → correction (stale→current delta) + directional signal | belief MSE beats the stale-echo floor (CI) |
| T4 | edge-level recurrent state + unblock actor head | edge-state ablation changes belief MSE / action; GRU→LSTM if runtime allows |
| T5 | physical residual model + uncertainty head | physics-residual beats pure-neural belief; calibrated variance |
| T6 | integrated temporal A/B vs degraded stale anchor, 5-seed random+urban | feasibility (temporal − stale_anchor) CI vs 0 |
| T7 | docs close-out + 中文 analysis+data report | all stages have Claim Card + decision.md; report delivered |

---

## 4. Hard constraints (unchanged; re-affirmed)

Deploy fully decentralized, 0 eval at decision; the deployed actor **never** reads true current CSI / critic /
evaluator / central solver / global decoder (it observes STALE CSI in the target regime). Training MAY use true
current CSI as a supervised **label** only (mark training-only). True PBFT C/E/L always on the current real
channel + closed-form quorum tail; proxies/aux never replace the true evaluator. Imitate only anchor-relative
positive-gain LOCAL edits, never the full oracle topology. Failing-test-first; held/test never for
teacher/checkpoint/training. Every conclusion carries a Claim Card + Mechanism-Path Matrix; report whether a
mechanism changes the final action/topology, not just the forward pass; negatives are path-specific. ≥5 seeds +
CI for any headline. Commit per stage, do NOT push.

---

## 5. Progress ledger

- **T0 (this commit):** audit above; `docs/temporal_recovery/T0/`; `tests/unit/test_temporal_recovery_T0_audit.py`
  pins the 4 defects (direct-`p_t` target / `tanh` saturation / per-node-GRU vs edge-CSI / no edge recurrence)
  as load-bearing regression tripwires that flip when T2–T4 fix them. Suite green. — *in progress this commit.*
