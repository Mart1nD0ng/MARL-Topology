# T4 — decision (edge-level recurrent state; task 3.4)

**Disposition: KEEP the edge_recurrent mechanism (opt-in, byte-identical default) / HONEST NEGATIVE — edge-level
recurrence does NOT recover the magnitude the per-node recurrence could not.** The edge GRU carries a per-edge
temporal hidden (the CSI dynamics are edge attributes), but it lands at the stale-echo floor (≈ node on MSE),
is slightly worse on direction, and **significantly HURTS** the recovered-psucc anchor feasibility (worse than
the node arm). The recurrence locus is not the magnitude bottleneck.

## Result (5 seeds × {random, urban}, delay-1, in-policy correction belief; `edge_recurrence_metrics.json`)

| metric | random | urban | reading |
|---|---|---|---|
| `edge_beats_floor` | −2e-4 [−2.7e-3,+2.3e-3] | +2.1e-3 [−4.8e-3,+9.0e-3] | **spans 0 both** — edge at the floor (like node) |
| `edge_vs_node_mse` | −2.4e-4 [−2.7e-3,+2.3e-3] | +1.8e-3 [−5.5e-3,+9.0e-3] | **spans 0 both** — edge ≈ node on MSE (no gain) |
| `edge_dir_acc` / `node_dir_acc` | 0.898 / 0.932 | 0.893 / 0.914 | edge slightly WORSE at direction |
| `edge_feas_gain` | **−0.25 [−0.387,−0.113]** | **−0.325 [−0.493,−0.157]** | **CI<0 both — edge significantly HURTS the anchor** |
| `node_feas_gain` | −0.219 [−0.494,+0.057] | −0.094 [−0.355,+0.168] | node spans 0 (T3) — edge is WORSE than node |

## What T4 establishes (Claim Cards)
1. **Edge-level recurrence does not beat node-level recurrence or the floor** (`edge_vs_node_mse` and
   `edge_beats_floor` both span 0). Moving the recurrent state from per-node to per-edge — the locus fix for
   task 3.4 — recovers no additional CSI magnitude.
2. **More temporal capacity → worse decisions when the magnitude signal is absent.** The edge GRU adds per-edge
   memory + parameters, yet its recovered psucc HURTS the anchor significantly (feas_gain CI<0), *more* than the
   node arm. Extra capacity overfits the per-edge stale trajectory and perturbs the anchor ranking further —
   confirming the bottleneck is the *signal*, not the model capacity.
3. **Third independent confirmation of the magnitude limit.** T1 (standalone regressor), T3 (in-policy node
   belief), and T4 (in-policy edge belief) all land at the floor: the decision-critical magnitude (fast-fading
   residual) is not recoverable from the leak-free features (stale channel + current geometry), regardless of
   the recurrence locus.

## Honest scope / caveats (Contract v4 §5/§14)
- NEGATIVE on task 3.4: the edge locus is not the fix. The `edge_recurrent` mechanism is correct and load-bearing
  (per-edge hidden [E,H] carried across frames, belief changes when carried — test-verified) but does not help.
- **LSTM not run** (task 4 raised it as optional): an LSTM differs from the GRU by its gating / separate
  cell-state structure, not only its parameter count, so the GRU result does not *strictly* bound LSTM behavior.
  The deferral therefore rests on the **feature** limit, not a capacity argument: T1–T4 show the decision-critical
  magnitude is absent from the leak-free features (the standalone T1 regressor, the node T3 belief, and the edge
  T4 belief all land at the floor), so no temporal architecture — GRU or LSTM — can recover a signal that is not
  in its inputs. An LSTM would change how the history is gated, not what information the history contains. This
  is a reasoned deferral grounded in the feature limit, not a capacity hand-wave.
- Edges are a fixed candidate set per scene (complete graph, verified stable across frames), so the edge hidden
  is carried by index; this is exact, not an approximation.
- `edge_recurrent=False` (default) is byte-identical to the frozen R1–R8 / T2 / T3 actor (edge modules appended
  last, allocated only when enabled) — the whole prior suite stays green; the T0 tripwire (default has no edge
  recurrence) is preserved, and the T4 tests assert the enabled structure.

## Effect on the campaign plan
- **KEEP** `edge_recurrent` opt-in (default off). It is not the lever; the node correction belief (T3) is the
  reference temporal module.
- **The binding limit is now firmly the FEATURES, not the architecture** (activation T2, target T3, recurrence
  locus T4 all tested; none recover magnitude). Two levers remain:
  - **T5 (uncertainty):** stop trying to recover the magnitude point-estimate; predict its *uncertainty* and
    only commit a correction where confident (feed the anchor a calibrated psucc). This changes the objective
    from "recover magnitude" to "know when you can't" — the one angle T1–T4 did not test.
  - **Env temporal-hidden-features (task-1 fallback):** if T5 also stalls, add leak-safe predictable structure
    to the env so the magnitude becomes recoverable (the true-CSI oracle from T1 shows the *headroom* is real —
    it is realizability, not existence, that fails).

## Verification (Ultracode adversarial Workflow `w82f3rn07`)
4-lens + synthesis. **Synthesis: PASS, 0 MAJOR — safe to commit.** back-compat **PASS** (empirical: every shared
param bit-identical OFF vs ON, only edge modules added; forward/belief OFF==ON; edge path unreachable at default);
single-variable **PASS** (node vs edge differ ONLY in the recurrence locus; edge GRU load-bearing — carrying the
[E,H] hidden changes the belief; edge_ids ORDER verified stable across frames so carry-by-index is exact);
leak-free **PASS** (edge belief input = stale ef + spatial embed + velocity; true CSI only label/metric);
honest-scope **PASS** (numbers match; over-generalization avoided — limit is THIS signal, T5 + env-features open).
1 MINOR applied: reground the LSTM deferral on the feature limit (an LSTM differs by gating, not just capacity).
1 hardening applied: the back-compat test now asserts full state_dict bit-identity.

## Next
**T5 — physical residual model + uncertainty head:** predict a correction AND its variance; shrink the magnitude
by confidence so the recovered psucc only deviates from stale where the model is confident. One variable vs the
T3/T4 mean-only correction belief.
