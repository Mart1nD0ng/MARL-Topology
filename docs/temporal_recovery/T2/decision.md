# T2 — decision (non-saturating activation)

**Disposition: KEEP (enabling fix, opt-in, byte-identical default).** T2 adds a linear skip to the residual
activation so the temporal signal's gradient never vanishes at the tanh rail. Honestly scoped: this is a
logit/gradient-level enabling fix (like R1), not a topology conversion.

## What T2 delivers
- `BeliefResidualActor(residual_leak=λ)`: `z = z_max·tanh(raw/z_max) + λ·raw`. Gradient `sech²(·)+λ ≥ λ > 0`
  never vanishes → the recurrent signal reaches the ACTED logit regardless of `raw` (T0 defect 2 fixed for
  `λ>0`). `λ=0` (default) is **byte-identical** to the frozen R1–R8 head — zero blast radius (prior suite green).
- The campaign's actor (T3+) uses `λ=0.1`.

## Evidence
- **Failing-first / load-bearing** (4 tests): actor applies the leaky map (`λ>0`); gradient never vanishes at
  `|raw|=100` (vs pure tanh → <1e-3); `λ=0` byte-identical.
- **Effect-on-Decision** (saturation regime, deterministic): recurrent-vs-memoryless `logit_delta` — tanh
  **< 0.05** (behaviorally inert) vs leaky **> 0.15** (>5×). The activation was suppressing the temporal signal;
  the skip restores it.
- **Real-data pilot** (`mechanism_activation.json`, delay-1 stale urban, untrained, same init): raw is large on
  the untrained actor (`raw_abs_mean ≈ 3449`, tanh 37.5% railed — the Q14 saturation regime), and the leaky head
  carries **≥** the recurrent→logit signal (`0.0164 ≥ 0.0149`, `leak_carries_ge_signal=True`, a marginal +10%).
  **`recurrent_action_delta_mean = 0.0` for BOTH heads** on this untrained pilot — T2's real-data effect is
  logit-level ONLY (zero action change); the action/topology conversion is entirely T3/T6.

## Honest scope / caveats (Contract v4 §5/§14)
- ENABLING fix only. T2 does NOT change the topology/feasibility by itself (no A/B headline; edit_rate/topology
  conversion is T6; the direction signal is T3). On the untrained actor the recurrent effect on `raw` is tiny
  relative to `raw`'s magnitude (the head is dominated by `ef`, not the GRU — the "actor head suppresses the GRU"
  point, task 3.3), so the activation's marginal effect there is small; the structural GRU-contribution lever is
  T4 (edge recurrence).
- On the TRAINED actor the R1 raw-L2 keeps `raw` small (≈4), so both activations are near-linear there anyway;
  T2's payoff is ROBUSTNESS — the gradient still flows when the T3 correction signal drives `raw` up — and
  removing the dependence on the raw-L2 band-aid to avoid the rail.
- Default `λ=0` retained for byte-identical back-compat; the T0 saturation tripwire remains valid for that
  default (documents the retained behavior), and the T2 tests assert the `λ>0` fix.

## Verification (Ultracode adversarial Workflow `w7or4jsv2`)
3-lens + synthesis. **Synthesis: PASS, 0 MAJOR — cleared to commit.** back-compat **PASS** (leak=0 bit-identical
to pure tanh over 1000+ values incl. saturation/±0/1e±30; only divergence at the UNREACHABLE raw=±inf where
0·inf=NaN); gradient-correctness **PASS** (dz/draw = sech²+leak ≥ leak; on the real decode path via
`residual_action.py`; tests fair); honest-scope **MINOR** (no overclaim; nit: state the pilot's action_delta=0).
3 MINOR fixes applied: (1) forward SHORT-CIRCUITS leak=0 → byte-identical even at raw=±inf; (2) byte-identical
test tightened to `torch.equal` + an inf-boundary assertion; (3) decision states `recurrent_action_delta_mean=0`
on the real pilot (logit-level only).

## Next
**T3 — belief target → correction:** the highest-value stage, directly validated by T1 (`C−B` CI>0, `B−A` CI<0).
Change `L_CSI` from the absolute current-`p_t` target to a stale→current CORRECTION target (echo = zero-baseline)
so stale-echo is no longer loss-optimal, with a directional signal; acceptance = belief beats the stale-echo
floor AND the recovered-psucc anchor improves (ranking, not MSE). The T3 actor uses `residual_leak>0` (this T2 fix).
