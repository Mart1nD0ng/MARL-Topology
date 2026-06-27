# D12 — decision

**Result: KEEP (mechanism wired + verified + active; A/B deferred to D13).** A decentralized,
preference-conditioned PNA directional dynamic actor is now available as an opt-in drop-in for the MLP
actor — the LAST Phase-8–11 mechanism ported to the dynamic task.

## What changed (one variable: the dynamic actor backbone, opt-in)
- New `models/dynamic_pna_actor.py::DynamicPNAActor`: a drop-in for `DynamicRecurrentActor` (identical
  `forward(nf, ef, ei, hidden=None) -> (logits[E], h[N,H])` contract) that upgrades the single
  mean-aggregation message round to the full PNA readout (4 aggregators × 3 degree scalers, reusing the
  verified Phase-11a `pna_combine` + the gradient-safe `scatter_directional_pna`) over DIRECTIONAL
  physical-neighbour messages, with a SHARED cross-frame `GRUCell`. The deployment preference
  `ω = (ω_E, ω_L)` is a per-node PUBLIC input (broadcast), so a single policy can sweep the energy-latency
  trade-off (consumes the D11 preference primitive). `hidden=None` → memoryless on the same architecture.
- `run_dynamic_training`: builds `DynamicPNAActor` when `--dynamic-actor-arch pna` (computes the PNA
  `delta` from the train graph degrees; sets `ω` from `--dyn-pref-energy/--dyn-pref-latency`); records
  `actor_arch` + `preference_omega` in the activation. `--dynamic-actor-arch` is ORTHOGONAL to
  `--dynamic-actor` (recurrence), so all four {mlp,pna}×{recurrent,memoryless} arms are expressible.
- Default `mlp` → `DynamicRecurrentActor` (byte-identical to pre-D12).

## Decentralization (D1 — the actor IS the deployed path)
Each node uses ONLY its own (preference-augmented) features + its physical in-neighbours' DIRECTED
messages + its per-node cross-frame hidden. No global aggregate / global decoder / global argsort / node
ids / critic / evaluator at inference; `ω` is a public scalar pair (deploy-legal). The activation stays
owned by the torch-free `local_mutual_assemble` decoder (train == deploy). NaN-safe at isolated /
single-neighbour nodes (the reused std `clamp_min` + degree-0 scaler mask); logits softly bounded (the
BCSP sampler can't overflow).

## Tests (failing-first; fail on `b42b744`, pass after)
- `test_dynamic_pna_actor_signature_compatible`, `test_dynamic_pna_cross_frame_hidden_changes_output`,
  `test_dynamic_pna_preference_changes_output`, `test_dynamic_pna_backward_no_nan_isolated`,
  `test_dynamic_pna_is_decentralized`.
- Unit suite **697/0**; contract **63/0**; `--dynamic --dynamic-actor-arch pna --dyn-pref-energy 0.5
  --dyn-pref-latency 0.5` smoke exit 0 (PNA actor trains end-to-end; activation `actor_arch=pna`,
  `preference_omega=[0.5,0.5]`).

## Adversarial verification (focused single agent, 4 claims, PASS, no gaps)
1. **Decentralization**: local features + in-neighbour directed messages + per-node hidden only; ω is a
   public broadcast (no global-state leak); activation owned by the local decoder.
2. **NaN-safety**: gradient-safe std + degree-0 attenuation mask + bounded logits.
3. **Drop-in + byte-identity off**: identical I/O; default mlp byte-identical; hidden=None memoryless;
   T=1 path untouched.
4. **No bypass**: critic stays training-only; no regression (697/0).

## Scope / next
- D12 delivers the MECHANISM (correct, decentralized, active). It does NOT claim PNA beats the MLP actor —
  consistent with the static Phase-11 pattern (PNA verified-correct but opt-in, no headline gain at N≤16)
  and the D8 finding. The MLP-vs-PNA × {recurrent,memoryless} A/B (per-seed/CI + param count / runtime /
  comm degree) is part of D13.
- **All Phase-8–11 mechanisms are now ported to the dynamic task (D9 COMA, D10 SCQ, D11 chance/CVaR/Pareto,
  D12 PNA), each opt-in / verified / budget-honest.** Next: **D1** (the deferred REAL 4-RSU urban-grid
  dynamic data — gap #1, mandatory before any urban headline), then D13 (the full campaign) and D14 (docs).
