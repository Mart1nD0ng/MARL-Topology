# Stage 25 Small-Scale MAPPO Training Pilot

Stage 25 executed `stage25_pilot_base_config` as the only formal pilot
protocol. No diagnostic probes were run, no hyperparameters were tuned, and no
checkpoint was written.

## Controlled System

- Controlled object: fixed-protocol MAPPO fine-tuning loop for the selected
  physical-link actor, sampler, assembler, centralized critic, evaluator, and
  frozen reward surrogate.
- Desired state: prove whether MAPPO improves the supervised GNN baseline on a
  held-out eval split without materially degrading reliability.
- Sensors: train/eval metrics, pass/fail safety gate, manifest validation,
  source boundary tests, reward-surface checks, and visualization artifacts.
- Actuators: bounded MAPPO updates and manifest-approved report writes.
- Disturbances: stochastic seeds, limited source contexts, sampler projection,
  critic value fit, and reward variance.
- Coupling: policy gradients alter actor scores; actor scores alter proposal
  topology; topology projection alters objective metrics and surrogate reward.

## Execution Summary

Artifact root:
`result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config`

The run used three seeds: `2501`, `2502`, and `2503`. Two seeds completed the
fixed protocol. One seed stopped early on the declared tau-feasible drop safety
condition. The pass gate requires at least two completed seeds, so the seed gate
passed while retaining the stopped-seed risk as Stage 26 review input.

The train/eval split used disjoint unique source rows expanded cyclically into
16 train slots and 8 eval slots. The supervised GNN baseline, MAPPO fine-tuned
actor, projected greedy baseline, projected full graph baseline,
objective-aware teacher, and diagnostic sparse/random/empty records were
evaluated on the fixed eval surface.

## Eval Comparison

Mean supervised GNN eval metrics:

- tau-feasible rate: `0.6276041667`
- violation rate: `0.3723958333`
- consensus success probability: `0.6276041667`
- latency: `0.0042705665`
- energy: `0.0072100000`
- top proposal rejection rate: `0.0434027778`
- surrogate reward: `-33.3572951004`

Mean MAPPO fine-tuned eval metrics across completed seeds:

- tau-feasible rate: `0.5963541667`
- violation rate: `0.4036458333`
- consensus success probability: `0.5963541667`
- latency: `0.0041814993`
- energy: `0.0070125000`
- top proposal rejection rate: `0.0477430556`
- surrogate reward: `-35.8234434768`

Eval deltas, MAPPO minus supervised:

- tau-feasible rate: `-0.03125`
- violation rate: `+0.03125`
- latency: `-0.0000890672`
- energy: `-0.0001975000`
- top proposal rejection rate: `+0.0043402778`
- surrogate reward: `-2.4661483765`

## Gate Result

Stage 25 PASS was achieved.

The MAPPO actor improved latency and energy while reliability degradation stayed
inside the 0.05 absolute bound. Surrogate reward and projection rejection did
not improve, so the result is not evidence for scale-up. It is evidence that
the fixed small-scale MAPPO pilot can produce objective-side improvement under
the current baseline and safety checks.

## Boundaries Held

- Active sampler remained `physical_plackett_luce_top_k_sampler`.
- Action semantics remained `undirected_physical_link_v1`.
- Reward weights and tau requirement were unchanged.
- No final tau was selected.
- No COMA, Transformer, GRU/LSTM, or recurrent PPO was implemented.
- No checkpoint was written.
- No scale-up training was run.
- No v5 code was modified or migrated.
- Artifact writes stayed under the manifest-approved report root.

## Residual Risk

One seed stopped on tau degradation, reward worsened despite latency and energy
improving, and source context diversity remains limited. Stage 26 must review
scale readiness and failure modes before any larger training run.
