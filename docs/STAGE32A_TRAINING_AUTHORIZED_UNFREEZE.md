# Stage 32a: Training-Authorized Cleanup and Real MAPPO Reuse

This document records the owner-approved Stage 32a corrections to the Stage 32
production training, prompted by three owner findings:

1. Stage 32 used a hand-rolled REINFORCE loop instead of the project's designed
   MAPPO trainer.
2. The deep-learning code used a placeholder evasion
   `getattr(loss, "back" + "ward")()` instead of a real `loss.backward()`.
3. The graphs were too small to exercise the GNN actor.

## Owner decision

- Training execution is owner-authorized (Stage 32 option B). The design-era
  convention that forbade the literal `backward(` anywhere in `src` (a proxy for
  "no training execution yet", enforced by ~33 frozen contract gates across
  Stages 2-9) is therefore **lifted**.
- Decision on scope: `full unfreeze` - remove the `backward(` ban from the
  src-wide gates and use real `loss.backward()`. Architecture bans (COMA,
  Transformer, recurrent PPO, GRU/LSTM where applicable) and the v5-migration ban
  remain in force.

## Change 1 - real MAPPO trainer reuse (finding 1)

Stage 32a drives the project's real clipped-PPO / MAPPO loss
(`src/marl_topology/training/mappo/losses.py::clipped_policy_value_loss`) instead
of the hand-rolled single-step REINFORCE. The training loop now performs a proper
rollout (budget-aware sampler, recorded old log-probabilities and standardized
feasibility-first returns), then PPO update epochs that recompute new
log-probabilities, apply the clipped surrogate objective with the centralized
graph-critic value loss and entropy bonus, and early-stop on approximate KL > 0.03
(the Stage 25 protocol bound). The actor remains the message-passing GNN
(`local_message_passing_gnn_edge_scorer_v2`) and the critic the repaired graph
value critic (`centralized_message_passing_graph_value_critic_v1`).

## Change 2 - honest `loss.backward()` (finding 2)

All 21 `getattr(loss, "back" + "ward")()` / `getattr(loss, "backward")()`
evasions across the deep-learning modules were replaced with real
`loss.backward()` / `loss_result.total_loss.backward()`. The src-wide banned-term
gates had `backward(` removed from their banned lists; layer-scoped gates
(scenario/geometry/channel/network/protocol specific-file scans) keep their
`backward(` ban so non-training layers still may not perform backward.

Files cleaned: `training/mappo/trainer.py`, `training/mappo/stage25_pilot.py`,
`training/mappo/stage28_repaired_critic_pilot.py`,
`training/critic_pretrainer.py`, `training/critic_repair_trainer.py`,
`training/ppo_pilot.py`, `training/supervised_actor_trainer.py`,
`training/temporal_actor_ablation.py`, `training/stage19_supervised_actor_stack.py`,
`training/policy_gradient/pilot_runner.py`, `training/stage31_readiness.py`,
`training/stage32_production_training.py`, `evaluation/stage21_fair_evaluation.py`.

The other string-concatenation patterns in `src` (e.g. `"co"+"ma"`,
`"trans"+"former"`, `"checkpoint_"+"path"`, `"directed_"+"outgoing_v1"`) are NOT
evasions to be removed: they live in scanner/diagnostic code that must reference
forbidden terms without containing them literally, so the scanner does not flag
itself. They remain.

## Change 3 - larger graphs (finding 3)

The procedural generator's `node_count_choices` widened from `(4, 5, 6, 7)` to
`(5, 6, 7, 8, 9, 10)`, producing larger and denser candidate graphs so the GNN
actor's relational message passing is exercised against the per-edge MLP.

## Boundaries preserved

tau fixed at 0.9; feasibility-first surrogate and budget-aware sampler kept;
mean-field PBFT kept; Dec-POMDP locality preserved; no v5 migration; COMA,
Transformer, and recurrent PPO remain out of scope; no model-weight checkpoints
(the no-checkpoint discipline, i.e. the `torch.save` ban, is unchanged).
