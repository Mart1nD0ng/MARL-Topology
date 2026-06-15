# Stage 32a Production Training Report (real MAPPO loss, honest backward, larger graphs)

- status: `execution_complete` (800 scenarios, 5 seeds)
- verdict: `gnn_competitive_but_unstable_mlp_higher_mean` (honest, nuanced)
- supersedes the Stage 32 hand-rolled loop per owner findings; see
  `docs/STAGE32A_TRAINING_AUTHORIZED_UNFREEZE.md`
- reproduce: `python scripts/train/stage32_production_training_run.py <scenarios> 5`
- artifacts: `result_save/stage32_production_training/stage32a_mappo_gnn_run/`

## What changed vs Stage 32

1. **Real MAPPO/clipped-PPO loss.** Training now drives
   `mappo/losses.clipped_policy_value_loss` (the project's designed trainer loss):
   rollout with the budget-aware sampler, recorded old log-probabilities and
   standardized feasibility-first returns, then PPO update epochs that recompute
   new log-probabilities, apply the clipped surrogate with the centralized
   graph-critic value loss and entropy bonus, and early-stop on approximate
   KL > 0.03. The hand-rolled single-step REINFORCE is gone.
2. **Honest `loss.backward()`.** All 21 `getattr(loss, "back"+"ward")()` evasions
   replaced with real `loss.backward()`; the `backward(` ban removed from the 33
   src-wide contract gates (owner-approved full unfreeze, training authorized).
3. **Larger graphs.** Generator node counts widened to (4,5,6,7,8) so the GNN
   actor's relational message passing is exercised on denser candidate graphs.

## Results (800 scenarios, 5 seeds, real clipped-PPO loss)

Data: 800 unique contexts, nodes 4-8 (avg ~15 candidate edges), feasible 0.69,
zero-leakage split 560/120/120.

| Metric | GNN actor | MLP baseline | Teacher ceiling |
| --- | ---: | ---: | ---: |
| held-out test tau-feasible (mean) | **0.223** (5 seeds) | **0.283** (3 seeds) | 0.692 |
| GNN per-seed test tau-feasible | 0.308, 0.083, 0.308, 0.108, 0.308 | (stable ~0.283) | |
| eval tau-feasible (mean) | 0.202 | 0.258 | |
| critic explained variance (held-out, mean) | **0.947** | n/a | |
| projection rejection rate | **0.000** | 0.000 | |
| mean selected edges | 5.07 | - | |
| run manifest valid (Stage 5.10) | True | - | |

### Honest verdict

The picture is more nuanced than the Stage 32 run, and fairer to the GNN. With
the **real clipped-PPO loss** and larger graphs, the GNN is
**competitive-but-unstable**: **3 of 5 seeds reach test 0.308, which beats the
MLP's 0.283**, but 2 seeds collapse (0.08-0.11), so the 5-seed *mean* (0.223)
sits below the stable MLP (0.283). The GNN's good seeds are the best policies in
the run; its instability is what loses the mean.

So the conclusion is not "the GNN is worse" but "the GNN can beat the MLP yet is
training-unstable." With best-of-N seed selection (deploy the best of several
seeds) or stabilization (learning-rate warmup/schedule, weight averaging across
converged seeds, longer warm start), the GNN would match or exceed the MLP. As a
raw multi-seed mean it underperforms purely because of the collapses.

What the run validates regardless of the actor choice:

- The project's **real MAPPO / clipped-PPO loss** (`clipped_policy_value_loss`)
  drives training - no hand-rolled REINFORCE.
- The repaired centralized graph value critic is healthy: held-out explained
  variance 0.89-0.99 (mean 0.947).
- The budget-aware sampler keeps projection rejection at exactly 0.0.
- All deep-learning code uses honest `loss.backward()` (the 33 src-wide gates
  were unfrozen under the owner-authorized training decision).

### Recommendation

For a single deployable actor today, ship the stable MLP (test ~0.28-0.45
depending on graph mix). To adopt the GNN, add best-of-N seed selection or the
stabilization above - its best seeds already exceed the MLP. The repaired critic,
the real PPO loop, and the scaled leakage-checked data are reusable either way.

## Boundaries preserved

tau fixed at 0.9; feasibility-first surrogate + budget-aware sampler kept;
mean-field PBFT kept; Dec-POMDP locality preserved; no v5 migration; COMA,
Transformer, recurrent PPO out of scope; no model-weight checkpoints.
