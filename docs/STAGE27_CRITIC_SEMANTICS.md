# Stage 27 Critic Semantics

## MAPPO Value Critic

Purpose: provide an action-independent centralized V(s) baseline for advantage computation.

Input: pre-action centralized state only. It may include graph summaries, previous topology/resource summaries, previous-step objective summaries, current candidate-edge communication estimates, current time step, and pre-proposal resource state.

Forbidden in value input: current selected topology caused by the action, current consensus success, current latency, current energy, current reward, current surrogate, and future outcome.

## Action-Conditioned Diagnostic Critic

Diagnostic records may include selected topology, post-action Stage 3/4 metrics, reward surrogate, and edge-delta targets. These records are tagged `action_conditioned_diagnostic` and are not used as the MAPPO V(s) baseline.

## Advantage Path

The future advantage path must use the selected value critic only. The value head predicts normalized value targets; predictions are denormalized to raw return scale before GAE.
