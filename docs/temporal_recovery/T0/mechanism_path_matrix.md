# T0 — Mechanism-Path Matrix (Contract v4 §2)

HEAD snapshot of the mechanisms this campaign will touch. Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE,
ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}. Paths audited at commit `c74a71d`.

| mechanism | code_path | status @ HEAD | evidence (file:line) |
|---|---|---|---|
| CSI-belief target = absolute current `p_t` | `training/csi_belief.py` | ACTIVE_IN_LOSS | `belief_target_logits` [csi_belief.py:58]; `csi_belief_loss` [csi_belief.py:88] |
| `tanh` residual saturation (acted logit) | `models/belief_residual_actor.py` | ACTIVE_IN_EVAL | `forward` `z=z_max*tanh(raw/z_max)` [belief_residual_actor.py:94-95] |
| per-node `GRUCell` recurrence | `models/belief_residual_actor.py` | ACTIVE_IN_EVAL | `self.gru = nn.GRUCell(hidden,hidden)` [belief_residual_actor.py:41]; `h=self.gru(x+agg,h0)` [:79] → `[N,H]` |
| edge-level recurrent state / edge hidden | — | NOT_PRESENT | (no `edge_gru`; only the per-node cell) |
| physical residual model | — | NOT_PRESENT | belief head is pure MLP over [ef, hu*hv, |hu-hv|] [:114] |
| uncertainty (variance) head | `models/belief_residual_actor.py` | NOT_PRESENT | `belief_head` outputs a single logit [:50-52] |
| CSI-recovery → anchor injection (oracle) | — | NOT_PRESENT | (T1 builds it) |
| anchor consumes psucc col 0 only | `training/dynamic_baselines.py` | ACTIVE_IN_DEPLOY | `local_hysteresis_proposals` sorts on `ef[i,psucc_col]` [dynamic_baselines.py:59-62] |
| leak-free motion features | `training/csi_belief.py` | CALLABLE | `leak_free_motion_features` [csi_belief.py:20] |
| stale/partial CSI overlay | `training/csi_observation_model.py` + `dynamic_frames.py` | ACTIVE_IN_EVAL (opt-in) | `_apply_stale_csi` [dynamic_frames.py:197]; cols 0-3 staleified, evaluator on true channel [:172] |

No path changes at T0 — this is the audited baseline the campaign edits one variable at a time.
