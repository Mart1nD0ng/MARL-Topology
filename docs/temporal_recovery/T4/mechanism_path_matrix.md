# T4 — Mechanism-Path Matrix (Contract v4 §2)

Status ∈ {NOT_PRESENT, IMPLEMENTED_ONLY, CALLABLE, ACTIVE_IN_LOSS, ACTIVE_IN_EVAL, ACTIVE_IN_DEPLOY}.

| mechanism | code_path | status @ T4 | evidence |
|---|---|---|---|
| edge-level recurrent state (`edge_gru`, `belief_edge`) | `models/belief_residual_actor.py` | ACTIVE_IN_LOSS (opt-in; campaign edge_recurrent=True) | per-edge GRUCell [E,H] carried across frames; test_belief_edge_returns_edge_hidden_and_carries |
| spatial node embedding (no recurrence) | `_spatial_node_embed` | ACTIVE_IN_EVAL | enc + MP, feeds edge_gru input [ef, z_u, z_v] |
| edge belief head (reads edge hidden directly) | `edge_belief_head` | ACTIVE_IN_LOSS | [ef, extra, edge_hidden] → correction |
| per-node GRU recurrence (T3 baseline) | `belief` / `gru` | ACTIVE_IN_LOSS | the node arm; unchanged |
| recovered-psucc → anchor (T1 injection) | `t1_oracle_recovery_gen._anchor_with_psucc` (reused) | ACTIVE_IN_EVAL | edge feas_gain CI<0 (hurts) |
| edge_recurrent=False default (back-compat) | `models/belief_residual_actor.py` | ACTIVE (default) | edge modules appended last; T0 tripwire preserved; prior suite byte-identical |

**Blast radius: zero at the default.** `edge_recurrent=False` (default) allocates no edge modules and does not
change `belief()`/`forward()` — the frozen R1–R8 / T2 / T3 actor is byte-identical (test_edge_modules_appended_
last_preserve_init + full suite). `models/` is gate-exempt (CTDE).
