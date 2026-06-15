# CTDE And Decentralized Deployment Decision

## Decision

MARL-Topology will use Dec-POMDP deployment constraints with CTDE as the allowed training paradigm:

- Deployment is decentralized: each vehicle or RSU actor may use only local observations, local history, and permitted local messages.
- Training may be centralized: training-only evaluators, critics, replay diagnostics, and oracle analysis may use global state when declared.
- CTDE is a boundary and evidence model, not an immediate commitment to MAPPO, COMA, GNN, LSTM, or any other algorithm.

## Rationale

The project goal is a distributed topology-control system. If actor inputs include global topology, oracle labels, future outcomes, or centralized critic tensors, deployment evidence is invalid even if training metrics improve. CTDE fits the target because it keeps centralized evidence available during training while preserving decentralized execution.

## Algorithm Posture

- First line later: simple decentralized non-learning baselines, then IPPO/MAPPO-style CTDE baselines if contracts are stable.
- Deferred: COMA, Q critics, direct edge-delta critics, GNN actor, LSTM actor, and GNN+LSTM actor.
- COMA/Q-style credit assignment requires the `credit_calibration_gate` before use.
- Fixed `0.5` thresholding remains a baseline to test, not a deployment rule.

## Current Stage 2.1 Action

Stage 2.1 records this decision and implements only the observation/action schema contract plus leakage validators. It does not implement:

- actor model
- critic model
- COMA
- GNN or LSTM policy
- training loop
- reward implementation

## Required Gates

- `dec_pomdp_leakage_gate`
- `fixed_threshold_is_baseline_gate` before deployment threshold claims
- `credit_calibration_gate` before COMA/Q/direct edge-delta work
- `training_precondition_gate` before any training

## Owner Decision

Future algorithm selection requires a separate owner-approved task after schema, oracle, baseline, metric, reward, and leakage evidence exist.
