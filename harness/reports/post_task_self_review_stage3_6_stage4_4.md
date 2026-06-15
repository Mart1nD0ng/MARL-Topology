# Post-Task Self-Review: Stage 3.6 + Stage 4.4 Mathematical Model Refactor

## Completed Task

Stage 3.6 - URLLC finite-blocklength link reliability and Stage 4.4 -
expected-initiator PBFT reliability.

## Intended Desired State

Stage 3 should remove the active SINR-only packet-success surrogate and use a
finite-blocklength transmission model that couples reliability, latency, and
energy, including deadline-bounded retransmission. Stage 4 should stop treating
one fixed primary as topology-level PBFT reliability, preserve the fixed-primary
calculation as a helper, and expose a uniform expected initiator reliability
model without view-change, strict Byzantine adversary modeling, Monte Carlo,
random sampling, subset enumeration, reward, training, actor/critic code, or v5
migration.

## Actual Achieved State

Stage 3.6 now makes the channel layer produce path loss, interference, and
`sinr_db` only. Link transmission uses `urlcc_finite_blocklength_v1` with
finite-blocklength error probability, inverse target-reliability transmission
time solving, deadline retransmission, expected attempts, expected latency, and
expected energy. Network hop delivery now uses link
`deadline_delivery_probability`.

Stage 4.4 now adds `PBFTExpectedInitiatorConfig`,
`PBFTExpectedInitiatorReliabilityRecord`, `evaluate_pbft_given_primary`, and
`evaluate_expected_initiator_pbft_reliability`. The topology-level
`consensus_success_probability` is the uniform average of per-primary `Psi_p`
values under `pbft_expected_initiator_mean_field_v1`.

## Evidence

- `src/marl_topology/channel/model.py`
- `src/marl_topology/link/transmission.py`
- `src/marl_topology/network/communication.py`
- `src/marl_topology/protocol/pbft_reliability.py`
- `tests/unit/test_link_transmission_stage3.py`
- `tests/unit/test_pbft_expected_initiator_stage4_4.py`
- `tests/contract/test_stage3_6_urlcc_finite_blocklength_contract.py`
- `tests/contract/test_stage4_4_expected_initiator_pbft_contract.py`
- `docs/CHANNEL_MODEL_CONTRACT.md`
- `docs/LINK_TRANSMISSION_CONTRACT.md`
- `docs/STAGE3_LATENCY_ENERGY_RELIABILITY_COUPLING_REVIEW.md`
- `docs/STAGE4_2_PBFT_THREE_PHASE_RELIABILITY.md`
- `docs/PROTOCOL_CONTRACT.md`
- `docs/METRIC_CONTRACT.md`
- `docs/PROJECT_STATE.md`
- `harness/tasks/stage3_6_urlcc_finite_blocklength_link_reliability.yaml`
- `harness/tasks/stage4_4_expected_initiator_pbft_reliability.yaml`

## Tests

- `python -m pytest -q`
- `python harness\scripts\validate_tasks.py`
- `python harness\scripts\score_rubric.py docs\CONTROL_MODEL.md harness\rubrics\cybernetic_engineering_rubric.yaml --out harness\reports\control_model.score.json`
- Source/doc scan for removed active SINR-only packet-success route.
- Protocol source scan for no Monte Carlo, random sampling, subset enumeration,
  v5 imports, reward, training, or model routes.
- Hygiene and `result_save` scans.

## Gates Passed

- `stage3_6_urlcc_finite_blocklength_gate`
- `stage4_4_expected_initiator_pbft_gate`
- `stage3_channel_model_gate`
- `stage3_link_transmission_gate`
- `stage3_network_layer_gate`
- `stage4_pbft_three_phase_reliability_record_gate`
- `stage4_stage3_message_matrix_adapter_gate`
- `metric_governance_gate`
- `consensus_protocol_naming_gate`
- `training_precondition_gate`
- `phase_script_entropy_gate`
- `post_task_self_review`

## Gates Deferred

- Stage 4.5 baseline and oracle review using expected-initiator reliability.
- Stage 3.6a larger finite-blocklength boundary fixtures.
- Stage 4.4a larger or asymmetric primary-distribution boundary fixtures.
- Stage 4.1a log-space quorum-tail review for larger committees.
- Reward implementation, training, actor/critic, COMA, GNN, and LSTM work.

## New Risks

- The finite-blocklength normal approximation can become numerically step-like
  for small deterministic fixtures; more boundary fixtures are needed before
  using it for policy comparison.
- `attempt_duration_s` includes propagation, transmission, processing, and
  queueing for deadline accounting, while the finite-blocklength block duration
  is `transmission_delay_s`; this distinction must remain documented.
- The expected-initiator PBFT reliability still inherits the Stage 4.2
  mean-field independence approximation.
- `fault_filter_remove_largest` remains conservative but is not a strict
  Byzantine adversary model.

## Regressions

No reward, training, actor, critic, COMA, GNN, LSTM, checkpoint handling, or v5
code migration was added. Stage 3 still does not emit PBFT consensus metrics.
The old active SINR-only packet-success route is absent from active source and
docs.

## Candidate Next Tasks

- Stage 4.5 - baseline and oracle review using expected-initiator PBFT
  reliability.
- Stage 3.6a - finite-blocklength boundary fixture suite.
- Stage 4.4a - expected-initiator PBFT boundary fixtures.
- Stage 4.1a - log-space quorum-tail review.

## Recommended Next Task

Stage 4.5 - baseline and oracle review.

Reason: the active communication model and topology-level PBFT reliability
semantics are now aligned. The next safe step is to compare empty, full, sparse,
and oracle-candidate topologies while preserving full graph as a baseline, not
an oracle.

## Owner Decision Required

Yes. Codex may recommend Stage 4.5, but user approval is required before
executing the next task.
