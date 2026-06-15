# Post-Task Self-Review: Stage 34

## Completed Task

Stage 34 implemented the GNN evaluation baseline repair and V2/V3 ablation
diagnostic control surface. The work added the Stage34 graph-necessity dataset
contract, strict graph-necessity metrics, seven required GNN ablation registry
entries, logit/KL/gradient diagnostic schema, blocked diagnostic scripts,
owner-readable reports, harness task registration, and project-state closeout.

## Intended Desired State

Production GNN selection should occur only after a repaired baseline, credible
graph-necessity metrics, official MAPPO diagnostic training, seven ablation
baselines, seed stability diagnostics, and low-entropy production selection.
MLP must remain diagnostic only.

## Actual Achieved State

Stage34 achieved the implementation and contract sensors but did not complete
the full empirical training protocol. The default diagnostic script refuses to
silently downgrade to smoke-mode architecture selection and reports the full
seven-ablation protocol as blocked pending owner compute/protocol decision.

## Evidence

- Stage34 task: `harness/tasks/stage34_gnn_baseline_repair_and_ablation_diagnostics.yaml`
- Dataset contract: `src/marl_topology/data/stage34_graph_necessity_dataset.py`
- Metrics: `src/marl_topology/evaluation/graph_necessity_metrics.py`
- Diagnostics: `src/marl_topology/evaluation/stage34_gnn_diagnostics.py`
- Ablation registry: `src/marl_topology/models/gnn_ablation_registry.py`
- Blocked diagnostic artifacts: `result_save/stage34_gnn_ablation_diagnostics/stage34_blocked_protocol/`
- Project state: `docs/PROJECT_STATE.md`

## Required Questions

1. Was the Stage 33 baseline repaired?
   Contractually yes; empirically blocked. Stage34 now enforces at least seven
   families, at least 50 samples per family, at least 350 total scenarios, node
   counts 6-10, and stratified splits. The full materialized evaluator-backed
   baseline was not run because the full protocol exceeds a safe default compute
   budget.

2. How many graph-necessity samples per family were generated?
   The non-test Stage34 config plans 50 per required family for 350 total
   scenarios. The default report uses planned records and does not materialize
   the full corpus. Unit-test materialization uses a guarded small config only
   for contract coverage and cannot be used for architecture selection.

3. Are graph-necessity metrics credible?
   The metric definitions are credible enough for the next full run: labels are
   derived from local-heuristic gap, ambiguity, bridge/weak-primary/role
   sensitivity, rank gap, or MLP-hardness rather than family name. Full empirical
   threshold calibration remains unverified.

4. Did the training protocol avoid tiny smoke mode?
   Yes. `Stage34TrainingProtocolConfig` rejects smoke mode and tiny scenario
   counts for diagnostic selection. The required seven-ablation default implies
   3,136,000 rollout transitions before eval/test overhead.

5. Which of the 7 GNN ablations won?
   None. No production GNN was selected because the full MAPPO ablation protocol
   did not complete.

6. What caused collapses, if any?
   No new full training collapses were measured. The reported collapse issue is
   a selection-gate blocker caused by no selected GNN, not a measured Stage34
   seed collapse diagnosis.

7. Did logit-scale / KL diagnostics identify instability?
   Pre-update logit/probability/entropy diagnostics are finite for all seven
   ablations. Per-update KL/gradient instability is not diagnosed because the
   full MAPPO protocol did not run.

8. Is v2 still best, or did a repaired variant beat it?
   Unknown. V2 is registered as a diagnostic reference only. No repaired variant
   beat it under the full protocol because the full protocol was blocked.

9. Was one active production GNN selected?
   No. `active_stage34_production_gnn_entries()` returns no active Stage34
   production GNN entries.

10. Were losing variants removed or archived inactive?
    No ablation was promoted, so all Stage34 variants remain diagnostic-only.
    There are no multiple active Stage34 production GNNs.

11. Is Stage 35 allowed?
    No. `stage35_without_stage34_selected_gnn` is blocked in
    `docs/PROJECT_STATE.md`.

## Tests And Gates

- `python scripts\train\stage34_gnn_ablation_diagnostic_training.py` passed and
  returned `stage34_gnn_ablation_blocked_awaiting_owner_decision`.
- `python scripts\replay\stage34_gnn_failure_attribution_report.py` passed and
  returned owner decision required.
- `python -m pytest -q` passed: 976 tests.
- `python harness\scripts\validate_tasks.py` passed: 90 tasks.

## New Risks

- The full evaluator-backed Stage34 dataset has not been materialized at 350+
  scenarios.
- The full seven-ablation MAPPO protocol has not been run.
- Graph-necessity thresholds are test-covered but not empirically calibrated.
- No production GNN checkpoint exists.

## Regressions Checked

- MLP was not promoted to production.
- Reward weights, tau, Stage3 reliability, Stage4 PBFT, sampler, assembler, and
  action semantics were not changed.
- Stage32 custom loop remains inactive.
- No COMA, Transformer, GRU, LSTM, or recurrent PPO was introduced.
- No v5 writes were performed.

## Candidate Next Tasks

- Approve the full Stage34 compute budget and run the materialized seven-ablation
  MAPPO protocol.
- Revise Stage34 protocol scale if the owner rejects the compute budget.
- Calibrate graph-necessity metric thresholds after full dataset materialization.

## Recommended Next Task

`owner_decision_full_stage34_compute_or_protocol_repair`

## Owner Decision Required

Yes. The owner must choose whether Stage34 should proceed with the full compute
budget or revise the diagnostic protocol. Codex must not start Stage35 until a
Stage34 GNN is selected by a passed gate.
