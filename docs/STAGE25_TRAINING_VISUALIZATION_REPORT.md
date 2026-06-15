# Stage 25 Training Visualization Report

Stage 25 generated a human-review visualization package from the fixed base
training report. The visualization script validates the run manifest before
writing outputs and writes only under the approved artifact root.

## Controlled System

- Controlled object: Stage 25 report artifacts and their human-review views.
- Desired state: reviewers can inspect training curves, objective trends, PPO
  diagnostics, critic behavior, projection diagnostics, topology diagnostics,
  reward surface, and before/after comparisons without rerunning training.
- State variables: generated CSV files, optional plot files, manifest
  validation state, required-view coverage, and training-report visualization
  flag.
- Sensors: `visualization_report.json`, CSV row counts, plot-file existence,
  manifest validation, and contract tests.
- Actuators: CSV/JSON/PNG report generation only.
- Disturbances: missing matplotlib, missing or stale training report, manifest
  mismatch, and incomplete metric rows.
- Coupling: visualization depends on the training report and reward-surface
  analysis; training report pass/fail state records whether visualization was
  generated.

## Output Directory

`result_save/stage25_small_scale_mappo_training_pilot/stage25_pilot_base_config`

## Generated Views

- Training curves: reward, tau-feasible rate, violation rate.
- Objective curves: consensus success probability, latency, energy.
- PPO diagnostics: KL, entropy, clip fraction, policy loss, value loss.
- Critic diagnostics: explained variance and value loss.
- Projection diagnostics: top rejection, above-threshold rejection, and
  rejection-by-reason rows.
- Topology diagnostics: selected edge count, empty graph rate, full graph rate.
- Reward surface: reliability, latency, energy, surrogate reward, and
  dominated/non-dominated marking.
- Before/after comparison: supervised GNN, MAPPO fine-tuned GNN, greedy, full,
  and teacher baselines where available.

## Artifact Files

- `visualization_report.json`
- `training_curves.csv`
- `objective_curves.csv`
- `ppo_diagnostics.csv`
- `critic_diagnostics.csv`
- `projection_diagnostics.csv`
- `topology_diagnostics.csv`
- `reward_surface.csv`
- `before_after_comparison.csv`
- `training_curves.png`
- `objective_curves.png`
- `ppo_diagnostics.png`
- `projection_diagnostics.png`
- `topology_diagnostics.png`
- `reward_surface_scatter.png`
- `before_after_comparison.png`

Matplotlib was available during the Stage 25 run, so PNG plots were generated.
If plotting support is unavailable in a future reproduction, the CSV/JSON
outputs remain the fallback review artifacts.

## Verification

- `python scripts\replay\stage25_training_visualization_report.py`
- `tests/contract/test_stage25_visualization_outputs.py`

Residual risk: the plots are review sensors, not model-selection evidence. The
primary conclusion remains based only on `stage25_pilot_base_config`.
