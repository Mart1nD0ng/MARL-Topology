# Stage 30 Diagnostic MAPPO Pilot

Stage 30 stops before running a new policy update because the repair loop reaches owner-gated reward/projection/data blockers. This document records the latest fixed small-scale pilot evidence used by the Stage 30 readiness gate.

- source pilot: `stage28_repaired_critic_fixed_small_scale_pilot`
- new training run: `False`
- completed seed count: `3`
- all seeds completed: `True`
- critic EV: `0.9465631796667974`
- critic value-return correlation: `0.9733077206959327`
- latency improved: `True`
- energy improved: `True`
- surrogate improved: `False`
- projection improved: `False`
- reliability large-scale gate passed: `False`
- objective-alignment candidate improved: `True`
- diagnostic pilot passed: `False`
- large-scale readiness passed: `False`
- large-scale training allowed: `False`
- readiness issues: `['surrogate_alignment_not_active_without_owner_decision', 'reliability_margin_gate_failed', 'projection_friction_not_reduced', 'data_not_ready_for_scale', 'component_failures:reliability_margin,surrogate_objective']`

This script is intentionally a training-gate sensor, not a new training run.
