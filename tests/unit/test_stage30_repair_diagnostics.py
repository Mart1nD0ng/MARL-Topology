from pathlib import Path

from marl_topology.evaluation.stage30_repair_diagnostics import (
    STAGE30_BLOCKED_VERDICT,
    STAGE30_MAX_ITERATIONS,
    STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED,
    build_stage30_closed_loop_report,
    choose_dominant_blocker,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stage30_closed_loop_stops_at_owner_gated_blocker_after_four_iterations() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)

    assert report["verdict"] == STAGE30_BLOCKED_VERDICT
    assert report["pass_gate"] is False
    assert report["iteration_count"] == STAGE30_MAX_ITERATIONS
    assert report["recommended_next_task"] == STAGE30_RECOMMENDED_NEXT_TASK_BLOCKED
    assert report["owner_decision_required"] is True
    assert report["blocker_review"]["max_iterations_reached"] is True


def test_stage30_health_snapshot_covers_required_components() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    components = {row["component"] for row in report["component_health_snapshot"]}

    assert {
        "harness_state",
        "surrogate_objective",
        "reliability_margin",
        "projection_alignment",
        "data_scale",
        "critic_health",
        "sampler_health",
        "actor_health",
        "policy_gradient_loop",
        "communication",
        "consensus",
    } <= components


def test_stage30_dominant_blocker_priority_prefers_surrogate_objective_before_later_failures() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    dominant = report["dominant_blocker"]

    assert dominant["component"] == "surrogate_objective"
    assert dominant["status"] == "FAIL"
    assert dominant["recommended_repair_class"] == "surrogate_objective_alignment"


def test_stage30_diagnostic_pilot_reuses_latest_small_pilot_without_new_training() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    pilot = report["diagnostic_pilot"]

    assert pilot["new_training_run"] is False
    assert pilot["source_pilot"] == "stage28_repaired_critic_fixed_small_scale_pilot"
    assert pilot["completed_seed_count"] == 3
    assert pilot["latency_improved"] is True
    assert pilot["energy_improved"] is True
    assert pilot["diagnostic_pilot_passed"] is False


def test_stage30_dominant_blocker_function_prefers_failures_then_priority() -> None:
    report = build_stage30_closed_loop_report(project_root=ROOT)
    components = report["component_health_snapshot"]
    rebuilt = choose_dominant_blocker(
        tuple(
            type(
                "Component",
                (),
                {
                    "component": row["component"],
                    "status": row["status"],
                    "recommended_repair_class": row["recommended_repair_class"],
                    "evidence": row["evidence"],
                    "confidence": row["confidence"],
                },
            )()
            for row in components
        )
    )

    assert rebuilt["component"] == report["dominant_blocker"]["component"]
