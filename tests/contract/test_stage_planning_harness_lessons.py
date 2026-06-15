from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_engineering_planning_lessons_document_exists_and_covers_three_failures() -> None:
    text = _read("docs/ENGINEERING_PLANNING_LESSONS.md")

    required = [
        "Lesson A: Stage Sprawl / Design-Only Loop",
        "Lesson B: Weak Exit Gate / Patch Stage Cascade",
        "Lesson C: State / Harness / Self-Review Drift",
        "executable deliverable",
        "Completion Gate",
        "Next-Stage Readiness Gate",
        "implemented_unregistered",
        "PROJECT_STATE.md",
        "harness/tasks",
        "post-task self-review",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"engineering planning lessons missing terms: {missing}"


def test_stage_planning_skill_exists_and_contains_hard_rules() -> None:
    path = ROOT / ".agents" / "skills" / "cybernetic-stage-planning" / "SKILL.md"
    text = path.read_text(encoding="utf-8")

    required = [
        "stage sprawl",
        "next-stage readiness",
        "PROJECT_STATE.md",
        "Do not allow two consecutive planning-only stages",
        "Do not treat \"tests pass\" as the only exit condition",
        "implemented_unregistered",
        "harness registry drift",
        "model-before-data",
        "training-before-evidence",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"stage planning skill missing terms: {missing}"


def test_codex_workflow_contains_stage_planning_rules() -> None:
    text = _read("docs/CODEX_WORKFLOW.md")

    required = [
        "Stage Planning Rules",
        "Stage Closeout Rule",
        "No Consecutive Planning-Only Rule",
        "Next-Stage Readiness Rule",
        "Stage State Sync Rule",
        "Scope Compression Rule",
        "Owner Checkpoint Rule",
        "implemented_unregistered",
        "cybernetic-stage-planning",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"CODEX_WORKFLOW missing stage planning rules: {missing}"


def test_agents_contains_stage_closeout_project_state_and_readiness_rules() -> None:
    text = _read("AGENTS.md")

    required = [
        "Stage Planning Rules",
        "Stage closeout rule",
        "update `docs/PROJECT_STATE.md`",
        "run `python harness\\scripts\\validate_tasks.py`",
        "No consecutive planning-only rule",
        "Next-stage readiness rule",
        "Stage state sync rule",
        "implemented_unregistered",
        "Scope compression rule",
        "Owner checkpoint rule",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"AGENTS.md missing stage planning rules: {missing}"


def test_stage_planning_quality_review_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage_planning_quality_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage_planning_quality_review"
    checks = task["stage_planning_checks"]
    for field in [
        "stage_id",
        "stage_type",
        "executable_deliverable",
        "completion_gate",
        "next_stage_readiness_gate",
        "state_sync_required",
        "harness_task_required",
        "post_task_self_review_required",
        "planning_only_allowed",
        "blocker_removed",
        "owner_decision_required",
    ]:
        assert field in checks
    assert checks["state_sync_required"] is True
    assert checks["harness_task_required"] is True
    assert checks["post_task_self_review_required"] is True
    assert checks["owner_decision_required"] is True


def test_stage_closeout_review_template_exists() -> None:
    text = _read("harness/templates/stage-closeout-review.md.template")

    required_sections = [
        "1. Stage Summary",
        "2. Stage Type",
        "3. Executable Deliverables",
        "4. Completion Gate Result",
        "5. Next-Stage Readiness Result",
        "6. Evidence Produced",
        "7. What Remains Unproven",
        "8. State Sync Checklist",
        "9. Harness Registry Checklist",
        "10. Risks Introduced",
        "11. Recommended Next Task",
        "12. Owner Decision Required",
    ]
    missing = [section for section in required_sections if section not in text]
    assert not missing, f"stage closeout template missing sections: {missing}"


def test_project_state_is_present_and_stage16_closeout_is_owner_gated() -> None:
    text = _read("docs/PROJECT_STATE.md")

    required = [
        "current_stage:",
        "completed_stages:",
        "recommended_next_task:",
        "blocked_tasks:",
        "owner_decision_required: true",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"PROJECT_STATE missing core state fields: {missing}"


def test_planning_lessons_task_touches_no_model_training_or_v5_code() -> None:
    paths = [
        ROOT / "docs" / "ENGINEERING_PLANNING_LESSONS.md",
        ROOT / ".agents" / "skills" / "cybernetic-stage-planning" / "SKILL.md",
        ROOT / "docs" / "CODEX_WORKFLOW.md",
        ROOT / "AGENTS.md",
        ROOT / "harness" / "tasks" / "stage_planning_quality_review.yaml",
        ROOT / "harness" / "templates" / "stage-closeout-review.md.template",
    ]
    forbidden = [
        "optimizer.step(",
        ".backward(",
        "torch.save(",
        "checkpoint_path =",
        "write_checkpoint",
        "PPOTrainer(",
        "MAPPOTrainer(",
        "class COMA",
        "class Transformer",
        "D:\\PhD_works\\v5\\",
    ]
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"planning lessons changed forbidden implementation surface: {hits}"
