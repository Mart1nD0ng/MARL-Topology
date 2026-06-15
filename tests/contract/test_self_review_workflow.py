from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_SELF_REVIEW_FIELDS = [
    "completed task",
    "intended desired state",
    "actual achieved state",
    "evidence",
    "tests",
    "gates passed",
    "gates deferred",
    "new risks",
    "regressions",
    "candidate next tasks",
    "recommended next task",
    "owner decision required",
]


def test_post_task_self_review_task_exists_and_has_required_fields() -> None:
    path = ROOT / "harness" / "tasks" / "post_task_self_review.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "post_task_self_review"
    for field in REQUIRED_SELF_REVIEW_FIELDS:
        assert field in task["self_review_fields"]
    assert any("owner decision" in item for item in task["negative_checks"])
    assert any("self-authorized" in item or "authorizes" in item for item in task["negative_checks"])


def test_post_task_self_review_template_exists_and_has_sections() -> None:
    text = (ROOT / "harness" / "templates" / "post-task-self-review.md.template").read_text(
        encoding="utf-8"
    )

    for field in REQUIRED_SELF_REVIEW_FIELDS:
        heading = "## " + field.title()
        assert heading in text
    assert "Do not mark a recommendation as authorized" in text


def test_project_state_declares_owner_decision_and_blocked_tasks() -> None:
    text = (ROOT / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")

    required_terms = [
        "current stage",
        "completed stages",
        "active gates",
        "deferred gates",
        "allowed next tasks",
        "blocked tasks",
        "recommended next task",
        "owner decision required",
        "Codex may recommend",
        "user approval is required",
    ]
    lower = text.lower()
    missing = [term for term in required_terms if term.lower() not in lower]
    assert not missing, f"PROJECT_STATE missing required terms: {missing}"


def test_workflow_docs_require_self_review_and_owner_decision() -> None:
    combined = "\n".join(
        [
            (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "CODEX_WORKFLOW.md").read_text(encoding="utf-8"),
        ]
    ).lower()

    required_terms = [
        "self-review after every substantial task",
        "owner decision required",
        "must not self-authorize",
        "user approval is required",
    ]
    missing = [term for term in required_terms if term not in combined]
    assert not missing, f"workflow docs missing self-review rules: {missing}"
