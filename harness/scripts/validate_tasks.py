#!/usr/bin/env python3
"""Validate cybernetic harness task YAML files."""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to validate task files.") from exc


TASK_TYPES = {
    "project_analysis",
    "project_scaffold",
    "project_improvement",
    "harness_design",
    "skill_creation",
    "code_review",
}
REQUIRED_FIELDS = [
    "id",
    "name",
    "task_type",
    "prompt",
    "required_artifacts",
    "expected_evidence",
    "rubric_items",
    "pass_threshold",
    "minimum_required_items",
    "negative_checks",
    "source_awareness_required",
]
RULE_RE = re.compile(r"EC-RULE-\d{3}")


def kit_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rubric_items(rubric_path: Path) -> set[str]:
    rubric = load_yaml(rubric_path)
    return {item["id"] for item in rubric.get("items", [])}


def validate_task(path: Path, rubric_items: set[str]) -> list[str]:
    errors: list[str] = []
    task = load_yaml(path)
    if not isinstance(task, dict):
        return ["task file must contain a mapping"]

    for field in REQUIRED_FIELDS:
        if field not in task:
            errors.append(f"missing required field: {field}")

    if errors:
        return errors

    if task["task_type"] not in TASK_TYPES:
        errors.append(f"invalid task_type: {task['task_type']!r}")

    for list_field in ["required_artifacts", "expected_evidence", "rubric_items", "minimum_required_items", "negative_checks"]:
        if not isinstance(task[list_field], list) or not task[list_field]:
            errors.append(f"{list_field} must be a non-empty list")

    if task["source_awareness_required"] is not True:
        errors.append("source_awareness_required must be true")

    threshold = task["pass_threshold"]
    if not isinstance(threshold, int | float) or not (1 <= threshold <= 100):
        errors.append("pass_threshold must be a number from 1 to 100")

    unknown_items = sorted(set(task["rubric_items"]) - rubric_items)
    if unknown_items:
        errors.append(f"unknown rubric_items: {', '.join(unknown_items)}")

    unknown_min = sorted(set(task["minimum_required_items"]) - rubric_items)
    if unknown_min:
        errors.append(f"unknown minimum_required_items: {', '.join(unknown_min)}")

    min_not_in_task = sorted(set(task["minimum_required_items"]) - set(task["rubric_items"]))
    if min_not_in_task:
        errors.append(f"minimum_required_items not included in rubric_items: {', '.join(min_not_in_task)}")

    for field in ["id", "name", "prompt"]:
        if not isinstance(task[field], str) or not task[field].strip():
            errors.append(f"{field} must be a non-empty string")

    rule_basis = task.get("rule_basis", [])
    if rule_basis and not all(isinstance(rule, str) and RULE_RE.fullmatch(rule) for rule in rule_basis):
        errors.append("rule_basis must contain EC-RULE-### ids")

    return errors


def main() -> int:
    root = kit_root()
    tasks_dir = root / "harness" / "tasks"
    rubric_path = root / "harness" / "rubrics" / "cybernetic_engineering_rubric.yaml"
    rubric_items = load_rubric_items(rubric_path)
    task_paths = sorted(tasks_dir.glob("*.yaml"))
    if not task_paths:
        print(f"ERROR: no task YAML files found in {tasks_dir}")
        return 1

    all_errors: dict[str, list[str]] = {}
    covered_items: set[str] = set()
    covered_rules: set[str] = set()
    for path in task_paths:
        errors = validate_task(path, rubric_items)
        if errors:
            all_errors[path.name] = errors
        else:
            task = load_yaml(path)
            covered_items.update(task["rubric_items"])
            covered_rules.update(task.get("rule_basis", []))

    if all_errors:
        print("Task validation failed")
        for name, errors in all_errors.items():
            print(f"- {name}")
            for error in errors:
                print(f"  - {error}")
        return 1

    print(f"Task validation passed: {len(task_paths)} tasks")
    for path in task_paths:
        task = load_yaml(path)
        print(f"- {task['id']} ({task['task_type']})")
    print(f"Covered rubric items: {len(covered_items)}/{len(rubric_items)}")
    print(f"Covered rule IDs: {len(covered_rules)}")
    missing_items = sorted(rubric_items - covered_items)
    if missing_items:
        print("Rubric items not covered:")
        for item in missing_items:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
