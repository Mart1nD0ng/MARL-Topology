#!/usr/bin/env python3
"""Conservative keyword-assisted rubric scoring for markdown outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to score rubric files.") from exc


KEYWORDS = {
    "controlled_object_identified": ["controlled object", "boundary", "scope", "subsystem", "object"],
    "desired_state_defined": ["desired state", "acceptance", "success criteria", "target state", "metric"],
    "state_variables_defined": ["state variable", "state / inputs / outputs", "input", "output", "state"],
    "sensors_defined": ["sensor", "test", "log", "metric", "trace", "static check", "evidence"],
    "actuators_defined": ["actuator", "safe actuator", "editable", "config", "interface", "adapter"],
    "feedback_loop_present": ["feedback", "baseline", "post-change", "before", "after", "observe"],
    "verification_plan_present": ["verification", "test plan", "checks", "command", "expected evidence"],
    "stability_risk_checked": ["stability", "regression", "rollback", "oscillation", "scope", "risk"],
    "observability_gap_checked": ["observability gap", "blind spot", "missing sensor", "instrumentation", "gap"],
    "controllability_checked": ["controllability", "safe actuator", "approval", "unsafe", "fixed dependency"],
    "disturbance_checked": ["disturbance", "external service", "timing", "constraint", "environment"],
    "delay_or_async_risk_checked": ["async", "delay", "stale", "queue", "cache", "ci", "timestamp"],
    "noise_or_flakiness_checked": ["noise", "flaky", "random", "variance", "repeat", "statistical"],
    "decoupling_checked": ["coupling", "non-target", "impact matrix", "consumer", "side effect", "regression"],
    "reliability_or_error_control_checked": ["reliability", "redundant", "independent", "negative test", "false success", "isolation"],
    "persistent_learning_update_suggested": ["persistent", "follow-up", "agents.md", "skill", "harness backlog", "lesson"],
}

SOURCE_MARKERS = ["analogy", "software-engineering", "derived", "not direct", "source-awareness", "EC-RULE", "类比"]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(text: str) -> str:
    return text.lower()


def has_heading_or_field(text: str, item_id: str) -> bool:
    label = item_id.replace("_", " ")
    return bool(re.search(rf"(^|\n)#+\s+.*{re.escape(label)}", text)) or label in text


def score_item(text: str, item: dict) -> dict:
    item_id = item["id"]
    norm = normalize(text)
    keywords = KEYWORDS.get(item_id, [])
    hits = sorted({kw for kw in keywords if kw in norm})
    evidence_terms = [term.lower() for term in item.get("evidence_to_collect", [])]
    evidence_hits = sorted({term for term in evidence_terms if term and term in norm})
    heading_hit = has_heading_or_field(norm, item_id)

    pass_candidate = (heading_hit and (hits or evidence_hits)) or (len(hits) >= 2 and bool(evidence_hits))
    weak_candidate = not pass_candidate and (heading_hit or hits or evidence_hits)

    if pass_candidate:
        status = "pass_candidate"
        points = item["weight"]
    elif weak_candidate:
        status = "needs_human_review"
        points = 0
    else:
        status = "missing_candidate"
        points = 0

    return {
        "id": item_id,
        "weight": item["weight"],
        "points": points,
        "status": status,
        "keyword_hits": hits,
        "evidence_hits": evidence_hits,
        "heading_or_field_hit": heading_hit,
        "note": "Conservative keyword screen only; human review required for final judgment.",
    }


def score(markdown_path: Path, rubric_path: Path) -> dict:
    text = markdown_path.read_text(encoding="utf-8")
    rubric = load_yaml(rubric_path)
    item_scores = [score_item(text, item) for item in rubric["items"]]
    total_weight = sum(item["weight"] for item in rubric["items"])
    total_points = sum(item["points"] for item in item_scores)
    source_awareness = any(marker.lower() in normalize(text) for marker in SOURCE_MARKERS)
    return {
        "output_file": str(markdown_path),
        "rubric_id": rubric.get("rubric_id"),
        "total_points": total_points,
        "total_weight": total_weight,
        "percent": round((total_points / total_weight) * 100, 2) if total_weight else 0,
        "suggested_pass_threshold": rubric.get("scoring_guidance", {}).get("suggested_pass_threshold"),
        "source_awareness_marker_found": source_awareness,
        "automated_scoring_is_conservative": True,
        "requires_human_review": True,
        "items": item_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a markdown output against a cybernetic rubric.")
    parser.add_argument("markdown_output", type=Path)
    parser.add_argument("rubric_yaml", type=Path)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    report = score(args.markdown_output, args.rubric_yaml)
    data = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
