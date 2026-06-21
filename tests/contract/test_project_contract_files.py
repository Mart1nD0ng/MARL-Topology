from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_required_contract_files_exist() -> None:
    required = [
        ROOT / "docs" / "FOUR090_CAMPAIGN_PLAN.md",
        ROOT / "docs" / "TRUNK_MAP.md",
        ROOT / "docs" / "URBAN_V2X_RESEARCH_LOG.md",
        ROOT / "docs" / "REVIEW_2026-06-21_EVIDENCE_PASS.md",
        ROOT / "docs" / "NEXT_LOOP_INSTRUCTION.md",
    ]

    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"missing required contract files: {missing}"


def test_required_workflow_directories_exist() -> None:
    required = [
        ROOT / ".agents" / "skills",
        ROOT / "harness" / "tasks",
        ROOT / "harness" / "rubrics",
        ROOT / "harness" / "tasks" / "skill_calibration_audit.yaml",
        ROOT / "harness" / "tasks" / "post_task_self_review.yaml",
        ROOT / "harness" / "templates" / "post-task-self-review.md.template",
    ]

    missing = [
        str(path.relative_to(ROOT))
        for path in required
        if not (path.is_dir() if path.suffix == "" else path.exists())
    ]
    assert not missing, f"missing required workflow directories: {missing}"


def test_metric_contract_is_governance_first() -> None:
    text = (ROOT / "docs" / "METRIC_CONTRACT.md").read_text(encoding="utf-8")

    for concept in [
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    ]:
        assert concept in text

    for field in [
        "`name`",
        "`definition`",
        "`range/unit`",
        "`level`",
        "`used_for`",
        "`formula source`",
        "`dependencies`",
        "`tests`",
    ]:
        assert field in text

    assert "`P_eff_soft`" not in text
    assert "`P_eff_hard`" not in text


def test_goal_skeleton_layers_are_declared() -> None:
    text = (ROOT / "docs" / "GOAL_SKELETON.md").read_text(encoding="utf-8")
    layers = [
        "Scene3D",
        "CandidateGraph",
        "LinkModel",
        "TopologyEvaluator",
        "ConsensusSuccess",
        "LatencyEnergy",
        "TopologyOracle",
        "DecPOMDPEnv",
        "PolicyBaselines",
        "MARLTraining later",
    ]

    missing = [layer for layer in layers if layer not in text]
    assert not missing, f"missing goal skeleton layers: {missing}"


def test_v5_failure_lessons_have_confidence_labels() -> None:
    text = (ROOT / "docs" / "V5_FAILURE_LESSONS.md").read_text(encoding="utf-8")

    for label in ["seed", "confirmed", "contradicted", "unresolved"]:
        assert label in text

    for required in [
        "lesson_id",
        "affected GOAL_SKELETON layer",
        "symptom in v5",
        "evidence",
        "clean-project design rule",
        "required test or harness gate",
    ]:
        assert required in text
