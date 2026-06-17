from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_doc(name: str) -> str:
    return _read(ROOT / "docs" / name)


def test_stage5_1_plan_document_exists_and_is_plan_only() -> None:
    text = _read_doc("STAGE5_1_REWARD_IMPLEMENTATION_PLAN.md")

    required = [
        "Stage 5.1",
        "no reward code in Stage 5.1",
        "training surrogate, not a metric",
        "tau_requirement_min = 0.9",
        "reward_surrogate = - reliability_weight * reliability_penalty",
        "Reliability plateau rule",
        "Stage 5.1 does not choose",
        "Future implementation sequence",
        "Stage 5.2 - reward surrogate interface skeleton with contract tests",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.1 plan missing terms: {missing}"

    forbidden_active_claims = [
        "Stage 5.1 implements reward",
        "Stage 5.1 trains",
        "reward_code_allowed = true",
    ]
    hits = [item for item in forbidden_active_claims if item in text]
    assert not hits, f"Stage 5.1 plan authorizes forbidden work: {hits}"


def test_stage5_1_plan_documents_forbidden_reward_components() -> None:
    text = _read_doc("STAGE5_1_REWARD_IMPLEMENTATION_PLAN.md")

    required = [
        "standalone timeout reward",
        "standalone quorum reward",
        "standalone deadline reward",
        "standalone edge-count reward",
        "standalone density or sparsity reward",
        "full graph or full mask bonus",
        "reliability bonus above tau by default",
        "old v5 reward formulas",
        "`P_eff` aliases",
        "oracle labels as actor inputs",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"forbidden reward components missing: {missing}"


def test_stage5_1_plan_documents_dec_pomdp_and_replay_boundaries() -> None:
    text = _read_doc("STAGE5_1_REWARD_IMPLEMENTATION_PLAN.md")

    required = [
        "Reward is computed environment-side or training-side",
        "not a deployment actor observation",
        "ActorObservation",
        "`reward_surrogate`: training-only",
        "`return`, `advantage`, and `value_target`: unsupported until training",
        "None of these columns may enter deployment actor input projections",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Dec-POMDP/replay boundary missing terms: {missing}"


def test_stage5_1_updates_reward_metric_and_replay_contracts() -> None:
    reward = _read_doc("REWARD_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    replay = _read_doc("REPLAY_DATASET_CONTRACT.md")

    assert "Stage 5.1 Reward Implementation Plan Without Code" in reward
    assert "Stage 5.1 defines the future reward-surrogate implementation plan" in surrogate
    assert "Stage 5.1 does not add metric names" in metric
    assert "Stage 5.1 plans future reward columns but does not admit them yet" in replay
    assert "must keep them out of deployment actor input projections" in replay


def test_stage5_1_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_1_reward_implementation_plan.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_1_reward_implementation_plan"
    for field in [
        "required_artifacts",
        "expected_evidence",
        "negative_checks",
        "relevant_lessons",
        "required_tests",
        "forbidden_v5_inheritance",
        "expected_outputs",
    ]:
        assert field in task
        assert task[field]

    negative_text = " ".join(task["negative_checks"])
    assert "reward implementation code is added" in negative_text
    assert "reward weights are selected or calibrated" in negative_text
    assert "old v5 reward formulas" in negative_text
    assert "deployment actor inputs" in negative_text


def test_stage5_1_project_state_marks_plan_complete_and_next_stage() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_1_awaiting_owner_decision_for_stage_5_2",
        "post_stage_5_0m_ready_for_stage_5_1_owner_decision",
        "stage_5_1_reward_implementation_plan_without_code",
        "stage5_1_reward_implementation_plan_gate",
        "stage_5_2_reward_surrogate_interface_skeleton_with_contract_tests",
        "recommended_next_task: stage_5_2_reward_surrogate_interface_skeleton_with_contract_tests",
        "reward_implementation",
        "reward_weight_calibration",
        "training_runs",
        "v5_code_migration",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.1 state: {missing}"


def test_stage5_1_does_not_add_reward_or_training_implementation() -> None:
    forbidden_paths = [
        ROOT / "src" / "marl_topology" / "objectives" / "reward_surrogate.py",
        ROOT / "src" / "marl_topology" / "objectives" / "reward.py",
        ROOT / "src" / "marl_topology" / "training" / "reward.py",
        ROOT / "src" / "marl_topology" / "models" / "actor.py",
        ROOT / "src" / "marl_topology" / "models" / "critic.py",
    ]
    offenders = [path for path in forbidden_paths if path.exists()]
    assert not offenders, f"Stage 5.1 added implementation files: {offenders}"

    # 2026-06-17: the recovered production trunk is BEHAVIOUR-CLONING distillation
    # (training/decentralized_distillation.py) -- it does not implement a reward surrogate or a
    # policy-gradient optimizer, so no module needs the former exemption (the decentralized_marl.py
    # PPO exemption was retired together with that module). The scaffold-era guard scopes to all
    # of src again.
    authorized_production_training: set[str] = set()
    source_paths = [
        path
        for path in (ROOT / "src" / "marl_topology").rglob("*.py")
        if path.name not in authorized_production_training
    ]
    banned_terms = [
        "def compute_reward",
        "class Reward",
        "reward_surrogate =",
        "optimizer",
        "train_loop",
        "D:\\PhD_works\\v5",
        "P_eff_soft",
        "P_eff_hard",
    ]
    hits: list[str] = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 5.1 source introduced forbidden terms: {hits}"
