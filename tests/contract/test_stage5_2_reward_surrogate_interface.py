from pathlib import Path

import yaml

from marl_topology.data import REWARD_TRAINING_ONLY_COLUMNS
from marl_topology.metrics import REGISTERED_METRICS
from marl_topology.objectives import (
    SURROGATE_SIGNAL_INPUT_METRICS,
    SURROGATE_TRAINING_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage5_2_interface_document_exists_and_sets_boundary() -> None:
    text = _read_doc("STAGE5_2_REWARD_SURROGATE_INTERFACE.md")

    required = [
        "Stage 5.2 Reward Surrogate Interface Skeleton",
        "training-only reward surrogate",
        "src/marl_topology/objectives/surrogate_signal.py",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
        "reliability_penalty = 0",
        "no reliability bonus is added",
        "Training-only replay payload columns",
        "not evaluation metrics",
        "forbidden deployment actor inputs",
        "does not train",
        "does not migrate v5 code",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"Stage 5.2 interface doc missing terms: {missing}"


def test_stage5_2_updates_reward_metric_and_replay_contracts() -> None:
    reward = _read_doc("REWARD_CONTRACT.md")
    surrogate = _read_doc("REWARD_SURROGATE_CONTRACT.md")
    metric = _read_doc("METRIC_CONTRACT.md")
    replay = _read_doc("REPLAY_DATASET_CONTRACT.md")

    assert "Stage 5.2 Reward Surrogate Interface Skeleton" in reward
    assert "Stage 5.2 implements only the minimal pure interface skeleton" in surrogate
    assert "Stage 5.2 does not add metric names" in metric
    assert "Stage 5.2 admits the first reward-surrogate diagnostic columns" in replay
    assert "Generic `reward`, `return`, `advantage`, and `value_target` remain unsupported" in replay


def test_stage5_2_harness_task_exists() -> None:
    path = ROOT / "harness" / "tasks" / "stage5_2_reward_surrogate_interface_skeleton.yaml"
    task = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert task["id"] == "stage5_2_reward_surrogate_interface_skeleton"
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
    assert "reward weights are selected calibrated or presented as final" in negative_text
    assert "surrogate columns enter deployment actor inputs" in negative_text
    assert "surrogate outputs are registered as evaluation metrics" in negative_text


def test_stage5_2_project_state_waits_for_stage5_3_owner_decision() -> None:
    text = _read_doc("PROJECT_STATE.md")

    required = [
        "post_stage_5_2_awaiting_owner_decision_for_stage_5_3",
        "post_stage_5_1_awaiting_owner_decision_for_stage_5_2",
        "stage_5_2_reward_surrogate_interface_skeleton_with_contract_tests",
        "stage5_2_reward_surrogate_interface_gate",
        "stage_5_3_reward_normalization_reference_selection",
        "recommended_next_task: stage_5_3_reward_normalization_reference_selection",
        "reward_weight_calibration",
        "training_runs",
        "v5_code_migration",
        "owner_decision_required: true",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"PROJECT_STATE missing Stage 5.2 state: {missing}"


def test_stage5_2_surrogate_fields_are_not_registered_metrics() -> None:
    assert set(SURROGATE_SIGNAL_INPUT_METRICS) <= set(REGISTERED_METRICS)
    assert set(SURROGATE_TRAINING_COLUMNS) == REWARD_TRAINING_ONLY_COLUMNS
    assert not (set(SURROGATE_TRAINING_COLUMNS) & set(REGISTERED_METRICS))


def test_stage5_2_does_not_add_training_model_v5_or_legacy_metric_code() -> None:
    forbidden_paths = [
        ROOT / "src" / "marl_topology" / "objectives" / "reward.py",
        ROOT / "src" / "marl_topology" / "objectives" / "reward_surrogate.py",
        ROOT / "src" / "marl_topology" / "training" / "reward.py",
        ROOT / "src" / "marl_topology" / "models" / "actor.py",
        ROOT / "src" / "marl_topology" / "models" / "critic.py",
    ]
    offenders = [path for path in forbidden_paths if path.exists()]
    assert not offenders, f"Stage 5.2 added forbidden implementation files: {offenders}"

    banned_terms = [
        "def compute_reward",
        "class Reward",
        "reward =",
        "reward:",
        "optimizer",
        "train_loop",
        "D:\\PhD_works\\v5",
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "COMA",
        "MAPPO",
    ]
    hits: list[str] = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")
    assert not hits, f"Stage 5.2 source introduced forbidden terms: {hits}"
