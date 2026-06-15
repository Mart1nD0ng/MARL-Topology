from pathlib import Path

from marl_topology.data.stage21_objective_stack_evidence import STAGE21_TAU_REQUIREMENT_MIN
from marl_topology.training.production_mappo_adapter import Stage33GNNStabilityConfig


ROOT = Path(__file__).resolve().parents[2]


def test_stage33_config_keeps_tau_and_reward_weights_unchanged() -> None:
    payload = Stage33GNNStabilityConfig().to_payload()

    assert payload["tau_requirement_min"] == STAGE21_TAU_REQUIREMENT_MIN
    assert payload["reward_weights_tuned"] is False
    assert payload["entropy_coef"] == 0.01


def test_stage33_code_does_not_introduce_forbidden_architectures_or_algorithms() -> None:
    code_paths = [
        ROOT / "src" / "marl_topology" / "training" / "production_mappo_adapter.py",
        ROOT / "src" / "marl_topology" / "models" / "local_gnn_edge_scorer.py",
        ROOT / "src" / "marl_topology" / "data" / "stage33_graph_structure_dataset.py",
        ROOT / "scripts" / "train" / "stage33_gnn_stability_repair_training.py",
        ROOT / "scripts" / "replay" / "stage33_gnn_failure_attribution_report.py",
    ]
    banned_terms = [
        "class LSTM",
        "nn.LSTM",
        "class GRU",
        "nn.GRU",
        "class Transformer",
        "nn.Transformer",
        "class COMA",
        "COMACritic",
        "reward_weight_tuning_performed = True",
        "tau_requirement_min=0.89",
    ]
    hits = []
    for path in code_paths:
        text = path.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text:
                hits.append(f"{path.relative_to(ROOT)}:{term}")

    assert not hits
