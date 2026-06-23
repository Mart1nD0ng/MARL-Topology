from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_stage4_2_source_avoids_adapter_reward_training_and_v5_routes() -> None:
    source = (ROOT / "src" / "marl_topology" / "protocol" / "pbft_reliability.py").read_text(
        encoding="utf-8"
    )

    banned_terms = [
        "itertools",
        "combinations",
        "permutations",
        "monte_carlo",
        "Monte Carlo",
        "random",
        "sample",
        "import v5",
        "from v5",
        "torch",
        "class Actor(",
        "class Critic(",
        "COMA",
        "MAPPO",
        "optimizer",
        "backward(",
        "train_loop",
        "torch.save",
        "P_eff",
        "network_delivery_probability",
        "p2p_latency_s",
        "network_latency_s",
        "network_energy_j",
    ]
    hits = [term for term in banned_terms if term in source]

    assert not hits, f"Stage 4.2 source uses forbidden routes: {hits}"
