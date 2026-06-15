from __future__ import annotations

from math import sin

from marl_topology.training.critic_dataset import (
    CriticDataset,
    CriticDatasetRow,
    Stage27CriticDatasetConfig,
)
from marl_topology.training.critic_features import (
    CriticDiagnosticFeatureRecord,
    CriticValueFeatureRecord,
    DIAGNOSTIC_FEATURE_FIELDS,
    VALUE_CRITIC_FEATURE_FIELDS,
)


def make_stage27_synthetic_dataset(
    *,
    train_count: int = 512,
    eval_count: int = 128,
) -> CriticDataset:
    rows: list[CriticDatasetRow] = []
    total = train_count + eval_count
    for index in range(total):
        split = "train" if index < train_count else "eval"
        local_index = index if split == "train" else index - train_count
        x = float(local_index % 64) / 63.0
        y = float((local_index // 64) % 8) / 7.0
        wave = sin(float(index) * 0.17)
        return_value = -80.0 - 360.0 * x - 90.0 * y + 12.0 * wave
        reward = return_value / 12.0
        consensus = max(0.0, min(1.0, 1.0 - 0.7 * x + 0.05 * wave))
        latency = 0.02 + 0.45 * x + 0.04 * y
        energy = 0.1 + 3.5 * y + 0.2 * x
        value_values = _value_features(x=x, y=y, wave=wave)
        diagnostic_values = _diagnostic_features(
            consensus=consensus,
            latency=latency,
            energy=energy,
            reward=reward,
            x=x,
        )
        rows.append(
            CriticDatasetRow(
                split=split,
                scenario_id=f"synthetic_{local_index % 16}",
                time_step=local_index % 16,
                seed=2701 + index % 4,
                source_index=local_index % 16,
                scenario_slot=local_index % 32,
                value_features=CriticValueFeatureRecord(
                    values=value_values,
                    metadata={
                        "scenario_id": f"synthetic_{local_index % 16}",
                        "time_step": local_index % 16,
                    },
                ),
                diagnostic_features=CriticDiagnosticFeatureRecord(
                    values=diagnostic_values,
                    metadata={
                        "current_consensus_success_probability": consensus,
                        "current_latency": latency,
                        "current_energy": energy,
                        "current_reward_surrogate": reward,
                    },
                ),
                graph_node_features=_graph_node_features(x=x, y=y),
                graph_edge_features=_graph_edge_features(x=x, y=y),
                graph_edge_index=((0, 1), (1, 2), (0, 2)),
                reward=reward,
                return_value=return_value,
                done=local_index % 16 == 15,
                mask=0.0 if local_index % 16 == 15 else 1.0,
                selected_edges=("veh0--rsu0",) if x > 0.25 else (),
                sampler_id="physical_plackett_luce_top_k_sampler",
                proposal_logprob=-0.5 - x,
                proposal_entropy=0.3 + y,
                consensus_success_probability=consensus,
                latency=latency,
                energy=energy,
                reward_components={
                    "reliability_violation": max(0.0, 0.9 - consensus),
                    "reliability_penalty": -max(0.0, 0.9 - consensus),
                    "latency_penalty": -latency,
                    "energy_penalty": -energy,
                },
                topology_diagnostics={
                    "selected_edge_count": 1 if x > 0.25 else 0,
                    "empty_graph": x <= 0.25,
                    "full_graph": False,
                },
            )
        )
    return CriticDataset(
        rows=tuple(rows),
        config=Stage27CriticDatasetConfig(),
        actor_checksum_before=10.0,
        actor_checksum_after=10.0,
    )


def _value_features(*, x: float, y: float, wave: float) -> tuple[float, ...]:
    raw = [
        5.0,
        3.0,
        1.0,
        1.0,
        8.0,
        1.0 + x,
        x,
        2.0,
        4.0,
        x,
        y,
        x * y,
        1.0 - x,
        0.2 + 0.4 * (1.0 - x),
        abs(wave),
        1.0 - 0.5 * x,
        0.3,
        0.02 + x,
        0.04 + x,
        0.1 + y,
        0.4 + y,
        1.0 if x < 0.4 else 0.0,
        0.0,
        1.0 + x,
        2.0 + y,
        8.0,
        x,
        8.0 - x,
        5.0,
        y,
        2.0,
        4.0,
        0.1 * x,
        0.2 * y,
        x,
        y,
        x * y,
        0.0,
        0.0,
        wave,
        x + y,
        float(int(x * 10.0)),
        0.5,
        0.0,
        -20.0 * x,
        1.0 - x,
        0.02 + x,
        0.1 + y,
        1.0 if x > 0.7 else 0.0,
    ]
    assert len(raw) == len(VALUE_CRITIC_FEATURE_FIELDS)
    return tuple(float(value) for value in raw)


def _diagnostic_features(
    *,
    consensus: float,
    latency: float,
    energy: float,
    reward: float,
    x: float,
) -> tuple[float, ...]:
    raw = [
        consensus,
        latency,
        energy,
        reward,
        1.0 if x > 0.25 else 0.0,
        0.125 if x > 0.25 else 0.0,
        0.05 * x,
        0.03 * x,
        1.0 if consensus < 0.9 else 0.0,
        0.1 - x,
        0.02 + x,
    ]
    assert len(raw) == len(DIAGNOSTIC_FEATURE_FIELDS)
    return tuple(float(value) for value in raw)


def _graph_node_features(*, x: float, y: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, x, 1.0 + x, 2.0 - x, 0.5 + y, 0.0),
        (0.0, 1.0, 0.0, y, 0.5 + y, 1.5 - y, 0.2 + x, 0.0),
        (0.0, 0.0, 1.0, x + y, 0.2, 1.0, 0.3, 0.0),
    )


def _graph_edge_features(*, x: float, y: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0 - x, 1.0 - x, 0.02 + x, 0.1 + y, x, 10.0, 0.25, 0.5),
        (0.8 - 0.3 * y, 0.8 - 0.3 * y, 0.04 + y, 0.3 + x, y, 12.0, 0.5, 0.75),
        (0.6 - 0.2 * x, 0.6 - 0.2 * x, 0.05 + x + y, 0.4, x * y, 15.0, 0.25, 0.75),
    )
