"""Local LSTM edge scorer for Stage 14 ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tensorizers import ACTOR_EDGE_FEATURE_FIELDS


LOCAL_LSTM_EDGE_SCORER_MODEL_ID = "local_lstm_edge_scorer_v1"


@dataclass(frozen=True, slots=True)
class LocalLSTMEdgeScorerConfig:
    model_id: str = LOCAL_LSTM_EDGE_SCORER_MODEL_ID
    input_dim: int = len(ACTOR_EDGE_FEATURE_FIELDS)
    hidden_dim: int = 24


class LocalLSTMEdgeScorer(nn.Module):
    """Sequence scorer over local edge feature histories."""

    def __init__(self, config: LocalLSTMEdgeScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalLSTMEdgeScorerConfig()
        self.recurrent = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            batch_first=True,
        )
        self.head = nn.Linear(self.config.hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if features.ndim != 3 or features.shape[2] != self.config.input_dim:
            raise ValueError("features must have shape [sequence, time, feature]")
        output, hidden = self.recurrent(features, hidden_state)
        return self.head(output).squeeze(-1), hidden

    def reset_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.zeros(
            (1, batch_size, self.config.hidden_dim),
            dtype=torch.float32,
            device=device,
        )
        return state, state.clone()

    def boundary_report(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "history_source": "actor_safe_local_edge_history_only",
            "future_outcomes_used": False,
            "global_memory_used": False,
            "outputs_edge_scores_only": True,
        }
