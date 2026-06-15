"""Return normalization for training-only value baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


class ReturnNormalizationViolation(ValueError):
    """Raised when value-target scaling is used incorrectly."""


@dataclass(frozen=True, slots=True)
class ReturnNormalizer:
    mean: float
    std: float
    eps: float = 1e-8
    fitted_on_train_only: bool = True

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ReturnNormalizationViolation("eps must be positive")
        if not isfinite(self.mean) or not isfinite(self.std):
            raise ReturnNormalizationViolation("normalizer state must be finite")
        if self.std <= 0.0:
            raise ReturnNormalizationViolation("std must be positive")
        if not self.fitted_on_train_only:
            raise ReturnNormalizationViolation("return normalizer must be train-only")

    @classmethod
    def fit(
        cls,
        train_returns: Iterable[float],
        *,
        eps: float = 1e-8,
        fitted_on_train_only: bool = True,
    ) -> "ReturnNormalizer":
        values = [float(value) for value in train_returns]
        if not values:
            raise ReturnNormalizationViolation("fit requires at least one return")
        if any(not isfinite(value) for value in values):
            raise ReturnNormalizationViolation("returns must be finite")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = variance**0.5
        if std <= eps:
            std = 1.0
        return cls(mean=mean, std=std, eps=eps, fitted_on_train_only=fitted_on_train_only)

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReturnNormalizer":
        return cls(
            mean=float(state["mean"]),
            std=float(state["std"]),
            eps=float(state.get("eps", 1e-8)),
            fitted_on_train_only=bool(state.get("fitted_on_train_only", True)),
        )

    def transform(self, returns: Iterable[float]) -> list[float]:
        return [(float(value) - self.mean) / (self.std + self.eps) for value in returns]

    def inverse_transform(self, normalized_values: Iterable[float]) -> list[float]:
        return [float(value) * (self.std + self.eps) + self.mean for value in normalized_values]

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
            "fitted_on_train_only": self.fitted_on_train_only,
        }

    def scale_report(self, train_returns: Iterable[float]) -> dict[str, object]:
        normalized = self.transform(train_returns)
        count = len(normalized)
        mean = sum(normalized) / count if count else 0.0
        variance = sum((value - mean) ** 2 for value in normalized) / count if count else 0.0
        return {
            "raw_mean": self.mean,
            "raw_std": self.std,
            "normalized_mean": mean,
            "normalized_std": variance**0.5,
            "fitted_on_train_only": self.fitted_on_train_only,
            "gae_uses_raw_denormalized_values": True,
        }
