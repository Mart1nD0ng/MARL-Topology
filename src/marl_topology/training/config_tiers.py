"""Config tiers: smoke / pilot / research (v2 Engineering-Plan §R0).

Three operating tiers keep fast smoke params, single-seed pilot params, and the
heavy multi-seed research regime structurally separate. The point is an exit
condition of R0: **a smoke or pilot result must never become a headline.** Each
tier config carries an explicit ``headline_eligible`` flag; only ``research`` may
set it true (and only with >=5 seeds, D4). Headline-producing scripts call
:func:`assert_headline_eligible` on the tier config they ran under and refuse
smoke/pilot.

This lives under ``training/`` (gate-exempt, training-only governance); it touches
no deployed path. Configs live at the repo-root ``configs/<tier>/<name>.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

TIER_NAMES = ("smoke", "pilot", "research")

REQUIRED_TIER_FIELDS = (
    "tier",
    "headline_eligible",
    "purpose",
    "updates",
    "scene_count",
    "val_scene_count",
    "seeds",
    "samples_per_scene",
)

# A headline needs >=5 independent seeds with CI (AGENTS.md D4 / Spec §5-§9).
MIN_RESEARCH_SEEDS = 5


class HeadlineEligibilityError(RuntimeError):
    """Raised when a headline-producing path is handed a smoke/pilot config."""


def configs_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def validate_tier_config(config: Mapping[str, object]) -> None:
    """Validate a tier config's schema and the tier-specific invariants.

    Raises ``ValueError`` on any violation (missing field, unknown tier, a
    research config with too few seeds, or a non-research tier marked
    headline-eligible).
    """
    missing = [f for f in REQUIRED_TIER_FIELDS if f not in config]
    if missing:
        raise ValueError(f"tier config missing required fields: {missing}")

    tier = config["tier"]
    if tier not in TIER_NAMES:
        raise ValueError(f"unknown tier {tier!r}; must be one of {TIER_NAMES}")

    headline = config["headline_eligible"]
    if not isinstance(headline, bool):
        raise ValueError(f"headline_eligible must be a bool, got {type(headline).__name__}")

    seeds = config["seeds"]
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds:
        raise ValueError("seeds must be a non-empty list of ints")
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ValueError("seeds must all be ints")

    for int_field in ("updates", "scene_count", "val_scene_count", "samples_per_scene"):
        value = config[int_field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{int_field} must be a positive int, got {value!r}")

    if tier == "research":
        if not headline:
            raise ValueError("a research tier must be headline_eligible")
        if len(seeds) < MIN_RESEARCH_SEEDS:
            raise ValueError(
                f"a research headline needs >={MIN_RESEARCH_SEEDS} seeds (D4); got {len(seeds)}"
            )
    else:  # smoke / pilot
        if headline:
            raise ValueError(
                f"tier {tier!r} must NOT be headline_eligible; smoke/pilot results are never a headline"
            )


def load_tier_config(tier: str, name: str = "default") -> dict[str, object]:
    """Load and validate ``configs/<tier>/<name>.json``."""
    if tier not in TIER_NAMES:
        raise ValueError(f"unknown tier {tier!r}; must be one of {TIER_NAMES}")
    path = configs_root() / tier / f"{name}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_tier_config(config)
    return config


def assert_headline_eligible(config: Mapping[str, object]) -> None:
    """Guard a headline-producing path: refuse a smoke/pilot config.

    Raises :class:`HeadlineEligibilityError` unless ``config['headline_eligible']``
    is true (which, by :func:`validate_tier_config`, only a research tier can be).
    """
    if not bool(config.get("headline_eligible", False)):
        raise HeadlineEligibilityError(
            f"config tier {config.get('tier')!r} is not headline-eligible; "
            "smoke/pilot results must never become a headline (v2 plan R0 exit condition, D4)"
        )
