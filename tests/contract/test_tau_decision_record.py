"""Pin tau_requirement_min = 0.9 across all source modules (real D3 drift guard).

The value was previously a bare literal duplicated across ~10 modules with no shared
basis. The canonical value is re-pinned here as a literal: the prior single-source-of-truth
``docs/TAU_DECISION_RECORD.md`` lineage doc was retired on 2026-06-23 with the rest of the
multi-stage process docs. The *real* invariant this test protects -- that every in-source
definition of ``(STAGE21_)?TAU_REQUIREMENT_MIN`` equals the canonical 0.9, so the constant
cannot silently drift per-module (D3) -- survives unchanged against the literal.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Canonical tau requirement floor. Owner-approved Stage-31 production-readiness basis:
# 0.9 is a requirement *floor* (not a calibrated optimum), anchored to the 0.99 per-link
# reliability argument, and distinct from the still-open formal ``tau_consensus`` parameter.
# Re-pinned as a literal after the TAU_DECISION_RECORD.md lineage doc was retired.
CANONICAL_TAU_REQUIREMENT_MIN = 0.9

# matches a module-level (optionally indented) CONSTANT definition only, not usages,
# comparisons, or lowercase function-parameter defaults.
_SRC_DEF = re.compile(r"^\s*(?:STAGE21_)?TAU_REQUIREMENT_MIN\s*=\s*([0-9.]+)", re.MULTILINE)


def test_every_source_tau_constant_matches_the_record() -> None:
    canonical = CANONICAL_TAU_REQUIREMENT_MIN
    definitions: dict[str, float] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        for value in _SRC_DEF.findall(path.read_text(encoding="utf-8")):
            definitions[f"{path.relative_to(ROOT)}"] = float(value)

    # The drift hazard is real: the constant is defined in several modules. Keep the
    # test meaningful by requiring we actually found the cluster.
    assert len(definitions) >= 6, f"expected the tau constant in >=6 modules, found {definitions}"
    drift = {loc: val for loc, val in definitions.items() if val != canonical}
    assert not drift, f"tau constant drifted from the canonical {canonical}: {drift}"
