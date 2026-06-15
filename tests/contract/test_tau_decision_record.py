"""Pin tau_requirement_min = 0.9 to a single cited decision record.

The value was previously a bare literal duplicated across ~10 modules with no shared
basis. docs/TAU_DECISION_RECORD.md is now the single source of truth; this test makes
the magic number traceable and prevents silent per-module drift: every in-source
definition of (STAGE21_)?TAU_REQUIREMENT_MIN must equal the value declared in the
record, and the record must carry the owner-decision basis.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "docs" / "TAU_DECISION_RECORD.md"

# matches a module-level (optionally indented) CONSTANT definition only, not usages,
# comparisons, or lowercase function-parameter defaults.
_SRC_DEF = re.compile(r"^\s*(?:STAGE21_)?TAU_REQUIREMENT_MIN\s*=\s*([0-9.]+)", re.MULTILINE)
_CANONICAL = re.compile(r"CANONICAL_TAU_REQUIREMENT_MIN\s*=\s*([0-9.]+)")


def _canonical_value() -> float:
    text = RECORD.read_text(encoding="utf-8")
    match = _CANONICAL.search(text)
    assert match, "TAU_DECISION_RECORD.md must declare CANONICAL_TAU_REQUIREMENT_MIN = <value>"
    return float(match.group(1))


def test_tau_decision_record_exists_and_states_the_basis() -> None:
    assert RECORD.exists(), "docs/TAU_DECISION_RECORD.md must exist"
    text = RECORD.read_text(encoding="utf-8")
    required = [
        "owner_approved_stage31_production_readiness_unfreeze",  # approval id
        "requirement floor",  # 0.9 is a floor, not a calibrated optimum
        "0.99",  # the per-link reliability anchor argument
        "tau_consensus",  # relationship to the still-open formal parameter
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"TAU_DECISION_RECORD.md missing basis terms: {missing}"
    assert _canonical_value() == 0.9


def test_every_source_tau_constant_matches_the_record() -> None:
    canonical = _canonical_value()
    definitions: dict[str, float] = {}
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        for value in _SRC_DEF.findall(path.read_text(encoding="utf-8")):
            definitions[f"{path.relative_to(ROOT)}"] = float(value)

    # The drift hazard is real: the constant is defined in many modules. Keep the
    # test meaningful by requiring we actually found the cluster.
    assert len(definitions) >= 8, f"expected the tau constant in >=8 modules, found {definitions}"
    drift = {loc: val for loc, val in definitions.items() if val != canonical}
    assert not drift, f"tau constant drifted from the decision record ({canonical}): {drift}"
