"""Stage 2.7 link-model regime purity guard (real invariant; doc/harness lineage retired).

The link regime stays the simple ``exp(-d)`` model: deferred physics (path loss, shadowing,
SINR, interference) must not be *implemented* outside the authorized channel/ primitives, the
simple-link transmission module, and the equivalence-verified vectorized evaluator. The stale
doc / PROJECT_STATE / harness-task assertions from the multi-stage process were retired on
2026-06-23 when that lineage was consolidated; this real source-purity check survives.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stage_2_7_does_not_implement_deferred_physics_components() -> None:
    source_paths = []
    for path in (ROOT / "src" / "marl_topology").rglob("*.py"):
        relative_parts = path.relative_to(ROOT).parts
        if path.name == "regime.py":
            continue
        if "channel" in relative_parts:
            continue
        if "link" in relative_parts and path.name in {"transmission.py", "__init__.py"}:
            continue
        if "evaluation" in relative_parts and path.name == "stage3_fixture_suite.py":
            continue
        # Authorized post-2026-06-16 physics VECTORIZATION: it composes the canonical channel
        # primitives (no NEW deferred physics) and is equivalence-verified (|dp0| < 1e-9) vs the
        # canonical evaluator, so -- like channel/link -- it legitimately references SINR
        # composed from precomputed received powers.
        if path.name == "vectorized_objective_stack_evaluator.py":
            continue
        source_paths.append(path)
    banned_physics_terms = [
        "path_loss_db",
        "shadowing_db",
        "sinr_db",
        "interference_power",
    ]
    offenders: dict[str, list[str]] = {}
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_physics_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.7 implemented deferred physics components: {offenders}"
