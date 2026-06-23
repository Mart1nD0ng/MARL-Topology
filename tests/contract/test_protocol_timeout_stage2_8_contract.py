"""Stage 2.8 protocol-layer purity guard (real D2 invariant; doc/harness lineage retired).

``protocol/`` stays closed-form and deterministic: no simulated PBFT state-machine
(``class PBFT`` / ``view_change`` / ``byzantine``), no training (``train_loop`` / ``optimizer`` /
``backward(``), and no legacy v5 reward terms -- except the authorized closed-form modules
(quorum_tail, pbft_reliability, message_matrix_adapter, pbft_accounting, quorum_spec). This
keeps reliability a fixed Poisson-binomial quorum tail rather than a Monte-Carlo protocol sim
(D2). The stale doc / PROJECT_STATE / harness-task assertions were retired on 2026-06-23 when
the multi-stage process lineage was consolidated; this real source-purity check survives.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stage_2_8_protocol_review_does_not_add_forbidden_protocol_code() -> None:
    banned_terms = [
        "P_eff_soft",
        "P_eff_hard",
        "hard_eval",
        "soft_train",
        "timeout_reward",
        "quorum_reward",
        "class PBFT",
        "view_change",
        "byzantine",
        "train_loop",
        "optimizer",
        "backward(",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (ROOT / "src" / "marl_topology" / "protocol").rglob("*.py"):
        if path.name in {
            "quorum_tail.py",
            "pbft_reliability.py",
            "message_matrix_adapter.py",
            "pbft_accounting.py",
            # Phase 1 (spec-driven reconstruction): the authorized closed-form safe-quorum spec.
            "quorum_spec.py",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_terms if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"Stage 2.8 protocol review added forbidden code: {offenders}"
