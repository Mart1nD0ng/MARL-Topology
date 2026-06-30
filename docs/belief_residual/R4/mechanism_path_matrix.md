# R4 Mechanism-Path Matrix delta (Contract v4 §2)

| mechanism | before R4 | after R4 | path / evidence |
|---|---|---|---|
| beneficial-edit teacher dataset | NOT_PRESENT | **IMPLEMENTED (teacher-only)** | `oracle_edit_dataset.build_edit_dataset`; train scenes only |
| ΔC/ΔD_quorum/ΔE/ΔL/ΔJ per local edit | NOT_PRESENT | **ACTIVE (eval, training-only)** | true evaluator per edit; `test_edit_labels_include_C_D_E_L` |
| positive-edit supervision target | NOT_PRESENT | **IMPLEMENTED** (`supervised_records`: ΔJ>margin ∧ safe) | feeds R5 heads (ACTIVE_IN_LOSS at R5) |

This stage is a TEACHER (training-only, never at deployment, never on held). It produces the supervised target
for R5's repair/safety/utility/edit heads (which become ACTIVE_IN_LOSS at R5). Stores anchor + single edit
descriptor + Δ's only — NOT the full oracle topology (TechSpec §11; `test_no_full_oracle_topology_stored`).

Scheduled: R5 (train heads on the positive edits; held top-k edit hit rate > random); R6 (evidence-gated
deployable action from the heads); R7 (adaptive anchor-KL); R8 (full pilot). The deployable question — can the
heads learn this signal from LOCAL features (no evaluator) — is R5+, NOT settled by R4.
