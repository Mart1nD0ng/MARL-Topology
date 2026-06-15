from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / ".agents" / "skills"


def test_skill_calibration_contract_defines_global_gate() -> None:
    text = (ROOT / "docs" / "SKILL_CALIBRATION.md").read_text(encoding="utf-8")

    required = [
        "Global Calibration Gate",
        "Anti-Inheritance Rules",
        "Required Skill Output Additions",
        "Acceptance Criteria",
        "docs/V5_FAILURE_LESSONS.md",
        "docs/V5_DO_NOT_LEARN_BLINDLY.md",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"skill calibration contract missing: {missing}"


def test_project_skills_include_anti_inheritance_gate() -> None:
    skill_paths = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    assert skill_paths, "no project skills found"

    missing_gate = []
    missing_contract = []
    for path in skill_paths:
        text = path.read_text(encoding="utf-8")
        if "## V5 Anti-Inheritance Calibration" not in text:
            missing_gate.append(str(path.relative_to(ROOT)))
        if "docs/SKILL_CALIBRATION.md" not in text:
            missing_contract.append(str(path.relative_to(ROOT)))

    assert not missing_gate, f"skills missing V5 anti-inheritance gate: {missing_gate}"
    assert not missing_contract, f"skills missing calibration contract reference: {missing_contract}"


def test_project_skills_do_not_promote_old_effective_success_names() -> None:
    banned_exact_defaults = ["`P_eff_soft`", "`P_eff_hard`"]
    offenders: dict[str, list[str]] = {}

    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned_exact_defaults if term in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert not offenders, f"skills reintroduced old effective-success names: {offenders}"
