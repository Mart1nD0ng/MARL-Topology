from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_python_cache_artifacts_absent_from_scaffold() -> None:
    banned_dirs = {".pytest_cache", "__pycache__"}
    found_dirs = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name in banned_dirs
    ]
    found_pyc = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.pyc")
        if path.is_file()
    ]

    assert not found_dirs, f"cache directories should not be present: {found_dirs}"
    assert not found_pyc, f"compiled Python files should not be present: {found_pyc}"


def test_gitignore_blocks_generated_scaffold_artifacts() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        "harness/reports/*.json",
        "result_save/*",
        "!result_save/.gitkeep",
    ]
    missing = [pattern for pattern in required_patterns if pattern not in text]
    assert not missing, f".gitignore missing hygiene patterns: {missing}"


def test_pytest_cacheprovider_is_disabled() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "-p no:cacheprovider" in text
