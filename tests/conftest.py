from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.dont_write_bytecode = True


def _remove_python_cache_artifacts() -> None:
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in ROOT.rglob(".pytest_cache"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in ROOT.rglob("*.pyc"):
        if path.is_file():
            path.unlink()


def pytest_configure() -> None:
    _remove_python_cache_artifacts()


def pytest_collection_finish() -> None:
    _remove_python_cache_artifacts()


def pytest_sessionfinish() -> None:
    _remove_python_cache_artifacts()
