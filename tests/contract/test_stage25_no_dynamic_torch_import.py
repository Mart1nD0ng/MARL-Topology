import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_stage25_no_dynamic_torch_import_remains_in_source_or_scripts() -> None:
    hits: list[str] = []
    for root_name in ["src", "scripts"]:
        for path in (ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if _is_dynamic_torch_import(node):
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")

    assert not hits, f"dynamic torch imports remain: {hits}"


def test_stage25_direct_torch_imports_are_limited_to_model_and_training_modules() -> None:
    hits: list[str] = []
    for root_name in ["src", "scripts"]:
        for path in (ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if not _has_direct_torch_import(tree):
                continue
            relative = path.relative_to(ROOT)
            # Direct torch imports are allowed in the model/training packages and in the
            # runnable training drivers under scripts/train/. The latter is post-2026-06-16:
            # real large-scale training + checkpointing is now authorized, so its drivers
            # legitimately import torch (the prior scaffold-era assumption is retired).
            allowed = relative.parts[:3] in {
                ("src", "marl_topology", "models"),
                ("src", "marl_topology", "training"),
            } or relative.parts[:2] == ("scripts", "train")
            if not allowed:
                hits.append(relative.as_posix())

    assert not hits, f"direct torch imports outside model/training modules: {hits}"


def test_stage25_docs_and_harness_do_not_embed_dynamic_torch_import_patterns() -> None:
    patterns = [
        "__import__(\"to\" + \"rch\")",
        "__import__('to' + 'rch')",
        "importlib.import_module(\"torch\")",
        "importlib.import_module('torch')",
    ]
    hits: list[str] = []
    for root_name in ["docs", "harness"]:
        for path in (ROOT / root_name).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".yaml", ".yml", ".py"}:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                if pattern in text:
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{pattern}")

    assert not hits, f"dynamic torch import pattern found in docs or harness: {hits}"


def _has_direct_torch_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch" or module.startswith("torch."):
                return True
    return False


def _is_dynamic_torch_import(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name) and node.func.id == "__import__":
        return bool(node.args and _constant_string(node.args[0]) == "torch")
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
    ):
        return bool(node.args and _constant_string(node.args[0]) == "torch")
    return False


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None
