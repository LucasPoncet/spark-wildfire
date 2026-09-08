import ast
from pathlib import Path

import pytest

INVERSE_PACKAGE_PATH = Path(__file__).resolve().parents[1] / "src" / "spark" / "inverse"
FORBIDDEN_PACKAGES = ("src.spark.fire", "src.spark.terrain", "src.spark.fields", "src.spark.acoustic")


def collect_imported_modules(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("source_path", sorted(INVERSE_PACKAGE_PATH.glob("*.py")), ids=lambda p: p.name)
def test_inverse_never_imports_the_forward_simulation(source_path: Path) -> None:
    imported = collect_imported_modules(source_path)
    violations = sorted(
        module
        for module in imported
        for forbidden in FORBIDDEN_PACKAGES
        if module == forbidden or module.startswith(forbidden + ".")
    )
    assert violations == []


def test_the_inverse_package_is_not_empty() -> None:
    assert len(list(INVERSE_PACKAGE_PATH.glob("*.py"))) > 1
