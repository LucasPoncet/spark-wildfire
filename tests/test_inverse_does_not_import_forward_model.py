import ast
from pathlib import Path

import pytest

INVERSE_PACKAGE_PATH = Path(__file__).resolve().parents[1] / "src" / "spark" / "inverse"
FORBIDDEN_PACKAGES = ("spark.fire", "spark.terrain", "spark.fields", "spark.acoustic")


def collect_imported_modules(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize(
    "source_path", sorted(INVERSE_PACKAGE_PATH.glob("*.py")), ids=lambda path: path.name
)
def test_inverse_never_imports_the_forward_model(source_path: Path) -> None:
    imported = collect_imported_modules(source_path)
    violations = sorted(
        module
        for module in imported
        for forbidden in FORBIDDEN_PACKAGES
        if module == forbidden or module.startswith(forbidden + ".")
    )
    assert violations == []


def test_the_forbidden_packages_exist_so_the_guard_is_not_vacuous() -> None:
    source_root = INVERSE_PACKAGE_PATH.parents[1]
    for forbidden in FORBIDDEN_PACKAGES:
        assert (source_root / Path(*forbidden.split("."))).is_dir()


def test_the_inverse_package_is_not_empty() -> None:
    assert len(list(INVERSE_PACKAGE_PATH.glob("*.py"))) > 1
