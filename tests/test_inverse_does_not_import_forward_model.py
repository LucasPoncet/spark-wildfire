import ast
from pathlib import Path

import pytest

INVERSE_PACKAGE_PATH = Path(__file__).resolve().parents[1] / "src" / "spark" / "inverse"
SOURCE_ROOT_PREFIX = "src."
FORBIDDEN_PACKAGES = ("spark.fire", "spark.terrain", "spark.fields", "spark.acoustic")
ALLOWED_LEAF_PACKAGE = "spark.atmosphere"


def normalize_module(module: str) -> str:
    return (
        module[len(SOURCE_ROOT_PREFIX) :]
        if module.startswith(SOURCE_ROOT_PREFIX)
        else module
    )


def collect_imported_modules(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(normalize_module(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(normalize_module(node.module))
    return modules


def collect_all_inverse_imports() -> set[str]:
    modules: set[str] = set()
    for source_path in INVERSE_PACKAGE_PATH.glob("*.py"):
        modules |= collect_imported_modules(source_path)
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


def test_the_collector_sees_imports_under_the_current_convention() -> None:
    imported = collect_all_inverse_imports()
    assert any(module.startswith(ALLOWED_LEAF_PACKAGE) for module in imported), (
        "no import of the allowed leaf package was collected, so the import-root "
        "convention has changed and the forbidden-package match is comparing "
        "against strings that can never occur"
    )


def test_the_inverse_package_is_not_empty() -> None:
    assert len(list(INVERSE_PACKAGE_PATH.glob("*.py"))) > 1
