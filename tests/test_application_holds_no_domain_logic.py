"""The `app/` boundary, enforced rather than documented.

The application is a run picker, parameter widgets, calls into the composition
root and the plotters, and export buttons. Two rules keep it that way: nothing
in `src/` may import it, and it may not build physics objects of its own.
"""

import ast
from pathlib import Path

import pytest

APPLICATION_ROOT = Path("app")
SOURCE_ROOT = Path("src")
FORBIDDEN_CONSTRUCTIONS = (
    "SquareGridMesh",
    "RateOfSpreadEngine",
    "CellularAutomatonSpreadEngine",
    "StaticSourceSpreadEngine",
    "ExponentialAttenuationChannel",
    "AtmosphericAbsorptionChannel",
)


def application_modules() -> list[Path]:
    return sorted(APPLICATION_ROOT.glob("*.py"))


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_application_package_exists() -> None:
    assert APPLICATION_ROOT.is_dir()
    assert len(application_modules()) > 0


def test_nothing_in_src_imports_the_application() -> None:
    offenders = [
        str(path)
        for path in SOURCE_ROOT.rglob("*.py")
        if any(
            name == "app" or name.startswith("app.")
            for name in imported_module_names(parse_module(path))
        )
    ]
    assert offenders == [], f"src must not import app: {offenders}"


@pytest.mark.parametrize("module_path", application_modules(), ids=lambda p: p.name)
def test_the_application_builds_no_physics_objects(module_path: Path) -> None:
    tree = parse_module(module_path)
    constructed = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    offenders = constructed.intersection(FORBIDDEN_CONSTRUCTIONS)
    assert offenders == set(), (
        f"{module_path} constructs {sorted(offenders)}; the composition root owns that"
    )


@pytest.mark.parametrize("module_path", application_modules(), ids=lambda p: p.name)
def test_the_application_never_reaches_into_the_estimator(module_path: Path) -> None:
    offenders = [
        name
        for name in imported_module_names(parse_module(module_path))
        if name.startswith("src.spark.inverse")
    ]
    assert offenders == [], (
        f"{module_path} imports {offenders}; the application reads the "
        "estimator's metrics documents, it does not run it"
    )
