import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.config.component_registry import (
    PATCHY_TREES_FIELD,
    RANDOM_TREES_FIELD,
    UNIFORM_SCALAR_FIELD,
)
from src.config.simulation_configuration import (
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
)
from src.config.simulation_context_factory import build_fuel, build_simulation_context

SMALL_SCENE = Path("configs/e1")
TREE_FIELD_TYPES = (RANDOM_TREES_FIELD, PATCHY_TREES_FIELD)


def load_scene() -> ForwardSimulationConfiguration:
    return load_forward_simulation_configuration(SMALL_SCENE)


def with_fuel_field(field_type: str) -> ForwardSimulationConfiguration:
    config = load_scene()
    return dataclasses.replace(
        config, fire=dataclasses.replace(config.fire, fuel_field_type=field_type)
    )


def test_the_uniform_field_puts_the_whole_fuel_load_everywhere() -> None:
    context = build_simulation_context(with_fuel_field(UNIFORM_SCALAR_FIELD))
    sampled = context.fuel_field.sample(context.mesh.cell_positions_xyz)
    np.testing.assert_allclose(sampled, context.fuel.fuel_load_kg_per_m2)


@pytest.mark.parametrize("field_type", TREE_FIELD_TYPES)
def test_a_tree_field_varies_across_the_domain(field_type: str) -> None:
    context = build_simulation_context(with_fuel_field(field_type))
    sampled = context.fuel_field.sample(context.mesh.cell_positions_xyz)
    assert sampled.std() > 0.0
    assert sampled.min() >= 0.0


@pytest.mark.parametrize("field_type", TREE_FIELD_TYPES)
def test_a_tree_layout_is_reproducible_from_its_seed(field_type: str) -> None:
    config = with_fuel_field(field_type)
    first = build_simulation_context(config)
    second = build_simulation_context(config)
    np.testing.assert_array_equal(
        first.fuel_field.sample(first.mesh.cell_positions_xyz),
        second.fuel_field.sample(second.mesh.cell_positions_xyz),
    )


@pytest.mark.parametrize("field_type", TREE_FIELD_TYPES)
def test_a_different_seed_gives_a_different_layout(field_type: str) -> None:
    config = with_fuel_field(field_type)
    other = dataclasses.replace(
        config,
        fire=dataclasses.replace(
            config.fire, tree_layout_seed=config.fire.tree_layout_seed + 1
        ),
    )
    assert not np.array_equal(
        build_simulation_context(config).fuel_field.sample(
            build_simulation_context(config).mesh.cell_positions_xyz
        ),
        build_simulation_context(other).fuel_field.sample(
            build_simulation_context(other).mesh.cell_positions_xyz
        ),
    )


def test_patchy_trees_leave_more_bare_ground_than_a_uniform_forest() -> None:
    random_context = build_simulation_context(with_fuel_field(RANDOM_TREES_FIELD))
    patchy_context = build_simulation_context(with_fuel_field(PATCHY_TREES_FIELD))
    random_bare = float(
        np.mean(
            random_context.fuel_field.sample(random_context.mesh.cell_positions_xyz)
            <= 1e-6
        )
    )
    patchy_bare = float(
        np.mean(
            patchy_context.fuel_field.sample(patchy_context.mesh.cell_positions_xyz)
            <= 1e-6
        )
    )
    assert patchy_bare > random_bare


def test_an_unwired_fuel_field_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scalar field"):
        build_simulation_context(with_fuel_field("bracken"))


def test_moisture_reaches_the_fuel_bed() -> None:
    config = load_scene()
    wetter = dataclasses.replace(
        config,
        fire=dataclasses.replace(config.fire, fuel_moisture_content_fraction=0.30),
    )
    assert build_simulation_context(wetter).fuel.moisture_content_fraction == 0.30


def test_negative_moisture_is_rejected() -> None:
    with pytest.raises(ValueError, match="moisture content must be non-negative"):
        build_fuel("pine_needle_litter", -0.01)


def test_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fuel preset"):
        build_fuel("lava", 0.1)
