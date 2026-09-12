import dataclasses
from pathlib import Path

import pytest

from src.config.component_registry import (
    ATMOSPHERIC_ABSORPTION_CHANNEL,
    SPREAD_ENGINE_REGISTRY,
)
from src.config.simulation_configuration import (
    ForwardSimulationConfiguration,
    load_forward_simulation_configuration,
)
from src.config.simulation_context_factory import build_simulation_context
from src.spark.acoustic.channel_protocol import ChannelProtocol
from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.terrain.mesh_protocol import MeshProtocol

LADDER_DIRECTORIES = (
    Path("configs/e1"),
    Path("configs/e2"),
    Path("configs/e3"),
    Path("configs/e4"),
)
EXPECTED_ENGINE_NAMES = {
    Path("configs/e1"): "static_source",
    Path("configs/e2"): "static_source",
    Path("configs/e3"): "rate_of_spread",
    Path("configs/e4"): "rate_of_spread",
}
EXPECTED_IGNITION_POINT_COUNTS = {
    Path("configs/e1"): 1,
    Path("configs/e2"): 2,
    Path("configs/e3"): 1,
    Path("configs/e4"): 2,
}


def load(configuration_directory: Path) -> ForwardSimulationConfiguration:
    return load_forward_simulation_configuration(configuration_directory)


@pytest.mark.parametrize("configuration_directory", LADDER_DIRECTORIES)
def test_every_rung_builds_a_context(configuration_directory: Path) -> None:
    context = build_simulation_context(load(configuration_directory))
    assert isinstance(context.mesh, MeshProtocol)
    assert isinstance(context.spread_engine, SpreadEngineProtocol)
    assert isinstance(context.fuel_field, ScalarFieldProtocol)
    assert isinstance(context.elevation_field, ScalarFieldProtocol)
    assert isinstance(context.wind_field, VectorFieldProtocol)
    assert isinstance(context.channel, ChannelProtocol)


@pytest.mark.parametrize("configuration_directory", LADDER_DIRECTORIES)
def test_the_engine_is_the_class_the_configuration_names(
    configuration_directory: Path,
) -> None:
    context = build_simulation_context(load(configuration_directory))
    expected_name = EXPECTED_ENGINE_NAMES[configuration_directory]
    assert isinstance(context.spread_engine, SPREAD_ENGINE_REGISTRY[expected_name])


@pytest.mark.parametrize("configuration_directory", LADDER_DIRECTORIES)
def test_every_ignition_point_resolves_to_a_cell(
    configuration_directory: Path,
) -> None:
    context = build_simulation_context(load(configuration_directory))
    assert (
        context.ignition_cell_indices.size
        == EXPECTED_IGNITION_POINT_COUNTS[configuration_directory]
    )
    assert bool((context.ignition_cell_indices >= 0).all())
    assert bool((context.ignition_cell_indices < context.mesh.cell_count).all())


@pytest.mark.parametrize("configuration_directory", LADDER_DIRECTORIES)
def test_every_rung_places_at_least_two_receivers(
    configuration_directory: Path,
) -> None:
    context = build_simulation_context(load(configuration_directory))
    assert context.receiver_positions_xy_m.shape[0] >= 2
    assert context.receiver_positions_xy_m.shape[1] == 2


def test_ignition_points_land_on_the_nearest_cell_centre() -> None:
    config = load(Path("configs/e3"))
    context = build_simulation_context(config)
    requested_xy_m = (
        config.fire.ignition_points_xy_fraction[0][0] * config.mesh.extent_x_m,
        config.fire.ignition_points_xy_fraction[0][1] * config.mesh.extent_y_m,
    )
    resolved_xy_m = context.mesh.cell_positions_xyz[
        context.ignition_cell_indices[0], :2
    ]
    assert abs(resolved_xy_m[0] - requested_xy_m[0]) <= config.mesh.cell_spacing_m
    assert abs(resolved_xy_m[1] - requested_xy_m[1]) <= config.mesh.cell_spacing_m


def test_the_frequency_dependent_channel_is_reachable_by_name() -> None:
    config = load(Path("configs/e3"))
    context = build_simulation_context(
        dataclasses.replace(
            config,
            acoustic=dataclasses.replace(
                config.acoustic, channel_name=ATMOSPHERIC_ABSORPTION_CHANNEL
            ),
        )
    )
    assert isinstance(context.channel, ChannelProtocol)


def test_an_unregistered_engine_name_is_rejected() -> None:
    config = load(Path("configs/e3"))
    with pytest.raises(ValueError, match="unknown spread engine"):
        build_simulation_context(
            dataclasses.replace(
                config,
                fire=dataclasses.replace(config.fire, spread_engine_name="quantum"),
            )
        )


def test_an_unregistered_mesh_name_is_rejected() -> None:
    config = load(Path("configs/e3"))
    with pytest.raises(ValueError, match="unknown mesh"):
        build_simulation_context(
            dataclasses.replace(
                config, mesh=dataclasses.replace(config.mesh, mesh_type="hexagonal")
            )
        )


def test_an_unknown_fuel_preset_is_rejected() -> None:
    config = load(Path("configs/e3"))
    with pytest.raises(ValueError, match="unknown fuel preset"):
        build_simulation_context(
            dataclasses.replace(
                config, fire=dataclasses.replace(config.fire, fuel_preset_name="lava")
            )
        )


def build_automaton_fire(seed: int) -> object:
    """Runs the automaton for a few steps from a scene built with `seed`."""
    forward_config = load_forward_simulation_configuration(Path("configs/f2"))
    forward_config = dataclasses.replace(
        forward_config,
        fire=dataclasses.replace(forward_config.fire, spread_engine_random_seed=seed),
    )
    context = build_simulation_context(forward_config)
    state = context.spread_engine.initialize(
        context.mesh, context.fuel_field, context.wind_field
    )
    state = context.spread_engine.ignite_cells(state, context.ignition_cell_indices)
    for _ in range(8):
        state = context.spread_engine.step(state, 1.8)
    return state.is_burning.copy()


def test_the_automaton_gives_the_same_fire_twice_from_one_seed() -> None:
    """Without this the scene is a different experiment on every run.

    The engine's own default leaves its stream unseeded, so two runs of one
    configuration produced 159 and 166 burning cells at the same instant and a
    final-frame sector overlap of 0.68 against 0.28. Nothing measured on a
    probabilistic spread model means anything until this holds.
    """
    assert (build_automaton_fire(0) == build_automaton_fire(0)).all()


def test_two_seeds_give_two_fires() -> None:
    """The seed has to be reaching the engine, not merely being stored."""
    assert not (build_automaton_fire(0) == build_automaton_fire(7)).all()
