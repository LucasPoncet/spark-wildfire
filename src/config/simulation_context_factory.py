"""Turns a configuration into live objects. The only file that builds concretes.

Every other module in the repository sees protocols. This one reads the names
a configuration carries, looks them up in `component_registry`, instantiates,
and hands back a `SimulationContext`. If a concrete class is *constructed*
anywhere else in `src/` or `scripts/`, the composition root has leaked.
"""

import numpy as np

from src.config.component_registry import (
    ATMOSPHERIC_ABSORPTION_CHANNEL,
    CHANNEL_REGISTRY,
    EXPONENTIAL_ATTENUATION_CHANNEL,
    MESH_REGISTRY,
    PATCHY_DENSITY_FIELD,
    PATCHY_TREES_FIELD,
    RANDOM_TREES_FIELD,
    SCALAR_FIELD_REGISTRY,
    SPREAD_ENGINE_REGISTRY,
    SQUARE_GRID_MESH,
    UNIFORM_SCALAR_FIELD,
    VECTOR_FIELD_REGISTRY,
    resolve_registered_name,
)
from src.config.simulation_configuration import ForwardSimulationConfiguration
from src.config.simulation_context import SimulationContext
from src.spark.acoustic.channel_protocol import ChannelProtocol
from src.spark.acoustic.receiver_placement import place_receivers_from_configuration
from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.terrain.mesh_protocol import MeshProtocol
from src.utils.array_types import Int64Array

FUEL_PRESET_REGISTRY = {"pine_needle_litter": FuelProperties.pine_needle_litter}
CHANNEL_ANALYSIS_FREQUENCY_COUNT: int = 64
CHANNEL_MINIMUM_FREQUENCY_HZ: float = 20.0


def build_fuel(
    fuel_preset_name: str, moisture_content_fraction: float
) -> FuelProperties:
    """Look a fuel preset up by name and set its moisture.

    Args:
        fuel_preset_name: Key into the supported preset table.
        moisture_content_fraction: Water carried by the fuel, as a fraction of
            dry mass.

    Returns:
        The fuel bed.

    Raises:
        ValueError: If the preset name is not supported, or the moisture is
            negative.
    """
    if fuel_preset_name not in FUEL_PRESET_REGISTRY:
        supported = ", ".join(sorted(FUEL_PRESET_REGISTRY))
        raise ValueError(
            f"unknown fuel preset {fuel_preset_name!r}; registered: {supported}"
        )
    if moisture_content_fraction < 0.0:
        raise ValueError("fuel moisture content must be non-negative")
    return FUEL_PRESET_REGISTRY[fuel_preset_name](moisture_content_fraction)


def build_mesh(config: ForwardSimulationConfiguration) -> MeshProtocol:
    """Instantiate the mesh the configuration names.

    Args:
        config: Run configuration.

    Returns:
        The mesh, behind its protocol.

    Raises:
        ValueError: If the mesh name is not registered.
    """
    mesh_class = resolve_registered_name(MESH_REGISTRY, config.mesh.mesh_type, "mesh")
    if config.mesh.mesh_type == SQUARE_GRID_MESH:
        mesh: MeshProtocol = mesh_class.from_values(
            config.mesh.extent_x_m,
            config.mesh.extent_y_m,
            config.mesh.cell_spacing_m,
            config.mesh.use_diagonal_neighbors,
        )
        return mesh
    raise ValueError(f"no constructor wired for mesh {config.mesh.mesh_type!r}")


def build_scalar_field(field_type: str, constant_value: float) -> ScalarFieldProtocol:
    """Instantiate a scalar field by registered name.

    Args:
        field_type: Registered name of the field.
        constant_value: Value used by the uniform field.

    Returns:
        The field, behind its protocol.

    Raises:
        ValueError: If the name is registered but has no constructor wired.
    """
    field_class = resolve_registered_name(
        SCALAR_FIELD_REGISTRY, field_type, "scalar field"
    )
    if field_type == UNIFORM_SCALAR_FIELD:
        field: ScalarFieldProtocol = field_class(constant_value)
        return field
    raise ValueError(f"no constructor wired for scalar field {field_type!r}")


def build_fuel_field(
    config: ForwardSimulationConfiguration, fuel: FuelProperties
) -> ScalarFieldProtocol:
    """Instantiate the fuel field the configuration names.

    Three shapes are wired. `uniform` puts the whole fuel bed load everywhere,
    which is what every scene in the ladder uses so that only geometry drives
    the received levels. `random_trees` scatters a homogeneous Poisson forest,
    and `patchy_trees` thins that forest with Gaussian density bumps — both
    make the front visibly wander, which is what the live display is for.

    Args:
        config: Run configuration.
        fuel: The fuel bed, supplying the uniform load.

    Returns:
        The field, behind its protocol.

    Raises:
        ValueError: If the field name is not registered or not wired.
    """
    field_type = config.fire.fuel_field_type
    field_class = resolve_registered_name(
        SCALAR_FIELD_REGISTRY, field_type, "scalar field"
    )
    if field_type == UNIFORM_SCALAR_FIELD:
        return build_scalar_field(field_type, fuel.fuel_load_kg_per_m2)
    if field_type not in (RANDOM_TREES_FIELD, PATCHY_TREES_FIELD):
        raise ValueError(f"no constructor wired for fuel field {field_type!r}")

    density_field = (
        build_patch_density_field(config) if field_type == PATCHY_TREES_FIELD else None
    )
    field: ScalarFieldProtocol = field_class(
        extent_x_m=config.mesh.extent_x_m,
        extent_y_m=config.mesh.extent_y_m,
        tree_density_per_m2=config.fire.tree_density_per_m2,
        tree_fuel_load_kg_per_m2=config.fire.tree_fuel_load_kg_per_m2,
        influence_radius_m=config.fire.tree_influence_radius_m,
        seed=config.fire.tree_layout_seed,
        density_field=density_field,
    )
    return field


def build_patch_density_field(
    config: ForwardSimulationConfiguration,
) -> ScalarFieldProtocol:
    """Instantiate the density field that thins a patchy tree layout.

    Args:
        config: Run configuration, carrying the patch centres and radius.

    Returns:
        The density field, behind its protocol.
    """
    patch_class = resolve_registered_name(
        SCALAR_FIELD_REGISTRY, PATCHY_DENSITY_FIELD, "scalar field"
    )
    patch_centers_xy_m = np.array(
        [
            [x_fraction * config.mesh.extent_x_m, y_fraction * config.mesh.extent_y_m]
            for x_fraction, y_fraction in config.fire.tree_patch_centers_xy_fraction
        ],
        dtype=np.float64,
    )
    density_field: ScalarFieldProtocol = patch_class(
        patch_centers_xy=patch_centers_xy_m,
        patch_peak_density_per_m2=config.fire.tree_density_per_m2,
        patch_radius_m=config.fire.tree_patch_radius_fraction * config.mesh.extent_x_m,
        background_density_per_m2=config.fire.tree_background_density_per_m2,
    )
    return density_field


def build_wind_field(config: ForwardSimulationConfiguration) -> VectorFieldProtocol:
    """Instantiate the wind field the configuration names.

    Args:
        config: Run configuration.

    Returns:
        The field, behind its protocol.

    Raises:
        ValueError: If the wind field name is not registered.
    """
    field_class = resolve_registered_name(
        VECTOR_FIELD_REGISTRY, config.wind.wind_field_type, "vector field"
    )
    wind_vector_xyz = np.array(
        [
            config.wind.wind_speed_m_per_s * np.cos(config.wind.wind_bearing_rad),
            config.wind.wind_speed_m_per_s * np.sin(config.wind.wind_bearing_rad),
            0.0,
        ],
        dtype=np.float64,
    )
    field: VectorFieldProtocol = field_class(wind_vector_xyz)
    return field


def build_spread_engine(
    config: ForwardSimulationConfiguration, fuel: FuelProperties
) -> SpreadEngineProtocol:
    """Instantiate the spread engine the configuration names.

    The rate-of-spread engine takes the fuel bed; the cellular automaton and
    the static source engine take their own defaults.

    Args:
        config: Run configuration.
        fuel: The fuel bed.

    Returns:
        The engine, behind its protocol.

    Raises:
        ValueError: If the engine name is not registered.
    """
    engine_class = resolve_registered_name(
        SPREAD_ENGINE_REGISTRY, config.fire.spread_engine_name, "spread engine"
    )
    if config.fire.spread_engine_name == "rate_of_spread":
        engine: SpreadEngineProtocol = engine_class(fuel)
        return engine
    engine = engine_class()
    return engine


def build_channel(config: ForwardSimulationConfiguration) -> ChannelProtocol:
    """Instantiate the propagation channel the configuration names.

    Args:
        config: Run configuration.

    Returns:
        The channel, behind its protocol.

    Raises:
        ValueError: If the channel name is registered but not wired.
    """
    channel_class = resolve_registered_name(
        CHANNEL_REGISTRY, config.acoustic.channel_name, "channel"
    )
    if config.acoustic.channel_name == EXPONENTIAL_ATTENUATION_CHANNEL:
        channel: ChannelProtocol = channel_class(
            config.acoustic.attenuation_coefficient_per_m,
            config.acoustic.reference_distance_m,
        )
        return channel
    if config.acoustic.channel_name == ATMOSPHERIC_ABSORPTION_CHANNEL:
        channel = channel_class(
            config.atmosphere,
            np.geomspace(
                CHANNEL_MINIMUM_FREQUENCY_HZ,
                0.5 * config.acoustic.sample_rate_hz,
                CHANNEL_ANALYSIS_FREQUENCY_COUNT,
            ),
            config.acoustic.reference_distance_m,
        )
        return channel
    raise ValueError(
        f"no constructor wired for channel {config.acoustic.channel_name!r}"
    )


def compute_ignition_cell_indices(
    mesh: MeshProtocol,
    extent_x_m: float,
    extent_y_m: float,
    ignition_points_xy_fraction: tuple[tuple[float, float], ...],
) -> Int64Array:
    """Turn fractional grid coordinates into mesh cell indices.

    Works through positions rather than row and column arithmetic, so it holds
    for any mesh satisfying the protocol rather than only for a square grid.

    Args:
        mesh: The grid the fire runs on.
        extent_x_m: Domain extent along x, in metres.
        extent_y_m: Domain extent along y, in metres.
        ignition_points_xy_fraction: Fractions of the extent along x and y.

    Returns:
        Int64 array of shape (k,), the nearest cell to each ignition point.
    """
    requested_positions_xy_m = np.array(
        [
            [x_fraction * extent_x_m, y_fraction * extent_y_m]
            for x_fraction, y_fraction in ignition_points_xy_fraction
        ],
        dtype=np.float64,
    )
    squared_distances_m2 = np.sum(
        (mesh.cell_positions_xyz[None, :, :2] - requested_positions_xy_m[:, None, :])
        ** 2,
        axis=2,
    )
    return np.asarray(np.argmin(squared_distances_m2, axis=1), dtype=np.int64)


def build_simulation_context(
    config: ForwardSimulationConfiguration,
) -> SimulationContext:
    """Build every live object one forward run needs.

    Args:
        config: The resolved run configuration.

    Returns:
        The context, holding only protocol-typed references.

    Raises:
        ValueError: If any component name is unregistered or unwired.
    """
    mesh = build_mesh(config)
    fuel = build_fuel(
        config.fire.fuel_preset_name, config.fire.fuel_moisture_content_fraction
    )
    return SimulationContext(
        mesh=mesh,
        elevation_field=build_scalar_field(
            config.mesh.elevation_field_type, config.mesh.elevation_m
        ),
        fuel_field=build_fuel_field(config, fuel),
        wind_field=build_wind_field(config),
        spread_engine=build_spread_engine(config, fuel),
        channel=build_channel(config),
        fuel=fuel,
        ignition_cell_indices=compute_ignition_cell_indices(
            mesh,
            config.mesh.extent_x_m,
            config.mesh.extent_y_m,
            config.fire.ignition_points_xy_fraction,
        ),
        receiver_positions_xy_m=place_receivers_from_configuration(
            config.receiver, config.mesh.extent_x_m, config.mesh.extent_y_m
        ),
    )
