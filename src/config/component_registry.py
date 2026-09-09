"""Name-to-class maps. No logic, and nothing here is ever instantiated.

A configuration file names a component as a string; these tables are the only
place that string becomes a class. Adding an implementation means adding one
entry here and one line in `simulation_context_factory`, never editing a
script.

This module necessarily imports concrete classes, because a name-to-class map
has nothing else to map to. `simulation_context_factory` is the only module
that *calls* them.
"""

from src.spark.acoustic.exponential_attenuation_channel import (
    AtmosphericAbsorptionChannel,
    ExponentialAttenuationChannel,
)
from src.spark.fields.patchy_density_field import PatchyDensityField
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fields.uniform_vector_field import UniformVectorField
from src.spark.fire.cellular_automaton_spread_engine import (
    CellularAutomatonSpreadEngine,
)
from src.spark.fire.rate_of_spread_engine import RateOfSpreadEngine
from src.spark.fire.static_source_spread_engine import StaticSourceSpreadEngine
from src.spark.terrain.square_grid_mesh import SquareGridMesh

SQUARE_GRID_MESH: str = "square_grid"
RATE_OF_SPREAD_ENGINE: str = "rate_of_spread"
CELLULAR_AUTOMATON_ENGINE: str = "cellular_automaton"
STATIC_SOURCE_ENGINE: str = "static_source"
EXPONENTIAL_ATTENUATION_CHANNEL: str = "exponential_attenuation"
ATMOSPHERIC_ABSORPTION_CHANNEL: str = "atmospheric_absorption"
UNIFORM_SCALAR_FIELD: str = "uniform"
PATCHY_DENSITY_FIELD: str = "patchy"
CONSTANT_WIND_FIELD: str = "constant"

MESH_REGISTRY: dict[str, type[SquareGridMesh]] = {
    SQUARE_GRID_MESH: SquareGridMesh,
}

SPREAD_ENGINE_REGISTRY: dict[str, type] = {
    RATE_OF_SPREAD_ENGINE: RateOfSpreadEngine,
    CELLULAR_AUTOMATON_ENGINE: CellularAutomatonSpreadEngine,
    STATIC_SOURCE_ENGINE: StaticSourceSpreadEngine,
}

CHANNEL_REGISTRY: dict[str, type] = {
    EXPONENTIAL_ATTENUATION_CHANNEL: ExponentialAttenuationChannel,
    ATMOSPHERIC_ABSORPTION_CHANNEL: AtmosphericAbsorptionChannel,
}

SCALAR_FIELD_REGISTRY: dict[str, type] = {
    UNIFORM_SCALAR_FIELD: UniformScalarField,
    PATCHY_DENSITY_FIELD: PatchyDensityField,
}

VECTOR_FIELD_REGISTRY: dict[str, type] = {
    CONSTANT_WIND_FIELD: UniformVectorField,
}


def resolve_registered_name[RegisteredT](
    registry: dict[str, RegisteredT], name: str, kind: str
) -> RegisteredT:
    """Look one name up in one registry.

    Args:
        registry: The table to look in.
        name: The name a configuration file carries.
        kind: What is being resolved, used in the error message.

    Returns:
        The registered class.

    Raises:
        ValueError: If the name is not registered, listing what is.
    """
    if name not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"unknown {kind} {name!r}; registered: {supported}")
    return registry[name]
