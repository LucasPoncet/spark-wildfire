"""The live objects one run is built from, held behind their protocols.

Imports protocols only. A consumer holding a `SimulationContext` cannot tell
which concrete mesh, engine or channel it was handed, which is the whole
point: swapping any of them is a configuration change.
"""

from dataclasses import dataclass

from src.spark.acoustic.channel_protocol import ChannelProtocol
from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.fuel_properties import FuelProperties
from src.spark.fire.spread_engine_protocol import SpreadEngineProtocol
from src.spark.terrain.mesh_protocol import MeshProtocol
from src.utils.array_types import Float64Array, Int64Array


@dataclass(frozen=True)
class SimulationContext:
    """Everything a forward run needs, already instantiated.

    Attributes:
        mesh: Geometry and connectivity the fire propagates over.
        elevation_field: Ground height at each position.
        fuel_field: Fuel available at each position.
        wind_field: Wind vector at each position.
        spread_engine: The fire propagation model.
        channel: The propagation channel between sources and receivers.
        fuel: Fuel bed properties the engine and the source model share.
        ignition_cell_indices: Int64 array of shape (k,), where fires start.
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
    """

    mesh: MeshProtocol
    elevation_field: ScalarFieldProtocol
    fuel_field: ScalarFieldProtocol
    wind_field: VectorFieldProtocol
    spread_engine: SpreadEngineProtocol
    channel: ChannelProtocol
    fuel: FuelProperties
    ignition_cell_indices: Int64Array
    receiver_positions_xy_m: Float64Array
