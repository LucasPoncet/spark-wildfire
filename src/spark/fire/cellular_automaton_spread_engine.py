"""Alexandridis-style probabilistic cellular automaton fire spread engine.

This is a mesh-native translation of the original 2D-grid prototype: the
same three rules (extinction after a fixed burn duration, neighbor-driven
ignition probability modulated by wind and local fuel, no cell reignites
once burnt), expressed over `MeshProtocol.neighbor_indices` instead of a
2D array and its manual `(dx, dy)` shifts. Because the rules are phrased
in seconds rather than in turns, the ignition probability contributed by
any one already-burning neighbor is independent of `dt`.

This does not make the whole simulation dt-independent: one call to `step`
only lets fire cross one mesh edge, so calling it with a smaller `dt` more
often still resolves a fire front advancing through more cells over the
same wall-clock time. Pick `dt` relative to `cell_spacing_m` and the
fastest rate of spread you expect, the same way you would pick a
CFL-bounded timestep for any other explicit front-tracking scheme.

Queries the mesh for neighbors and directions, queries the wind field for
wind at each cell, applies probabilistic ignition. Never implements rate
of spread physics itself — `rate_of_spread_engine.py` is the physically
grounded alternative satisfying the same SpreadEngineProtocol.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.vector_field_protocol import VectorFieldProtocol
from src.spark.fire.fire_state import FireState
from src.spark.terrain.mesh_protocol import MeshProtocol

WIND_SPEED_EPSILON_M_PER_S = 1e-9
FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION = 1e-6


@dataclass(frozen=True, slots=True)
class CellularAutomatonSpreadEngineConfig:
    """Calibration constants for the probabilistic ignition rule.

    Attributes:
        base_ignition_probability: Probability that a cell with one burning
            neighbor, average fuel density and no wind ignites over one
            `probability_reference_time_step_s` of simulated time.
        relative_humidity_fraction: Ambient relative humidity, 0 to 1.
            Higher humidity suppresses ignition.
        humidity_suppression_coefficient: Strength of the humidity
            suppression effect on `base_ignition_probability`.
        probability_reference_time_step_s: The timestep, in seconds, that
            `base_ignition_probability` was calibrated against. Ignition
            probability at other timesteps is derived so the per-second
            ignition rate stays constant regardless of `dt`.
        burn_duration_s: How long a cell stays on fire before burning out.
        wind_influence_strength: How strongly alignment between the wind
            direction and the direction fire travels raises or lowers the
            ignition probability.
        ignition_probability_wind_factor_min: Lower clip on the wind
            multiplier applied to the ignition probability.
        ignition_probability_wind_factor_max: Upper clip on the wind
            multiplier applied to the ignition probability.
        fuel_factor_base: Ignition probability multiplier at zero fuel
            density.
        fuel_factor_gain: Additional ignition probability multiplier at
            full (1.0) fuel density.
        ignition_probability_max: Hard cap on the per-neighbor ignition
            probability.
        random_seed: Seed for the engine's own random number generator,
            reset on every call to `initialize`.
    """

    base_ignition_probability: float = 0.28
    relative_humidity_fraction: float = 0.20
    humidity_suppression_coefficient: float = 0.9
    probability_reference_time_step_s: float = 1.0
    burn_duration_s: float = 20.0
    wind_influence_strength: float = 0.48
    ignition_probability_wind_factor_min: float = 0.15
    ignition_probability_wind_factor_max: float = 2.2
    fuel_factor_base: float = 0.6
    fuel_factor_gain: float = 0.6
    ignition_probability_max: float = 0.98
    random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class _BoundDomain:
    """Mesh and sampled fields the engine is bound to by `initialize`."""

    mesh: MeshProtocol
    wind_field: VectorFieldProtocol
    fuel_density_fraction: npt.NDArray[np.float64]
    is_flammable: npt.NDArray[np.bool_]


class CellularAutomatonSpreadEngine:
    """Probabilistic ignition cellular automaton, satisfying SpreadEngineProtocol."""

    def __init__(
        self, config: CellularAutomatonSpreadEngineConfig | None = None
    ) -> None:
        """Store the calibration config.

        Args:
            config: Calibration constants. Defaults are used when omitted.
        """
        self._config = (
            config if config is not None else CellularAutomatonSpreadEngineConfig()
        )
        self._random_number_generator = np.random.default_rng(self._config.random_seed)
        self._domain: _BoundDomain | None = None

    def initialize(
        self,
        mesh: MeshProtocol,
        fuel_field: ScalarFieldProtocol,
        wind_field: VectorFieldProtocol,
    ) -> FireState:
        """Bind the engine to a mesh and its fields, and return the unburnt state.

        Resets the random stream so two runs from the same config and seed
        are identical.

        Args:
            mesh: Geometry and connectivity the fire will propagate over.
            fuel_field: Fuel density, clipped to 0 to 1, at each cell position.
            wind_field: Wind vector, in metres per second, at each cell position.

        Returns:
            A FireState with no cell burning, at time 0.
        """
        fuel_density_fraction = np.clip(
            fuel_field.sample(mesh.cell_positions_xyz), 0.0, 1.0
        )
        self._random_number_generator = np.random.default_rng(self._config.random_seed)
        self._domain = _BoundDomain(
            mesh=mesh,
            wind_field=wind_field,
            fuel_density_fraction=fuel_density_fraction,
            is_flammable=fuel_density_fraction
            > FUEL_DENSITY_IGNITION_THRESHOLD_FRACTION,
        )
        return FireState(
            ignition_times_s=np.full(mesh.cell_count, np.inf, dtype=np.float64),
            burnout_times_s=np.full(mesh.cell_count, np.inf, dtype=np.float64),
            is_burning=np.zeros(mesh.cell_count, dtype=np.bool_),
            has_ignited=np.zeros(mesh.cell_count, dtype=np.bool_),
            current_time_s=0.0,
        )

    def _require_bound_domain(self) -> _BoundDomain:
        """Return the bound domain, or fail with a message naming the cause.

        Returns:
            The mesh and fields bound by the last call to `initialize`.

        Raises:
            RuntimeError: If `initialize` has not been called.
        """
        if self._domain is None:
            raise RuntimeError(
                "CellularAutomatonSpreadEngine.initialize must be called "
                "before igniting cells or stepping the simulation"
            )
        return self._domain

    def ignite_cells(
        self, state: FireState, cell_indices: npt.NDArray[np.int64]
    ) -> FireState:
        """Start a fire at the given cells, at the state's current time.

        Cells that carry no fuel or have already ignited are skipped.

        Args:
            state: The state to ignite cells in.
            cell_indices: Int64 array of shape (k,), indices of cells to ignite.

        Returns:
            A new FireState at the same time, with the ignitable cells burning.
        """
        domain = self._require_bound_domain()
        is_requested = np.zeros(domain.mesh.cell_count, dtype=np.bool_)
        is_requested[cell_indices] = True
        newly_ignited = is_requested & domain.is_flammable & ~state.has_ignited
        return FireState(
            ignition_times_s=np.where(
                newly_ignited, state.current_time_s, state.ignition_times_s
            ),
            burnout_times_s=np.where(
                newly_ignited,
                state.current_time_s + self._config.burn_duration_s,
                state.burnout_times_s,
            ),
            is_burning=state.is_burning | newly_ignited,
            has_ignited=state.has_ignited | newly_ignited,
            current_time_s=state.current_time_s,
        )

    def step(self, state: FireState, dt: float) -> FireState:
        """Advance the simulation by `dt` seconds, applying the three CA rules.

        Rule 1: a burning cell whose burn duration has elapsed extinguishes.
        Rule 2: an unburnt, fuelled cell ignites with a probability that
            rises with its count of burning neighbors, its own fuel density,
            and wind blowing from those neighbors towards it.
        Rule 3: a cell that has ever ignited can never ignite again, which
            follows from only ever testing `~state.has_ignited`.

        Args:
            state: The state to advance from.
            dt: Timestep, in seconds.

        Returns:
            A new FireState at time `state.current_time_s + dt`.
        """
        domain = self._require_bound_domain()
        mesh = domain.mesh
        new_current_time_s = state.current_time_s + dt

        is_burning_after_extinction = state.is_burning & (
            state.burnout_times_s > new_current_time_s
        )

        wind_vectors_xyz = domain.wind_field.sample(mesh.cell_positions_xyz)
        wind_speeds_m_per_s = np.linalg.norm(wind_vectors_xyz, axis=1)
        has_wind = wind_speeds_m_per_s > WIND_SPEED_EPSILON_M_PER_S
        safe_wind_speeds_m_per_s = np.where(has_wind, wind_speeds_m_per_s, 1.0)
        wind_unit_vectors_xyz = wind_vectors_xyz / safe_wind_speeds_m_per_s[:, None]
        wind_unit_vectors_xyz[~has_wind] = 0.0

        neighbor_indices = mesh.neighbor_indices
        is_valid_neighbor = neighbor_indices >= 0
        safe_neighbor_indices = np.where(is_valid_neighbor, neighbor_indices, 0)
        is_neighbor_burning = (
            is_burning_after_extinction[safe_neighbor_indices] & is_valid_neighbor
        )

        wind_alignment_neighbor_to_cell = -np.einsum(
            "nkj,nj->nk", mesh.neighbor_unit_directions_xyz, wind_unit_vectors_xyz
        )
        wind_factor = np.clip(
            1.0
            + self._config.wind_influence_strength * wind_alignment_neighbor_to_cell,
            self._config.ignition_probability_wind_factor_min,
            self._config.ignition_probability_wind_factor_max,
        )

        humidity_suppression = 1.0 - (
            self._config.relative_humidity_fraction
            * self._config.humidity_suppression_coefficient
        )
        base_probability_per_reference_step = (
            self._config.base_ignition_probability * humidity_suppression
        )
        fuel_factor = (
            self._config.fuel_factor_base
            + domain.fuel_density_fraction * self._config.fuel_factor_gain
        )
        probability_per_reference_step = np.clip(
            base_probability_per_reference_step * wind_factor * fuel_factor[:, None],
            0.0,
            self._config.ignition_probability_max,
        )

        time_step_ratio = dt / self._config.probability_reference_time_step_s
        probability_per_actual_step = 1.0 - np.power(
            1.0 - probability_per_reference_step, time_step_ratio
        )

        probability_of_no_ignition_from_neighbor = np.where(
            is_neighbor_burning, 1.0 - probability_per_actual_step, 1.0
        )
        probability_of_ignition = 1.0 - np.prod(
            probability_of_no_ignition_from_neighbor, axis=1
        )

        can_ignite = domain.is_flammable & ~state.has_ignited
        ignition_roll = self._random_number_generator.random(mesh.cell_count)
        newly_ignited = can_ignite & (ignition_roll < probability_of_ignition)

        return FireState(
            ignition_times_s=np.where(
                newly_ignited, new_current_time_s, state.ignition_times_s
            ),
            burnout_times_s=np.where(
                newly_ignited,
                new_current_time_s + self._config.burn_duration_s,
                state.burnout_times_s,
            ),
            is_burning=is_burning_after_extinction | newly_ignited,
            has_ignited=state.has_ignited | newly_ignited,
            current_time_s=new_current_time_s,
        )
