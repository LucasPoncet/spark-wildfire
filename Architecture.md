# ARCHITECTURE.md

Reference for every file in the repository: what it does, what it depends on, what it must never do, and what extensions are planned. Read this before writing any code.

---

## Coding conventions

| Concern | Rule |
|---|---|
| Classes | `PascalCase` noun phrases: `SquareGridMesh`, `FuelProperties` |
| Functions | `snake_case` verb phrases: `compute_rate_of_spread_balbi_2009`, `build_simulation_context` |
| Variables | `snake_case`, physical quantities carry unit suffix: `wind_speed_m_per_s`, `cell_spacing_m` |
| Arrays | Carry shape meaning: `source_positions_xyz`, `gain_matrix_source_by_receiver` |
| Booleans | Read as assertions: `is_burning`, `has_ignited`, `uses_diagonal_neighbors` |
| Constants | `UPPER_SNAKE_CASE` with unit suffix: `STEFAN_BOLTZMANN_CONSTANT_W_PER_M2_K4` |
| Abbreviations | Never. `rate_of_spread`, not `ros`. `digital_elevation_model`, not `dem` |
| Paper notation | Allowed **only** inside the body of a function whose name states which equation it implements |
| Comments | None |
| Docstrings | Google style: one-line summary, then `Args:` / `Returns:` / `Raises:`. Concise — the signature carries the types |
| Type annotations | Mandatory on every function signature and dataclass field |
| Array operations | Vectorized numpy. No Python loops over cells |
| File scope | One primary class or one cohesive group of pure functions per file |
| Protocol files | Contain only the Protocol definition, no implementation |

---

## Dependency graph

```
src/spark/fields/  →  src/spark/terrain/  →  src/spark/fire/  →  src/spark/acoustic/
                                                                        │
                                                                   [signals only]
                                                                        │
                                                                   src/spark/inverse/

src/spark/atmosphere/   ← leaf; imported by BOTH acoustic/ and inverse/
src/audio/              ← leaf; signal operations, no domain imports
src/utils/array_types   ← leaf; array aliases, imported by nearly everything
src/config/             ← imported by scripts and app, imports Protocols only, except
                          simulation_context_factory, the one composition root
src/utils/io/           ← imported by scripts and app, imports domain types for serialization
src/utils/metrics/      ← leaf; scores position sets, imported by scripts
src/utils/visualization ← imports everything, imported by scripts and app
scripts/                ← imports everything; imported by nothing in src/, and by
                          one sibling script (the sweep reuses the run driver)
app/                    ← imports src/ and scripts' outputs, imported by nothing in src/
```

**`app/` is outside `src/` on purpose.** It holds no plotting logic and no physics: a run picker,
parameter widgets, calls into `simulation_context_factory` and the plotters, and export buttons.
Nothing in `src/` may import it, which
`tests/test_application_holds_no_domain_logic.py` checks by walking the import graph.

**Import root.** The repository root is on the path (`pythonpath = ["."]`) and `src/` is a real
package (`packages = ["src"]`, with `src/__init__.py`). Every import therefore carries the `src.`
prefix: `from src.spark.terrain...`, `from src.audio.octave_band_filter...`. A bare
`from spark....` resolves under pytest only by accident and breaks as soon as anything imports it
from the installed wheel, so it must not appear anywhere.

`src/__init__.py` must exist. Without it mypy cannot resolve `src.*` and aborts before checking
anything, which silently hides every other type error in the repository.

**Hard rule:** `src/spark/inverse/` must never import from `src/spark/fire/`, `src/spark/terrain/`, `src/spark/fields/`, or `src/spark/acoustic/`. Its only inputs are receiver signals and receiver positions. This prevents the pipeline from committing the leakage defect the paper criticizes. Enforced by `tests/test_inverse_does_not_import_forward_model.py`.

`inverse/` **may** import `audio/` and `atmosphere/`. Neither carries simulation state: `audio/` is
pure signal processing, and the air conditions an estimator uses are the ones its caller *assumes*,
supplied as an argument and never read back from the forward model. The isolation test names the
four forbidden packages explicitly and asserts they exist, so it cannot pass vacuously.

---

## `src/spark/fields/` — spatial fields

Every quantity that varies across the domain is a field: a function from position to value. The fire engine, the mesh, and the acoustic channel query fields by position and never know which implementation is behind them.

### `scalar_field_protocol.py` ✅ implemented

**Owns:** `ScalarFieldProtocol` with method `sample(positions_xyz: ndarray) -> ndarray`, mapping `(n, 3) -> (n,)`.
**Never:** Contains any implementation.

### `vector_field_protocol.py` ✅ implemented

**Owns:** `VectorFieldProtocol` with method `sample(positions_xyz: ndarray) -> ndarray`, mapping `(n, 3) -> (n, 3)`.
**Never:** Contains any implementation.

### `uniform_scalar_field.py` ✅ implemented

**Owns:** `UniformScalarField`. Returns the same float at every position. Used on day one for: elevation (0.0 = flat), fuel load (constant kg/m²), attenuation coefficient (constant m⁻¹).
**Never:** Knows about meshes or fire.

### `uniform_vector_field.py` ✅ implemented

**Owns:** `UniformVectorField`. Returns the same 3D vector everywhere. Day-one wind model: one direction and speed across the whole domain.
**Never:** Reads fire state.

### `constant_wind_field.py` ✅ implemented

**Owns:** `ConstantWindField`. Named constructor from `(wind_speed_m_per_s, wind_bearing_rad)` producing a 3D vector. Thin wrapper over `UniformVectorField` that makes the physical meaning explicit.
**Never:** Contains time-varying logic.

**Extension → `interpolated_wind_field.py`:** accepts a spatial grid of wind vectors (from a weather model or measured stations) and bilinearly interpolates at query positions. Same protocol, no downstream changes.

**Extension → `time_varying_wind_field.py`:** adds a `time_s` parameter to `sample`. This is the one extension that requires widening the Protocol signature. Defer until needed.

### `elevation_field_from_raster.py`

**Owns:** `ElevationFieldFromRaster`. Loads a DEM file (GeoTIFF, SRTM HGT), wraps a bilinear interpolator, answers elevation queries at arbitrary positions. Called once by the mesh constructor at build time to set z-coordinates.
**Never:** Computes slopes or normals — that is the mesh's job after it has the z-values.

**Extension → higher-resolution DEMs:** IGN RGE Alti 1 m, Copernicus EU-DEM 25 m. Same file, just a different file path at construction.

### `patchy_density_field.py` ✅ implemented

**Owns:** `PatchyDensityField`. Gaussian density bumps around hand-placed centers over a sparse background. Feeds `RandomTreePlacementField` as its `density_field`, producing dense stands and clearings instead of a uniform forest.
**Never:** Places trees itself — it answers density queries, nothing more.

### `random_tree_placement_field.py` ✅ implemented

**Owns:** Forest generation. Returns fuel load at any position from a spatial model of tree placement. This file is the single place where forest complexity lives.
**Never:** Simulates fire or computes propagation.

**Implementation note:** `sample` must never build a dense `(n_positions, n_trees)` array. It uses a `scipy.spatial.cKDTree` pair query truncated at 3σ, so cost scales with the trees actually near each query position. The dense form needs 642 GB at 1 km × 0.5 m spacing; the KD-tree form runs in 8 s.

**Extension path (each layer adds one feature, no rewrite):**

| Layer | What changes | Extra parameters |
|---|---|---|
| 1. Uniform | Use `UniformScalarField` instead, this file not needed yet | — |
| 2. Poisson placement | Trees as a Poisson point process, each with a Gaussian fuel footprint | `tree_density_per_m2`, `fuel_footprint_radius_m`, `seed` |
| 3. Clustered placement | Thomas cluster process (parent Poisson, children Gaussian around parent) | `cluster_radius_m`, `trees_per_cluster` |
| 4. Species heterogeneity | Each tree assigned a species, each species maps to a `FuelProperties` preset. Return type widens from `(n,)` to per-cell `FuelProperties` | `species_probability_distribution` |
| 5. Real vegetation raster | Load classified vegetation map (IGN BD Forêt, Copernicus), map class IDs to `FuelProperties` presets. New file `vegetation_raster_field.py`, same protocol | `raster_path`, `class_to_fuel_map` |

At layer 4 the protocol changes from `ScalarFieldProtocol` to a new `FuelFieldProtocol` returning structured data. Define it at that point, not before.

---

## `src/spark/terrain/` — meshes

Geometry and connectivity. The only place in the repo that knows what 2D vs 3D means.

### `mesh_protocol.py` ✅ implemented

**Owns:** `MeshProtocol` with read-only properties: `cell_count`, `cell_positions_xyz (n, 3)`, `neighbor_indices (n, k)`, `neighbor_distances_m (n, k)`, `neighbor_unit_directions_xyz (n, k, 3)`, `max_neighbors`.
**Never:** Contains implementation.

The 2D→3D upgrade is invisible to everything downstream: the mesh constructor reads an elevation field, sets z-coordinates, and `neighbor_unit_directions_xyz` automatically carries slope information. No consumer needs to know whether the mesh is flat or elevated.

### `square_grid_mesh.py` ✅ implemented

**Owns:** `SquareGridMeshConfig` dataclass and `SquareGridMesh` class. Regular grid, 4- or 8-connectivity, configurable `extent_x_m`, `extent_y_m`, `cell_spacing_m`. All arrays precomputed at construction.
**Never:** Knows about fire, fuel, or acoustics.

**Extension → elevation:** accept an `elevation_field: ScalarFieldProtocol` in the config. At construction, sample elevation at each cell position to fill the z-column. Recompute `neighbor_unit_directions_xyz` and `neighbor_distances_m` using the 3D positions. Same class, one extra optional field.

**Extension → variable spacing:** accept a `spacing_field: ScalarFieldProtocol` that returns desired cell size at each position. Build a non-uniform rectilinear grid. Significantly harder; defer unless needed.

### `triangular_mesh.py`

**Owns:** `TriangularMesh`. Unstructured Delaunay connectivity from scattered or regular points. Same `MeshProtocol`.
**Status:** Stub. Implement when square grid limitations become visible (irregular boundaries, local refinement).

**Extension → `hexagonal_mesh.py`:** hexagonal tiling, 6-connectivity. Natural for fire spread because all neighbors are equidistant. Same protocol.

---

## `src/spark/fire/` — fire propagation

### `spread_engine_protocol.py` ✅ implemented

**Owns:** `SpreadEngineProtocol` with methods: `initialize(mesh, fuel_field, wind_field) -> FireState`, `ignite_cells(state, cell_indices) -> FireState`, `step(state, dt) -> FireState`.

`ignite_cells` belongs on the protocol because burnout time and flammability are engine-owned knowledge: the cellular automaton reads a fixed `burn_duration_s` from its config, while the Balbi engine derives burnout from the fuel's residence time. A free function could know neither. It is vectorized and idempotent — cells with no fuel or already ignited are skipped — so a script ignites one cell or a lightning scatter through the same call.
**Never:** Contains implementation.

All engines satisfy this protocol. Scripts and the simulation context only see this type. Swapping engines is a config change, not a code change.

### `fire_state.py` ✅ implemented

**Owns:** `FireState` dataclass. Every array is marked `writeable = False` at construction, so immutability holds through the arrays and not only through the field bindings. Fields: `ignition_times_s (n,)`, `burnout_times_s (n,)`, `is_burning (n,)`, `has_ignited (n,)`, `current_time_s`. Immutable — `step` returns a new instance.
**Never:** Contains logic or computation.

**Extension → per-cell physical quantities:** when using the Balbi ROS engine, `FireState` gains optional fields: `flame_height_m (n,)`, `mean_flame_temperature_k (n,)`, `mass_loss_rate_kg_per_s (n,)`. These feed the acoustic source model with physically grounded amplitudes instead of constants.

### `fuel_properties.py` ✅ implemented

**Owns:** `FuelProperties` frozen dataclass. Six measured fields: `fuel_density_kg_per_m3`, `moisture_content_fraction`, `surface_area_to_volume_ratio_per_m`, `fuel_load_kg_per_m2`, `residence_time_s`, `fuel_bed_depth_m`. Plus named presets as classmethods: `FuelProperties.pine_needle_litter()`.

`packing_ratio` is **not** a field — it is `σ / (ρ_v e)`, so storing it would allow it to disagree with the fields it derives from. It and every other derived quantity live as pure functions in `rate_of_spread_equations.py`, which is what lets this file compute nothing.
**Never:** Computes anything.

**Extension → more presets:** add classmethods for Mediterranean maquis, grassland, Corsican scrub. Each is one classmethod returning hardcoded values from published fuel models.

### `rate_of_spread_equations.py` ✅ implemented

**Owns:** Pure functions only. The only file where paper notation (β, σ, γ, etc.) is permitted inside function bodies.

Functions to implement:

| Function | Source | Status |
|---|---|---|
| `compute_flame_tilt_angle_rad` | Balbi 2009 Eq. 2 | ✅ |
| `compute_rate_of_spread_balbi_2009` | Balbi 2009 Eq. 11a–11b | ✅ |
| `compute_reduced_rate_of_spread_balbi_2009` | Balbi 2009 Eq. 13 | ✅ |
| `compute_packing_ratio` | Balbi 2009 Table 2 | ✅ |
| `compute_optical_depth_m` | Balbi 2009 Table 2 | ✅ |
| `compute_absorption_coefficient` | Balbi 2009 Table 2 | ✅ |
| `compute_moisture_damping_factor` | Balbi 2009 §2.5 | ✅ |
| `compute_base_rate_of_spread_m_per_s` | Balbi 2009 Eq. 15 | ✅ |
| `compute_radiant_fraction_velocity_m_per_s` | Balbi 2009 §2.3 E7 | ✅ |
| `compute_rate_of_spread_balbi_2020` | Balbi 2020 Eq. 28 | Extension |
| `compute_ignition_energy_j_per_kg` | Balbi 2020 Eq. 9 | ✅ |
| `compute_extinction_depth_m` | Balbi 2020 Eq. 12 | Extension |
| `compute_radiative_coefficient` | Balbi 2009 Eq. 14 | ✅ |
| `compute_flame_height_m` | Balbi 2020 Eq. 23 | Extension |
| `compute_upward_gas_velocity_m_per_s` | Balbi 2009 Eq. 15 | ✅ |
| `compute_mean_flame_temperature_k` | Balbi 2020 Eq. B11 | Extension |
| `compute_rate_of_spread_rothermel` | Rothermel 1972 Eq. 42 | Optional baseline |

**Never:** Holds state, imports meshes or engines.
**Vectorized:** every function takes arrays and returns arrays. The engine calls them once per timestep for all cells, not per cell.

### `cellular_automaton_spread_engine.py` ✅ implemented

**Owns:** `CellularAutomatonSpreadEngine`. Alexandridis 2008 probabilistic ignition rules. Queries the mesh for neighbors and directions, queries the wind field for wind at each cell, applies probabilistic ignition.
**Never:** Implements ROS physics — it calls `rate_of_spread_equations.py` if it needs a physical ignition probability, or uses its own probabilistic rules.

**Extension → firebrand transport:** burning cells emit firebrands that land at a distance drawn from a distribution, igniting cells not adjacent to the fire front. One additional method, contained within this file.

### `rate_of_spread_engine.py` ✅ implemented

**Owns:** `RateOfSpreadEngine`. The physically-grounded engine. For each burning cell, projects wind onto each neighbor direction, calls `compute_rate_of_spread_balbi_2009` to get R in m/s, converts to ignition delay `cell_distance_m / rate_of_spread_m_per_s`, updates `FireState`.
**Never:** Contains the ROS equations themselves.

**Scheme:** fire crossing one edge is an arrival time problem. A cell igniting at `t` schedules an arrival at each neighbor at `t + distance / R`, with `R` evaluated from the wind and slope resolved along that edge; a cell ignites at the earliest arrival offered. Ignition times are therefore near-independent of `dt` — verified to 1e-9 relative between `dt = 1 s` and `dt = 10 s`. A burnt cell keeps its scheduled arrivals, because with pine needle litter a cell burns out in 20 s while fire needs 31 s to cross a 0.5 m edge; retracting them would stall the front entirely.

**Extension → Balbi 2020 upgrade:** swap `compute_rate_of_spread_balbi_2009` for `compute_rate_of_spread_balbi_2020` inside this engine. One function call change.

**Extension → per-cell fuel heterogeneity:** instead of one global `FuelProperties`, query the fuel field at each cell position. Requires the fuel field to return per-cell properties. Structurally: replace `self.fuel` with `self.fuel_field.sample(positions)`.

### `static_source_spread_engine.py` ✅ implemented

**Owns:** `StaticSourceSpreadEngine`. `initialize` returns an unlit state; `ignite_cells` lights the
given cells with an infinite burnout; `step` advances the clock and leaves every mask alone.
Satisfies `SpreadEngineProtocol`.
**Never:** Spreads.

**Why an engine and not a separate render path.** Rungs E1 and E2 need a source that sits still and
keeps sounding. Behind the protocol, "static source" and "spreading front" are one code path
differing by a registry entry: one forward script serves every rung and the static case inherits the
protocol's whole test suite. A separate path would duplicate the rendering loop and drift from it.

**Infinite burnout is safe for the front mask.** An isolated permanently burning cell keeps
un-ignited neighbours forever, so `compute_fire_front_mask` reports it as on the front at every
observation and the two emission models agree on it. That was the condition under which this design
would have had to be abandoned; it holds.

### `time_step_calculator.py` ✅ implemented

**Owns:** `compute_maximum_stable_time_step_s`, the discrete CFL condition applied to front
tracking: the time fire needs to cross the shortest mesh edge, scaled by a safety factor.
**Never:** Runs the fire.

**Also clamped by the residence time.** A timestep longer than a cell stays alight lets a cell
ignite and burn out between two samples, so the renderer never sees it. Passing `residence_time_s`
caps the step at `safety_factor * residence_time_s`; without it the default 100 m run samples 22.31 s
apart against a 20 s residence time and the first step renders silence.

### Extension → `game_of_life_spread_engine.py`

**Status:** Not yet in tree but planned. Simplest possible engine: a cell ignites if ≥ N neighbors are burning. Fixed burnout timer. No physics. Useful as a sanity check that the protocol, the mesh, and the visualization pipeline all work before introducing real ROS equations.

---

## `src/spark/atmosphere/` — ambient air physics

Stateless, leaf-level. Both `acoustic/` (forward) and `inverse/` (estimator) import it, which is
what keeps a single definition of the absorption coefficient: if the two sides each carried their
own, the estimator would silently be inverting a different physics than the renderer produced.

### `atmospheric_conditions.py` ✅ implemented

**Owns:** `AtmosphericConditions` frozen dataclass (temperature, humidity, pressure) and
`compute_speed_of_sound_m_per_s`. Also the universal constants the ISO formulas reference.
**Never:** Knows about signals, geometry or receivers.

### `atmospheric_absorption.py` ✅ implemented

**Owns:** `compute_absorption_coefficients_db_per_m`, the ISO 9613-1 absorption `α(f)` in dB/m,
vectorized over frequency. Reproduces the published table to six decimals at four
temperature/humidity pairs.
**Never:** Assumes a band structure — it evaluates at any frequency, band centre or FFT bin alike.

**Why recomputed per scenario:** `α` moves by roughly 30 % across realistic conditions, and the
high bands where it matters most move furthest. A generic table would bias exactly the bands that
carry the range information.

---

## `src/spark/acoustic/` — sound propagation

### `channel_protocol.py` ✅ implemented

**Owns:** `ChannelProtocol` with method `compute_gain_matrix(source_positions_xyz, receiver_positions_xyz) -> ndarray` returning shape `(n_sources, n_receivers)`.
**Never:** Contains implementation.

The forward acoustic render is: `received_levels = gain_matrix.T @ source_amplitudes`. This one line is the source–channel model made literal in code.

### `burning_cell_source_model.py` ✅ implemented

**Owns:** `BurningCellSources` and the two emission models that build it. `extract_burning_cell_sources` radiates every burning cell; `extract_fire_front_sources` radiates only the burning cells that still have an unignited neighbor. Also `compute_fire_front_mask` and `identify_connected_front_components`, which labels each separate front largest-first from the mesh adjacency. The only bridge between the fire domain and the acoustic domain.
**Never:** Knows about receivers, channels, or propagation.

**The two emission models coincide whenever the burning band is one cell thick.** With pine needle litter a cell burns out in 20 s while fire needs 25 s to cross a 0.5 m edge at 2 m/s wind, so no burning cell is ever interior and `front_only` returns exactly what `all_burning` returns. The distinction appears once the residence time exceeds the crossing time: at `residence_time_s = 200` the front is 92 cells where the burning area is 453.

**Component labels count fragments, not fires.** They are honest labels on whatever mask they are given; when the burning band is sparse the same physical ring reports as many components. A count is only "how many separate fires" when the band is connected.

**Day-one version:** amplitude is a constant for every burning cell. All fires sound the same.

**Extension → physics-based amplitude:** use `mass_loss_rate_kg_per_s` from `FireState` (available when using the Balbi ROS engine) as a proxy for acoustic power. Louder fires produce more sound.

**Extension → spectral source model:** instead of a scalar amplitude, return a spectrum per cell. Amplitude in the crackling band (1–15 kHz) driven by fuel type and burn rate. Amplitude in the puffing band (< 20 Hz) driven by flame diameter via Cetegen's scaling `f_puff ≈ 1.5 * D^(-0.5)`. Return type widens from `(n,)` to `(n, n_freq)`.

### `exponential_attenuation_channel.py` ✅ implemented

**Owns:** `ExponentialAttenuationChannel`. Implements `gain(r) = exp(-α * r) / r` where `r` is Euclidean distance and `α` is the attenuation coefficient. Builds the full `(n_sources, n_receivers)` gain matrix. Also the shared gain primitives `compute_source_receiver_distances_m`, `compute_geometric_spreading_gain` and `compute_atmospheric_absorption_gain`, which `free_field_propagation.py` reuses so the two forms of the channel cannot disagree.
**Never:** Knows about fire state or terrain.

**Extension → frequency-dependent attenuation ✅ implemented:** `AtmosphericAbsorptionChannel`, in this same file. `α` comes from ISO 9613-1 at the scenario's air conditions rather than from one fitted constant, and the gain matrix widens to `(n_sources, n_receivers, n_freq)`. Atmospheric absorption rises with frequency, so high-frequency crackle content decays faster than low-frequency roar. This is what makes range recoverable.

**Extension → terrain-aware path loss:** if source and receiver are not line-of-sight (terrain obstruction), apply diffraction loss. Requires querying the elevation field along the source–receiver path. Add as a wrapper around the base channel, not as a modification to it.

### `free_field_propagation.py` ✅ implemented

**Owns:** The same channel applied to a **waveform** rather than to amplitudes.
`apply_free_field_propagation` folds propagation delay, `1/r` spreading and frequency-dependent
absorption into one real FFT; `render_receiver_signals` returns one channel per receiver on a
common clock. The signal is zero-padded by the delay first, so the tail never wraps into the head.
**Never:** Knows the source is a fire, or that anyone will later invert it.

A gain matrix cannot carry a delay, so this is a separate entry point rather than a method on
`ChannelProtocol` — the time-difference-of-arrival the estimator needs lives only in the waveform.

**Extension → several concurrent sources ✅ implemented:**
`render_multi_source_receiver_signals` takes one waveform, one 3D position and
one amplitude scale per source and accumulates them on a common clock sized by
the longest path. Superposition is linear, so it reuses the same per-pair
operator and the single-source render is exactly its `n_sources = 1` case —
pinned by test, so the two cannot drift.

### `receiver_noise.py` ✅ implemented

**Owns:** `add_white_noise_at_snr_db` and its multi-channel form. Noise is drawn independently per
channel, so channels decorrelate as the ratio falls — which is what degrades the delay estimate.
**Never:** Shapes noise to a spectrum; that belongs to a sensor model.

### `receiver_placement.py` ✅ implemented

**Owns:** `place_receivers_in_ring`, `place_receivers_in_grid`, `place_receivers_randomly` and the
`place_receivers_from_configuration` dispatcher that resolves fractional layout centres against the
domain extent. Pure geometry; receivers sit on the ground plane, so z is not carried.
**Never:** Knows what will be rendered to them.

This is the one module in `acoustic/` that imports `config/`, because the dispatcher exists to turn
a strategy name into a layout. The three placement functions underneath it take plain floats and
import nothing.

**Extension → N-receiver layouts ✅ implemented:** `place_receivers_from_layout`
resolves a `ReceiverLayoutConfiguration` into `(n_receivers, 3)` positions and
runs both guards on the result. It lives here rather than in
`receiver_layout.py` so that this stays the only module in `acoustic/` reading
`config/`.

### `receiver_layout.py` ✅ implemented

**Owns:** `build_ring_receiver_positions_xyz`,
`build_grid_receiver_positions_xyz`, `build_random_receiver_positions_xyz`
(Poisson-disc rejection against `minimum_separation_m`),
`lift_positions_to_height_xyz`, and the two guards
`compute_minimum_pairwise_separation_m` and `compute_collinearity_measure`,
enforced by `validate_receiver_layout`.
**Never:** Renders anything, or reads `config/`.

Positions carry a height here because a multi-source run solves for `(x, y)`
while computing ranges in 3D, so the microphone height belongs in the geometry
rather than being dropped.

**The collinearity guard is the N-receiver form of the perpendicular-bisector
degeneracy.** With every receiver on a line, a position and its mirror across
that line produce identical delays at every receiver. It is skipped below three
receivers, where it carries no information: two points always lie on a line, and
that case is handled by the bisector flag in `level_ratio_triangulation.py`
instead.

### `measured_impulse_response_channel.py`

**Owns:** Stub for a channel built from experimentally measured impulse responses. Loads measured `h` from file, applies it as a convolution or frequency-domain multiply.
**Status:** Stub. Implement if/when measured data becomes available.

---

## `src/spark/inverse/` — kinematics estimation

**Hard rule:** this entire subpackage sees only receiver signals and receiver positions. It must never import from `fire/`, `terrain/`, `fields/`, or `acoustic/`.

### `kinematics_estimator_protocol.py`

**Owns:** `KinematicsEstimatorProtocol` with method `estimate(receiver_signal_levels, receiver_positions_xyz) -> FrontKinematics`. `FrontKinematics` is a dataclass holding `bearing_rad`, `position_xyz`, `rate_of_spread_m_per_s` and associated uncertainties.
**Never:** Contains implementation.

### `time_difference_of_arrival.py` ✅ implemented

**Owns:** `estimate_time_difference_of_arrival`. GCC-PHAT with parabolic sub-sample refinement,
plus the variance of that delay from effective bandwidth and coherence.
**Never:** Knows what the delay will be used for.

**Sign convention:** `τ > 0` means receiver 1 hears the event *later*, so `D = −c·τ`. Pinned by
test, because getting it backwards silently mirrors every estimate.

**Measure coherence after alignment.** On the raw pair, a baseline delay comparable to the Welch
segment collapses the coherence; measured that way `σ_D` came out at 6.5 m instead of 0.014 m and
the error ellipse was meaningless.

**Extension → the curve as the primitive ✅ implemented.** The multi-source side
needs the whole correlation function, not its peak: several sources live in one
curve, and a located source is peeled out of it. This file therefore also owns
`GeneralizedCrossCorrelationCurve`, `compute_whitened_cross_spectrum`,
`compute_generalized_cross_correlation`,
`accumulate_generalized_cross_correlation_over_windows`,
`compute_pair_correlation_curves`, `refine_peak_index`,
`estimate_delay_from_correlation_curve` and `estimate_delay_near_prediction`.

The original scalar `estimate_time_difference_of_arrival` is **left exactly as
it was** rather than being replaced. It is what the single-source pipeline runs
on, and its metrics reproduce bit-for-bit against the committed reference — the
guarantee is worth more than the naming symmetry.

Four things the curve form does that the scalar form did not:

| Change | Why |
|---|---|
| Whitening band-limited to `[lowest, highest]` | Above a codec brickwall the content is encoder noise floor; a phase transform would raise it to full weight in the correlation |
| Parameterised phase transform `\|X_i X_j*\|^β + γ` | `β = 1` recovers conventional PHAT and shows the largest performance fluctuation; near 0.7 some magnitude survives |
| Analytic-signal envelope, optional | A band-passed input drives the correlation toward a sinc whose ripples become spurious map peaks. It costs some sub-sample sharpness, so it is measured rather than assumed |
| Exponential peak interpolation | Reported best of parabolic, exponential and Fourier for this map |

**Windows are accumulated in the frequency domain,** averaging the whitened
cross-spectrum and transforming once, not averaging curves. Fire is continuous
noise, so one window gives a peak buried in its own variance; 50 s at 0.5 s
windows with half overlap gives about 200.

**`estimate_delay_near_prediction` exists because of mixtures.** A pair curve's
global maximum belongs to whichever source dominates that pair, so a source the
map has already located has to be read out of its own neighbourhood instead.

### `band_level_difference.py` ✅ implemented

**Owns:** The per-band, per-window geometric term `G_bm = ΔL_bm − α_b·D`; its mean, its variance
across windows, and the variance *of that mean*.
**Never:** Computes `α_b` — it is handed them, which is what keeps `atmosphere/` the single source.

**Fuse the standard error, not the per-window variance.** `G_b` is a mean of `M` windows, so its
uncertainty is `σ²_b / M_eff`. The factor is common to every band and cancels in `G_hat`, but using
`σ²_b` directly inflates `var_G` by `M_eff ≈ 12` and deflates `χ²_ν` by the same factor. Both are
kept on the result object.

### `inverse_variance_fusion.py` ✅ implemented

**Owns:** Generic inverse-variance weighted mean with its variance, residuals and `χ²_ν`.
**Never:** Knows the quantity being fused is a level. Reusable for the multi-source case.

### `level_ratio_triangulation.py` ✅ implemented

**Owns:** `G, D → r1, r2 → (x, y)` in the baseline frame, the mirror-side choice, the near-singular
guard, and the position covariance from a finite-difference Jacobian.
**Never:** Touches signals.

**The perpendicular bisector is a hard degeneracy, not poor precision.** There `r1 = r2`, so `D = 0`
and `k = 1` at once and `r1 = D/(k−1)` is `0/0`. Every point along the bisector produces identical
observables, so no estimator can separate them. Sensitivity goes as `r1²/D`, which is why the flag
fires on a *band* around the bisector, not just on it.

### `single_source_estimator.py` ✅ implemented

**Owns:** `SingleSourceEstimator` and the `localize_single_source` driver that wires the four
modules above together, plus the error ellipse. Assumes one fire front. Estimates the range ratio
from relative level decay across receivers (the unknown source amplitude cancels in the ratio) and
the range difference from the delay, then triangulates.
**Never:** Accesses ground-truth fire state, or contains any of the four modules' maths itself.

**Input contract differs from `KinematicsEstimatorProtocol`.** This estimator consumes receiver
*waveforms*, not scalar `receiver_signal_levels`: the range difference comes from a time delay,
which levels cannot supply. The protocol is left for a level-based estimator; reconciling the two
(widening the protocol, or two protocols) is an open design decision, not an oversight.

**Extension → bearing estimation:** use the spatial gradient of received levels across the receiver array to estimate the direction to the source.

**Extension → rate of spread estimation:** compare estimated positions at consecutive time steps to estimate front velocity.

### `receiver_pair_index.py` ✅ implemented

**Owns:** `enumerate_receiver_pairs` in canonical `i < j` order,
`compute_pair_baseline_distances_m` and `compute_maximum_absolute_lag_s`. One
definition of pair ordering for the whole subpackage.
**Never:** Touches signals.

A signed delay only means something against a stated ordering, so the
`τ > 0 means receiver i hears it later` convention is pinned here rather than
re-derived at each call site.

### `steered_response_power.py` ✅ implemented

**Owns:** the candidate grid (`build_candidate_grid_xyz`,
`subdivide_candidate_cells`), the delay tables (`compute_pair_delay_table_s`,
`compute_distance_bounds_to_cells_m`, `compute_cell_delay_bounds_s`), the map
(`pool_curve_over_intervals`, `combine_pairwise_maps`,
`compute_steered_response_power_map`), and the coarse-to-fine driver
(`run_coarse_to_fine_search`, `extract_map_peak`, `select_retained_cells`).
**Never:** Assumes a source count.

**The grid is pooled over, not sampled at.** The time-difference-of-arrival
gradient has magnitude at most `2/c`, so a 2 m cell spans about 16 ms of delay —
some 730 samples at 44.1 kHz — while a correlation peak carrying 10 kHz of
bandwidth is about 4 samples wide. Point sampling at that spacing steps over the
peak and returns a confident maximum somewhere else, with no symptom that
anything went wrong. Point sampling would need roughly 1.7 cm cells, a
36-million-point grid per deflation round.
`tests/spark/inverse/test_steered_response_power.py` holds the regression: at
the same spacing, volumetric pooling puts the map maximum on the true cell and
point sampling does not.

**The bounds are guaranteed to contain the cell's delays,** because each
receiver distance is bounded over the cell box and the extremes are differenced.
That is what makes discarding a cell safe rather than a heuristic.

**The product combinator returns a geometric mean.** Taking the root is monotone
so it moves no maximum, but it keeps the map on the scale of one pairwise map
however many pairs there are. Without it the raw product of fifteen maps runs to
`1e-180` and every ratio against the map median stops meaning anything — the
prominence test included.

**The first level runs on low-frequency content only.** Peak width goes
inversely with frequency, so that map is deliberately smooth and its basins are
wide enough that a coarse cell cannot fall between two of them.

### `multilateration.py` ✅ implemented

**Owns:** `compute_predicted_time_differences_s`,
`compute_time_difference_jacobian`, `compute_degrees_of_freedom` and
`refine_position_gauss_newton`, weighted by inverse delay variance, with the
covariance from the weighted normal matrix. The N-receiver analogue of
`level_ratio_triangulation.py`.
**Never:** Touches signals.

**Seeded by the map, never run alone.** Two-step delay-then-solve methods
discard the rest of the correlation function and are fragile under noise, which
is why the map comes first.

**The residual needs `N ≥ 5` to say anything.** The cocycle constraint means `N`
receivers supply only `N − 1` independent delays however many pairs are formed,
so against two unknowns there are `N − 3` degrees of freedom. At `N = 3` the fit
is exact by construction, the residual is uninformative, and the two hyperbola
branches can intersect twice — reported through `is_ambiguous` rather than
hidden.

### `source_deflation.py` ✅ implemented

**Owns:** `build_source_template`, `apply_tdoa_notch_deflation`,
`apply_subspace_projection_deflation`, the `deflate_located_source` dispatcher,
`compute_delay_and_sum_beamformed_signal` and
`compute_residual_correlation_energy_ratio`.
**Never:** Reads ground truth or the forward model's per-source signals.

**Deflation acts on the correlations, not the waveforms.** This is what keeps
the isolation rule intact: subtracting an estimated source from the waveforms
would need a propagation operator, while notching or projecting it out of the
cross-correlations needs only receiver positions, an assumed speed of sound and
the correlations themselves. No shared propagation leaf was required.

**The notch takes everything under it; the projection takes only what the source
explains.** That is the whole difference, and it is what makes the projection
survive a power disparity — the documented failure mode of every method in this
family.

**The template width must match the correlation peak.** Set five times too wide,
the least-squares coefficient underestimates the peak, the projection leaves
most of it behind, and the loop re-detects the source it just peeled off. This
was observed, not predicted.

**Documented ceiling, inherited:** the originators of the notch approach report
that at three sources the noise in the correlation function becomes prohibitive.
`windowed_position_clustering.py` exists because of it.

### `multiple_source_estimator.py` ✅ implemented

**Owns:** `MultipleSourceEstimator` and the `localize_multiple_sources` driver,
structured as the X-SRP loop: build the pair correlations once, search the
domain, refine against every pair delay, peel the source out of the
correlations, search again. `update_signal_features` is the deflation step and
`update_grid` is the refinement step, so one loop covers both the coarse-to-fine
search and the multi-source peeling.
**Never:** Requires the true source count.

`K̂` is an output. The loop stops when the residual map peak is not prominent
against the map median, when the peak lands within `minimum_source_separation_m`
of an accepted source, when `maximum_source_count` is reached, or when peeling
stops removing correlation energy — and it reports which.

**A source's delays are read from its own neighbourhood of each curve,** not
from each curve's global maximum, which in a mixture belongs to whichever source
dominates that pair.

**Extension → dense receiver selection:** when many receivers are available,
select the subset with highest SNR or best geometric diversity. One additional
method in this file.

**Extension → group-sparse joint fitting:** the contingency if sequential
deflation breaks down at three sources. Fit the whole map at once over the
candidate grid so all sources are estimated simultaneously and the sequential
error compounding disappears.

### `windowed_position_clustering.py` ✅ implemented

**Owns:** `cluster_positions_by_radius` and
`estimate_positions_by_windowed_clustering`. Runs a single-source search on many
short windows, clusters the per-window maxima, and reports each dense cluster as
a source with its centroid and an empirical spread.
**Never:** Assumes the windows are independent of each other in the fusion step.

**A second route to `K̂`, independent of deflation,** so a disagreement between
the two is a finding rather than a restatement. Much of the multi-source
literature leans on speech sparsity — disjoint time-frequency support, one
source dominant per bin, voice activity detection — none of which transfers to
continuous, stationary, spectrally similar broadband noise. The one speech-like
property fire does have is impulsivity: crackle is a sequence of transients, so
within a short window one source frequently dominates outright, which is exactly
the condition this needs.

**It is the first thing a power disparity breaks.** On the `m1` scene, where the
second source is 6 dB down, 48 of 49 windows land on the louder source and this
route reports `K̂ = 1` where sequential deflation reports 2. That is the
predicted behaviour, and it is reported rather than suppressed.

### `joint_position_refinement.py` ✅ implemented

**Owns:** `align_channels_to_beamformer`, `compute_pair_level_differences_db`,
`compute_predicted_level_differences_db`, `compute_level_difference_jacobian`,
`fuse_pair_geometric_level_differences` and
`refine_position_with_band_levels` — one Gauss-Newton solve over delay residuals
and band-level residuals together, each weighted by its own variance and fused
through `inverse_variance_fusion.py`.
**Never:** Recomputes `α` — it is handed the coefficients, keeping
`atmosphere/` the single source.

**This is where the two observables earn their keep together.** Delays constrain
range *differences* very precisely, but a compact array constrains absolute
range poorly, so a delay-only ellipse stretches radially away from the array.
The absorption slope depends on absolute path length, which is exactly that
direction.

**The reference and the channels must share a clock.** A residual delay puts a
phase ramp across each band and cancels part of the matched-filter projection;
aligning to the nearest sample is enough, and the beamformer's own shifts are
what to align by.

---

## `src/audio/` — real audio preprocessing

Standalone pipeline for processing real fire recordings. No imports from the simulation domain.

### `recording_segmenter.py` ✅ implemented

**Owns:** Cuts audio files into fixed-length segments (5 s and the overlap come from `configs/`) . A trailing partial segment is dropped rather than zero-padded.

### `octave_band_filter.py` ✅ implemented

**Owns:** ISO octave band edges (`f_c/√2`, `f_c·√2`, clipped below Nyquist) and a zero-phase
Butterworth band-pass. Second-order-section form, not `b, a`: the 125 Hz band is narrow enough
relative to 44.1 kHz that the transfer-function form loses accuracy.

Kept separate from `lowpass_filter.py` on purpose. That file owns the *preprocessing* filter
applied once to a recording; this one owns the *analysis* filterbank the estimator runs per window.
Different consumers, different lifecycles.

**Extension → fractional octaves ✅ implemented.** `compute_band_half_width_factor`,
`build_fractional_octave_centre_frequencies_hz` (from the tabulated ISO 266
preferred series), `compute_fractional_octave_band_edges_hz`,
`design_fractional_octave_bandpass` and `apply_fractional_octave_bandpass`
generalise the edges to `f_c · 2^(±1/(2·fraction))`. The octave functions are now
thin wrappers at `fraction_denominator = 1`, so existing callers are unaffected —
pinned by test.

**Why thirds above 2 kHz.** The full 8 kHz octave spans 5.6–11.3 kHz and `α`
roughly doubles across it, so the energy-weighted effective `α` drifts downward
as range grows and biases exactly the bands that carry the range information.

### `codec_bandwidth_detector.py` ✅ implemented

**Owns:** `compute_long_term_average_spectrum` and `detect_codec_cutoff_hz`,
shared by the Stage 0 audit and the band selector.
**Never:** Filters or modifies the signal.

**The crossing is refined on the unsmoothed spectrum.** The search runs on a
median-smoothed curve so one loud bin above the wall cannot set the answer, but
a centred median filter holds its in-band value for half its width past a step
edge, so reading the crossing off the smoothed curve alone reports the cutoff
several bins too high.

**It is accurate to the analysis window's skirt, not to one bin.** A Hann window
spreads a step edge over about four bins however sharp the wall really is; the
test asserts that and says why, rather than asserting something no windowed
spectrum can deliver.

Measured on `data/raw_recordings/kaggle/`: every one of the twenty files
brickwalls at 15.0–15.7 kHz, confirming one common encoder.

### `usable_band_selector.py` ✅ implemented

**Owns:** `select_usable_band_centres_hz`, applying the upper-edge rule against a
measured cutoff, and `filter_bands_by_signal_to_noise_ratio`, applying the
run-time test.
**Never:** Knows about propagation or ranges; it is handed levels.

A band whose upper edge reaches past the brickwall measures the encoder noise
floor over part of its width, which biases its level downward by an amount that
grows with range. Dropping the band is cheaper than modelling that bias.

### `excerpt_coherence.py` ✅ implemented

**Owns:** `compute_maximum_normalized_cross_correlation`,
`compute_pairwise_excerpt_coherence_matrix` and
`compute_maximum_off_diagonal_coherence`.
**Never:** Chooses excerpts.

The pre-flight guard. Two source excerpts that share waveform content put a peak
into every receiver pair's cross-correlation at a lag no source occupies, and
nothing downstream can tell that artefact from a real source.

### `source_excerpt_selector.py` ✅ implemented

**Owns:** `SourceExcerpt`, `read_leading_excerpt`, `list_pool_recordings`,
`choose_least_coherent_indices` and `select_source_excerpts`, which raises when
the chosen excerpts exceed the configured coherence limit.
**Never:** Applies gain, propagation, or normalization.

Three policies: `explicit` takes the configured order, `distinct_provenance`
shuffles the pool under a seed and never reuses a file, `minimum_coherence`
greedily picks the subset whose worst mutual coherence is smallest — the last is
`O(n²)` in the pool size and is what the `m1` scene uses.

### `matched_filter_band_level.py` ✅ implemented

**Owns:** `compute_band_bin_masks` and `compute_matched_filter_band_levels_db`.
Recovers per-source, per-receiver band levels from a mixture given a beamformed
reference, as the band-restricted cross-spectrum magnitude squared normalised by
the reference auto-spectrum.
**Never:** Knows about geometry.

**This is what replaces waveform subtraction as the input to the level stage.**
Once deflation moved to the correlation domain there is no per-source waveform to
difference, so the level stage projects each receiver channel onto the beamformed
reference instead. Other sources are rejected to the extent that they are
uncorrelated with it.

**The reference and the channels must already share a clock:** a residual delay
puts a phase ramp across each band and cancels part of the projection. Nearest-sample
alignment is enough, and it is the caller's job — which is what keeps this file
free of geometry.

### `band_level_meter.py` ✅ implemented

**Owns:** The Hann analysis window, the windowed RMS level in dB (window power divided out, so the
level does not depend on window shape), and the window start indices for a range.

### `signal_alignment.py` ✅ implemented

**Owns:** Integer-sample shift, the valid overlap range after a shift, and the aligned channel pair.
Used by both the estimator's window loop and the coherence measurement.

### `lowpass_filter.py`

**Owns:** Filter design and application. Verifies sample rate consistency across sources. Default: high-pass at 50 Hz (wind rumble), keep everything above — do not low-pass aggressively, the high band is what makes range recoverable.

### `amplitude_normalizer.py`

**Owns:** Per-segment normalization. Scheme to be determined (peak, RMS, LUFS).

### `grouped_split_manifest.py`

**Owns:** Creates train/test split manifests with group IDs derived from source recording provenance (channel signature, noise-floor shape, spectral rolloff), not from filenames. This is the project's own consistency check against the leakage critique (claim 1 in the paper).

---

## `src/config/` — configuration and wiring

### `simulation_configuration.py` ✅ partly implemented

**Owns:** Nested frozen dataclasses describing what to build, and the TOML loaders that fill them from `configs/`. Fully serializable to JSON.

Implemented for the forward render: `MeshConfiguration`, `WindConfiguration`, `FireSimulationConfiguration`, `ReceiverConfiguration` and `AcousticRenderingConfiguration`, one module each, grouped under `ForwardSimulationConfiguration`. Still planned: `FuelConfiguration` and `SpreadEngineConfiguration`, which today are a preset name and a hard-wired engine.

**Three fuel fields are wired.** `uniform` puts the whole fuel bed load everywhere, which every
rung of the ladder uses so that only geometry drives the received levels. `random_trees` scatters a
homogeneous Poisson forest and `patchy_trees` thins it with Gaussian density bumps; both make the
front visibly wander, which is what the live display is for, and both are reproducible from
`tree_layout_seed`.

`ForwardSimulationConfiguration` sits **alongside** `SimulationConfiguration`, not inside it. The two pipelines share only the air conditions, and folding the fire model into the object the estimator loads would put `fire/` one attribute away from the code that must never reach it.

Implemented for the localization pipeline: `BandConfiguration`, `WindowConfiguration`, `DelayEstimationConfiguration`, `TriangulationConfiguration` (grouped under `LocalizationConfiguration`), plus `DataConfiguration`, `GeometryConfiguration`, `ForwardModelConfiguration` and the top-level `SimulationConfiguration`.
**Never:** Imports concrete implementations.

**No tunable value is written in the source.** Everything a run can change lives in `configs/`, and
`--configs <dir>` swaps the whole set:

| File | Holds |
|---|---|
| `environment.toml` | Air temperature, relative humidity, pressure |
| `data.toml` | Recording path, clip duration and overlap, metrics output path |
| `geometry.toml` | Domain size, receiver positions, true source positions |
| `forward_model.toml` | Reference distance, receiver-noise SNR and on/off, random seed |
| `localization.toml` | Octave bands, analysis window, delay estimation, triangulation guards |
| `mesh.toml` | Domain extent, cell spacing, 4- or 8-connectivity |
| `wind.toml` | Wind speed and bearing |
| `fire.toml` | Fuel preset, ignition point, run length, CFL safety factor, emission model |
| `receiver.toml` | Placement strategy and its parameters |
| `acoustic_rendering.toml` | Sample rate, segment duration, reference distance, source seed |

Two categories deliberately stay in the source as named module-level constants, because they are
not properties of a run: **published equation coefficients** (the ISO 9613-1 relaxation terms,
`20.05` in the speed of sound) and **numerical guards** (`SPECTRUM_FLOOR`, `POWER_FLOOR`,
`LEVEL_RATIO_FLOOR`). Leaf modules in `audio/` and `atmosphere/` carry **no defaults at all**, so a
missing value is a `TypeError` at the call site rather than a silent fallback; config objects are
unpacked at the boundary, which is why those two packages never import `config/`.

**Extension → receiver layout configuration ✅ implemented.**
`GeometryConfiguration` no longer fixes two receivers. It carries a
`ReceiverLayoutConfiguration` (`explicit`, `ring`, `grid` or `random`, with the
crowding and collinearity limits and the microphone height) and a tuple of
`SourceConfiguration`, each a position and an amplitude scale. `explicit` keeps
every two-receiver directory working unmodified, and is the only layout that
carries its positions in the configuration — the rest are built by
`receiver_placement.py`, because building a layout is geometry rather than
configuration.

**Two source descriptions sit side by side because two pipelines read them.**
`true_positions_xy_m` is a list of scenarios, run one at a time by the
single-source estimator; `[[sources.concurrent]]` is one scene whose sources all
sound at once, which is what the multi-source estimator sees. They are separate
keys rather than one, because collapsing them would silently change what the
single-source sweep means.

**Extension → multi-source estimator settings ✅ implemented.**
`src/config/multi_source_localization_configuration.py` owns
`CorrelationConfiguration`, `FractionalBandConfiguration`,
`SteeredResponsePowerConfiguration`, `DeflationConfiguration`,
`PositionRefinementConfiguration`, `WindowedClusteringConfiguration` and
`LocalizationMetricsConfiguration`, grouped under
`MultiSourceLocalizationConfiguration` and reached through
`LocalizationConfiguration.multi_source`.
`src/config/receiver_layout_configuration.py` and
`src/config/source_scene_configuration.py` hold the two geometry additions.

Three new configuration blocks, all required, all present in every directory:

| File | Section | Holds |
|---|---|---|
| `data.toml` | `[excerpts]` | Recording pool, assignment policy, coherence limit, selection seed |
| `geometry.toml` | `[receivers]` | Layout name and its parameters, the two guards, microphone height |
| `geometry.toml` | `[[sources.concurrent]]` | One table per concurrent source: position and amplitude scale |
| `localization.toml` | `[correlation]` | Whitening band, `β`, envelope, accumulation window, lag margin, interpolation |
| `localization.toml` | `[fractional_bands]` | Bands per octave, centre range, codec margin, band SNR floor, cutoff detection |
| `localization.toml` | `[steered_response_power]` | Pooling, combinator, coarse spacing, refinement, low-frequency seed |
| `localization.toml` | `[deflation]` | Method, source ceiling, prominence, separation, residual threshold, notch width |
| `localization.toml` | `[refinement]` | Gauss-Newton limits and whether to use the level terms |
| `localization.toml` | `[windowed_clustering]` | The cross-check's window, radius and minimum membership |
| `localization.toml` | `[localization_metrics]` | Optimal sub-pattern assignment cutoff and order |

**`configs/m1/` is the multi-source scene:** two sources at (30, 40) and
(70, 65) with the second 6 dB down, heard by a ring of six receivers of radius
45 m, each source carrying its own 50 s excerpt.

### `simulation_context.py` ✅ implemented

**Owns:** `SimulationContext` frozen dataclass holding live Protocol instances: `mesh: MeshProtocol`, `elevation_field: ScalarFieldProtocol`, `fuel_field: ScalarFieldProtocol`, `wind_field: VectorFieldProtocol`, `spread_engine: SpreadEngineProtocol`, `channel: ChannelProtocol`, plus the `FuelProperties` the engine and the source model share, the resolved `ignition_cell_indices` and the `receiver_positions_xy_m`.
**Never:** Contains logic. Imports only Protocols, never concrete classes.

### `simulation_context_factory.py` ✅ implemented

**Owns:** `build_simulation_context(config) -> SimulationContext`. The composition root, and the **only file in the repository that constructs a concrete class**. Reads the configuration, looks the names up in the registry, instantiates, returns a context.
**When you add a new implementation:** register it in the registry and add one branch here. Nothing else moves.

**Guarded by grep, not by good intentions.** `grep -rn "SquareGridMesh(\|RateOfSpreadEngine(\|UniformScalarField(" src/ scripts/` must return hits only inside `square_grid_mesh.py` itself. A script that narrows a `MeshProtocol` back to `SquareGridMesh` for a raster display is fine — narrowing is not construction.

### `component_registry.py` ✅ implemented

**Owns:** Name-to-class maps, and `resolve_registered_name`, which raises listing what *is* registered:

```
MESH_REGISTRY            square_grid
SPREAD_ENGINE_REGISTRY   rate_of_spread, cellular_automaton, static_source
CHANNEL_REGISTRY         exponential_attenuation, atmospheric_absorption
SCALAR_FIELD_REGISTRY    uniform, patchy
VECTOR_FIELD_REGISTRY    constant
```

This module necessarily *imports* concrete classes, because a name-to-class map has nothing else to map to. It never calls them.

### `experiment_configuration.py` ✅ implemented

**Owns:** `ExperimentConfiguration` — the scene's `name`, `title` and `description`. The name is
validated against `^[a-z0-9][a-z0-9_-]*$`, because it becomes a directory name for both runs and
figures.
**Never:** Holds anything the physics reads.

**Why a configuration section exists purely for a name.** Without one, a run directory and a figure
file carry only a timestamp, and nothing says which rung of the ladder they belong to. With one,
`results/simulations/e3/<stamp>/` and `results/figures/e3/` label themselves, and
`figure_metadata.json` can be checked against the scene it claims to come from.

A configuration directory with no `experiment.toml` still loads; it falls back to the default name,
so an older directory keeps working.

Every run dumps its resolved config to `results/simulations/<experiment>/<run_id>/config.json` for
reproducibility.

---

## `src/utils/` — shared vocabulary

### `array_types.py` ✅ implemented

**Owns:** the array aliases every signature uses: `Float64Array`, `Float32Array`, `Int64Array`,
`BoolArray`, `ComplexArray`. One import instead of `npt.NDArray[np.float64]` spelled out in four
hundred places.
**Never:** Holds a function.

---

## `src/utils/metrics/` — scoring a result against the truth

### `localization_metrics.py` ✅ implemented

**Owns:** `compute_position_distance_matrix_m`,
`match_estimated_to_true_sources` (Hungarian assignment),
`compute_optimal_subpattern_assignment_distance` and
`build_localization_metric_record`.
**Never:** Runs an estimator.

**Optimal sub-pattern assignment rather than a root-mean-square error over
matched pairs,** because that error silently ignores missed and spurious
sources — the failure mode a multi-source sweep exists to expose. A run that
finds one source of three cannot look good by locating that one precisely.

---

## `src/utils/io/` — file operations

Moves bytes. Never transforms scientific content.

### `simulation_run_writer.py` ✅ implemented

**Owns:** `write_simulation_run`. Creates the run directory and writes the five files below. Validates that the signal array is three-dimensional and that its observation count matches the ground truth, because both mismatches are silent otherwise.
**Never:** Formats anything for display.

**Signals are float32 on disk.** At a 20 dB signal-to-noise ratio no estimator can use float64 precision, and it halves every run.

### `simulation_run_reader.py` ✅ implemented

**Owns:** the frozen `SimulationRun` dataclass, `read_simulation_run(run_directory)` and `list_simulation_runs(root)`. Every consumer of a run — the estimator, the plotters, the application — loads it through here, so none of them imports a script or hard-codes a filename.
**Never:** Transforms scientific content.

**Signals are memory-mapped, not read.** A run is hundreds of megabytes and the application must not block on opening one; slice the array to pull only the observations you need.

**The run directory is the whole contract between the forward side and the inverse side:**

```
config.json                     resolved ForwardSimulationConfiguration
receiver_positions_xy_m.npy     float64, (n_receivers, 2)
receiver_signals.npy            float32, (n_observations, n_receivers, n_samples)
ground_truth.json               one record per observation: simulation_time_s,
                                component_count, component_centroids_xy_m,
                                front_centroid_xy_m, front_radius_m,
                                front_cell_count, burning_cell_count,
                                ignited_cell_count, burnt_area_m2, source_count,
                                receiver_range_spread_m
metadata.json                   time_step_s, observation_interval_s,
                                observation_count, sample_rate_hz, receiver_count
```

Changing one side without the other breaks every consumer silently, so change both in the same commit.

### `audio_file_reader.py` ✅ implemented

**Owns:** Reads audio files (wav, flac), returns `(samples: ndarray, sample_rate_hz: int)`. Optional
mono downmix, and a header-only `read_audio_metadata`.
**Never:** Filters, normalizes, or segments.

### `metrics_writer.py` ✅ implemented

**Owns:** Appends metric records (JSON) to `results/metrics/`, or writes a whole document. Records are committed to git as the experiment record. Carries a numpy-to-JSON converter, which is why it holds the repo's only `ANN401` exemption.

---

## `src/utils/visualization/` — plotting

May import from everywhere. Nothing imports it back. Every function takes data and returns a `matplotlib.Figure`. **None of them calls `savefig`** — that is `scripts/`' job.

The no-`savefig` rule is enforced, not just stated: `tests/utils/visualization/conftest.py` monkeypatches `Figure.savefig` to raise for the whole plotter suite.

### `fire_state_plotter.py` ✅ implemented

**Owns:** the grid-state RGB image (burning, burnt and unburnt colormap) and `plot_ignition_time_map`, the ignition-time heatmap with front cells overlaid at chosen instants — figure F3.

Takes a concrete `SquareGridMesh` rather than a `MeshProtocol` because a raster image needs the row and column structure the protocol deliberately does not expose. An unstructured mesh needs a different renderer, not this one.

**Extension → 3D surface plot:** when the mesh has elevation, plot the fire state draped over the terrain.

### `mesh_plotter.py` ✅ implemented

**Owns:** `plot_scene_geometry` and `plot_scene_geometry_panels` — domain, ignition points, receivers, the perpendicular bisector of the first receiver pair and its near-singular band — plus `compute_range_ratio_field`, `compute_near_singular_band_mask` and `compute_perpendicular_bisector_endpoints_xy_m`. Figure F2.
**Never:** Imports `inverse/`.

**The shaded band is what makes it a figure rather than a map.** It is pure geometry, `|r1 / r2 - 1| < tolerance`, so it can be drawn without asking the estimator anything, and it says where no estimator can work rather than where this one happens to be imprecise.

### `rate_of_spread_plotter.py` ✅ implemented

**Owns:** rate of spread against wind speed, slope and moisture — figure F1 — against the Balbi 2009 Table 2 Case 1 fuel the equation tests already validate, so the figure and the unit tests agree by construction.

### `receiver_signal_plotter.py` ✅ implemented

**Owns:** received level traces over the observations of a run (F5); the per-band analysis chain on one clip, `ΔL_b`, `α_b·D` and `G_b` (F5); estimates against ground truth with error ellipses (F6); error against signal-to-noise ratio and error against distance from the bisector (F7); reduced chi-square per scenario (F8); front travel with its rate-of-spread fit and residual (F9).
**Never:** Imports from both `inverse/` and `fire/`. Every quantity from either side arrives as a function argument.

**Extension → multi-source diagnostics ✅ implemented:**
`plot_pair_correlation_curves` draws one generalised cross-correlation per
receiver pair with the true delay marked, so a peak sitting somewhere else is
visible as such rather than averaged into a position error;
`plot_band_level_difference_regression` draws the per-band level difference
against its absorption coefficient, where the model says the points fall on a
line whose intercept is the geometric term and whose slope is the path
difference.

### `steered_response_power_plotter.py` ✅ implemented

**Owns:** `reshape_map_to_grid`, `add_error_ellipse`,
`plot_steered_response_power_map` — the map as a heat map with receivers, truth,
estimates and error ellipses — and `plot_windowed_position_clusters`, the
per-window maxima with the clusters they formed.

### `localization_error_plotter.py` ✅ implemented

**Owns:** `plot_metric_against_parameter`, the sweep curves, and
`plot_source_count_confusion`, estimated against true source count. A point off
the diagonal there is the failure the sweep exists to find: a merged pair below
it, a spurious detection above it.

### `channel_plotter.py` ✅ implemented

**Owns:** ISO 9613-1 `α(f)` at the scenario's air conditions, and channel gain against range per octave band — figure F4.
**Never:** Knows what is being propagated.

This is the figure that shows *why* the high bands carry the range information: the spread between bands grows with range, and that spread is what an estimator inverts. No other plotter owned it.

---

## `scripts/` — runners

Zero logic. Import from `src/`, call one function. Each script is a thin CLI entry point.

### `run_fire_simulation.py`

Loads config → builds context → runs the spread engine loop → saves fire state snapshots → generates figures.

### `export_report_figures.py` ✅ implemented

Takes `--run <run_id>` and `--metrics <path>`, calls the plotters, saves SVG under stable names, and
writes the front animation as GIF. Zero plotting logic of its own. Running it regenerates every
figure in the report from scratch with no manual step.

**Figures are filed by what produced them,** in three kinds of directory under `results/figures/`:

| Directory | Holds |
|---|---|
| `general/` | F1 and F2 — the fire model, and every rung side by side. No one scene owns them |
| `<experiment>/` | F3, F4, F5, F9 and the front GIF — everything derived from the chosen run |
| `<metrics stem>/` | F6, F7, F8 — everything derived from a localization metrics document |

Each carries a `figure_metadata.json` holding the resolved configuration, the run identifier and the
run's own metadata. A figure on its own says nothing about which configuration made it; the sidecar
is how an `e1/` figure gets checked against the E1 scene rather than taken on trust.

**The scene replayed for F3 comes from the run's own `config.json`,** not from a separate flag, so
the front figure and the level traces cannot silently describe different scenes. `--configs`
overrides it deliberately.

### `run_audio_preprocessing.py`

Reads raw recordings → segments → filters → normalizes → writes processed segments and a grouped-split manifest.

### `run_kinematics_estimation.py`

Loads a saved simulation run → renders acoustic field at receivers → runs the inverse estimator → writes metrics and comparison figures.

### `run_acoustic_rendering.py` ✅ implemented

Loads `configs/` → builds the mesh, fuel, wind and receivers → derives `dt` from the CFL condition →
runs the arrival-time engine → extracts sources under the configured emission model → sums every
source at every receiver → writes `receiver_signals.npy` of shape `(n_steps, n_receivers, n_samples)`
alongside `receiver_positions_xy_m.npy`, `ground_truth.json`, `config.json` and `metadata.json` into
`results/simulations/<experiment>/<timestamp>/`, where the experiment name comes from that
scene's own `experiment.toml`. Takes `--configs <dir>`.

**Blocks are padded to one fixed length.** Propagation delay makes each rendered block a different
length, so the run sizes once against the worst-case source–receiver distance in the domain and pads
or trims every block to it. Without this the saved array cannot be rectangular and the silent steps
do not match the rendered ones.

### `run_single_source_localization.py` ✅ implemented

Loads `configs/` → reads a recording → cuts it into clips → renders each clip to the two receivers
→ localizes from those two signals alone → prints one table per source scenario and writes the
metrics document. Takes no parameters of its own beyond `--configs <dir>`.

### `run_source_excerpt_audit.py` ✅ implemented

Stage 0 of the multi-source pipeline, and the one to run before anything else.
Reads the pool, measures each recording's codec cutoff, resolves both band lists
(one wide passband for the correlation stage, fractional octaves for the level
stage), selects one excerpt per source, computes the mutual coherence matrix and
**exits non-zero** when two excerpts share waveform content beyond the permitted
limit. Writes `results/metrics/source_excerpt_audit.json`.

### `run_multi_source_localization.py` ✅ implemented

Loads `configs/` → gives each configured source its own recording excerpt →
builds the receiver layout → renders the mixture → adds receiver noise →
`localize_multiple_sources` → optional windowed-clustering cross-check →
optional level fusion → matches against ground truth → writes metrics and
figures. Takes `--configs <dir>` and nothing else.

### `run_localization_sweep.py` ✅ implemented

Loops one scene over receiver count, signal-to-noise ratio, source separation,
`β` and pairwise combinator, writing one metrics record per cell and the
resolution curve.

**This is the one script that imports another.** Reusing
`run_multi_source_localization`'s run driver keeps a single definition of what a
run is, so a sweep cell and a single run cannot drift apart. `scripts/` gained an
`__init__.py` to make that import resolvable; nothing in `src/` imports either of
them, which is the direction the dependency rule is about.

---

## `tests/`

Mirror the `src/` tree, directory for directory: `tests/audio/`, `tests/config/`,
`tests/spark/{acoustic,atmosphere,fields,fire,inverse,terrain}/`, `tests/utils/visualization/`.
Each test file tests one source file and is named after it. Cross-cutting architectural tests
(such as the isolation guard) sit at the top level, since they belong to no single source file.
Shared fixtures live in `tests/conftest.py` and load the real `configs/`, so the configuration
files are exercised on every run rather than duplicated in test constants.

### Mandatory tests

| Test file | What it asserts |
|---|---|
| `spark/terrain/test_square_grid_mesh.py` ✅ | Cell count, positions, neighbor connectivity, distances, unit directions |
| `spark/fire/test_rate_of_spread_equations.py` ✅ | Table 2 startup values, Fig. 6 reduced curve, closed form against the implicit Eq. 13b |
| `spark/acoustic/test_exponential_attenuation_channel.py` ✅ | Gain matrix shape, inverse-square-law sanity, symmetry, frequency widening |
| `test_inverse_does_not_import_forward_model.py` ✅ | `inverse/` has no import path to `fire/`, `terrain/`, `fields/`, or `acoustic/`, and those packages exist so the guard is not vacuous |
| `test_fire_state_immutability.py` | `step()` returns a new `FireState`, original is unchanged |
| `test_component_registry.py` | Every registered name resolves to a class that satisfies its Protocol |
| `spark/atmosphere/test_atmospheric_absorption.py` ✅ | ISO 9613-1 against the published table at four temperature/humidity pairs |
| `spark/inverse/test_single_source_estimator.py` ✅ | Noiseless synthetic sources recovered; error and ellipse grow with noise; bisector flagged |
| `audio/test_codec_bandwidth_detector.py` ✅ | Synthetic brickwall recovered to the analysis window's skirt; a lower wall detected lower |
| `audio/test_octave_band_filter.py` ✅ | `fraction_denominator = 1` reproduces the octave edges exactly; `= 3` matches the ISO third-octave series |
| `audio/test_usable_band_selector.py` ✅ | A 15.5 kHz cutoff at 0.9 margin admits 10 kHz and rejects 12.5 kHz |
| `audio/test_excerpt_coherence.py` ✅ | Identical excerpts score 1.0, independent noise near zero, the matrix is symmetric |
| `audio/test_source_excerpt_selector.py` ✅ | Distinct-provenance never reuses a file; the coherence limit raises; minimum-coherence avoids a duplicated pair |
| `audio/test_matched_filter_band_level.py` ✅ | Two sources at known gains: recovered level differences within 0.5 dB |
| `spark/acoustic/test_receiver_layout.py` ✅ | Ring, grid and random honour the separation guard; a collinear array of three or more raises; two receivers skip that guard |
| `spark/acoustic/test_free_field_propagation.py` ✅ | `n_sources = 1` matches the previous path to machine precision; two sources equal the sum of two single renders |
| `spark/inverse/test_receiver_pair_index.py` ✅ | Canonical ordering, pair count `n(n-1)/2`, the lag covers the longest baseline |
| `spark/inverse/test_time_difference_of_arrival.py` ✅ | The curve peak equals the previous scalar estimate; band-limited whitening ignores out-of-band noise; the envelope removes sign changes; the sign convention still pinned; reading near a prediction finds the quieter of two peaks |
| `spark/inverse/test_steered_response_power.py` ✅ | **Volumetric pooling puts the map maximum on the true cell where point sampling at the same spacing does not** — the regression for the revision-2 correction; the delay bounds contain every delay the cell can produce; every pooling rule and combinator peaks at the truth |
| `spark/inverse/test_multilateration.py` ✅ | Noiseless recovery to a micrometre; the Jacobian matches a finite difference; the covariance grows with the delay variance; `N = 3` flagged ambiguous with no spare degrees of freedom |
| `spark/inverse/test_source_deflation.py` ✅ | Deflating removes the located source from every pair curve; deflating twice changes nothing further; the projection spares what sits under the source where the notch does not |
| `spark/inverse/test_windowed_position_clustering.py` ✅ | Two separated point clouds form two clusters; a thin cluster is dropped; one rendered source recovered as one cluster |
| `spark/inverse/test_multiple_source_estimator.py` ✅ | One and two rendered sources recovered; `K̂` is an output, not an input; the class and the driver agree |
| `spark/inverse/test_joint_position_refinement.py` ✅ | Adding the level terms shrinks the error ellipse; the level Jacobian matches a finite difference |
| `utils/metrics/test_localization_metrics.py` ✅ | Hungarian matching on a permuted set; a missed source priced at the cutoff, the same as a spurious one |
| `utils/visualization/test_steered_response_power_plotter.py` ✅ | The map folds back into its grid; both figures survive having found nothing |
| `utils/visualization/test_localization_error_plotter.py` ✅ | One line per series; a label mismatch raises |

---

## Extension summary

All extensions in one table, sorted by likely implementation order.

| Extension | Files touched | Protocol change | Priority |
|---|---|---|---|
| Balbi 2009 ROS equations | `rate_of_spread_equations.py` | None | Week 1 |
| ROS-driven ignition delay engine | `rate_of_spread_engine.py` | None | Week 1 |
| Exponential attenuation channel | `exponential_attenuation_channel.py` | None | Week 1 |
| Single-source triangulation | `single_source_estimator.py` | None | Week 1 |
| Receiver layout in config | `simulation_configuration.py` | None | Week 1 |
| Per-cell physical quantities in FireState | `fire_state.py` | None (additive fields) | Week 1–2 |
| Physics-based source amplitude | `burning_cell_source_model.py` | None | Week 2 |
| Elevation on square grid (3D) | `square_grid_mesh.py`, `elevation_field_from_raster.py` | None | Week 2 |
| Cellular automaton engine | `cellular_automaton_spread_engine.py` | None | Week 2 |
| Firebrand transport | `cellular_automaton_spread_engine.py` | None | Week 2 |
| Game of life engine | New file | None | Week 2 |
| Poisson tree placement | `random_tree_placement_field.py` | None | Week 2 |
| Clustered tree placement | `random_tree_placement_field.py` | None | Week 2 |
| Frequency-dependent attenuation | `exponential_attenuation_channel.py` | Gain widens to 3D | Week 2 |
| Spectral source model | `burning_cell_source_model.py` | Return widens to 2D | Week 2 |
| Two-source estimation | `multiple_source_estimator.py` | None | ✅ done |
| Fractional-octave filterbank | `octave_band_filter.py` | None | ✅ done |
| Codec bandwidth detection | `codec_bandwidth_detector.py` | None | ✅ done |
| Per-source excerpt selection and its coherence guard | `excerpt_coherence.py`, `source_excerpt_selector.py` | None | ✅ done |
| Matched-filter per-source band levels | `matched_filter_band_level.py` | None | ✅ done |
| N-receiver layouts and their guards | `receiver_layout.py`, `receiver_placement.py` | None | ✅ done |
| Multi-source forward render | `free_field_propagation.py` | None | ✅ done |
| Correlation curve as the primitive | `time_difference_of_arrival.py` | Additive | ✅ done |
| Steered response power with volumetric pooling | `steered_response_power.py` | None | ✅ done |
| Correlation-domain deflation | `source_deflation.py` | None | ✅ done |
| N-receiver multilateration | `multilateration.py` | None | ✅ done |
| Windowed-clustering cross-check on `K̂` | `windowed_position_clustering.py` | None | ✅ done |
| Delay-and-level joint refinement | `joint_position_refinement.py` | None | ✅ done |
| Optimal sub-pattern assignment scoring | `utils/metrics/localization_metrics.py` | None | ✅ done |
| Balbi 2020 fixed-point ROS | `rate_of_spread_equations.py` | None | If time allows |
| Interpolated wind field | New file | None | If time allows |
| Terrain-aware path loss | New wrapper file | None | If time allows |
| Triangular mesh | `triangular_mesh.py` | None | If time allows |
| Vegetation raster field | New file | New `FuelFieldProtocol` | If time allows |
| Dense receiver selection | `multiple_source_estimator.py` | None | If time allows |
| Group-sparse joint map fitting | `multiple_source_estimator.py` | None | Contingency if deflation breaks at K = 3 |
| N-source general assignment | `multiple_source_estimator.py` | None | Future work |
| Time-varying wind | New file | Protocol signature widens | Future work |
| Measured impulse response channel | `measured_impulse_response_channel.py` | None | Future work |
