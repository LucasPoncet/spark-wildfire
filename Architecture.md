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
 
src/config/             ← imported by scripts, imports Protocols only
src/utils/io/           ← imported by scripts, imports domain types for serialization
src/utils/visualization ← imports everything, imported by nothing
src/audio/              ← standalone, no domain imports
scripts/                ← imports everything, imported by nothing
```
 
**Hard rule:** `src/spark/inverse/` must never import from `src/spark/fire/`, `src/spark/terrain/`, `src/spark/fields/`, or `src/spark/acoustic/`. Its only inputs are receiver signals and receiver positions. This prevents the pipeline from committing the leakage defect the paper criticizes. Enforced by `tests/test_inverse_does_not_import_forward_model.py`.
 
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
 
### Extension → `game_of_life_spread_engine.py`
 
**Status:** Not yet in tree but planned. Simplest possible engine: a cell ignites if ≥ N neighbors are burning. Fixed burnout timer. No physics. Useful as a sanity check that the protocol, the mesh, and the visualization pipeline all work before introducing real ROS equations.
 
---
 
## `src/spark/acoustic/` — sound propagation
 
### `channel_protocol.py`
 
**Owns:** `ChannelProtocol` with method `compute_gain_matrix(source_positions_xyz, receiver_positions_xyz) -> ndarray` returning shape `(n_sources, n_receivers)`.
**Never:** Contains implementation.
 
The forward acoustic render is: `received_levels = gain_matrix.T @ source_amplitudes`. This one line is the source–channel model made literal in code.
 
### `burning_cell_source_model.py` ✅ implemented
 
**Owns:** `BurningCellSourceModel`. Takes a `FireState`, returns `(source_positions_xyz, source_amplitudes)`. The only bridge between the fire domain and the acoustic domain.
**Never:** Knows about receivers, channels, or propagation.
 
**Day-one version:** amplitude is a constant for every burning cell. All fires sound the same.
 
**Extension → physics-based amplitude:** use `mass_loss_rate_kg_per_s` from `FireState` (available when using the Balbi ROS engine) as a proxy for acoustic power. Louder fires produce more sound.
 
**Extension → spectral source model:** instead of a scalar amplitude, return a spectrum per cell. Amplitude in the crackling band (1–15 kHz) driven by fuel type and burn rate. Amplitude in the puffing band (< 20 Hz) driven by flame diameter via Cetegen's scaling `f_puff ≈ 1.5 * D^(-0.5)`. Return type widens from `(n,)` to `(n, n_freq)`.
 
### `exponential_attenuation_channel.py`
 
**Owns:** `ExponentialAttenuationChannel`. Implements `gain(r) = exp(-α * r) / r` where `r` is Euclidean distance and `α` is the attenuation coefficient. Builds the full `(n_sources, n_receivers)` gain matrix.
**Never:** Knows about fire state or terrain.
 
**Extension → frequency-dependent attenuation:** `α` becomes a function of frequency. The gain matrix widens to `(n_sources, n_receivers, n_freq)`. Atmospheric absorption rises with frequency, so high-frequency crackle content decays faster than low-frequency roar. This is what makes range recoverable.
 
**Extension → terrain-aware path loss:** if source and receiver are not line-of-sight (terrain obstruction), apply diffraction loss. Requires querying the elevation field along the source–receiver path. Add as a wrapper around the base channel, not as a modification to it.
 
### `measured_impulse_response_channel.py`
 
**Owns:** Stub for a channel built from experimentally measured impulse responses. Loads measured `h` from file, applies it as a convolution or frequency-domain multiply.
**Status:** Stub. Implement if/when measured data becomes available.
 
---
 
## `src/spark/inverse/` — kinematics estimation
 
**Hard rule:** this entire subpackage sees only receiver signals and receiver positions. It must never import from `fire/`, `terrain/`, `fields/`, or `acoustic/`.
 
### `kinematics_estimator_protocol.py`
 
**Owns:** `KinematicsEstimatorProtocol` with method `estimate(receiver_signal_levels, receiver_positions_xyz) -> FrontKinematics`. `FrontKinematics` is a dataclass holding `bearing_rad`, `position_xyz`, `rate_of_spread_m_per_s` and associated uncertainties.
**Never:** Contains implementation.
 
### `single_source_estimator.py`
 
**Owns:** `SingleSourceEstimator`. Assumes one fire front. Estimates distance from relative level decay across receivers (the unknown source amplitude cancels in the ratio). Triangulates position from multiple receiver pairs.
**Never:** Accesses ground-truth fire state.
 
**Extension → bearing estimation:** use the spatial gradient of received levels across the receiver array to estimate the direction to the source.
 
**Extension → rate of spread estimation:** compare estimated positions at consecutive time steps to estimate front velocity.
 
### `multiple_source_estimator.py`
 
**Owns:** `MultipleSourceEstimator`. Handles superposition of signals from multiple concurrent fire fronts. Separates the mixture into per-source contributions, then applies single-source estimation to each.
**Status:** Stub. Scope to two well-separated sources first.
 
**Extension path:**
 
| Step | What changes |
|---|---|
| 1. Two sources, well-separated bearings | Cluster receivers by dominant source, apply single-source to each cluster |
| 2. Two sources, overlapping | Source separation (e.g. NMF on the spatial level matrix), then single-source per component |
| 3. N sources | General multi-source assignment. Future work in the paper |
 
**Extension → dense receiver selection:** when many receivers are available, select the subset with highest SNR or best geometric diversity. One additional method in this file.
 
---
 
## `src/audio/` — real audio preprocessing
 
Standalone pipeline for processing real fire recordings. No imports from the simulation domain.
 
### `recording_segmenter.py`
 
**Owns:** Cuts audio files into fixed-length segments (default 5 s) with configurable overlap.
 
### `lowpass_filter.py`
 
**Owns:** Filter design and application. Verifies sample rate consistency across sources. Default: high-pass at 50 Hz (wind rumble), keep everything above — do not low-pass aggressively, the high band is what makes range recoverable.
 
### `amplitude_normalizer.py`
 
**Owns:** Per-segment normalization. Scheme to be determined (peak, RMS, LUFS).
 
### `grouped_split_manifest.py`
 
**Owns:** Creates train/test split manifests with group IDs derived from source recording provenance (channel signature, noise-floor shape, spectral rolloff), not from filenames. This is the project's own consistency check against the leakage critique (claim 1 in the paper).
 
---
 
## `src/config/` — configuration and wiring
 
### `simulation_configuration.py`
 
**Owns:** Nested frozen dataclasses describing what to build. Fully serializable to JSON. One dataclass per component: `MeshConfiguration`, `FuelConfiguration`, `WindConfiguration`, `SpreadEngineConfiguration`, `ChannelConfiguration`, `ReceiverConfiguration`.
**Never:** Imports concrete implementations.
 
**Extension → receiver layout configuration:** number of receivers, placement strategy (regular grid, random, ring), spacing. This is a swept parameter in the paper's degradation analysis.
 
### `simulation_context.py`
 
**Owns:** `SimulationContext` frozen dataclass holding live Protocol instances: `mesh: MeshProtocol`, `elevation_field: ScalarFieldProtocol`, `fuel_field: ScalarFieldProtocol`, `wind_field: VectorFieldProtocol`, `spread_engine: SpreadEngineProtocol`, `channel: ChannelProtocol`.
**Never:** Contains logic. Imports only Protocols, never concrete classes.
 
### `simulation_context_factory.py`
 
**Owns:** `build_simulation_context(config) -> SimulationContext`. The composition root. The **only file in the entire repo** that imports concrete classes (`SquareGridMesh`, `UniformScalarField`, `ExponentialAttenuationChannel`, etc.). Reads config, looks up classes from the registry, instantiates them, returns a context.
**When you add a new implementation:** register it in the registry and add one branch here. Nothing else moves.
 
### `component_registry.py`
 
**Owns:** Name-to-class maps for every Protocol:
 
```
MESH_REGISTRY: dict[str, type[MeshProtocol]]
SPREAD_ENGINE_REGISTRY: dict[str, type[SpreadEngineProtocol]]
CHANNEL_REGISTRY: dict[str, type[ChannelProtocol]]
ESTIMATOR_REGISTRY: dict[str, type[KinematicsEstimatorProtocol]]
```
 
Every run dumps its resolved config to `results/simulation_runs/<run_id>/config.json` for reproducibility.
 
---
 
## `src/utils/io/` — file operations
 
Moves bytes. Never transforms scientific content.
 
### `simulation_run_writer.py`
 
**Owns:** Creates `results/simulation_runs/<run_id>/`, writes resolved config JSON, saves `FireState` snapshots as numpy arrays at configurable intervals.
 
### `simulation_run_reader.py`
 
**Owns:** Loads a saved run back for replay or post-hoc analysis.
 
### `audio_file_reader.py`
 
**Owns:** Reads audio files (wav, flac), returns `(samples: ndarray, sample_rate_hz: int)`.
**Never:** Filters, normalizes, or segments.
 
### `metrics_writer.py`
 
**Owns:** Appends metric records (JSON) to `results/metrics/`. Records are committed to git as the experiment record.
 
---
 
## `src/utils/visualization/` — plotting
 
May import from everywhere. Nothing imports it back. Every function takes data and returns a `matplotlib.Figure`. **None of them calls `savefig`** — that is `scripts/`' job.
 
### `fire_state_plotter.py` ✅ implemented
 
**Owns:** Grid-state image (burning/burnt/unburnt colormap), ignition-time heatmap, fire perimeter animation frames.
 
**Extension → 3D surface plot:** when the mesh has elevation, plot the fire state draped over the terrain.
 
### `mesh_plotter.py`
 
**Owns:** Mesh geometry wireframe, cell positions, receiver positions overlaid, elevation surface when available.
 
### `rate_of_spread_plotter.py`
 
**Owns:** ROS vs wind speed, ROS vs slope, ROS vs moisture curves. Used to reproduce Balbi 2009 Fig. 3 / Fig. 4 as a unit-test-with-visual-output.
 
### `receiver_signal_plotter.py`
 
**Owns:** Received level traces over time, spatial level maps across the receiver array, estimated bearing vs ground-truth bearing.
**Never:** Imports from both `inverse/` and `fire/` simultaneously without both datasets arriving as function arguments.
 
---
 
## `scripts/` — runners
 
Zero logic. Import from `src/`, call one function. Each script is a thin CLI entry point.
 
### `run_fire_simulation.py`
 
Loads config → builds context → runs the spread engine loop → saves fire state snapshots → generates figures.
 
### `run_audio_preprocessing.py`
 
Reads raw recordings → segments → filters → normalizes → writes processed segments and a grouped-split manifest.
 
### `run_kinematics_estimation.py`
 
Loads a saved simulation run → renders acoustic field at receivers → runs the inverse estimator → writes metrics and comparison figures.
 
---
 
## `tests/`
 
Mirror the `src/` tree. Each test file tests one source file.
 
### Mandatory tests
 
| Test file | What it asserts |
|---|---|
| `test_square_grid_mesh.py` ✅ | Cell count, positions, neighbor connectivity, distances, unit directions |
| `test_rate_of_spread_equations.py` ✅ | Table 2 startup values, Fig. 6 reduced curve, closed form against the implicit Eq. 13b |
| `test_exponential_attenuation_channel.py` | Gain matrix shape, inverse-square-law sanity, symmetry |
| `test_inverse_does_not_import_forward_model.py` | `inverse/` has no import path to `fire/`, `terrain/`, `fields/`, or `acoustic/` |
| `test_fire_state_immutability.py` | `step()` returns a new `FireState`, original is unchanged |
| `test_component_registry.py` | Every registered name resolves to a class that satisfies its Protocol |
 
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
| Two-source estimation | `multiple_source_estimator.py` | None | Week 2 |
| Balbi 2020 fixed-point ROS | `rate_of_spread_equations.py` | None | If time allows |
| Interpolated wind field | New file | None | If time allows |
| Terrain-aware path loss | New wrapper file | None | If time allows |
| Triangular mesh | `triangular_mesh.py` | None | If time allows |
| Vegetation raster field | New file | New `FuelFieldProtocol` | If time allows |
| Dense receiver selection | `multiple_source_estimator.py` | None | If time allows |
| N-source general assignment | `multiple_source_estimator.py` | None | Future work |
| Time-varying wind | New file | Protocol signature widens | Future work |
| Measured impulse response channel | `measured_impulse_response_channel.py` | None | Future work |
