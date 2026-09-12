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
explains.** That is the whole difference, and neither is uniformly better. Which
one to use depends on how far apart the sources are, and the two scenes disagree
by design:

| Scene | Sources | Method | Width | Outcome |
|---|---|---|---|---|
| `m1` | Two, 40 m apart, 6 dB gap | `subspace_projection` | `1e-4` | `K̂ = 2`, exact |
| `m3` | Six, 8 m apart on a ring | `subspace_projection` | `1e-4` | `K̂ = 3` of 6 |
| `m3` | Six, 8 m apart on a ring | `subspace_projection` | `5e-4` | `K̂ = 1` of 6 |
| `m3` | Six, 8 m apart on a ring | `tdoa_notch` | `5e-4` | `K̂ = 6`, exact |

**On a compact cluster the projection leaves structured residue and the notch
does not.** The projection's coefficient is a least-squares fit of one template
to the curve, and when neighbouring sources sit inside that template they inflate
the fit, so it subtracts the wrong amount and leaves a peak sitting at the delay
it just located. The residual map then peaks on that residue, the loop calls it a
re-detection and stops. Widening the template makes this worse, not better,
because it admits more neighbours into the fit: on `m3` it takes `K̂` from three
down to one. The hard notch has no coefficient to get wrong — it clears the
footprint outright — and recovers all six.

**The separation stop rule is not what is binding there.** Dropping
`minimum_source_separation_m` from 3 m to 0.5 m on `m3` changes nothing at all,
which says the residual map peak lands within half a metre of a source already
accepted. Neither the pairwise combinator nor the receiver count changes it
either: `product` and `sum`, at six and eight receivers, all return the same
three sources. Only the deflation method does.

**So the guidance inverts with source separation.** Well separated and unequal in
power, prefer the projection, which spares a weak neighbour the notch would take.
Close together, prefer the notch, which clears residue the projection cannot.
The published claim that the projection outperforms the notch is stated for the
power-disparity case, and that is the case where it holds here too.

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

**Where the sequential loop actually stops, at ten sources and beyond.** On the
`m4` ring at 20 dB, positions are exact whenever they are found and the whole
score is cardinality:

| K \ N | 15 | 20 | 25 |
|---|---|---|---|
| 10 | 10 of 10 | 10 of 10 | 10 of 10 |
| 15 | 8 of 15 | 8 of 15 | 8 of 15 |
| 20 | 6 of 20 | 5 of 20 | 6 of 20 |

Two things are visible at once. **Receiver count buys nothing**: the rows are
identical to three decimals across fifteen, twenty and twenty-five receivers, so
whatever caps `K̂` is not the array. And **every cell of the grid stopped on
`deflation_stopped_removing_energy`**, which names the cap.

**`residual_power_reduction_threshold` is a fixed fraction, and a fixed fraction
cannot scale with the source count.** Each source contributes roughly `1/K` of
the correlation energy, so the drop one deflation produces shrinks as `K` grows
until it falls under the threshold with sources still un-found. Measured on an
earlier, wider-spread variant of this scene at `K = 15`, `N = 20`:

| Threshold | 0.02 | 0.01 | 0.005 | 0.002 | 0.0005 |
|---|---|---|---|---|---|
| `K̂` of 15 | 5 | 14 | 22 | 22 | 22 |

It is knife-edge: halving the threshold takes `K̂` from 5 to 14, halving it again
overshoots to 22 with matched positions degrading as spurious sources crowd in.
Neither the shipped value nor any single replacement is right, because the rule
is the wrong shape. A rule scaled to the energy a single source is expected to
carry, or an absolute prominence test on the residual map, would be the fix.
That is a design change beyond the plan this work implements, and it is recorded
here rather than made silently.

**`minimum_source_separation_m` is a floor on what a scene may report, and it
will be mistaken for a resolution limit if it is not set below the separations
of interest.** Swept over two sources at 2, 4, 8 and 16 m with the shipped value
of 3 m, the run returns `K̂ = 1` at 2 m and `K̂ = 2` everywhere above, at every
receiver count from three to eight — which reads exactly like a resolution limit
between 2 and 4 m that more receivers cannot fix. It is not one. Dropping the
guard to 0.5 m resolves two sources **1 m apart to a millimetre**. The limit
being measured was the configuration.

Where the true limit lies is therefore still open: it is below one metre at
20 dB in free field, and nothing here has found it.

The same trap caught `m4` twice. Its first geometry spread the sources 7 m apart
and shipped the 3 m guard; tightening the ring to 2.7 m spacing while dropping
the guard to 1.5 m took `K̂` at twenty sources from 3 to 6, which reads as closer
sources being easier and is nothing of the kind. **Set the guard from the
scene's own closest pair, and say so in the configuration.**

**A source's delays are read from its own neighbourhood of each curve,** not
from each curve's global maximum, which in a mixture belongs to whichever source
dominates that pair.

**Three sources works, against the literature's prediction.** The deflation
literature reports that at three sources the noise in the correlation function
becomes prohibitive. On the `m2` scene — three equal-amplitude sources at least
40 m apart, each carrying a genuinely decorrelated excerpt, rendered noiselessly
through free field — the loop returns `K̂ = 3` with every position exact. That is
the most favourable case there is, so it does not refute the ceiling; it locates
it above three for well-separated, equal, decorrelated sources. `e2` shows the
same at `K = 2, N = 3`.

**A compact contour works too, once the deflation suits it.** On the `m3` ring —
six sources 8 m apart on a circle of radius 8 m, eight receivers — the loop
returns `K̂ = 6` with every point exact to a centimetre.

Getting there took the finding in `source_deflation.py`: with the subspace
projection the same scene returns three of six, and the three it misses are the
far half of the ring. That is not the geometry failing. It is the projection
leaving residue at the delay it just located, which the residual map then reads
as the strongest remaining candidate. Neither the combinator, nor the receiver
count, nor a six-fold tightening of the separation threshold moves it; switching
to the hard notch recovers all six.

The lesson generalises past this scene: **the source count a sequential loop can
reach is set by how cleanly deflation removes what it has already found**, not by
how many sources the map could in principle separate.

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

**What breaks it is excerpt coherence, not power disparity.** That was not
obvious, and the first measurements pointed the wrong way:

| Scene | Sources | Excerpt coherence | Window split | Reported |
|---|---|---|---|---|
| `m1`, coherent excerpts | Two, second 6 dB down | 0.478 | 48 of 49 | `K̂ = 1` |
| `m1`, decorrelated excerpts | Two, second 6 dB down | 0.026 | 35 / 14 | `K̂ = 2` |
| `e2` | Two, equal amplitude | 0.026 | 97 of 99 | `K̂ = 1` |
| `m2` | Four, equal amplitude | 0.097 | 47 of 49 | `K̂ = 1` |

The same `m1` geometry and the same 6 dB disparity give `K̂ = 1` on excerpts that
share waveform content and `K̂ = 2` on excerpts that do not. A power disparity was
the predicted cause and it is not the binding one: the quieter source still wins
fourteen windows outright once its excerpt is its own.

`e2` and `m2` show the route is not simply reliable either — decorrelated
excerpts, equal amplitudes, and one source still takes nearly every window. Two
sources with a 6 dB gap split the windows; two and four equal ones do not, which
is the opposite of what a power-disparity explanation predicts and is not
currently explained.

So this remains a cross-check to read alongside the sequential loop rather than
a second opinion to trust on its own, and a disagreement is a question about the
scene, not a verdict on `K̂`.

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

### Fire characterisation — Tier 0 foundations

Tier 0 of `Related_works/FIRE_CHARACTERIZATION_PLAN.md`. The plan's reframing is
that deflation fits a **sparse point-source model** to a source that is
genuinely **distributed** — a closed curve of radiating cells — and that the
observed ten-source ceiling is where that mismatch binds rather than where the
information runs out. Every characterisation tier therefore reads **the map**,
not the accepted detections, and Tier 0 is what makes the map safe to read that
way.

Two map families now run from the same correlation curves. The detection family
(`beta = 0.7`, product, in `[steered_response_power]`) is **untouched**, so every
committed localisation result stays reproducible. The imaging family lives in
`[imaging_map]` of `configs/<scene>/fire_characterization.toml` and is described
as a set of *overrides* on the scene's own correlation and search settings
rather than a second copy of them, so a scene that changes its whitening band
changes it for both.

#### `map_normalization.py` ✅ implemented

**Owns:** `subtract_map_background`, `normalize_map_to_unit_mass` and
`prepare_map_for_moments` — the single preparation path every moment in the
package runs through.
**Never:** Knows what produced the map.

**That single path is load-bearing, not tidiness.** The extent stage differences
two second moments, one from a frame and one from the analytic response, so any
preparation applied to one and not the other appears as a bias in the
difference rather than as a visible discrepancy.

**Background subtraction is idempotent** because after one pass exactly the
bottom `background_percentile` per cent of cells are zero, so the same
percentile of the result is itself zero.

#### `map_moments.py` ✅ implemented

**Owns:** `MapMoments`, `compute_map_moments`, `compute_directional_third_moment`,
`compute_directional_skewness` and `compute_skewness_over_directions`. Mass,
centroid, second central moment with its principal axes, and the directional
third moment.
**Never:** Loops over grid points in Python.

**Pulled forward from the plan's Tier 1.** Milestones 0.3 and 0.4 both need a
centroid and a second moment, so the plan's ordering is not runnable as written;
this is a genuine dependency rather than scope creep.

**The plan lists the directional third moment as a `Callable` field on the
result.** It is a free function here instead: a callable cannot be serialised
into a metrics record and would be the only field in the package that is not
data.

**A ring of radius `R` has second central moment `R² / 2` on each axis**, so
`R = √(2λ)`. That is what makes a front's radius recoverable from a map that
never resolves the ring as a ring, and it is the relation the whole extent tier
rests on, so it is pinned exactly by test rather than approximately.

#### `steered_response_power_sequence.py` ✅ implemented

**Owns:** `SteeredResponsePowerSequence`, `compute_imaging_grid_shape`,
`compute_frame_start_indices`, `compute_steered_response_power_sequence`,
`build_imaging_candidate_grid` and `reshape_sequence_map_to_grid`.
**Never:** Searches coarse-to-fine — imaging wants one fixed grid at every
frame so frames can be differenced and regressed against time.

**Frames are cut from the waveforms, not from the maps.** The frame length sets
both how far a moving front smears and how many correlation windows average into
each curve, and those pull against each other, which is why it is configuration
and why the speckle floor is measured against it rather than assumed.

**A frame is timed at its centre**, which is the instant a linear fit over
frames is unbiased about. A trailing partial frame is dropped rather than padded,
matching `recording_segmenter`: a short frame averages fewer windows and would
otherwise sit on the same axis as full ones with nothing marking it as noisier.

**Named `reshape_sequence_map_to_grid`**, not `reshape_map_to_grid`, because the
plotter already owns a function of that name which infers the grid from the
positions of a single map; here the shape is known and carried on the sequence.

#### `point_spread_function.py` ✅ implemented

**Owns:** `compute_whitened_autocorrelation_kernel`,
`build_point_spread_correlation_curves`, `compute_point_spread_function`,
`compute_point_spread_covariance`, `build_point_spread_covariance_field` and
`compute_point_spread_semi_axes_m`.
**Never:** Requires a forward model — the response is fixed by array geometry,
the whitening band, the speed of sound and the pooling, so this works unchanged
on recorded data and belongs in `inverse/`.

**Each pair's curve carries the level that pair would actually receive**, and
that is only true because the imaging family stopped rescaling. Under a
combinator that normalises every pair map before merging, pair amplitude
cancels and an equally-weighted response is exact; under `unscaled_sum` it is
not, because pairs contribute in proportion to what they hear. Leaving the
weighting out cost 12 per cent on the response semi-axis at half the ring
radius and 19 per cent at 0.85 of it, and took the analytic-against-rendered
correlation to 0.94. Adding it returns 2 per cent and 0.9996.

Two other explanations were measured and rejected first. Grid discretisation:
refining from 0.5 m to 0.25 m, which takes the response from six cells across
to twelve, made the agreement slightly *worse*. Spectral tilt from atmospheric
absorption: that would widen the rendered response, and it was narrower. The
scale is free-field spreading computed from receiver positions, which is
geometry the estimator assumes, so `inverse/` still imports nothing from
`acoustic/`.

**The response is synthesised, not derived.** Rather than writing an analytic
expression for the lobe, this builds the cross-spectrum a noiseless point source
would produce and hands it to the *shipped* curve builder and the *shipped* grid
evaluator. An analytic lobe would silently omit the analytic envelope, the lag
truncation and the per-pair rescaling inside the combinator — and the extent
stage subtracts this from an observed covariance, so each omission returns as a
bias.

**The phase transform exponent does not change the response of flat in-band
content.** Whitening divides by the magnitude of the *mixture*, so `|X|^(1−β)`
is constant across the band whatever `β` is. The exponent acts only through
spectral non-flatness, which is a property of the sources present rather than of
the array. Where that matters, a measured magnitude can be passed in.

**The response is widest at the centre, not at the edge.** Measured on the
twenty-receiver ring: 12.1 m semi-axis at the domain centre, about 8.0 m at half
the ring radius and 8.8 m at 0.85 of it. The plan's expectation of a response
that broadens towards the edge does not hold for the `sum` combinator, whose
pairwise maps agree over a wide region at the centre of a symmetric array.

**Measured, and it constrains Tier 2: the response width is set by the
background percentile, not by the band.** Widening the correlation band fourfold
moves the semi-axis by about two per cent; moving the percentile from 0 to 80
moves it by a factor of three. The width of this map is a property of the array
geometry and of the preparation, not of the correlation lobe. So
`Σ_source = Σ_observed − Σ_psf` is only meaningful when both sides are prepared
identically, and the shape factor must be calibrated at the percentile it will
be applied at.

**That percentile is what closed the bias failure, and it cuts the other way
for the extent.** At 60 the response is wide enough that the domain boundary
truncates it and drags the centroid 3.1 m over the region a fifty-second front
occupies; at 98 the response is 1.5 m and the bias is 0.23 m. Splitting the two
stages — a low percentile for the second moment, a high one for the first — was
tried on the reasoning that a second moment *is* the tails a high percentile
removes. It fails completely: at 60 the response semi-axis grows to between 7.5
and 10.5 m, far faster than the front does, so `Σ_observed − Σ_psf` goes
non-positive and every frame is correctly reported unresolved. The response has
broader tails than a compact front, so lowering the percentile hands it more
than it hands the fire. The configuration keeps the two fields separate, and on
this scene they hold the same value.

#### `centroid_bias_correction.py` ✅ implemented

**Owns:** `CentroidBiasField`, `save_centroid_bias_field`,
`load_centroid_bias_field`, `interpolate_bias_offset_xy_m`,
`apply_centroid_bias_correction`, `compute_maximum_offset_magnitude_m` and
`has_no_sign_flip_between_adjacent_nodes`.
**Never:** Extrapolates outside the tabulated box — a position beyond it is
clamped, because a bias already growing towards the boundary is least
trustworthy exactly where extrapolation would amplify it.

**A centroid bias masquerades as a bearing.** A response that is not symmetric
about its source displaces the centroid of that source, and an uncorrected
displacement growing with radius is indistinguishable from a fire spreading
outward — the one thing the bearing tier exists to measure.

**Measured, and larger than the plan expected.** On the twenty-receiver ring the
bias is zero at the exact domain centre, which is a symmetry point of the
imaging grid where it vanishes by construction rather than by merit; 0.86 m
three and a half metres away; and about 3.1 m at ten to fifteen metres out,
which is where a fifty-second front sits. The plan's GATE 0 line asks for under
0.3 m at the centre, and read literally it passes on a symmetry artefact, so
`run_tier0_report.py` reports the same threshold over the region a centroid can
actually land in as well — and that line fails at 3.1 m.

**Correction is iterated because the offset is a function of where the source
is, not of where its centroid landed.** One subtraction only moves the lookup
closer to the right place.

### Fire characterisation — Tiers 1 and 2

Bearing and rate of spread, both read off the imaging map rather than off any
detection. Both work: on `configs/f1` the bearing lands within a degree and the
head rate within twenty per cent of the Balbi physics that produced it.

#### `fire_bearing.py` ✅ implemented

**Owns:** `BearingEstimate`, `estimate_ignition_position_xy_m`,
`estimate_bearing_from_centroid_drift`, `estimate_bearing_from_centroid_offset`,
`estimate_bearing_from_skewness`, `estimate_bearing_from_principal_axis`,
`fuse_bearing_estimates`, and the circular statistics they all run through.
**Never:** Averages an angle linearly. The linear mean of 350 and 10 degrees is
180, so every mean, spread and interval here is circular.

**Measured on `configs/f1`:** centroid drift −0.67°, centroid offset a median
0.85° per frame with every frame inside 10°, fused −0.65°, against a true
bearing of 0°. The ignition point is estimated from the first frame's centroid
at 0.52 m from truth, inside the plan's 1.0 m line, and nothing downstream is
told where the fire actually started.

**The plan has the skewness sign backwards, and it would have passed
unnoticed.** Milestone 1.3 says to take the argmax of the directional skewness.
A third moment points along a distribution's *long tail*, and concentrating
mass toward the head leaves the tail behind it: on a ring whose downwind half
is twice as bright, the skewness along the true bearing is −0.51 and along its
reverse is +0.51. The argmax returns the bearing rotated by half a turn. Taking
the argmin instead moves the per-frame error from 137° to under a degree.

**`estimate_bearing_from_centroid_offset` is not in the plan and was added
because the scene needed it.** It reads the direction from the estimated
ignition point to the frame's centroid — a *first* moment where skewness is a
third. A front that burns out behind itself sits downwind of its origin whether
or not its head burns brighter, so this works on a source model with no
intensity gradient at all, which is what the burning-cell model is.

#### `fire_extent.py` ✅ implemented

**Owns:** `ExtentEstimate` and `estimate_fire_extent`, computing
`Sigma_source = Sigma_observed - Sigma_response` and turning its eigenvalues
into semi-axes through the shape factor.
**Never:** Clips a negative eigenvalue to zero and calls the result a
measurement. A front smaller than the response along an axis returns
`is_resolved = False`, `upper_bound_only = True` and the observed semi-axis as
a ceiling — which on `configs/f1` happens on the first two frames and is
verified honest, meaning those frames really were below the response.

**The deconvolved semi-axis runs 19 to 43 per cent under truth**, worst when
the fire is smallest and improving monotonically as it grows: the ratio of true
to estimated falls 1.75, 1.50, 1.44, 1.40, 1.32, 1.26, 1.24 across the run.
That is the plan's 25 per cent extent criterion failed on the early resolved
frames, and it is what the head rate's residual error is made of.

#### `fire_rate_of_spread.py` ✅ implemented

**Owns:** `RateOfSpreadEstimate`, `project_head_and_back_m` and
`estimate_rate_of_spread`. Head, back, centroid speed and expansion rate as
Theil-Sen slopes against time, with a bootstrap interval on the head rate.
**Never:** Fits on an unresolved frame, or on fewer than the configured
minimum.

**Measured on `configs/f1`:** head 0.2969 m/s against a true 0.2500 (18.7 per
cent, inside the plan's 20), back 0.0858 against 0.1000 (14.2 per cent, inside
its 40), on eight resolved frames of ten. The true head rate is itself a good
check — Balbi at this scene's 12 m/s wind gives 0.245 m/s.

**The closure residual is very nearly empty, and the plan oversells it three
times over.** It is presented as a free check that catches a mis-scaled shape
factor. It cannot: head and back are both built from one centroid and one
semi-axis, so `head - back = 2 * semi_axis` identically, and substituting a
reversed bearing or a rescaled semi-axis leaves the identity exactly satisfied.
Theil-Sen does not generally distribute over a sum, but it does whenever either
addend is exactly linear — and two quadratic series make every pairwise slope a
monotone function of `t_i + t_j`, so both medians fall on the same pair and
curvature does not disturb it either. What remains is a residual that departs
from zero only when the two series disagree about which pair carries the median
slope, which takes irregularity rather than shape. Four tests pin each
narrowing. It is kept because it costs one subtraction, and reported for what
it is.

### Fire characterisation — Tier 3

Front position: where the burning edge is in every direction it covers. Read
three ways on the same frames, because the plan's own gate asks whether the
free-form route beats the parametric one and says to prefer the parametric fit
and report it if not.

**Two measurements reshaped this tier before any of it was written.**

*The radiating front is an open arc, never a closed contour.* Measured at all
ten observations of `configs/f1`, it spans about ninety degrees with a
two-hundred-and-seventy degree gap: the fire burns out behind itself, so the
upwind perimeter stops radiating. That invalidates three of the plan's
prescriptions at once — the two-lobe model has no back lobe to find, a closed
elliptical perimeter would place front across the silent three quarters, and an
enclosed-area intersection over union is undefined because neither contour
encloses anything.

*The response is shift-invariant to between 0.935 and 0.966* across the fire's
own region. A shift-variant operator needs a response evaluated at every cell
at eight seconds each — thirty-two hours — so the deconvolution is
transform-based about a single centred response, and that approximation now
carries its measured fidelity.

**A third measurement arrived after the tier was built and overturned its first
verdict.** Both contour routes were reported as degrading with fire size, which
read as a model failing where a front outgrows the array. It was not: the
response is evaluated at each frame's own centroid and the operator was not
re-centring it, so the reconstruction was displaced by however far the fire had
travelled. Centred, Gate 3 holds five of five. The lesson worth keeping is that
a defect growing smoothly with the independent variable looks exactly like a
physical limit, and the way to tell them apart was to check the estimate in the
one direction where an independent route already worked — the matched filter
read the head to 0.87 m off the same map that the contour put 6.66 m wrong.

#### `front_radial_profile.py` ✅ implemented

**Owns:** `sample_map_bilinear`, `extract_radial_profile`,
`build_two_lobe_model`, `estimate_front_distance_by_matched_filter`,
`build_band_model` and `estimate_front_band_by_matched_filter`.
**Never:** Reads a distance off the profile's maximum. The response is a metre
and a half wide and the front a few metres across, so a smeared peak sits
wherever two lobes overlap.

**A single lobe finds the band's centre, and a head distance means its leading
edge.** The first run under-read by a nearly constant 1.5 m, which happens to
equal the response semi-axis — and adding that would have passed the gate.
Measuring it on synthetic bands disproved the explanation: the shortfall tracks
the *band's own half-width* exactly (3.50 against 3.53, 2.00 against 2.03, 0.50
against 0.52) and is completely independent of the response, identical at 0.8,
1.5 and 2.5 m. The two coincided only because this scene's band is about 1.5 m
wide along the bearing. Fitting both edges instead recovers the leading edge to
0.03 m on synthetic data and took the worst error on the fire from 1.89 m to
0.87 m, inside the gate.

**The band's edges are soft for the same reason the perimeter is rendered
bilinearly.** A hard box is piecewise constant in its own edges at the sampling
resolution, so the finite-difference Jacobian is exactly zero and the optimiser
reports success without leaving its seed.

#### `front_perimeter_fit.py` ✅ implemented

**Owns:** `PerimeterParameters`, `PerimeterFit`, `render_perimeter_density`,
`compute_front_distance_by_angle` and `fit_front_perimeter`. An elliptical arc
with its angular support fitted, rendered and blurred through the same operator
the deconvolution uses so both are scored on identical terms.
**Never:** Deposits a rendered sample into the nearest cell. That made the
whole objective piecewise constant and the first fit returned its seed with
`converged = True`; bilinear deposition and an explicit difference step fixed
it.

#### `map_deconvolution.py` ✅ implemented

**Owns:** `PointSpreadOperator`, `build_point_spread_operator`,
`apply_operator`, `apply_operator_transpose`, `deconvolve_richardson_lucy`,
`compute_total_variation_drag`, `deconvolve_with_total_variation` and
`compute_l_curve`.
**Never:** Materialises the operator. The dense matrix is 14400 squared, about
1.6 GB, and even an on-the-fly matvec is that much arithmetic per iteration.

Total variation is physically motivated rather than generic here: only the
perimeter radiates and the interior is silent, so the true support is
one-dimensional.

**The kernel is re-centred before its transform is taken, and omitting that was
the largest error in this tier.** A convolution kernel carries its own offset.
Callers evaluate the response at the source they are imaging rather than at the
middle of the domain — the accurate thing to do for its *shape* — so a kernel
built straight from it translates by the separation between the two, and the
density it recovers comes back displaced by that separation in the opposite
direction. On `configs/f1` the separation is the distance the fire has
travelled: 0.71 m at five seconds, 8.02 m at fifty. Both contour routes share
this operator and both tracked it, the free-form route's mean radial error
running 0.35 m to 6.66 m and the parametric perimeter's 0.48 m to 7.01 m. That
looked exactly like a model degrading as a front outgrows the array's
resolution, and it was not one. Centred, the final-frame radial error is
0.24 m and the whole of Gate 3 holds.

Centring is done in two steps, because a roll moves whole cells only. The roll
puts the brightest cell on the grid centre and a linear phase ramp on the
spectrum takes out the fraction of a cell left over — a shift is a
multiplication in the transform domain, so the sub-cell part costs one complex
array and no resampling. The residue is then set by how well the peak's own
curvature locates it rather than by the grid: measured on a Gaussian response,
0.012 cells at two cells wide and 0.003 at four, against the half a cell the
roll alone leaves. At the imaging spacing that is 0.25 m down to under a
centimetre, which matters once the error it sits inside is 0.24 m.

#### `front_contour.py` ✅ implemented

**Owns:** `compute_angular_support`, `compute_contour_distance_by_angle`,
`extract_front_contour` and `compute_angular_coverage_fraction`. Extraction is
radial rather than by marching squares, so a direction the front does not cover
reports none instead of being invented.

**A crossing alone does not make a direction covered.** The level is a fraction
of the density's *global* peak, so a direction whose only claim is the skirt of
a bright arc elsewhere still crosses it, close in, and returns a radius
belonging to that arc. `compute_angular_support` measures each direction on its
own instead — the wedge mass `sum(density(r) * r * dr)` along its ray, scaled
so the best direction is one — and a direction below `contour_support_fraction`
carries no front. The radius weight is what makes it a mass rather than a line
sum: a wedge's area grows with radius, so a dim residue beside the origin
cannot outvote an arc twelve metres out.

**The gate is inert on `configs/f1` and is kept deliberately.** The failure it
refuses was real here for as long as the deconvolution kernel was mis-centred,
and the tests reproduce it directly: an arc with a dim halo about the origin
claims the whole circle ungated and its own sector once gated. A centred kernel
leaves no such residue, the level gate alone suffices, and any positive support
now costs overlap — 0.865 at zero, 0.840 at 0.15, 0.312 at 0.40. It ships at
zero, to be raised only on a scene whose coverage exceeds the truth's.

`contour_level_fraction` is 0.15 rather than the plan's half maximum. A
correctly placed density is also correctly concentrated, and at 0.5 the contour
kept only the arc's bright middle: coverage 0.106 against a true 0.283, sector
overlap 0.407. The optimum is flat from 0.10 to 0.20 — mean overlap 0.578,
0.573, 0.563 — and 0.15 is the middle of it.

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

Measured over all sixty files of `data/raw_recordings/kaggle/`: every one
brickwalls between 15.67 and 15.86 kHz, across all three provenances. One common
encoder, and comfortably above the 10 kHz top band, so all seventeen
third-octave centres from 250 Hz survive the codec rule.

### `usable_band_selector.py` ✅ implemented

**Owns:** `select_usable_band_centres_hz`, applying the upper-edge rule against a
measured cutoff, and `filter_bands_by_signal_to_noise_ratio`, applying the
run-time test.
**Never:** Knows about propagation or ranges; it is handed levels.

A band whose upper edge reaches past the brickwall measures the encoder noise
floor over part of its width, which biases its level downward by an amount that
grows with range. Dropping the band is cheaper than modelling that bias.

### `excerpt_coherence.py` ✅ implemented

**Owns:** `compute_transform_length`, `prepare_excerpt_spectrum`,
`compute_coherence_from_spectra`,
`compute_maximum_normalized_cross_correlation`,
`compute_pairwise_excerpt_coherence_matrix` and
`compute_maximum_off_diagonal_coherence`.
**Never:** Chooses excerpts.

The pre-flight guard. Two source excerpts that share waveform content put a peak
into every receiver pair's cross-correlation at a lag no source occupies, and
nothing downstream can tell that artefact from a real source.

**The matrix transforms each row's excerpt once and reuses it.** A sixty-file
pool is 1 770 pairs at three transforms of 2²³ samples each; sharing the row's
forward transform is what keeps that a one-off Stage 0 cost rather than a
per-run one. The numbers are unchanged, and a test pins each matrix entry
against the pairwise function: zero-padding past the linear correlation length
cannot move the peak or change its height.

### `source_excerpt_selector.py` ✅ implemented

**Owns:** `SourceExcerpt`, `read_leading_excerpt`, `read_excerpt_at_offset`,
`enumerate_pool_windows`, `list_pool_recordings`,
`choose_least_coherent_indices` and `select_source_excerpts`, which raises when
the chosen excerpts exceed the configured coherence limit.
**Never:** Applies gain, propagation, or normalization.

Four policies:

| Policy | What it picks | When it is the right one |
|---|---|---|
| `explicit` | The configured filenames, in order | A scene whose excerpts the audit has already chosen — fast and reproducible |
| `distinct_provenance` | A seeded shuffle, never reusing a file | Sweeps that want a fresh draw per run |
| `minimum_coherence` | The greedy subset of whole files with the smallest worst-case coherence | Finding a good set in a pool nobody has audited |
| `distinct_windows` | The greedy subset of *non-overlapping windows* across the pool | More sources than the pool has independent files |

**`distinct_windows` exists because of how these recordings are actually built.**
The Stage 0 audit over all sixty files of `data/raw_recordings/kaggle/` puts the
pool's structure beyond doubt — three provenances of twenty files, with the
coherence blocked by provenance:

| Block | Median | Minimum |
|---|---|---|
| A–A | 0.582 | 0.478 |
| B–B | 0.907 | 0.884 |
| C–C | 0.255 | 0.084 |
| A–B, A–C, B–C | 0.023 to 0.027 | 0.020 |

Across provenances the recordings are independent, at two to three per cent.
Within one they are not: each provenance is one recording cut into twenty
overlapping windows, and B's twenty are near-duplicates of each other. C is the
exception — some of its files are far enough apart in the underlying recording
to fall to 0.084.

That fixes the ceiling at a 50 s clip length precisely. The greedy subset
reaches **four** sources at 0.097, and a fifth jumps it to 0.244: one usable
file from A, one from B, two from C. `m2` is built on exactly that set.

Beyond four, whole files run out and `distinct_windows` is the way past it.
Different windows of one recording carry different content, so shortening the
clip buys sources, at the cost of accumulation windows in the correlation:

| Clip | Candidate windows | K = 6 | K = 10 | K = 15 | K = 20 | K = 25 |
|---|---|---|---|---|---|---|
| 5 s | 140, from 14 files | 0.097 | 0.116 | 0.143 | 0.175 | 0.202 |
| 2.5 s | 280, from 14 files | 0.108 | 0.121 | 0.132 | 0.136 | 0.152 |

At 5 s the pool tops out at fifteen sources inside the 0.15 target; at 2.5 s it
reaches twenty. `m3` needs six and uses 5 s; `m4` needs twenty and uses 2.5 s,
which leaves about nine accumulation windows per correlation instead of the
hundred and ninety-nine a 50 s clip gives.

**A scene names the windows rather than searching for them.** The greedy search
over 280 candidates costs about ten minutes, and a sweep would pay it again on
every cell. `recording_start_offsets_s` lets `explicit` name a `(file, offset)`
set, so `m4` selects in seconds. The list is written in the order the search
chose it, which is what makes any prefix usable: the first ten and first fifteen
of `m4`'s twenty are themselves least-coherent subsets of their size, at 0.121
and 0.132.

`minimum_coherence` is `O(n²)` in the pool size: sixty files at 50 s is 1 770
pairs and about half an hour. That search belongs in the audit, run once, which
is why every scene in `configs/` names its excerpts explicitly.

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
| `fire_characterization.toml` | Imaging map family, array response calibration, bearing, extent, spread model and front settings |
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

**The multi-source scenes that ship in `configs/`:**

| Scene | What it is | Why it exists |
|---|---|---|
| `m1` | Two sources, the second 6 dB down, six receivers | The power-disparity case. Its excerpts cannot meet the 0.15 target, and its configuration says so in a comment |
| `m2` | Three equal sources, one recording provenance each, six receivers | The clean case: coherence 0.028, no power disparity, so `K = 3` is tested on its own |
| `m3` | Six sources on a circle of radius 8 m, eight receivers | A burning contour rather than six fires. Excerpts are 5 s windows, which is what lets six decorrelate, and the deflation is the hard notch because a compact cluster needs it |
| `e2` | Two sources, three receivers, 60 m domain | The ladder's own E2 scene, now readable by the multi-source pipeline. `N = 3` is the ambiguous baseline the plan says to report rather than hide |
| `m4` | Up to twenty sources around a circle of radius 10 m, fifteen to twenty-five receivers | A compact ring, far tighter than `m3`: closest pair 4.1 m at ten sources and 2.7 m at twenty. The scene the source-count and receiver-count sweep runs on |
| `f1` | One spreading front on a 60 m domain, twenty receivers on a 25 m ring, ten 5 s windows | The only scene whose sources are not configured at all: they are whatever cells are alight, so the count changes every window and is never known in advance |
| `f2` | One spreading front on a 100 x 100 grid, twenty receivers on a 22 m ring, ten 3 s windows, driven by the **cellular automaton** | The probabilistic spread model rather than the propagation equation. Its front is nearly closed where `f1`'s is a quarter arc, so it exercises the geometry the plan originally assumed. Gate 3 holds 3 of 5 here, and what fails is the free-form route |
| `f3` | `f2` in every respect but the spread engine, which is the propagation equation | The control. `f2` changes both the spread model and the grid against `f1`, and without this there is no way to say which of them moved a number. It holds 5 of 5, which is what makes every `f2` failure attributable to the spread model |

**What the automaton scene established.** Three defects and one limit, none of
them reachable from `f1`:

*The spread engine was never seeded.* `CellularAutomatonSpreadEngineConfig`
defaults `random_seed` to `None` and the composition root constructed the
engine with no arguments, so every run simulated a different fire — 159 against
166 burning cells at the same instant, and a final-frame sector overlap of 0.68
in one realisation against 0.28 in another, which is wider than most of the
effects the scene is used to measure. `spread_engine_random_seed` now sits on
`FireSimulationConfiguration` and reaches the engine through
`build_spread_engine`. The rate-of-spread and static-source engines draw no
random numbers and are unaffected.

*A head distance could be read from behind the origin.* The band fit seeded
from the profile's global maximum and bounded both edges symmetrically. A fire
that burns out behind itself puts nothing there to find; one that spreads
upwind puts a lobe on each side, and four frames of ten came back with head
distances of −2.09 to −4.63 m. The seed is now taken over non-negative radii
and the outer edge is floored at zero.

*Richardson-Lucy concentrates, and on an extended source that costs coverage.*
Measured on the final frame against a true angular coverage of 0.95, the
contour covers 1.000, 0.639, 0.422, 0.356 and 0.272 of the circle at 4, 5, 10,
50 and 100 iterations. `f1`'s compact arc does not suffer and takes 50; this
closed ring takes 4.

*And the free-form route does not earn its place on a closed front.* Read at
the same level, the **undeconvolved** imaging map gives a sector overlap of
0.708 on the final frame where the deconvolution's best setting gives 0.690 and
its shipped one 0.575. The parametric perimeter beats it outright, 0.91 m
against 2.03 m. The plan's own gate asks whether the free-form route beats the
parametric one and says to prefer the parametric fit and report it when it does
not, which is exactly the verdict here.
| `audit_pool` | The whole pool, `distinct_provenance` | A scratch scene for Stage 0 only; it localizes nothing |

**`f1` raises the wind to 12 m/s and that is the one number in it that is not a
default.** The reason is measured. At the shipped 2 m/s the Balbi rate of spread
is 0.020 m/s, so over the fifty seconds the scene runs the front moves one metre
— two cells, a point source, no shape to estimate and nothing to see change
between windows. At 12 m/s the rate is 0.245 m/s, the front reaches twelve
metres across by `t = 50 s`, and the burning band is about ten cells thick, so
`front_only` and `all_burning` genuinely differ. Fifty seconds of simulated fire
is a short time; a scene that wants a shape out of it has to spread fast enough
to grow one.

**Its `minimum_source_separation_m` is a display budget, not a resolution
claim,** and the warning recorded under `multiple_source_estimator.py` applies
directly. The scene's own closest pair is one cell, 0.5 m. The guard is set to
2 m anyway, to spread a handful of accepted points along a twelve-metre contour
rather than stack them on its brightest few metres. Read against this scene it
measures nothing about resolution.

`e2`'s geometry is taken from the rest of its own directory rather than invented:
the domain and receiver ring come from `mesh.toml` and `receiver.toml`, and the
two sources sit at the ignition fractions `fire.toml` already declares.

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

### `map_linearity_plotter.py` ✅ implemented

**Owns:** `build_series_label` and `plot_map_linearity_audit` — one panel per
source count, the normalised residual against the phase transform exponent, one
line per combinator and pooling rule, with the acceptance target as a rule.

**The residual is plotted rather than the correlation.** Correlation saturates
near one long before a map is additive enough to deconvolve; a figure showing
every family at 0.99 would suggest the choice does not matter, and the whole
audit exists because it does.

### `point_spread_plotter.py` ✅ implemented

**Owns:** `plot_point_spread_calibration` — the centroid bias field as arrows,
the response width field as a heat map, and the analytic-against-rendered
agreement at the validated positions.

Arrows are drawn at their own scale, not the axes', because the offsets are
sub-metre over a sixty-metre domain. What the panel is for is the *pattern*: a
field that grows outward and turns with the geometry is being sampled
correctly, one that changes direction between neighbours is noise being
tabulated.

### `speckle_floor_plotter.py` ✅ implemented

**Owns:** `plot_speckle_floor` — map grain against frame duration on log axes,
with the configured duration marked.

Only one side of the trade is visible in it. A longer frame averages more
correlation windows and lowers the floor; it also smears a moving front over
further ground, which no single-frame measurement can show.

### `fire_shape_plotter.py` ✅ implemented

**Owns:** `draw_receivers`, `draw_fire_panel`, `draw_map_panel` and
`plot_fire_and_steered_response_power` — one frame carrying the true fire state
on the left and the steered response power map on the right, on shared axes,
with the burning contour drawn on both.
**Never:** Runs an estimator, or saves anything.

Composed from the two plotters that already own those panels rather than
redrawing either, so `render_fire_state_rgb_image` and `reshape_map_to_grid`
stay the single definition of what each panel looks like.

**The true front is drawn on the map panel as well as the fire panel.** The
question the frame exists to answer is whether the map's ridge sits on the
contour, and that comparison cannot be made across two sets of axes.

### `fire_characterization_plotter.py` ✅ implemented

**Owns:** `plot_fire_characterization` — the per-frame bearing against its
truth with the fused answer as a rule, beside the extent series with the array
response drawn alongside it and resolved frames marked apart from ceilings.

The response line is on the extent panel because an extent below it is not a
measurement, and a figure should say where that line falls rather than leave
the reader to infer it.

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

### `run_map_linearity_audit.py` ✅ implemented

Tier 0, Milestone 0.2. Renders each source alone, renders them together, and
reports how far the joint map is from the sum of the individual ones, sweeping
the source count, the phase transform exponent, the combinator and the pooling
rule. Writes `results/metrics/tier0_map_linearity.json`. Takes `--configs` and
one flag per swept axis.

**The sweep is coarse and the gate is fine.** Linearity is measured on a 2 m
grid for cost, then the chosen family is re-measured at the imaging spacing. The
two are *not* equal — the finer grid resolves more of the structure the maps
disagree about — so the coarse sweep ranks families and the fine measurement is
what a gate is read against. That was checked rather than assumed, and the
assumption it replaced was wrong.

**Renders are noiseless on purpose.** Receiver noise is common to the joint
render and absent from the individual ones, so it would appear as a linearity
failure having nothing to do with how the map is formed.

### `run_point_spread_calibration.py` ✅ implemented

Tier 0, Milestones 0.3 and 0.4. Tabulates the response covariance and the
centroid offset across the domain and writes both to `results/calibration/`,
plus `results/metrics/tier0_point_spread_calibration.json`.

**The fields are analytic and the shortcut is earned rather than assumed.** A
rendered node costs a twenty-channel propagation plus a one-hundred-and-ninety
pair correlation before any map is formed; the analytic node costs the map
alone, which is the difference between half an hour and most of a day. What
earns it is rendering point sources at three validation positions and reporting
how well the analytic response reproduces them — the check GATE 0 asks for. If
that agreement fails the fields are still written and the report says they are
not trustworthy; it does not quietly substitute something else.

### `run_tier0_report.py` ✅ implemented

Assembles GATE 0. Reads what the audit and the calibration wrote, measures the
speckle floor against frame duration, evaluates every criterion against its
stated number, and writes `results/metrics/tier0_foundations.json`.

**Nothing here adapts a target to what was measured.** The plan is explicit that
a failed gate is a finding to record rather than a threshold to move, so a
criterion that fails is reported as failing and the run still writes its report.
Where a criterion turned out to be readable in a way that passes trivially — the
centroid bias at a symmetry point — the plan's own threshold is applied a second
time somewhere it means something, rather than the threshold being changed.

**The speckle floor does not trade against frame duration, which the plan
assumed it would.** Differencing two disjoint frames separates the deterministic
sidelobe structure from the estimation noise, and the random part is 0.4 % of
the floor at 4 s while the floor itself is flat from 0.5 s to 8 s. The random
part does fall as `1/sqrt(T)` — 4.4-fold across a 16-fold change in duration —
it is simply negligible against what it is competing with. So the frame duration
is set by front smearing alone, and Tier 1 can shorten it for temporal
resolution at almost no cost in map quality.

### `run_fire_shape_localization.py` ✅ implemented

Loads `configs/` → runs the propagation-equation engine from a single ignition →
every `observation_interval_s` extracts the radiating cells, gives each its own
seeded broadband waveform, renders the whole front at once to the receiver
layout and adds sensor noise → hands those waveforms to
`localize_multiple_sources`, which is told nothing about how many cells produced
them → writes one two-panel frame per window, the animation over them, and a
metrics document. Takes `--configs <dir>` and `--max-observations <n>`.

**The forward and inverse halves meet in memory here, not through a run
directory.** Every other path from fire to estimate goes through
`simulation_run_writer` and back; this one renders and localizes the same window
in one process because there are ten windows and each is read once. The
isolation rule is untouched — the estimator is handed waveforms and receiver
positions, exactly as it is from a saved run.

**The source count is not the deliverable and cannot be.** A front carries 38 to
64 cells at `t = 50 s` and no sequential deflation loop recovers that many; the
architecture's own measurements put the ceiling near ten. What the frames report
is the map, whose ridge is the shape estimate, with the handful of accepted
positions marked on it. The two numbers that score a window are therefore
distance from each estimate to the nearest burning cell, and the share of the
contour with an estimate within `FRONT_COVERAGE_RADIUS_M` of it. Neither is an
optimal sub-pattern assignment, because there is no source set to assign
against — one estimate sitting exactly on a twelve-metre front is accurate and
says almost nothing about its shape.

### `run_fire_characterization.py` ✅ implemented

Tiers 1 and 2. Runs the fire, images every window with the imaging family,
reduces each to its moments, and reports bearing and rate of spread against
ground truth. Takes `--configs`, `--max-observations` and
`--background-percentile`.

**The array response is evaluated at each frame's own centroid** rather than
interpolated from the calibration field. Ten frames cost ten responses, which
is cheaper than the field and exact where the field would interpolate. The
response *map* is computed once per frame and its moments taken at whichever
percentiles the two stages need.

### `run_front_characterization.py` ✅ implemented

Tier 3. Reads every frame's imaging map three ways — a matched filter on one
radial profile, a parametric elliptical arc, and a free-form deconvolution —
and scores each against the cells that were actually alight. Writes
`results/metrics/<scene>_tier3_front_position.json`. The scene name is part
of the filename because three scenes now run this script and a shared name
meant each overwrote the last, leaving a committed record that silently
belonged to whichever ran most recently.

**The angular sweep is the whole circle and the front covers a quarter of it**,
so coverage is a measured output on both sides rather than an assumption on
either. Each frame also records the estimate's own coverage beside the truth's,
which is what makes an over-claiming contour visible as something other than a
radial error.

Gate 3 holds 5 of 5 on `configs/f1`: head distance worst 0.87 m over nine
qualifying frames, parametric perimeter 0.67 m on the final frame, the
free-form route 0.24 m against it, sector overlap 0.87, and the onset curve
monotone at 0.39 m for a fire 0.7 response widths across falling to 0.24 m at
7.8. Before the deconvolution kernel was centred the same run gave 7.01 m,
6.66 m, an overlap of 0.07 and a rising onset curve.

### `run_localization_sweep.py` ✅ implemented

Loops one scene over receiver count, signal-to-noise ratio, **source count**,
source separation, `β` and pairwise combinator, writing one metrics record per
cell. `--source-counts` sounds the first `K` of the scene's sources, so one
scene answers several `K` at once; `--source-separations-m` applies only to
two-source cells and, left empty, keeps the scene's own positions.

It draws whatever the swept grid supports: accuracy and stated precision against
receiver count with one line per source count, the two-source resolution curve
against separation, and the `K̂`-against-`K` confusion scatter, where a point off
the diagonal is the failure the sweep exists to find.

**Every cell also saves its own search map**, named for the whole cell rather
than for the axes that grid happened to vary
(`steered_response_power_map_n20_k15_snr20db_sepscene_beta0.7_product.svg`). A
summary curve says a cell scored badly; only the map says whether the sources
were merged, missed outright, or found somewhere else entirely, and a name that
carries its own settings stays readable away from the command line that produced
it.

**Scripts may import one sibling script, and two do.** Reusing
`run_multi_source_localization`'s run driver keeps a single definition of what a
run is, so a sweep cell and a single run cannot drift apart; for the same reason
`run_fire_shape_localization` imports `run_acoustic_rendering`'s emission-model
dispatch rather than restating which cells radiate. `scripts/` has an
`__init__.py` to make those imports resolvable; nothing in `src/` imports any of
them, which is the direction the dependency rule is about.

---

## `tests/`

Mirror the `src/` tree, directory for directory: `tests/audio/`, `tests/config/`,
`tests/spark/{acoustic,atmosphere,fields,fire,inverse,terrain}/`, `tests/utils/{metrics,visualization}/`.
Each test file tests one source file and is named after it. Cross-cutting architectural tests
(such as the isolation guard) sit at the top level, since they belong to no single source file.

`tests/scripts/` is the one directory with no `src/` counterpart. Runners hold no
domain logic, but the sweep does hold the arithmetic that expands a command line
into cells and reduces cells into curves, and both are silent when wrong: a
mis-expanded grid runs an experiment nobody asked for and a mis-reduced series
draws a figure that looks perfectly reasonable.
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
| `audio/test_excerpt_coherence.py` ✅ | Identical excerpts score 1.0, independent noise near zero, the matrix is symmetric, and every matrix entry equals the pairwise function |
| `audio/test_source_excerpt_selector.py` ✅ | Distinct-provenance never reuses a file; the coherence limit raises; minimum-coherence avoids a duplicated pair; an offset excerpt starts where it was asked to; distinct-windows feeds more sources than the pool holds files |
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
| `scripts/test_run_localization_sweep.py` ✅ | The grid is the product of every swept axis; an unswept axis falls back to the scene; a separation sweep moves two sources symmetrically and keeps their amplitudes; a series averages only its own cells; a cell slug distinguishes every axis and stays filename-safe |
| `scripts/test_run_fire_shape_localization.py` ✅ | The window clock lands exactly on its observation time and a step longer than the window does not overshoot it; distance is measured to the nearest burning cell; coverage counts the front and not the estimates, so two estimates on one cell do not read as two cells covered |
| `spark/inverse/test_map_normalization.py` ✅ | Unit mass after normalisation; background subtraction is idempotent and zeroes exactly its own share of the frame; preparation is invariant to the input's overall scale |
| `spark/inverse/test_map_moments.py` ✅ | **A ring of radius `R` has variance `R²/2` per axis and is recovered by the `√2` shape factor exactly** — the relation the extent tier rests on; four point masses give their closed-form covariance; a sampled Gaussian recovers its own standard deviations; the skewness sweep finds the brighter side of an asymmetric ring to within a degree |
| `spark/inverse/test_steered_response_power_sequence.py` ✅ | Frame starts advance by the hop and a trailing partial frame is dropped; each frame is timed at its own centre; every frame of a stationary source peaks on it; a frame folds back into the grid it came from |
| `spark/inverse/test_point_spread_function.py` ✅ | **Every synthesised curve peaks at the delay the steering table predicts** — the sign convention, which if mirrored would silently invert every covariance subtracted downstream; the response peaks on the cell its source sits in; **the response width is set by the background percentile and not by the band**, asserted so that finding cannot quietly change |
| `spark/inverse/test_map_linearity.py` ✅ | Interval pooling is linear in the curve under `sum` and `mean` and is not under `max`; **the per-pair rescaling inside the combinator is what breaks additivity even when the pooling beneath it is linear**; the product is further from additive than the sum |
| `spark/inverse/test_centroid_bias_correction.py` ✅ | A linear field is interpolated exactly between nodes and clamped outside them; a uniform bias is removed exactly and a position-dependent one to its own second order; iterating beats a single subtraction; an alternating field is reported as noise rather than tabulated |
| `config/test_fire_characterization_configuration.py` ✅ | Every section loads; the imaging pooling names a rule the map code actually accepts and is not point sampling; the imaging overrides replace the exponent, combinator and grid while leaving the detection band alone |
| `scripts/test_run_map_linearity_audit.py` ✅ | Sources are placed evenly on a ring of the requested radius; the comparison ignores overall scale and reports an empty map as infinitely far off; the selection reads the sweep at the gate's source count and takes the smallest residual |
| `utils/visualization/test_map_linearity_plotter.py` ✅ | One panel per source count and one line per series plus the target rule; residuals reach the axis in ascending exponent order; mismatched columns raise |
| `utils/visualization/test_point_spread_plotter.py` ✅ | One panel per field plus the validation; one bar per checked position; a node grid that does not match its shape raises |
| `utils/visualization/test_speckle_floor_plotter.py` ✅ | The curve carries every measured point on log axes; the configured duration is marked and named |
| `spark/inverse/test_fire_bearing.py` ✅ | A translating map gives its own direction to a degree, at four bearings; a map that has not moved is refused rather than guessed; **taking the argmax of the skewness returns the reverse bearing**, pinned so the plan's sign cannot come back; a uniform ring reports no lean while an asymmetric one does; a zero weight keeps a method out of the fusion and opposed inputs report a low resultant |
| `spark/inverse/test_fire_extent.py` ✅ | A blurred annulus is recovered to within 5 per cent at three radii; subtracting the response is what makes it accurate; a source below the response returns `upper_bound_only` with a bound that contains the truth and never a clipped zero |
| `spark/inverse/test_fire_rate_of_spread.py` ✅ | A linear series returns the rates it was built from; one bad early frame does not move them, which is why the slopes are Theil-Sen; unresolved frames are left out; **and four tests establish that the closure residual survives a reversed bearing, a mis-scaled semi-axis, a curving centroid and a curving semi-axis** — it moves only under irregularity |
| `utils/metrics/test_fire_front_ground_truth.py` ✅ | Head, back and bearing correct on a hand-built cell mask and on an arc at four bearings; a ring recovers its radius through the shape factor; collinear cells fall back to counting rather than raising |
| `utils/metrics/test_bearing_metrics.py` ✅ | An error across the wrap is the short way round; a summary counts the frames inside each tolerance; agreement is measured circularly too |
| `utils/metrics/test_spread_metrics.py` ✅ | The true rate is fitted robustly; a zero truth reports no relative error rather than an infinite one; an unresolved frame above the response is reported as merely conservative rather than honest |
| `utils/visualization/test_fire_characterization_plotter.py` ✅ | Both panels run on one clock; resolved frames are drawn apart from ceilings; a run with nothing resolved still draws |
| `spark/inverse/test_front_radial_profile.py` ✅ | A two-lobed map gives back both distances and a map with no trailing lobe is reported one-sided; **the band model recovers the leading edge across every band width and response width, and a one-lobe fit falls short by exactly the band's half-width** — the measurement that ruled out correcting it with a response width; **a head distance is never read from behind the origin**, on a profile whose trailing lobe is deliberately the brighter of the two |
| `spark/inverse/test_front_perimeter_fit.py` ✅ | The ray-ellipse solve is exact on circles, rotated ellipses and an offset origin; directions outside the fitted sector carry no distance; a rendered arc carries unit mass and a fit recovers the arc it was given |
| `config/test_simulation_context_factory.py` ✅ | Every ladder scene builds its named components; **the cellular automaton gives the same fire twice from one seed and two fires from two seeds** — without the first, nothing measured on a probabilistic spread model means anything, and without the second a seed that is stored but never passed looks identical to one that works |
| `spark/inverse/test_map_deconvolution.py` ✅ | A point source blurs to the response itself and the operator conserves mass; **the transpose is the adjoint**, which Richardson-Lucy needs to converge at all; **a response evaluated away from the grid centre blurs in place and does not displace what it recovers**, which is the defect that set the whole front position tier's error, and a *fractional* offset is taken out too, to under a hundredth of a cell, with the adjoint and the mass both surviving the ramp; a blurred annulus comes back at its own radius; total variation smooths a noisy recovery and a zero weight recovers the plain iteration exactly |
| `spark/inverse/test_front_contour.py` ✅ | A closed shape is covered everywhere and **an arc reports no front where it has none**; the extractor finds a finite-width ring's outer half-maximum crossing rather than its centre; **an arc with a dim halo about the origin claims every direction ungated and its own sector once gated on wedge mass**, and the gate refuses directions without moving a radius it keeps; a ragged closed ring loses its dim half to a global level and gets it back from a per-ray one, while the global level still floors a direction carrying only a halo |
| `utils/metrics/test_contour_metrics.py` ✅ | Hausdorff catches one stray point; the radial error skips directions only one side covers; **the sector overlap reduces to the plan's enclosed-area ratio when both contours close** and charges for coverage one side lacks |
| `utils/visualization/test_fire_shape_plotter.py` ✅ | The frame carries one panel each; both share the domain and the aspect ratio, without which the two shapes cannot be compared by eye; every estimate gets its own ellipse; the frame survives a window that located nothing |

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
| Windowed excerpt selection, past the one-file-per-provenance ceiling | `source_excerpt_selector.py` | None | ✅ done |
| Source-count and receiver-count sweep with its figures | `run_localization_sweep.py`, `localization_error_plotter.py` | None | ✅ done |
| Fire shape from a spreading front, window by window | `run_fire_shape_localization.py`, `fire_shape_plotter.py` | None | ✅ done |
| Fire characterisation Tier 0: imaging map family, moments, array response, bias field | `inverse/map_normalization.py`, `inverse/map_moments.py`, `inverse/steered_response_power_sequence.py`, `inverse/point_spread_function.py`, `inverse/centroid_bias_correction.py` | None | ✅ done |
| Fire characterisation Tier 1: bearing | `inverse/fire_bearing.py`, `utils/metrics/fire_front_ground_truth.py`, `utils/metrics/bearing_metrics.py` | None | ✅ done |
| Fire characterisation Tier 2: rate of spread | `inverse/fire_extent.py`, `inverse/fire_rate_of_spread.py`, `utils/metrics/spread_metrics.py` | None | ✅ done |
| Fire characterisation Tier 2 extension: Richards elliptical model fit | `inverse/elliptical_spread_model.py` | None | Not needed yet; the moment path already meets the gate |
| Fire characterisation Tier 3: front position | `inverse/front_radial_profile.py`, `inverse/front_perimeter_fit.py`, `inverse/map_deconvolution.py`, `inverse/front_contour.py`, `utils/metrics/contour_metrics.py` | None | ✅ done; Gate 3 holds 5 of 5 |
| Fire characterisation Tier 3 extension: angular support gate | `inverse/front_contour.py` | None | ✅ built and tested; inert on `configs/f1` once the kernel was centred |
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
