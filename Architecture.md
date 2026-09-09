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
src/utils/visualization ← imports everything, imported by scripts and app
scripts/                ← imports everything, imported by nothing
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

### `recording_segmenter.py` ✅ implemented

**Owns:** Cuts audio files into fixed-length segments (5 s and the overlap come from `configs/`) . A trailing partial segment is dropped rather than zero-padded.

### `octave_band_filter.py` ✅ implemented

**Owns:** ISO octave band edges (`f_c/√2`, `f_c·√2`, clipped below Nyquist) and a zero-phase
Butterworth band-pass. Second-order-section form, not `b, a`: the 125 Hz band is narrow enough
relative to 44.1 kHz that the transfer-function form loses accuracy.

Kept separate from `lowpass_filter.py` on purpose. That file owns the *preprocessing* filter
applied once to a recording; this one owns the *analysis* filterbank the estimator runs per window.
Different consumers, different lifecycles.

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

**Extension → receiver layout configuration:** number of receivers, placement strategy (regular grid, random, ring), spacing. This is a swept parameter in the paper's degradation analysis. `GeometryConfiguration` currently fixes exactly two receivers; widening it is where `ReceiverConfiguration` lands.

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
`BoolArray`. One import instead of `npt.NDArray[np.float64]` spelled out in four hundred places.
**Never:** Holds a function.

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
