## Responsibility of each file

### `fields/`

Position-to-value functions. Every environmental heterogeneity axis lives here.

| File | Owns | Never |
|---|---|---|
| `scalar_field_protocol.py` | `ScalarField` Protocol: `sample(positions_xyz) -> Float64Array` | Contains any implementation |
| `vector_field_protocol.py` | `VectorField` Protocol, returns `(n, 3)` | — |
| `uniform_scalar_field.py` | Constant value everywhere | Knows about meshes |
| `uniform_vector_field.py` | Constant vector everywhere | — |
| `elevation_field_from_raster.py` | Loads DEM, bilinear interpolation at arbitrary positions | Computes slopes or normals — that is the mesh's job |
| `random_tree_placement_field.py` | Poisson/clustered tree positions, converts to a fuel-load field | Simulates fire |
| `constant_wind_field.py` | Uniform wind vector; later the base for interpolated fields | Reads fire state |

### `terrain/`

Geometry and connectivity. **The only place that knows what 3D means.**

| File | Owns | Never |
|---|---|---|
| `mesh_protocol.py` | `Mesh` Protocol: positions, neighbor indices, distances, unit directions | — |
| `square_grid_mesh.py` | Regular grid construction, 4- or 8-neighborhood, elevation applied at construction so `neighbor_unit_directions_xyz` already carries slope | Knows about fire, fuel or acoustics |
| `triangular_mesh.py` | Same contract, unstructured connectivity | — |

The 2D→3D upgrade is one argument to the mesh constructor: pass an `ElevationFieldFromRaster` instead of `UniformScalarField(0.0)`. Nothing downstream changes.

### `fire/`

| File | Owns | Never |
|---|---|---|
| `spread_engine_protocol.py` | `SpreadEngine` Protocol: `initialize(...)`, `step(state, dt) -> FireState` | — |
| `fire_state.py` | `FireState` dataclass: ignition times, burnout times, `is_burning`, current time. Immutable, `step` returns a new one | Contains logic |
| `fuel_properties.py` | `FuelProperties` dataclass plus named presets (pine needles, excelsior) | Computes anything |
| `rate_of_spread_equations.py` | **Pure functions only.** Balbi 2009, Balbi 2020, and their sub-equations. Vectorized over arrays. The only file where paper notation is allowed | Holds state, imports meshes, imports engines |
| `game_of_life_spread_engine.py` | Simplest engine: neighbor-count rules, fixed burnout | — |
| `cellular_automaton_spread_engine.py` | Alexandridis 2008 probabilistic rules, optional firebrand transport | Implements ROS physics — imports it |
| `rate_of_spread_engine.py` | Consumes ROS to compute per-neighbor ignition delays; projects wind onto each neighbor direction | Contains the ROS equations themselves |

### `acoustic/`

| File | Owns | Never |
|---|---|---|
| `channel_protocol.py` | `Channel` Protocol: `compute_gain_matrix(source_positions_xyz, receiver_positions_xyz) -> Float64Array` shape `(n_src, n_rec)` | — |
| `burning_cell_source_model.py` | `FireState` → `(source_positions_xyz, source_amplitudes)`. The only bridge from fire to acoustics | Knows about receivers or propagation |
| `exponential_attenuation_channel.py` | `exp(−αr)/r` gain matrix | — |
| `measured_impulse_response_channel.py` | Stub for measured `h`; extends the gain to `(n_src, n_rec, n_freq)` | — |

### `inverse/`

| File | Owns | Never |
|---|---|---|
| `kinematics_estimator_protocol.py` | `estimate(receiver_signal_levels, receiver_positions_xyz) -> FrontKinematics` | — |
| `single_source_estimator.py` | Level-ratio distance, triangulation across receivers | — |
| `multiple_source_estimator.py` | Superposition, separation, per-source attribution | — |

**Hard rule, enforced by test:** nothing under `inverse/` imports from `fire/`, `terrain/`, `fields/`, or `acoustic/`. Its only inputs are receiver signals and receiver positions. This is the structural guarantee that your own pipeline does not commit the leakage defect the paper criticizes.

### `audio/`

| File | Owns |
|---|---|
| `recording_segmenter.py` | 5 s windows, configurable overlap |
| `lowpass_filter.py` | Filter design and application, sample-rate verification |
| `amplitude_normalizer.py` | Per-window normalization |
| `grouped_split_manifest.py` | Train/test manifest with group IDs from source recording provenance |

### `configuration/`

| File | Owns |
|---|---|
| `simulation_configuration.py` | Nested dataclasses, one per component, fully serializable |
| `component_registry.py` | Name → class maps for meshes, engines, channels, estimators |

---

## The two utility packages you asked about

Both are justified, but they need boundaries or they become the dumping ground.

### `io/`

| File | Owns | Never |
|---|---|---|
| `simulation_run_writer.py` | Creates `results/simulation_runs/<run_id>/`, writes resolved config JSON, saves `FireState` snapshots | Formats anything for display |
| `simulation_run_reader.py` | Loads a run back for replay or analysis | — |
| `audio_file_reader.py` | Reads audio, returns `(samples, sample_rate_hz)` | Filters or normalizes |
| `metrics_writer.py` | Appends metrics as JSON to `results/metrics/` | Computes metrics |

`io/` moves bytes. It never transforms scientific content.

### `visualization/`

| File | Owns | Never |
|---|---|---|
| `fire_state_plotter.py` | Grid state as a 2D image, ignition-time heatmap | Runs the simulation |
| `mesh_plotter.py` | Mesh geometry, elevation surface, receiver positions | — |
| `rate_of_spread_plotter.py` | ROS vs wind, ROS vs moisture — the Balbi Fig. 3 / Fig. 4 reproductions | — |
| `receiver_signal_plotter.py` | Level traces, bearing estimates against ground truth | Imports from `inverse/` and `fire/` in the same figure without an explicit argument |

`visualization/` may import from everywhere — it sits at the top of the dependency graph and nothing imports it back. Every function takes data and returns a `Figure`; **none of them calls `savefig`**, which is `scripts/`' job. That keeps plotting testable and lets the same function serve a notebook and a batch run.

---