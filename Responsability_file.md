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

### `atmosphere/`

Ambient air physics. **Leaf-level pure functions with no simulation state**, so both `acoustic/`
(forward) and `inverse/` (estimator) may import it without breaching the isolation rule below —
the estimator's atmosphere is the one its caller *assumes*, never one read back from the simulation.
A single definition also prevents the forward and inverse sides drifting to different `α_b`.

| File | Owns | Never |
|---|---|---|
| `atmospheric_conditions.py` | `AtmosphericConditions` dataclass (T, RH, P), speed of sound from temperature | Knows about signals or geometry |
| `atmospheric_absorption.py` | ISO 9613-1 absorption `α(f)` in dB/m, vectorized over frequency | Assumes a band structure — it takes any frequency |

### `acoustic/`

| File | Owns | Never |
|---|---|---|
| `channel_protocol.py` | `Channel` Protocol: `compute_gain_matrix(source_positions_xyz, receiver_positions_xyz) -> Float64Array` shape `(n_src, n_rec)` | — |
| `burning_cell_source_model.py` | `FireState` → `(source_positions_xyz, source_amplitudes)`. The only bridge from fire to acoustics | Knows about receivers or propagation |
| `exponential_attenuation_channel.py` | `exp(−αr)/r` gain matrix | — |
| `measured_impulse_response_channel.py` | Stub for measured `h`; extends the gain to `(n_src, n_rec, n_freq)` | — |
| `free_field_propagation.py` | Applies a channel **to a waveform**: propagation delay, `1/r` spreading and frequency-dependent ISO 9613-1 absorption, all in one rFFT. Renders one signal per receiver on a common clock | Knows the source position is a fire, or that anyone will invert it |
| `receiver_noise.py` | Additive white sensor noise at a requested SNR | Shapes noise to a spectrum — that belongs to a sensor model |

### `inverse/`

| File | Owns | Never |
|---|---|---|
| `kinematics_estimator_protocol.py` | `estimate(receiver_signal_levels, receiver_positions_xyz) -> FrontKinematics` | — |
| `time_difference_of_arrival.py` | GCC-PHAT delay with parabolic sub-sample refinement; delay variance from effective bandwidth and coherence, both measured **after** alignment | Knows what the delay will be used for |
| `band_level_difference.py` | Per-band, per-window geometric term `G_bm = ΔL_bm − α_b·D`; its mean, per-window variance and standard error of the mean | Computes `α_b` — it is handed them |
| `inverse_variance_fusion.py` | Generic inverse-variance weighted mean, its variance, residuals and `χ²_ν` | Knows the quantity being fused is a level |
| `level_ratio_triangulation.py` | `G, D → r1, r2 → (x, y)` in the baseline frame, mirror side, near-singular guard, covariance from the finite-difference Jacobian | Touches signals |
| `single_source_estimator.py` | The end-to-end driver wiring the four modules above, and the error ellipse | Contains any of their maths itself |
| `multiple_source_estimator.py` | Superposition, separation, per-source attribution | — |

**Two deliberate departures from `single_source_localization_plan.md`.** Both were confirmed
empirically; do not "restore" the plan's version without re-running the validation ladder.

1. **§5.2 fuses the standard error of the mean, not the per-window variance.** `G_b` is the mean
   of `M` window estimates, so its uncertainty is `σ²_b / M_eff`, not `σ²_b`. The factor is common
   to every band and therefore cancels in `G_hat`, but the plan's version inflates `var_G` by
   `M_eff ≈ 12` and deflates `χ²_ν` by the same factor. Both are kept on the result object.
2. **§6 shifts channel 2 the other way.** With `τ > 0` meaning receiver 1 hears the event later,
   `y1[n] ≈ y2[n − τ·fs]`, so channel 2 must be **delayed** by `τ·fs` to land on channel 1's clock;
   the plan's driver advances it. The sign is pinned by `test_time_difference_of_arrival.py`.

Separately, `§5.5`'s delay variance is measured **after** alignment. Measured on the raw pair, a
40 m baseline delay sits inside the Welch segment and collapses the coherence, which inflated
`σ_D` from 0.014 m to 6.5 m and made the error ellipse meaningless.

**Hard rule, enforced by test** (`tests/test_inverse_module_isolation.py`): nothing under `inverse/` imports from `fire/`, `terrain/`, `fields/`, or `acoustic/`. Its only inputs are receiver signals and receiver positions. This is the structural guarantee that your own pipeline does not commit the leakage defect the paper criticizes. `inverse/` may import `audio/` (signal operations) and `atmosphere/` (assumed air physics), neither of which carries simulation state.

### `audio/`

| File | Owns |
|---|---|
| `recording_segmenter.py` | 5 s windows, configurable overlap |
| `lowpass_filter.py` | Filter design and application, sample-rate verification |
| `amplitude_normalizer.py` | Per-window normalization |
| `grouped_split_manifest.py` | Train/test manifest with group IDs from source recording provenance |
| `octave_band_filter.py` | ISO octave band centres and edges; zero-phase 4th-order Butterworth band-pass in second-order-section form |
| `band_level_meter.py` | Hann analysis window, windowed RMS level in dB, window start indices |
| `signal_alignment.py` | Integer-sample shift, valid overlap range, aligned channel pair |

`audio/` holds signal operations that carry no physics. Both `acoustic/` and `inverse/` import it.

### `configuration/`

| File | Owns |
|---|---|
| `simulation_configuration.py` | Nested frozen dataclasses, one per component, plus the TOML loaders that fill them from `configs/` |
| `component_registry.py` | Name → class maps for meshes, engines, channels, estimators |

**No tunable value is written in the source.** Everything a run can change lives in `configs/`
(`environment`, `data`, `geometry`, `forward_model`, `localization`), and `--configs <dir>` swaps
the whole set. The dataclasses in `simulation_configuration.py` are the single definition of what a
run can be told; components receive the config object for their own concern
(`BandConfiguration`, `WindowConfiguration`, `DelayEstimationConfiguration`,
`TriangulationConfiguration`) and never reach for a global.

Two categories deliberately stay in the source as named module-level constants, because they are
not properties of a run:

- **Published equation coefficients** — the ISO 9613-1 relaxation terms, `NEPER_TO_DECIBEL`,
  `SPEED_OF_SOUND_COEFFICIENT_M_PER_S_PER_SQRT_K`. Changing one means implementing a different
  equation, which belongs in code and in `Notation.md`, not in a config file.
- **Numerical guards** — `SPECTRUM_FLOOR`, `POWER_FLOOR`, `LEVEL_RATIO_FLOOR`, `MAXIMUM_COHERENCE`.
  These exist to keep a division finite; they are not knobs.

Leaf modules (`audio/`, `atmosphere/`) take plain scalars and **carry no defaults at all**, so a
missing value is a `TypeError` at the call site rather than a silent fallback. Config objects are
unpacked at the boundary, which is why `audio/` and `atmosphere/` never import `config/`.

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