# PLAN.md

Implementation plan for the forward, tooling and visualization side of SPARK. Read this together
with `Architecture.md` (what every file owns), `Responsability_file.md` (boundaries between
packages) and `Notation.md` (symbols and units). Where this file and those disagree, this file is
newer; fix the older document in the same commit rather than leaving the contradiction.

Work through the blocks in order. Each block states a hypothesis before its implementation, because
several of them are wrong-until-tested and the test is what settles them. Do not start a block
before the block it depends on is green.

---

## 0. Standing constraints

**Do not touch `src/spark/inverse/`.** That package and its tests belong to the teammate working on
the K-source / N-microphone estimator, including the reduced-χ² calibration problem recorded in
§2.4. Read it, call it, never edit it. The same applies to `tests/spark/inverse/`.

**No tunable value in source.** Everything a run can change lives in `configs/`; `--configs <dir>`
swaps the whole set. Published equation coefficients and numerical guards stay as named
module-level constants. Leaf packages (`audio/`, `atmosphere/`) carry no defaults at all.

**Import root is `src.`** Every import starts `from src.…`. A bare `from spark…` resolves under
pytest by accident and breaks outside it.

**Conventions**, enforced by `ruff` and `mypy --strict`: `PascalCase` classes, `snake_case`
functions, unit suffixes on every physical quantity (`wind_speed_m_per_s`, `cell_spacing_m`),
shape meaning on arrays (`receiver_positions_xy_m`, `gain_matrix_source_by_receiver`), booleans as
assertions (`is_burning`), no abbreviations (`rate_of_spread`, never `ros`), Google-style
docstrings, no comments, full type annotations, vectorized numpy with no Python loop over cells.

**Tests mirror `src/` directory for directory.** One test file per source file, named after it.
Cross-cutting architectural tests sit at the top level of `tests/`.

**Nothing in this plan may run the estimator's numbers.** The forward side produces scenes; the
inverse side consumes them. The run directory is the only contract between the two.

---

## 1. Where the repository actually stands

Verified by reading the tree on 2026-09-09, not by trusting the checkmarks in `Architecture.md`.

**Working end to end.** `fields/`, `terrain/`, `fire/`, `atmosphere/`, `acoustic/` and `inverse/`
are implemented with mirrored tests. `run_fire_simulation.py` and `run_acoustic_rendering.py`
produce a run directory. `run_single_source_localization.py` inverts a real recording rendered to
two microphones across five scenarios.

**Empty files that the documentation claims exist.** `src/config/simulation_context.py`,
`simulation_context_factory.py`, `component_registry.py`; `src/utils/io/simulation_run_writer.py`,
`simulation_run_reader.py`; `src/utils/visualization/mesh_plotter.py`,
`rate_of_spread_plotter.py`, `receiver_signal_plotter.py`; `scripts/run_kinematics_estimation.py`,
`scripts/run_audio_preprocessing.py`; `src/audio/lowpass_filter.py`, `amplitude_normalizer.py`,
`grouped_split_manifest.py`; `src/spark/terrain/triangular_mesh.py`;
`src/spark/inverse/kinematics_estimator_protocol.py`, `multiple_source_estimator.py`.
`src/spark/fields/elevation_field_from_raster.py` is a 358-byte stub.

**Documentation drift to repair as you go.** `Architecture.md` does not list `receiver_placement.py`,
`time_step_calculator.py`, `run_acoustic_rendering.py`, `array_types.py` or `patchy_density_field.py`.
`README.md`'s roadmap leaves the Balbi engine, the cellular automaton, the square grid, the timestep
calculation and the exponential channel unchecked although all are implemented.

---

## 2. Findings that motivate this plan

Each is a claim with its evidence. Any of them can be refuted; refute it in writing rather than
silently working around it.

### 2.1 The existing run is a zero-information geometry

`results/simulation_runs/20260908T215505Z/config.json` sets `ring_center_x_fraction = 0.5`,
`ring_center_y_fraction = 0.5`, `ignition_cell_x_fraction = 0.5`, `ignition_cell_y_fraction = 0.5`.
The fire is ignited at the exact centre of the receiver ring. A front spreading symmetrically from
the ring centre presents an identical level and an identical arrival time at all eight receivers:
every level ratio is 1 and every time difference of arrival is 0. This is the multi-receiver
generalisation of the perpendicular-bisector degeneracy that `level_ratio_triangulation.py` already
flags. **That run carries no information and must not be used for any result.**

### 2.2 The front does not travel far enough to support a rate-of-spread estimate

`metadata.json` records `time_step_s = 18.0` over `simulation_duration_s = 120.0`, six steps. The
timestep is set by the fuel residence time, which pins the maximum rate of spread at or below
`0.5 / 18 ≈ 0.028 m/s`; `Architecture.md` records `R ≈ 0.016 m/s` for pine needle litter. The front
therefore reaches roughly 2 m radius over the whole run, observed from 40 m, while the estimator's
position error at 20 dB signal-to-noise ratio is 5 cm to 1.7 m. A rate of spread fitted to six
positions spanning 2 m fits estimator noise.

**Decision.** Extend `simulation_duration_s` to about 1800 s, keeping pine needle litter and the
validated Balbi Table 2 Case 1 parameters. A faster fuel preset (maquis, grassland) is a later
option, taken only if time allows, and it requires sourcing and defending published fuel values.

### 2.3 The observation cadence is welded to the fire timestep

`run_acoustic_rendering.py` renders one acoustic block per fire step. The fire timestep is bounded
by physics (residence time, CFL); how often the array listens is an experimental choice. Coupling
them means a 1800 s run renders about a hundred blocks at roughly 15 MB each. These are separate
concerns and need separate knobs.

### 2.4 Reported uncertainties are not calibrated

Across `results/metrics/single_source_localization.json` and `…_snr20.json`, reduced χ² runs from
18 to 78 on every non-degenerate scenario, noiseless and noisy alike, while position errors are
millimetres to centimetres. Per-band residuals are one to two orders of magnitude larger than the
fused uncertainty predicts. A term common to every band cancels in `G_hat`, so this is
band-dependent: an unmodelled systematic, not a scale factor. Positions are accurate; the error
ellipses are not yet trustworthy.

**This is the estimator owner's problem, not this plan's.** Record it, plot it (§8, F8), do not fix
it.

### 2.5 A real recording cannot be shared verbatim across cells

`generate_source_signal_for_burning_cell` seeds on `source_signal_seed + burning_cell_index`, so
every cell currently emits an independent white-noise waveform. That incoherence is what makes
summing on the order of a hundred point sources physically meaningful. If every cell emitted the
same recording, the receiver would see a coherent sum of delayed copies of one waveform — a comb
filter whose nulls are set by the front geometry — and the per-band levels the estimator consumes
would be dominated by interference rather than propagation.

The counter-argument favours the recording: white noise is spectrally flat, while α(f) rises with
frequency and the range information lives in the high bands, so the source spectrum sets the
per-band signal-to-noise ratio and therefore the fusion weights. Real fire spectrum is the more
honest test.

**Deferred.** The source-signal question is settled jointly with the estimator owner after the
ladder in §3 runs on synthetic sources. Candidate schemes when it is taken up: an independent
random offset per cell into a long recording (keeps spectrum and crackle transients); per-cell
random phase in the frequency domain (keeps the magnitude spectrum exactly, destroys transient
structure). Do not implement either yet.

### 2.6 The acoustic transfer function is assumed, not measured

The channel is exponential attenuation with ISO 9613-1 coefficients. Rendering and inversion share
that same channel, so every reported accuracy is an upper bound on field performance rather than an
estimate of it. The study is an identifiability analysis, not a field validation. This must be
explicit in Methods and in Limitations; the abstract's "an estimated attenuation response" is the
correct phrasing. Do not let any figure caption imply a measured response.

### 2.7 Storage is already safe

`.gitignore` line 123 is `*.npy`, and `results/figures/` and `results/simulations/` are ignored
while `results/metrics/` is explicitly un-ignored. `results/simulation_runs/` is not ignored, so
`config.json`, `ground_truth.json` and `metadata.json` are committed while the heavy arrays are
not. That is the right split. Do not change it. Store signals as `float32`: at 20 dB
signal-to-noise ratio no estimator can use `float64` precision, and it halves every run.

---

## 3. The experiment ladder

Four scenes, simplest first, each a complete `configs_*` directory so `--configs` swaps the whole
scene. The forward side owns the scenes; the estimator owner consumes their run directories.

| Rung | Scene | Receivers | Purpose |
|---|---|---|---|
| E1 | One static, non-spreading source | 2 | Baseline identifiability. Already covered by `run_single_source_localization.py` |
| E2 | Two static, non-spreading sources | 3 | Superposition without motion. First test of source separation |
| E3 | One spreading fire front | ring | Distributed and moving source. Rate of spread becomes estimable |
| E4 | Two spreading fronts, different sounds | ring | Superposition and motion together. The hardest case the report attempts |

**Hypothesis for the ladder's implementation.** E1 and E2 are cleanest as a
`StaticSourceSpreadEngine` satisfying `SpreadEngineProtocol`, whose `step` returns the state
unchanged and whose `initialize` marks a configured set of cells burning forever. Then "static
source" and "spreading front" are one code path differing only by a registry entry: one forward
script serves all four rungs, and `tests/spark/fire/test_engines_are_interchangeable.py` covers the
static engine for free. The alternative — a separate static-source render path — duplicates the
rendering loop and will drift from it.

*Refutation condition:* if a static source needs a burnout time of infinity and that breaks
`FireState`'s burnout bookkeeping or the front mask (`compute_fire_front_mask` requires an
un-ignited neighbour, which a permanently burning isolated cell has), write the separate path
instead and record why here.

---

## 4. Block 0 — make the scene informative

**Goal.** Stop producing runs that cannot carry a result. Small, surgical, no new modules.

**Hypothesis.** With ignition at fraction (0.3, 0.3) against a ring centred at (0.5, 0.5), the eight
receivers span distinct source-receiver ranges and non-zero time differences of arrival, so the
geometry is non-degenerate; and with `simulation_duration_s = 1800`, the front reaches roughly 30 m,
an order of magnitude above the 1.7 m position error at 20 dB, so a rate-of-spread fit is
meaningful.

**Changes.**

- `configs/fire.toml`: `ignition_cell_x_fraction = 0.3`, `ignition_cell_y_fraction = 0.3`,
  `simulation_duration_s = 1800.0`.
- `src/config/acoustic_rendering_configuration.py`: add `observation_interval_s: float`. Render an
  acoustic block every `observation_interval_s` of simulated time, not every fire step. The fire
  still steps at its physics-bounded timestep.
- `configs/acoustic_rendering.toml`: `observation_interval_s = 60.0`.
- `scripts/run_acoustic_rendering.py`: select observation steps from the interval; write
  `receiver_signals` as `float32`.
- `src/utils/array_types.py`: add `Float32Array` if the annotation needs it.

**Tests.** `tests/config/test_acoustic_rendering_configuration.py` covers the new field's default,
round-trip and rejection of a non-positive interval. Add a case to the forward-configuration test
asserting an observation interval shorter than the fire timestep raises rather than silently
rendering every step.

**Acceptance.** A fresh run's `metadata.json` reports a step count equal to
`simulation_duration_s / observation_interval_s`, its `ground_truth.json` shows the largest
component centroid moving monotonically away from the ignition point, and the distance from that
centroid to each of the eight receivers differs across receivers by more than a metre at every
observation step. Assert that last condition in the run script or in a smoke test; it is the
guard against silently regenerating §2.1.

---

## 5. Block 1 — the run directory as a first-class object

**Goal.** Lift serialisation out of `run_acoustic_rendering.py` so the estimator, the plotters and
the application all load runs through one reader and none of them imports a script. Everything
after this block depends on it.

**Hypothesis.** Every consumer of a run needs exactly: the resolved configuration, receiver
positions, the signal array, per-observation ground truth and the metadata sidecar. If a consumer
needs anything else, the forward script is not writing enough and the fix belongs here rather than
in the consumer.

**Files.**

`src/utils/io/simulation_run_writer.py` owns `write_simulation_run`. Creates
`results/simulation_runs/<run_id>/`, writes `config.json`, `receiver_positions_xy_m.npy`,
`receiver_signals.npy` (`float32`, shape `(n_observations, n_receivers, n_samples)`),
`ground_truth.json` (one record per observation) and `metadata.json`. Never formats anything for
display.

`src/utils/io/simulation_run_reader.py` owns a frozen `SimulationRun` dataclass and
`read_simulation_run(run_directory) -> SimulationRun`, plus `list_simulation_runs(root)` returning
run identifiers newest first, which the application needs for its run picker. Signals load lazily
or memory-mapped — a run is hundreds of megabytes and the application must not block on opening
one. Never transforms scientific content.

**Contract, to be stated in the module docstring and never changed silently.**

```
config.json                     resolved ForwardSimulationConfiguration
receiver_positions_xy_m.npy     float64, (n_receivers, 2)
receiver_signals.npy            float32, (n_observations, n_receivers, n_samples)
ground_truth.json               list of n_observations records:
                                  simulation_time_s, component_count,
                                  component_centroids_xy_m, front_cell_count,
                                  burning_cell_count, source_count
metadata.json                   time_step_s, observation_interval_s,
                                  observation_count, sample_rate_hz, receiver_count
```

**Tests.** `tests/utils/io/test_simulation_run_writer.py` and `…_reader.py`: round-trip a small
synthetic run through both and assert every field returns identical, that the signal dtype is
`float32` on disk, that a missing file raises a clear error naming the file, and that
`list_simulation_runs` orders newest first.

**Acceptance.** `run_acoustic_rendering.py` contains no `json.dumps` and no `np.save`; it calls
`write_simulation_run` and nothing else. `Architecture.md` and `Responsability_file.md` are updated
in the same commit.

---

## 6. Block 2 — the composition root and the ladder

**Goal.** Make `--configs configs_e1 … configs_e4` swap an entire scene, so the four rungs are four
configuration directories rather than four scripts.

**Hypothesis.** Stated in §3: E1 and E2 are a `StaticSourceSpreadEngine` behind the existing
protocol, not a separate render path.

**Files.**

`src/spark/fire/static_source_spread_engine.py` owns `StaticSourceSpreadEngine`. `initialize`
returns a `FireState` with no cell burning; `ignite_cells` marks the given cells burning with a
burnout time of infinity; `step` advances `current_time_s` and returns a new `FireState` with the
same masks. Satisfies `SpreadEngineProtocol`.

`src/config/component_registry.py` owns the name-to-class maps: `MESH_REGISTRY`,
`SPREAD_ENGINE_REGISTRY`, `CHANNEL_REGISTRY`, `SCALAR_FIELD_REGISTRY`, `VECTOR_FIELD_REGISTRY`. No
logic.

`src/config/simulation_context.py` owns the frozen `SimulationContext` holding live protocol
instances: `mesh`, `elevation_field`, `fuel_field`, `wind_field`, `spread_engine`, `channel`.
Imports protocols only, never concrete classes.

`src/config/simulation_context_factory.py` owns `build_simulation_context(config)`. **The only file
in the repository that imports concrete classes.** Reads the configuration, looks classes up in the
registry, instantiates, returns a context.

`configs_e1/`, `configs_e2/`, `configs_e3/`, `configs_e4/`: complete configuration directories.
E1 and E2 name the static engine and list source cells; E3 and E4 name the rate-of-spread engine.
E4 lists two ignition points and, once §2.5 is settled, two source signal descriptions.

`scripts/run_acoustic_rendering.py`: replaces its hand-wiring with `build_simulation_context`.

**Configuration shape.** The ignition point is currently a single (x, y) fraction pair. E2 and E4
need a list. Widen `FireSimulationConfiguration` to `ignition_points_xy_fraction: tuple[tuple[float,
float], ...]` and keep a `from_dict` path that accepts the two scalar keys so existing `configs/`
and every committed `config.json` still load. Do not break the round-trip test.

**Tests.** `tests/config/test_component_registry.py`: every registered name resolves to a class
satisfying its protocol, and no registry is empty. `tests/config/test_simulation_context_factory.py`:
each of the four configuration directories builds a context, and the engine each yields is the class
its configuration names. `tests/spark/fire/test_static_source_spread_engine.py`: sources stay
burning across steps, `current_time_s` advances, `FireState` immutability holds. Extend
`test_engines_are_interchangeable.py` to cover the static engine.

**Acceptance.** `grep -rn "SquareGridMesh(\|RateOfSpreadEngine(\|UniformScalarField(" src/ scripts/`
returns hits only in `simulation_context_factory.py` and the tests. All four configuration
directories render without a code change.

**Out of scope.** `scripts/run_kinematics_estimation.py` stays empty. That file is the estimator
owner's entry point and writing it here would prejudge his interface.

---

## 7. Block 3 — the plotters and a reproducible figure export

**Goal.** Every figure the report needs, generated from a run identifier, as SVG for LaTeX and GIF
for the defense.

**Rule inherited from `Responsability_file.md`.** Every function in `src/utils/visualization/`
takes data and returns a `matplotlib.Figure`. **None of them calls `savefig`.** Saving is the
script's job. Every function that needs both forward and inverse data takes both as explicit
arguments; a plotter never imports `inverse/` and `fire/` and reaches for state itself.

**Files.**

`src/utils/visualization/mesh_plotter.py`: grid extent, cell positions, ignition points, receiver
positions, and the perpendicular bisector of a receiver pair with its near-singular band shaded.
That last element is what turns a geometry diagram into an identifiability diagram.

`src/utils/visualization/rate_of_spread_plotter.py`: rate of spread against wind speed, slope and
moisture. Reproduces Balbi 2009 Fig. 3 / Fig. 4 as a unit test with visual output, against the
Table 2 Case 1 parameters the equation tests already validate.

`src/utils/visualization/receiver_signal_plotter.py`: received level traces per receiver over
observations; the per-band analysis chain for one clip (band levels at each receiver, ΔL_b, α_b·D,
G_b); estimate against ground truth in the plane with error ellipses; error against
signal-to-noise ratio; error against distance from the bisector; reduced χ² per scenario.

`src/utils/visualization/channel_plotter.py` (new): ISO 9613-1 α(f) at the scenario's temperature
and humidity, and channel gain against range per octave band. This is the figure that shows *why*
the high bands carry the range information, and no existing plotter owns it.

`scripts/export_report_figures.py`: takes `--run <run_id>` and `--metrics <path>`, calls the
plotters, saves SVG to `results/figures/` under stable names, and writes animations as GIF. Zero
plotting logic of its own.

**Animation.** Front spreading with receiver levels ticking alongside; estimate landing against
ground truth over observations. `matplotlib.animation.PillowWriter` avoids an ffmpeg dependency;
if quality is insufficient, add `imageio` to the application dependency group rather than the core
one.

**Tests.** `tests/utils/visualization/test_*_plotter.py`: each function returns a `Figure`, has the
expected number of axes, and does not touch the filesystem. Assert the no-`savefig` rule by
monkeypatching `matplotlib.figure.Figure.savefig` to raise inside the plotter tests.

**Acceptance.** `uv run python scripts/export_report_figures.py --run <id>` regenerates every figure
in §8 from scratch with no manual step.

---

## 8. Figure inventory

Mapped to the report's fixed structure. Vector SVG for Overleaf; the two-column layout means a
single-column figure is roughly 88 mm wide, so size and set font sizes accordingly rather than
scaling in LaTeX.

**Methods — scene synthesis**

- **F1** Rate of spread against wind speed, reproducing Balbi 2009 Table 2 Case 1. Justifies the
  fire model against published values.
- **F2** Domain and receiver geometry, one panel per rung E1–E4, with the bisector band shaded.
- **F3** Front evolution: ignition-time heatmap plus front cells at selected observations.
- **F4** ISO 9613-1 α(f) at the scenario's air conditions, and channel gain against range per octave
  band. Shows why range is recoverable at all.

**Methods — learning task and evaluation protocol**

- **F5** The analysis chain on one clip: octave-band levels at each receiver, ΔL_b, α_b·D and the
  resulting G_b. The estimator made visible rather than described.

**Results — recovery of spread attributes**

- **F6** Estimate against ground truth in the plane with error ellipses, per scenario. E1 data
  already exists in `results/metrics/single_source_localization.json`.
- **F7** Error against signal-to-noise ratio, and error against distance from the bisector, with the
  singular band shaded. The identifiability figure; the core of claim 3.
- **F8** Reduced χ² per scenario. The calibration figure. It reports §2.4 honestly whether or not
  the cause is found.
- **F9** Estimated front position over observations against ground truth, with the rate-of-spread
  fit and its residual. E3 and E4 only.

**Discussion — limitations**

- **F10** Reserved for claim 2 (band-energy ratios varying with distance as much as with combustion
  regime). No code and no owner yet; do not build it until someone takes the workstream.

F7 and F8 are the figures a rigor-graded report rewards, and neither depends on the result being
good.

---

## 9. Block 4 — the Streamlit application

**Goal.** One application serving both the defense demo and figure export, able to replay a saved
run or launch a simulation live.

**Hypothesis.** The application holds no plotting logic and no physics. It is a run picker, a set of
parameter widgets, calls into `simulation_context_factory` and the plotters, and export buttons. If
a feature cannot be expressed that way it belongs in `src/`, not in the application.

**Structure.** `app/` at the repository root, not under `src/`; nothing in `src/` may import it.
`app/main.py` is the entry point; one module per panel.

**Panels, in the order the defense should walk through them.**

1. *Scene* — pick a rung or edit parameters; shows F2, the geometry with the bisector band.
2. *Physics* — front spreading over the grid, front cells lighting up, animated; shows F1 and F3.
3. *Channel and receivers* — F4 and the level traces; the source-channel model made concrete.
4. *Estimate* — F6 and F9, estimate against ground truth over observations.
5. *Failure modes* — drag the source toward the bisector and watch the singular flag fire; move the
   signal-to-noise ratio slider and watch the ellipse inflate. F7 and F8.

Every panel has an export control writing the same SVG or GIF that
`scripts/export_report_figures.py` writes, through the same plotter call, so a figure shown in the
demo and a figure in the report cannot differ.

**Live mode.** Guard it. A full render is minutes; run it in a background thread with a progress
readout and a cancel, and default the demo to replay. Live mode is for exploration, not for
standing in front of a supervisor.

**Dependencies.** Add `streamlit` to a new `app` dependency group in `pyproject.toml`, not to
`[project.dependencies]`, so `uv sync` for the science stays lean. Exclude `app/` from `mypy`'s
`files` list or annotate it fully; do not weaken `strict`.

---

## 10. Repair tasks, to be done inside the blocks that touch them

- `Architecture.md`: add `receiver_placement.py`, `time_step_calculator.py`,
  `run_acoustic_rendering.py`, `array_types.py`, `patchy_density_field.py`; add every file this plan
  creates.
- `README.md`: tick the roadmap items that are implemented, add the ladder and the application.
- `Responsability_file.md`: add the static engine, the composition root, the run reader and writer,
  the new plotter, and the `app/` boundary.
- `Notation.md`: add any symbol these blocks introduce.
- `tests/toerase.md` and `data/**/toerase.md`: delete the placeholders once their directories hold
  real content.
- Reconcile `tests/test_inverse_does_not_import_forward_model.py` with the name
  `tests/test_inverse_module_isolation.py` used in `Responsability_file.md`; the cache shows both
  have existed. One file, one name, referenced consistently.

---

## 11. Deferred, with the condition that reopens them

| Item | Reopens when |
|---|---|
| Real recording as the source signal, and per-cell decorrelation (§2.5) | The ladder runs on synthetic sources and the estimator owner is ready to choose jointly |
| Faster fuel preset (maquis, grassland) | Time allows, and published fuel values can be sourced and defended |
| 3D terrain: `elevation_field_from_raster.py`, elevation on the square grid | Multiple sources work end to end |
| Claim 1, the leakage audit: `grouped_split_manifest.py`, `run_audio_preprocessing.py` | Someone on the team takes the workstream. Currently unowned, and it is one of the report's three claims |
| Claim 2, the descriptor–propagation confound (F10) | Same |
| Reduced-χ² calibration (§2.4) | Never, on this side. It belongs to the estimator owner |
| `triangular_mesh.py`, `measured_impulse_response_channel.py` | Not before submission |
