# spark-wildfire

Acoustic wildfire kinematics: fire spread simulation and inverse source-channel estimation of front bearing, position and rate of spread from distributed microphone arrays.


# READ THE RESPONSABILITY_FILE AND ADD ANY NOTATION IN NOTATION.MD WORK IN YOUR BRANCH
---

## Overview

SPARK models a wildfire front propagating over a spatial grid, renders the acoustic field at distributed receivers through a source–channel model, and solves the inverse problem to recover kinematic properties of the front (bearing, position, rate of spread) from level ratios across receivers.

The pipeline has three independent stages that can be developed and tested in isolation:

```
Audio preprocessing  →  Fire + acoustic simulation  →  Inverse estimation
```

---

## Architecture

```
spark-wildfire/
├── configs/                                    # every tunable value, nothing in the source
│   ├── environment.toml                        # temperature, humidity, pressure
│   ├── data.toml                               # recording path, segmentation, output
│   ├── geometry.toml                           # domain, receivers, source scenarios
│   ├── forward_model.toml                      # propagation reference, noise, seed
│   └── localization.toml                       # bands, window, delay, triangulation
├── data/{raw_recordings,processed_segments,split_manifests}/
├── scripts/
│   ├── run_audio_preprocessing.py
│   ├── run_fire_simulation.py
│   ├── run_kinematics_estimation.py
│   └── run_single_source_localization.py
├── src/
│   ├── audio/                                  # signal operations, no physics
│   │   ├── recording_segmenter.py
│   │   ├── lowpass_filter.py
│   │   ├── amplitude_normalizer.py
│   │   ├── grouped_split_manifest.py
│   │   ├── octave_band_filter.py               # ISO octave bands, zero-phase band-pass
│   │   ├── band_level_meter.py                 # Hann window, windowed RMS level in dB
│   │   └── signal_alignment.py                 # integer-sample shift, valid overlap
│   ├── config/
│   │   ├── simulation_configuration.py
│   │   └── component_registry.py
│   ├── spark/
│   │   ├── fields/                             # ScalarField / VectorField Protocols
│   │   ├── terrain/                            # Mesh Protocol, square and triangular
│   │   ├── fire/                               # SpreadEngine Protocol, FireState, ROS
│   │   ├── atmosphere/                         # shared, stateless air physics
│   │   │   ├── atmospheric_conditions.py       # T / RH / P, speed of sound
│   │   │   └── atmospheric_absorption.py       # ISO 9613-1 alpha(f)
│   │   ├── acoustic/
│   │   │   ├── channel_protocol.py
│   │   │   ├── burning_cell_source_model.py
│   │   │   ├── exponential_attenuation_channel.py
│   │   │   ├── measured_impulse_response_channel.py
│   │   │   ├── free_field_propagation.py       # applies the channel to a waveform
│   │   │   └── receiver_noise.py
│   │   └── inverse/
│   │       ├── kinematics_estimator_protocol.py
│   │       ├── time_difference_of_arrival.py   # GCC-PHAT + delay variance
│   │       ├── band_level_difference.py        # per-band, per-window G
│   │       ├── inverse_variance_fusion.py      # weighted mean, chi2_nu
│   │       ├── level_ratio_triangulation.py    # G, D -> ranges -> position, covariance
│   │       ├── single_source_estimator.py      # end-to-end driver
│   │       └── multiple_source_estimator.py
│   └── utils/
│       ├── array_types.py
│       ├── io/
│       └── visualization/
├── results/{figures,metrics,simulations}/
├── tests/
└── pyproject.toml
```

Dependency direction: `audio/` and `atmosphere/` are leaves. `acoustic/` (forward) and `inverse/`
(estimator) both sit above them and **never import each other** — see `Responsability_file.md`.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/<org>/spark-wildfire.git
cd spark-wildfire
uv python pin 3.13
uv sync
```

To run a script:

```bash
uv run python scripts/run_acoustic_rendering.py
```

To run the test suite:

```bash
uv run pytest
```

### Single-source localization

Renders a 50 s fire recording through the forward channel to two microphones on the domain edge,
then localizes the source from the two receiver signals alone, once per 5 s clip:

```bash
uv run python scripts/run_single_source_localization.py
```

The script takes no parameters of its own — everything comes from `configs/` (see below). To run a
variant without touching the defaults, copy the directory and point at it:

```bash
uv run python scripts/run_single_source_localization.py --configs configs_noisy
```

Results are printed as a per-scenario table and written to `results/metrics/`. Read `chi2_nu` and
the `SINGULAR` flag before trusting a row: `chi2_nu ≈ 1` means the bands agree, and `SINGULAR`
means the source sits near the perpendicular bisector of the microphone baseline, where the two
observables both go to zero and the range cannot be recovered at all.

### The experiment ladder

Four scenes, simplest first. Each is a complete configuration directory, so `--configs` swaps the
whole scene and no script changes between rungs.

| Rung | Scene | Receivers | What it tests |
|---|---|---|---|
| `configs/e1` | One static, non-spreading source | 2 | Baseline identifiability |
| `configs/e2` | Two static, non-spreading sources | 3 | Superposition without motion |
| `configs/e3` | One spreading fire front | ring of 8 | A distributed, moving source |
| `configs/e4` | Two spreading fronts | ring of 8 | Superposition and motion together |

```bash
uv run python scripts/run_acoustic_rendering.py --configs configs/e3
```

Each run writes `results/simulations/<experiment>/<timestamp>/` — the experiment name comes from
that scene's own `experiment.toml`, so a run says which rung it belongs to without anyone having to
remember. The directory holds the resolved configuration, receiver positions, a
`(n_observations, n_receivers, n_samples)` float32 signal array, per-observation ground truth and a
metadata sidecar. That directory is the whole contract between the forward side and the estimator;
read it back with `src/utils/io/simulation_run_reader.py`.

### Report figures

```bash
uv run python scripts/export_report_figures.py --run e3/20260909T120000Z
```

With no `--run` it takes the newest run. Figures land in three kinds of directory:

| Directory | Holds |
|---|---|
| `results/figures/general/` | F1 rate of spread, F2 all four rungs side by side — figures that belong to no single scene |
| `results/figures/<experiment>/` | F3 front evolution, F4 channel, F5 receiver levels, F9 front position, and the front GIF — everything derived from the chosen run |
| `results/figures/<metrics stem>/` | F6 estimates against truth, F7 error against degeneracy, F8 reduced χ² — everything derived from a localization metrics document |

Every one of those directories carries a `figure_metadata.json` holding the full resolved
configuration that produced it, so an `e1/` figure can be checked against the E1 scene rather than
taken on trust. SVGs and GIFs are gitignored — they regenerate — but the metadata sidecars are
committed, because they are the reproducibility record.

The plotters return figures and never write files; only this script saves.

### Application

```bash
uv run --group app streamlit run app/main.py
```

Then open <http://localhost:8501>. Add `--server.port 8600` to move it, or `--server.headless true`
to stop it opening a browser.

Five panels — scene, physics, channel and receivers, estimate, failure modes — over the same
plotters the export script calls, so a figure in the demo and a figure in the report cannot differ.
Live mode renders a new scene in the background; the demo should replay a saved run.

**Where to change how it looks.** Three separate places, depending on what you want to change:

| What | Where |
|---|---|
| Page title, wide/centred layout, panel order and names | `app/main.py` — `st.set_page_config` and `PANEL_NAMES` |
| Theme: colours, fonts, light or dark | `.streamlit/config.toml`, `[theme]` section |
| What is inside one panel: widgets, headings, captions | `app/<name>_panel.py` |
| How a figure itself looks: colours, axes, labels, sizes | `src/utils/visualization/*.py` — never in `app/` |

The last row is the important one. A panel calls a plotter and shows what comes back, so changing a
figure's appearance in `app/` would make the demo and the report disagree. Change the plotter, and
both follow.

---

## Configuration

Every tunable value lives in `configs/`, never in the source. A run is fully described by the files
below, and `--configs <dir>` swaps the whole set.

```
configs/                 the shipped defaults
├── experiment.toml      name, title and description of the scene
├── environment.toml     …and the nine other files below
├── e1/                  ┐
├── e2/                  │ one complete copy of the whole set per rung,
├── e3/                  │ so a variant is a copied directory, never an edited file
└── e4/                  ┘
```

| File | Holds |
|---|---|
| `experiment.toml` | Scene name, title and description. The name becomes the run and figure directory |
| `environment.toml` | Air temperature, relative humidity, pressure |
| `data.toml` | Recording path, clip duration and overlap, metrics output path |
| `geometry.toml` | Domain size, receiver positions, true source positions |
| `forward_model.toml` | Reference distance, receiver-noise SNR and on/off, random seed |
| `localization.toml` | Octave bands, analysis window, delay estimation, triangulation guards |
| `mesh.toml` | Domain extent, cell spacing, connectivity, mesh implementation |
| `wind.toml` | Wind speed and bearing |
| `fire.toml` | Fuel preset, spread engine, ignition points, run length, emission model |
| `receiver.toml` | Placement strategy and its parameters |
| `acoustic_rendering.toml` | Sample rate, segment duration, observation interval, channel |

What stays in the source as a named module-level constant is only what a run cannot change:
published equation coefficients (the ISO 9613-1 relaxation terms, `20.05` in the speed of sound)
and numerical guards (spectrum and power floors). Notation and values are indexed in
`Notation.md`.

---

## Roadmap

### Audio preprocessing

- [x] Segment recordings into 5 s windows with configurable overlap
- [ ] Low-pass filter, sample-rate verification across all sources
- [ ] Normalization (scheme to be determined)
- [ ] Explicit train/test manifest with group IDs derived from source recording, not filename
- [x] Octave band-pass filterbank (125 Hz – 8 kHz), zero-phase
- [x] Windowed band level meter and integer-sample channel alignment

### Acoustic channel

- [x] Exponential decay propagation model `p(r) = p₀ · exp(−αr)` as a gain matrix
- [x] Waveform-level free-field propagation: delay, `1/r` spreading and ISO 9613-1 absorption
- [x] ISO 9613-1 atmospheric absorption `α(f)` from scenario temperature, humidity and pressure
- [x] Additive white receiver noise at a requested SNR

### Terrain and grid

- [x] 2D square grid: 100 m × 100 m, 50 cm node spacing (201 × 201)
- [x] Time step determination (CFL against the shortest edge, clamped by residence time)
- [ ] 3D extension: height field added to visual output and acoustic path length
- [ ] Optional triangle or unstructured mesh

### Fire propagation

#### Cellular automaton (Alexandridis 2008)
- [x] 2D surface propagation with ignition and burnout conditions
- [ ] Optional firebrand transport

#### PDE rate-of-spread (Balbi 2009 / 2020)
- [x] Closed-form scalar ROS from Balbi 2009 Eq. 11a/11b as default engine
- [ ] Balbi 2020 convective-radiative fixed-point as upgrade path
- [x] Swappable engine interface behind a common `Protocol`, chosen by configuration
- [x] Probabilistic cellular automaton engine, for comparison against the physical one
- [x] Static source engine, so a non-spreading source is a configuration and not a code path

### Inverse problem

- [x] Single source, two receivers: TDOA (GCC-PHAT), per-band level ratio, inverse-variance
      fusion, triangulation and error ellipse
- [x] Fire front as a distributed source: front extraction and connected-component labelling
- [ ] Multiple sources: signal superposition, per-source separation and attribution
- [ ] Dense array: automatic selection of highest-SNR receivers

---

## Conventions

| Concern | Convention |
|---|---|
| Classes | `PascalCase` |
| Functions, variables, modules | `snake_case` |
| Type annotations | All function signatures, all dataclass fields |
| Comments and docstrings | None |
| Engine swap | `Protocol`-based, engines are interchangeable behind a single interface |
| Package management | `uv` |
| Python | 3.13 |

---

## References

- Balbi et al. (2009) — *A physical model for wildland fires*, Combustion and Flame 156(12). doi:10.1016/j.combustflame.2009.07.010
- Balbi et al. (2020) — *A convective-radiative propagation model for wildland fires*, IJWF 29(8). doi:10.1071/WF19103
- Alexandridis et al. (2008) — Cellular automaton fire spread engine
- Sheng & Hu (2003) — Energy-based source localization
- Cobos et al. (2017) — Wireless acoustic sensor network localization survey