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
├── data/{raw,processed,manifests}/
├── scripts/
│   ├── preprocess.py
│   ├── simulate.py
│   └── estimate.py
├── src/spark/
│   ├── fields/
│   │   ├── base.py              # ScalarField, VectorField Protocols
│   │   ├── uniform.py
│   │   ├── dem.py
│   │   ├── vegetation.py        # tree generation, density
│   │   └── wind.py
│   ├── terrain/
│   │   ├── base.py              # Mesh Protocol
│   │   ├── square.py
│   │   └── triangular.py
│   ├── fire/
│   │   ├── base.py              # SpreadEngine Protocol, FireState
│   │   ├── fuel.py              # FuelModel dataclass
│   │   ├── ros.py               # pure functions: balbi2009, rothermel
│   │   ├── life.py              # game of life
│   │   ├── cellular.py          # Alexandridis CA
│   │   └── physical.py          # ROS-driven ignition delay
│   ├── acoustic/
│   │   ├── base.py              # Channel Protocol
│   │   ├── source.py            # FireState -> (positions, amplitudes)
│   │   ├── exponential.py
│   │   └── measured.py          # stub
│   ├── inverse/
│   │   ├── base.py              # Estimator Protocol
│   │   ├── single.py
│   │   └── multiple.py
│   ├── audio/                   # segment, filter, normalize, manifest
│   └── config/
│       ├── schema.py            # dataclass configs per component
│       └── registry.py          # name -> class lookup
├── results/{figures,metrics,simulations}/
├── tests/
└── pyproject.toml
```

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
uv run python scripts/simulate.py
```

To run the test suite:

```bash
uv run pytest
```

### Development tooling

| Command | Purpose |
|---|---|
| `uv run ruff check .` | Lint: naming, import order, annotation coverage, docstring style, numpy idioms |
| `uv run ruff format .` | Format |
| `uv run mypy` | Strict static type check; enforces the `Protocol` contracts |
| `uv run pytest --cov` | Tests with coverage |

`data/` and `results/` are excluded from every tool.

---

## Roadmap

### Audio preprocessing

- [ ] Segment recordings into 5 s windows with configurable overlap
- [ ] Low-pass filter, sample-rate verification across all sources
- [ ] Normalization (scheme to be determined)
- [ ] Explicit train/test manifest with group IDs derived from source recording, not filename

### Acoustic channel

- [ ] Exponential decay propagation model `p(r) = p₀ · exp(−αr)`

### Terrain and grid

- [ ] 2D square grid: 100 m × 100 m, 50 cm node spacing (201 × 201)
- [ ] Time step determination
- [ ] 3D extension: height field added to visual output and acoustic path length
- [ ] Optional triangle or unstructured mesh

### Fire propagation

#### Cellular automaton (Alexandridis 2008)
- [ ] 2D surface propagation with ignition and burnout conditions
- [ ] Optional firebrand transport

#### PDE rate-of-spread (Balbi 2009 / 2020)
- [ ] Closed-form scalar ROS from Balbi 2009 Eq. 11a/11b as default engine
- [ ] Balbi 2020 convective-radiative fixed-point as upgrade path
- [ ] Swappable engine interface behind a common `Protocol`

### Inverse problem

- [ ] Single source: distance from attenuation function, triangulation across microphones
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