# MorphoNAND

**MorphoNAND** = **Morphology + NAND**.

A deliberately small artificial-life sandbox for one question:

> Can spatial morphology become part of an evolvable computational substrate?

Each particle is a tiny computing element: it carries a position, a velocity, and one bit of state. The bit decides how the particle moves (via local forces), and the resulting geometry decides when and against whom the particle runs a local **NAND**. There is no genome, species, organism object, reproduction API, energy variable, or external code execution.

## The feedback loop

```text
bits -> force signs -> geometry -> neighborhoods -> NAND -> bits -> ...
```

- **Physical rule**: unlike bits (`0`/`1`) attract, like bits repel, plus a short-range hard-core repulsion. Motion is passive (inertia + damping).
- **Logic rule**: each particle computes `my_bit <- NAND(nearest_1, nearest_2)` against its local neighbors.

The engine does **not** hard-code either rule; both are injected (see [Rule interface](#rule-interface)).

## Version history

The code has moved past the original v0 to a series of event-semantics upgrades. The `logic_mode` field selects the family; extra flags layer on top:

| Version | `logic_mode` | What changed |
|---|---|---|
| v0.1.0 | `periodic` | Global clock: every `logic_interval` steps, each particle NANDs its two nearest neighbors inside `logic_radius`. Synchronous. |
| v0.1.1 | `contact` | Bond-triggered NAND. A particle fires when its bonded degree crosses up to `logic_min_neighbors`. Bonds are hysteretic (`bind_radius` < `break_radius`). Asynchronous, shuffled within a step. |
| v0.1.2 | `signal` | Signal-driven NAND. Topology (bonds) decides *who* can talk to *whom*; signals decide *when*. Fires on bond create/break or a neighbor's bit change, after `gate_delay`, with `refractory` dead time. |
| v0.1.3 | `signal` + `bond_stiffness > 0` | A formed bond also exerts a spring force (reactive mechanical bond). |
| v0.1.4 | `signal` + `batch_events=True`, `allow_not=True` | Same-timestamp events batch-evaluate against a frozen state then commit together (preserves local oscillators). `allow_not` lets a degree-1 particle compute `NAND(A,A)=NOT(A)`, restoring functional completeness. |

The runtime version is auto-labelled from the config (see `version_label` in `morphonand/render.py`).

## Install

```bash
pip install -r requirements.txt
```

Runtime dependencies are only NumPy and Matplotlib. Python 3.10+ recommended.

## Quick start: visualization

Each version ships as a JSON config. Run the interactive 2D window with:

```bash
python run.py --config <file>
```

| Version | Feature | Command |
|---|---|---|
| v0.1.0 | Periodic NAND | `python run.py --config configs/v0_default.json` |
| v0.1.1 | Contact NAND | `python run.py --config configs/contact.json` |
| v0.1.2 | Signal NAND | `python run.py --config configs/signal.json` |
| v0.1.3 | Signal NAND + reactive bond spring | `python run.py --config configs/v0.1.3.json` |
| v0.1.4 | Signal NAND + batch events + NOT | `python run.py --config configs/v0.1.4.json` |
| — | Mechanics only (bits frozen) | `python run.py --config configs/mechanics_only.json` |

You can also run without a config (uses `SimConfig` defaults = v0.1.0 periodic mode) and override on the command line:

```bash
# Mechanics only
python run.py --no-logic

# More particles
python run.py --particles 400

# Reproducible alternate initial condition
python run.py --seed 42

# Headless snapshot (PNG)
python run.py --snapshot snapshot.png --warmup 1200
```

### Interactive keys

- `Space` — pause / resume
- `R` — reset to the initial condition
- `B` — toggle physical bond layer
- `L` — toggle active NAND-input bond layer
- `S` — toggle pending-signal halo layer

### Color convention

- blue = `0`
- red = `1`
- grey line = physical bond
- yellow line = active NAND-input bond
- white ring = particle that just flipped this frame

## Quick start: experiment (phase sweep)

The sweep scans density `rho` × initial speed `v0` × seed, and records time-series metrics plus a per-trial summary. The runner overrides the base config's `box_size`, `initial_speed`, and `seed` from the sweep axes:

```bash
python -m morphonand.experiment
```

```text
L  = sqrt(N / rho)        # box_size from density
v0 = initial_speed        # initial velocity scale
```

### Default sweep parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--config` | `configs/v0.1.4.json` | Base config (physics + logic) |
| `--rho` | `0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0` | Densities to scan |
| `--v0` | `0.01,0.05,0.1,0.2,0.4,0.8,1.2` | Initial speeds to scan |
| `--seeds` | `20` | Seed count (`20` = seeds `0..19`; or a comma list) |
| `--max-steps` | `8000` | Hard step ceiling; survivors labelled `chaos` |
| `--measure-interval` | `100` | Steps between recorded samples |
| `--die-window` | `500` | Steps per death-detection window |
| `--die-consecutive` | `3` | Consecutive dead windows required to stop |
| `--die-speed-threshold` | `0.005` | Mean speed below this counts as mechanical quiescence |
| `--out` | `exp_result` | Output directory |
| `--jobs` | `0` | Parallel workers (`0` = auto = CPU count) |

A full default sweep is `10 × 7 × 20 = 1400` trials. Outputs:

- `exp_result/trials/<rho...>_<v0...>_<seed>.csv` — per-trial time series
- `exp_result/summary.csv` — one row per trial with averaged metrics

### Recorded metrics

Time-series columns (`SERIES_FIELDS`):

| Column | Meaning |
|---|---|
| `step`, `time` | Simulation step and time |
| `ones` | Fraction of bits equal to `1` |
| `K` | Mean kinetic energy per particle `mean(0.5·|v|²)` |
| `A` | Logic-flip rate per particle per unit time |
| `B` | Bond-event rate per particle per unit time |
| `f_giant` | Fraction of particles in the largest cluster |
| `n_clusters` | Number of connected components |
| `mean_size` | Mean cluster size |
| `mean_rg` | Mean radius of gyration |
| `mean_anisotropy` | Mean shape anisotropy (`0` = round, `1` = worm/chain) |
| `mean_lifetime` | Mean cluster lifetime (Jaccard-tracked across samples) |
| `mean_com_speed` | Mean cluster centre-of-mass speed |
| `size_dist` | Cluster-size histogram (JSON string) |

Summary rows additionally carry `rho`, `v0`, `seed`, `N`, `L`, `status` (`died` = quiescent, `chaos` = survived to `max_steps`), `final_step`, `n_samples`, and the trial means of the metrics above.

A trial is declared `died` when, for `die_consecutive` consecutive `die_window` windows, the mean speed is below `die_speed_threshold` **and** there are zero logic flips, zero bond events, and zero pending signals.

## Parameter reference

All parameters live in `SimConfig` (`morphonand/config.py`) and can be loaded from JSON (unknown keys are rejected). Defaults:

| Group | Parameter | Default |
|---|---|---|
| World | `n_particles` | `180` |
| | `box_size` | `28.0` |
| | `dt` | `0.035` |
| | `seed` | `7` |
| | `boundary` | `"periodic"` |
| Initial | `initial_one_fraction` | `0.50` |
| | `initial_speed` | `0.05` |
| Mechanics | `interaction_radius` | `2.5` |
| | `core_radius` | `0.38` |
| | `attraction_strength` | `1.25` |
| | `same_bit_repulsion` | `0.80` |
| | `core_repulsion` | `6.0` |
| | `damping` | `0.992` |
| | `max_speed` | `4.0` |
| Logic / event | `logic_enabled` | `true` |
| | `logic_mode` | `"periodic"` |
| | `logic_radius` | `1.25` |
| | `logic_interval` | `12` |
| | `logic_min_neighbors` | `2` |
| | `logic_flip_probability` | `1.0` |
| Contact | `bind_radius` | `0.48` |
| | `break_radius` | `0.65` |
| Signal | `gate_delay` | `0.175` |
| | `refractory` | `0.07` |
| v0.1.4 | `batch_events` | `false` |
| | `allow_not` | `false` |
| v0.1.3 bond | `bond_stiffness` | `0.0` |
| | `bond_rest_length` | `0.40` |
| | `bond_max_force` | `2.0` |
| Die / save | `die_window` | `500` |
| | `die_consecutive` | `3` |
| | `die_speed_threshold` | `0.005` |
| | `die_dir` | `D:\桌面\MorphoNAND\die graph` (local path; override in JSON) |
| | `snapshot_interval` | `500` |
| Visualization | `fps` | `40` |
| | `trail_length` | `0` |
| | `marker_size` | `22.0` |

## Rule interface

The engine does not hard-code the force law or logic law.

- `ForceRule.pair_scalar(bits, distances)` decides pairwise mechanical interaction.
- `LogicRule.update(...)` decides local bit updates (periodic mode).
- `ContactNAND.update(...)` and `SignalNAND.update(...)` implement the contact/signal event semantics.

See `examples/custom_rules.py` for a drop-in replacement logic rule (nearest-neighbour copy). Later versions can add energy fields, bonds, tokens, or alternative event semantics without rewriting the world engine.

## Repository layout

```text
MorphoNAND/
├─ morphonand/
│  ├─ config.py      # all parameters / JSON interface / validation
│  ├─ rules.py       # replaceable force + logic rules (periodic/contact/signal)
│  ├─ world.py       # state and time integration
│  ├─ render.py      # 2D Matplotlib visualization + death auto-save
│  ├─ analysis.py    # cluster metrics (components, gyration, anisotropy, lifetimes)
│  ├─ experiment.py  # phase-sweep runner (rho x v0 x seed)
│  └─ cli.py
├─ configs/          # one JSON per version/mode
├─ examples/
│  └─ custom_rules.py
├─ tests/
│  └─ test_rules.py
├─ exp_result/       # sweep output (trials/ + summary.csv)
├─ requirements.txt
├─ pyproject.toml
└─ run.py
```

## Design boundary for v0.x

Included now:

- 2D periodic space, conserved particle count
- one bit per particle
- unlike attraction / like repulsion + hard-core exclusion
- three event semantics (periodic / contact / signal), plus reactive bonds, batch events, and NOT
- replaceable rules and parameters
- interactive visualization with bond / active-gate / signal overlays
- cluster analysis and a headless phase-sweep runner

Deliberately postponed:

- energy / free-energy budget
- gravitational or external field
- opposite 0/1 response to a field
- mobile information tokens
- particle creation/destruction
- reproduction semantics
- 3D rendering
- spatial acceleration structures

## Performance note

The physics uses a clear vectorized all-pairs interaction (`O(N²)`) so the model remains easy to audit; it is suitable for a few hundred particles. If the conceptual experiment works, a later version should replace it with a cell list / spatial hash before scaling to tens of thousands of particles.
