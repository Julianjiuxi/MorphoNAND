# MorphoNAND v0

**MorphoNAND** = **Morphology + NAND**.

A deliberately small artificial-life sandbox for testing one question:

> Can spatial morphology become part of an evolvable computational substrate?

This is **v0**, not an open-ended evolution system yet. It only establishes the minimal feedback loop between **1-bit information**, **local NAND updates**, and **particle motion**.

## V0 universe

Each particle has only:

- a 2D position
- a 2D velocity
- one bit of state (`0` or `1`)

There is no genome, species label, organism object, reproduction API, energy variable, network access, or external code execution.

### Physical rule

Inside a finite interaction radius:

- `0` and `1` attract
- `0` and `0` repel
- `1` and `1` repel
- all particles have strong short-range hard-core repulsion

Motion is passive: particles move only because of these local forces plus inertia/damping. There is no self-propulsion in v0.

### Logic rule

Every `logic_interval` physics steps, each particle asks:

> **WHEN** at least two particles are inside my logic radius, what are the bits of my two nearest neighbors?

It then performs exactly one Boolean primitive:

```text
my_bit <- NAND(nearest_bit_1, nearest_bit_2)
```

All logic updates are synchronous.

This closes the loop:

```text
bits -> force signs -> geometry -> neighborhoods -> NAND -> bits -> ...
```

## Why no IF instruction?

NAND is functionally complete, so v0 avoids baking higher-level Boolean operators into the universe. `WHEN` is treated as an event condition supplied by the simulation substrate, while spatial topology supplies most of the control flow.

## Dependencies

Only two runtime dependencies:

```bash
pip install -r requirements.txt
```

- NumPy
- Matplotlib

Python 3.10+ is recommended.

## Run

Interactive 2D visualization:

```bash
python run.py
```

Or use a JSON configuration:

```bash
python run.py --config configs/v0_default.json
```

Useful variants:

```bash
# Mechanics only: freeze the bits and observe 0/1 self-organization
python run.py --no-logic

# More particles
python run.py --particles 400

# Reproducible alternate initial condition
python run.py --seed 42

# Headless snapshot
python run.py --snapshot snapshot.png --warmup 1200
```

Interactive keys:

- `Space`: pause/resume
- `R`: reset

Color convention:

- blue = `0`
- red = `1`

## Parameter interface

All v0 parameters live in `SimConfig` (`morphonand/config.py`) and can be loaded from JSON. Important knobs include:

- `interaction_radius`
- `core_radius`
- `attraction_strength`
- `same_bit_repulsion`
- `core_repulsion`
- `damping`
- `logic_radius`
- `logic_interval`
- `logic_min_neighbors`
- `logic_flip_probability`

## Rule interface

The engine does not hard-code the force law or logic law.

- `ForceRule` decides pairwise mechanical interaction.
- `LogicRule` decides local bit updates.

See `examples/custom_rules.py` for a replacement logic rule.

This is intentional: later versions can add gravity-like energy fields, bit-dependent field response, 3D geometry, bonds, tokens, or alternative event semantics without rewriting the world engine.

## Repository layout

```text
MorphoNAND-v0/
├─ morphonand/
│  ├─ config.py      # all parameters / JSON interface
│  ├─ rules.py       # replaceable force + logic rules
│  ├─ world.py       # state and time integration
│  ├─ render.py      # 2D Matplotlib visualization
│  └─ cli.py
├─ configs/
│  ├─ v0_default.json
│  └─ mechanics_only.json
├─ examples/
│  └─ custom_rules.py
├─ tests/
│  └─ test_rules.py
├─ requirements.txt
├─ pyproject.toml
└─ run.py
```

## Design boundary for v0

Included now:

- 2D periodic space
- conserved particle count
- one bit per particle
- unlike attraction / like repulsion
- hard-core exclusion
- local synchronous NAND events
- replaceable rules and parameters
- interactive visualization

Deliberately postponed:

- energy / free-energy budget
- gravitational or external field
- opposite 0/1 response to a field
- explicit bonds
- mobile information tokens
- particle creation/destruction
- reproduction semantics
- 3D rendering
- spatial acceleration structures

## Performance note

V0 uses a clear vectorized all-pairs interaction (`O(N^2)`) so the model remains easy to audit. It is suitable for a few hundred particles. If the conceptual experiment works, a later version should replace this with a cell list / spatial hash before scaling to tens of thousands of particles.
