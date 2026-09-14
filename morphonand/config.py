from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SimConfig:
    # World
    n_particles: int = 180
    box_size: float = 28.0
    dt: float = 0.035
    seed: int = 7
    boundary: str = "periodic"

    # Initial state
    initial_one_fraction: float = 0.50
    initial_speed: float = 0.05

    # Mechanics
    interaction_radius: float = 2.5
    core_radius: float = 0.38
    attraction_strength: float = 1.25
    same_bit_repulsion: float = 0.80
    core_repulsion: float = 6.0
    damping: float = 0.992
    max_speed: float = 4.0

    # Logic / event layer
    logic_enabled: bool = True
    logic_mode: str = "periodic"  # "periodic": clock-tick scan; "contact": bond-triggered
    logic_radius: float = 1.25
    logic_interval: int = 12
    logic_min_neighbors: int = 2
    logic_flip_probability: float = 1.0

    # Contact-mode bond physics (hysteresis: bind_radius < break_radius)
    bind_radius: float = 0.48
    break_radius: float = 0.65

    # Signal-mode event timing
    gate_delay: float = 0.175  # tau_gate: propagation delay after a bit change
    refractory: float = 0.07   # tau_refractory: minimum gap between a particle's NANDs

    # Reactive mechanical bond (v0.1.3): a bond also exerts a spring force.
    # bond_stiffness=0 disables the force, keeping v0.1.0..v0.1.2 behaviour.
    bond_stiffness: float = 0.0
    bond_rest_length: float = 0.40
    bond_max_force: float = 2.0

    # Die detection / auto-save
    die_window: int = 500            # steps per detection window
    die_consecutive: int = 3         # consecutive dead windows required to stop
    die_speed_threshold: float = 0.005  # mean speed below this marks mechanical quiescence
    die_dir: str = r"D:\桌面\MorphoNAND\die graph"
    snapshot_interval: int = 500     # record a position snapshot every N steps into the .npy

    # Visualization
    fps: int = 40
    trail_length: int = 0
    marker_size: float = 22.0

    def validate(self) -> None:
        if self.n_particles < 3:
            raise ValueError("n_particles must be >= 3")
        if self.box_size <= 0 or self.dt <= 0:
            raise ValueError("box_size and dt must be positive")
        if not 0 <= self.initial_one_fraction <= 1:
            raise ValueError("initial_one_fraction must be in [0, 1]")
        if not 0 < self.core_radius < self.interaction_radius:
            raise ValueError("Require 0 < core_radius < interaction_radius")
        if not 0 < self.logic_radius <= self.interaction_radius:
            raise ValueError("logic_radius must lie inside interaction_radius")
        if self.logic_interval < 1 or self.logic_min_neighbors < 1:
            raise ValueError("logic_interval and logic_min_neighbors must be >= 1")
        if not 0 <= self.logic_flip_probability <= 1:
            raise ValueError("logic_flip_probability must be in [0, 1]")
        if self.logic_mode not in ("periodic", "contact", "signal"):
            raise ValueError("logic_mode must be 'periodic', 'contact' or 'signal'")
        if not 0 < self.bind_radius < self.break_radius:
            raise ValueError("Require 0 < bind_radius < break_radius")
        if self.gate_delay <= 0:
            raise ValueError("gate_delay must be positive")
        if self.refractory < 0:
            raise ValueError("refractory must be >= 0")
        if self.bond_stiffness < 0 or self.bond_rest_length < 0 or self.bond_max_force < 0:
            raise ValueError("bond_stiffness, bond_rest_length and bond_max_force must be >= 0")
        if self.die_window < 1 or self.die_consecutive < 1:
            raise ValueError("die_window and die_consecutive must be >= 1")
        if self.die_speed_threshold < 0:
            raise ValueError("die_speed_threshold must be >= 0")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be >= 1")
        if not 0 < self.damping <= 1:
            raise ValueError("damping must be in (0, 1]")
        if self.boundary != "periodic":
            raise ValueError("v0 currently supports boundary='periodic' only")

    @classmethod
    def from_json(cls, path: str | Path) -> "SimConfig":
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
