from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimConfig
from .rules import BinaryAttractionForce, ContactNAND, ForceRule, LogicRule, NearestTwoNAND, SignalNAND, minimum_image


@dataclass(slots=True)
class StepStats:
    step: int
    time: float
    fraction_ones: float
    mean_speed: float
    logic_flips: int


class World:
    """State container and dynamics engine. Rules are injected, not hard-coded."""

    def __init__(
        self,
        config: SimConfig,
        force_rule: ForceRule | None = None,
        logic_rule: LogicRule | None = None,
        contact_rule: ContactNAND | None = None,
        signal_rule: SignalNAND | None = None,
    ) -> None:
        self.cfg = config
        self.cfg.validate()
        self.rng = np.random.default_rng(config.seed)
        n = config.n_particles

        self.positions = self.rng.uniform(0.0, config.box_size, size=(n, 2))
        self.velocities = self.rng.normal(0.0, config.initial_speed, size=(n, 2))
        self.bits = (self.rng.random(n) < config.initial_one_fraction).astype(np.uint8)

        self.force_rule = force_rule or BinaryAttractionForce(
            interaction_radius=config.interaction_radius,
            core_radius=config.core_radius,
            attraction_strength=config.attraction_strength,
            same_bit_repulsion=config.same_bit_repulsion,
            core_repulsion=config.core_repulsion,
        )
        self.logic_rule = logic_rule or NearestTwoNAND()
        self.contact_rule = contact_rule or ContactNAND(
            bind_radius=config.bind_radius,
            break_radius=config.break_radius,
            min_neighbors=config.logic_min_neighbors,
            flip_probability=config.logic_flip_probability,
        )
        self.signal_rule = signal_rule or SignalNAND(
            bind_radius=config.bind_radius,
            break_radius=config.break_radius,
            min_neighbors=config.logic_min_neighbors,
            gate_delay=config.gate_delay,
            refractory=config.refractory,
            flip_probability=config.logic_flip_probability,
        )
        self.bonds = np.zeros((n, n), dtype=bool)
        self.signal = np.zeros(n, dtype=bool)
        self.ready_time = np.full(n, np.inf)
        self.bond_age = np.zeros((n, n), dtype=float)
        self.step_index = 0
        self.last_logic_flips = 0

    def accelerations(self) -> np.ndarray:
        # disp[i,j] = vector from i to j, using minimum periodic image.
        disp = self.positions[None, :, :] - self.positions[:, None, :]
        disp = minimum_image(disp, self.cfg.box_size)
        dist2 = np.einsum("ijk,ijk->ij", disp, disp)
        distances = np.sqrt(dist2)
        safe = np.where(distances > 1e-12, distances, 1.0)
        unit = disp / safe[:, :, None]

        scalar = self.force_rule.pair_scalar(self.bits, distances)
        return np.einsum("ij,ijk->ik", scalar, unit)

    def step(self, n: int = 1) -> StepStats:
        for _ in range(n):
            a = self.accelerations()
            self.velocities = (self.velocities + a * self.cfg.dt) * self.cfg.damping

            speed = np.linalg.norm(self.velocities, axis=1)
            too_fast = speed > self.cfg.max_speed
            if np.any(too_fast):
                self.velocities[too_fast] *= (self.cfg.max_speed / speed[too_fast])[:, None]

            self.positions = (self.positions + self.velocities * self.cfg.dt) % self.cfg.box_size
            self.step_index += 1
            self.last_logic_flips = 0

            if self.cfg.logic_enabled:
                if self.cfg.logic_mode == "contact":
                    self.bits, self.bonds, self.last_logic_flips = self.contact_rule.update(
                        self.positions,
                        self.bits,
                        self.bonds,
                        self.cfg.box_size,
                        self.rng,
                    )
                elif self.cfg.logic_mode == "signal":
                    t = self.step_index * self.cfg.dt
                    (self.bits, self.bonds, self.signal, self.ready_time,
                     self.bond_age, self.last_logic_flips) = self.signal_rule.update(
                        self.positions,
                        self.bits,
                        self.bonds,
                        self.signal,
                        self.ready_time,
                        self.bond_age,
                        self.cfg.box_size,
                        t,
                        self.cfg.dt,
                        self.rng,
                    )
                elif self.step_index % self.cfg.logic_interval == 0:
                    self.bits, self.last_logic_flips = self.logic_rule.update(
                        self.positions,
                        self.bits,
                        self.cfg.box_size,
                        self.cfg.logic_radius,
                        self.cfg.logic_min_neighbors,
                        self.rng,
                        self.cfg.logic_flip_probability,
                    )

        return self.stats()

    def stats(self) -> StepStats:
        return StepStats(
            step=self.step_index,
            time=self.step_index * self.cfg.dt,
            fraction_ones=float(self.bits.mean()),
            mean_speed=float(np.linalg.norm(self.velocities, axis=1).mean()),
            logic_flips=self.last_logic_flips,
        )
