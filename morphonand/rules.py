from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

Array = np.ndarray


def minimum_image(displacement: Array, box_size: float) -> Array:
    """Shortest periodic displacement in a square torus."""
    return displacement - box_size * np.round(displacement / box_size)


class ForceRule(Protocol):
    def pair_scalar(self, bits: Array, distances: Array) -> Array:
        """Return signed force magnitude for every ordered pair i<-j."""
        ...


class LogicRule(Protocol):
    def update(
        self,
        positions: Array,
        bits: Array,
        box_size: float,
        logic_radius: float,
        min_neighbors: int,
        rng: np.random.Generator,
        flip_probability: float,
    ) -> tuple[Array, int]:
        ...


@dataclass(slots=True)
class BinaryAttractionForce:
    """
    v0 physical law.

    - unlike bits (0/1) attract inside interaction_radius
    - like bits (0/0, 1/1) repel inside interaction_radius
    - every pair has strong short-range hard-core repulsion
    - all forces taper linearly to zero at interaction_radius
    """

    interaction_radius: float
    core_radius: float
    attraction_strength: float
    same_bit_repulsion: float
    core_repulsion: float

    def pair_scalar(self, bits: Array, distances: Array) -> Array:
        r = distances
        inside = (r > 0.0) & (r < self.interaction_radius)
        taper = np.clip(1.0 - r / self.interaction_radius, 0.0, 1.0)

        unlike = bits[:, None] != bits[None, :]
        signed = np.where(unlike, self.attraction_strength, -self.same_bit_repulsion)
        scalar = np.where(inside, signed * taper, 0.0)

        # Hard core always repels. Negative means "away from j" under our vector convention.
        core = (r > 0.0) & (r < self.core_radius)
        core_shape = np.clip(1.0 - r / self.core_radius, 0.0, 1.0)
        scalar = scalar - np.where(core, self.core_repulsion * core_shape, 0.0)
        np.fill_diagonal(scalar, 0.0)
        return scalar


@dataclass(slots=True)
class NearestTwoNAND:
    """
    Local event rule: a particle reads the two nearest neighbors inside logic_radius.
    If enough neighbors exist, its next bit is NAND(neighbor_a, neighbor_b).

    Updates are synchronous: all outputs are computed from the same old bit-state.
    The simulation engine decides WHEN to invoke this rule (logic_interval).
    """

    def update(
        self,
        positions: Array,
        bits: Array,
        box_size: float,
        logic_radius: float,
        min_neighbors: int,
        rng: np.random.Generator,
        flip_probability: float,
    ) -> tuple[Array, int]:
        n = len(bits)
        disp = positions[None, :, :] - positions[:, None, :]
        disp = minimum_image(disp, box_size)
        dist2 = np.einsum("ijk,ijk->ij", disp, disp)
        np.fill_diagonal(dist2, np.inf)
        within = dist2 < logic_radius * logic_radius

        new_bits = bits.copy()
        candidates = np.flatnonzero(within.sum(axis=1) >= min_neighbors)
        for i in candidates:
            neighbor_ids = np.flatnonzero(within[i])
            if neighbor_ids.size < 2:
                continue
            order = np.argpartition(dist2[i, neighbor_ids], 1)[:2]
            a, b = neighbor_ids[order]
            nand_value = 1 - int(bool(bits[a]) and bool(bits[b]))
            if nand_value != bits[i] and rng.random() <= flip_probability:
                new_bits[i] = nand_value

        flips = int(np.count_nonzero(new_bits != bits))
        return new_bits, flips


@dataclass(slots=True)
class ContactNAND:
    """
    V0.1 bond-triggered NAND event.

    NAND no longer runs on a global clock. It fires only when local topology
    changes: a particle's bonded degree crosses from below min_neighbors to at
    least min_neighbors. The inputs are the particle's nearest *bonded*
    neighbors, never arbitrary passers-by inside a scan radius.

    Bonds are geometric and hysteretic:
        - form when distance < bind_radius
        - break when distance > break_radius (break_radius > bind_radius)
    This gives the network structural memory without per-particle storage.

    Events within one step are processed in shuffled order, so later events see
    earlier outputs (asynchronous local semantics, not a universe-wide tick).
    """

    bind_radius: float = 0.48
    break_radius: float = 0.65
    min_neighbors: int = 2
    flip_probability: float = 1.0

    def update(
        self,
        positions: Array,
        bits: Array,
        bonds: Array,
        box_size: float,
        rng: np.random.Generator,
    ) -> tuple[Array, Array, int]:
        disp = positions[None, :, :] - positions[:, None, :]
        disp = minimum_image(disp, box_size)
        distances = np.sqrt(np.einsum("ijk,ijk->ij", disp, disp))
        np.fill_diagonal(distances, np.inf)

        degree_before = bonds.sum(axis=1)
        new_bond = (~bonds) & (distances < self.bind_radius)
        broken = bonds & (distances > self.break_radius)
        new_bonds = (bonds | new_bond) & ~broken
        degree_after = new_bonds.sum(axis=1)

        # Fire only on the "second bond formed" topology change.
        trigger = (degree_before < self.min_neighbors) & (degree_after >= self.min_neighbors)

        new_bits = bits.copy()
        flips = 0
        order = np.flatnonzero(trigger)
        rng.shuffle(order)  # asynchronous: later events see earlier outputs
        for i in order:
            neighbors = np.flatnonzero(new_bonds[i])
            if neighbors.size < self.min_neighbors:
                continue
            two = np.argsort(distances[i, neighbors])[: self.min_neighbors]
            a, b = neighbors[two]
            nand_value = 1 - int(bool(new_bits[a]) and bool(new_bits[b]))
            if nand_value != new_bits[i] and rng.random() <= self.flip_probability:
                new_bits[i] = nand_value
                flips += 1

        return new_bits, new_bonds, flips
