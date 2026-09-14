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


@dataclass(slots=True)
class SignalNAND:
    """
    V0.2 signal-driven asynchronous NAND.

    Topology decides WHO can talk to WHOM (the bond graph). Signals decide WHEN
    computation happens. A particle computes only on local causal events:
        - a bond is created
        - a bond is broken
        - a bonded neighbor's bit changed

    The two gate inputs are the two *oldest* active bonds (wiring persistence).

    V0.1.4 (batch_events=True) separates the two timing roles that were
    previously conflated in a single `ready_time`:
        - event_due_time: when a pending signal actually arrives
        - refractory_until: when the particle is allowed to fire again
    A gate fires at max(event_due_time, refractory_until). Events that share the
    same timestamp are evaluated against a frozen bit-state and committed
    together, instead of being shuffled into a sequential order that can destroy
    local oscillators. `allow_not=True` additionally lets a degree-1 particle
    compute NAND(A,A)=NOT(A), restoring NAND's functional completeness.
    """

    bind_radius: float = 0.48
    break_radius: float = 0.65
    min_neighbors: int = 2
    gate_delay: float = 0.175
    refractory: float = 0.07
    flip_probability: float = 1.0
    batch_events: bool = False
    allow_not: bool = False

    def update(
        self,
        positions: Array,
        bits: Array,
        bonds: Array,
        signal: Array,
        event_due_time: Array,
        refractory_until: Array,
        bond_age: Array,
        box_size: float,
        time: float,
        dt: float,
        rng: np.random.Generator,
    ) -> tuple[Array, Array, Array, Array, Array, Array, dict]:
        disp = positions[None, :, :] - positions[:, None, :]
        disp = minimum_image(disp, box_size)
        distances = np.sqrt(np.einsum("ijk,ijk->ij", disp, disp))
        np.fill_diagonal(distances, np.inf)

        # 1. Update bonds with hysteresis, and age each surviving bond.
        new_bond = (~bonds) & (distances < self.bind_radius)
        broken = bonds & (distances > self.break_radius)
        new_bonds = (bonds | new_bond) & ~broken
        surviving = bonds & ~broken
        bond_age = np.where(surviving, bond_age + dt, 0.0)

        # 2. Bond events (created or broken) excite their two endpoints now.
        signal = signal.copy()
        event_due_time = event_due_time.copy()
        refractory_until = refractory_until.copy()
        changed = new_bond | broken
        if changed.any():
            ii, jj = np.nonzero(changed)
            signal[ii] = True
            signal[jj] = True
            event_due_time[ii] = np.minimum(event_due_time[ii], time)
            event_due_time[jj] = np.minimum(event_due_time[jj], time)

        new_bits = bits.copy()
        evals = 0
        flips = 0
        signals = 0

        def eval_gate(i: int, state_bits: Array):
            """Return (nand_value, neighbor_ids) for gate i, or None if no inputs."""
            neighbors = np.flatnonzero(new_bonds[i])
            if neighbors.size == 0:
                return None
            if neighbors.size == 1:
                if not self.allow_not:
                    return None
                a = neighbors[0]
                return 1 - int(bool(state_bits[a]) and bool(state_bits[a])), neighbors
            by_age = np.argsort(-bond_age[i, neighbors])
            a, b = neighbors[by_age[0]], neighbors[by_age[1]]
            return 1 - int(bool(state_bits[a]) and bool(state_bits[b])), neighbors

        def propagate(i: int, neighbors: Array) -> None:
            nonlocal signals
            for j in neighbors:
                signal[j] = True
                event_due_time[j] = np.minimum(event_due_time[j], time + self.gate_delay)
                signals += 1

        if self.batch_events:
            # v0.1.4: batch evaluate against the frozen state, then commit all flips.
            due = signal & (event_due_time <= time) & (refractory_until <= time)
            due_idx = np.flatnonzero(due)
            pending: dict[int, tuple[int, Array]] = {}
            for i in due_idx:
                result = eval_gate(i, new_bits)
                if result is None:
                    signal[i] = False
                    refractory_until[i] = time + self.refractory
                    continue
                nand_value, neighbors = result
                evals += 1
                pending[i] = (nand_value, neighbors)
            for i, (nand_value, neighbors) in pending.items():
                if nand_value != new_bits[i] and rng.random() <= self.flip_probability:
                    new_bits[i] = nand_value
                    flips += 1
                    propagate(i, neighbors)
                refractory_until[i] = time + self.refractory
                signal[i] = False
        else:
            # Legacy v0.1.2/v0.1.3: shuffled sequential update within this step.
            while True:
                due = signal & (event_due_time <= time) & (refractory_until <= time)
                if not due.any():
                    break
                order = np.flatnonzero(due)
                rng.shuffle(order)
                for i in order:
                    if event_due_time[i] > time or refractory_until[i] > time:
                        continue
                    signal[i] = False
                    result = eval_gate(i, new_bits)
                    if result is None:
                        refractory_until[i] = time + self.refractory
                        continue
                    nand_value, neighbors = result
                    evals += 1
                    if nand_value != new_bits[i] and rng.random() <= self.flip_probability:
                        new_bits[i] = nand_value
                        flips += 1
                        propagate(i, neighbors)
                    refractory_until[i] = time + self.refractory

        stats = {
            "evals": evals,
            "flips": flips,
            "signals": signals,
            "bond_created": int(new_bond.sum()),
            "bond_broken": int(broken.sum()),
        }
        return new_bits, new_bonds, signal, event_due_time, refractory_until, bond_age, stats
