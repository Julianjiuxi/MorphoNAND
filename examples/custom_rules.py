"""Example showing how to replace the v0 logic rule without touching the engine."""

import numpy as np
from morphonand.config import SimConfig
from morphonand.rules import minimum_image
from morphonand.world import World
from morphonand.render import show


class NearestNeighborCopy:
    def update(self, positions, bits, box_size, logic_radius, min_neighbors, rng, flip_probability):
        disp = positions[None, :, :] - positions[:, None, :]
        disp = minimum_image(disp, box_size)
        dist2 = np.einsum("ijk,ijk->ij", disp, disp)
        np.fill_diagonal(dist2, np.inf)
        nearest = np.argmin(dist2, axis=1)
        valid = dist2[np.arange(len(bits)), nearest] < logic_radius**2
        out = bits.copy()
        out[valid] = bits[nearest[valid]]
        return out, int(np.count_nonzero(out != bits))


if __name__ == "__main__":
    cfg = SimConfig(seed=19)
    show(World(cfg, logic_rule=NearestNeighborCopy()))
