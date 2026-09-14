from __future__ import annotations

from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .world import World


BIT_COLORS = np.array(["#33A1FD", "#FF5A5F"], dtype=object)


def _setup_axes(world: World):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, world.cfg.box_size)
    ax.set_ylim(0, world.cfg.box_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MorphoNAND v0")
    return fig, ax


def show(world: World, steps_per_frame: int = 1) -> None:
    fig, ax = _setup_axes(world)
    colors = BIT_COLORS[world.bits]
    scatter = ax.scatter(
        world.positions[:, 0],
        world.positions[:, 1],
        s=world.cfg.marker_size,
        c=colors,
        edgecolors="none",
        alpha=0.9,
    )
    status = ax.text(0.01, 0.98, "", transform=ax.transAxes, fontsize=9, verticalalignment="top")

    trails: deque[np.ndarray] | None = None
    trail_artist = None
    if world.cfg.trail_length > 0:
        trails = deque(maxlen=world.cfg.trail_length)
        trail_artist = ax.scatter([], [], s=2, alpha=0.12)

    paused = {"value": False}

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
        elif event.key in ("r", "R"):
            # Keep the same rule/config, but reset the random world with the same seed.
            fresh = World(world.cfg, world.force_rule, world.logic_rule)
            world.positions[:] = fresh.positions
            world.velocities[:] = fresh.velocities
            world.bits[:] = fresh.bits
            world.step_index = 0

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        if not paused["value"]:
            stats = world.step(steps_per_frame)
        else:
            stats = world.stats()
        scatter.set_offsets(world.positions)
        scatter.set_color(BIT_COLORS[world.bits])
        status.set_text(
            f"step={stats.step}   ones={stats.fraction_ones:.3f}   "
            f"mean|v|={stats.mean_speed:.3f}   flips={stats.logic_flips}   "
            f"[space=pause, r=reset]"
        )
        if trails is not None and trail_artist is not None:
            trails.append(world.positions.copy())
            pts = np.concatenate(tuple(trails), axis=0) if trails else np.empty((0, 2))
            trail_artist.set_offsets(pts)
            trail_artist.set_color(np.tile(BIT_COLORS[world.bits], len(trails)))
        return scatter, status

    interval = 1000.0 / world.cfg.fps
    animation = FuncAnimation(fig, update, interval=interval, blit=False, cache_frame_data=False)
    # Keep a live reference for backends that otherwise garbage-collect animations.
    fig._morphonand_animation = animation  # type: ignore[attr-defined]
    plt.show()


def save_snapshot(world: World, path: str | Path, warmup_steps: int = 800) -> Path:
    world.step(warmup_steps)
    fig, ax = _setup_axes(world)
    ax.scatter(
        world.positions[:, 0],
        world.positions[:, 1],
        s=world.cfg.marker_size,
        c=BIT_COLORS[world.bits],
        edgecolors="none",
        alpha=0.9,
    )
    stats = world.stats()
    ax.set_title(
        f"MorphoNAND v0 — step {stats.step}, ones={stats.fraction_ones:.2f}, "
        f"mean|v|={stats.mean_speed:.2f}"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
