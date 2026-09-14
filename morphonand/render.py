from __future__ import annotations

from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .rules import minimum_image
from .world import World


BIT_COLORS = np.array(["#33A1FD", "#FF5A5F"], dtype=object)

VERSION_BY_MODE = {
    "periodic": "v0.1.0",
    "contact": "v0.1.1",
    "signal": "v0.1.2",
}


def version_label(world: World) -> str:
    return VERSION_BY_MODE.get(world.cfg.logic_mode, "v0.1.0")


def _setup_axes(world: World, version: str | None = None):
    version = version or version_label(world)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, world.cfg.box_size)
    ax.set_ylim(0, world.cfg.box_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"MorphoNAND {version}")
    try:
        fig.canvas.manager.set_window_title(f"MorphoNAND {version}")
    except Exception:
        pass
    return fig, ax


def show(world: World, steps_per_frame: int = 1) -> None:
    version = version_label(world)
    fig, ax = _setup_axes(world, version)
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

    # Die detection state: compare position snapshots every snapshot_interval steps.
    snapshot_interval = world.cfg.snapshot_interval
    die_threshold = world.cfg.die_threshold
    die_dir = Path(world.cfg.die_dir)
    last_snapshot = world.positions.copy()
    next_snapshot_at = world.step_index + snapshot_interval

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
        nonlocal last_snapshot, next_snapshot_at
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

        if not paused["value"] and world.step_index >= next_snapshot_at:
            disp = minimum_image(world.positions - last_snapshot, world.cfg.box_size)
            mean_change = float(np.linalg.norm(disp, axis=1).mean())
            if mean_change < die_threshold:
                die_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(die_dir / f"{version}.png", dpi=160, bbox_inches="tight")
                animation.event_source.stop()
                plt.close(fig)
                return scatter, status
            last_snapshot = world.positions.copy()
            next_snapshot_at = world.step_index + snapshot_interval

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
        f"MorphoNAND {version_label(world)} — step {stats.step}, ones={stats.fraction_ones:.2f}, "
        f"mean|v|={stats.mean_speed:.2f}"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
