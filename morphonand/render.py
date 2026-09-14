from __future__ import annotations

from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .world import World


BIT_COLORS = np.array(["#33A1FD", "#FF5A5F"], dtype=object)

VERSION_BY_MODE = {
    "periodic": "v0.1.0",
    "contact": "v0.1.1",
    "signal": "v0.1.2",
}


def version_label(world: World) -> str:
    if world.cfg.bond_stiffness > 0.0:
        return "v0.1.3"
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

    die_window = world.cfg.die_window
    die_consecutive = world.cfg.die_consecutive
    die_speed_threshold = world.cfg.die_speed_threshold
    die_dir = Path(world.cfg.die_dir)
    snapshot_interval = world.cfg.snapshot_interval

    window_flips = 0
    window_bond_events = 0
    window_start_step = world.step_index
    consecutive_dead = 0

    snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    snapshot_times: list[float] = []

    def reset_world() -> None:
        nonlocal window_flips, window_bond_events, window_start_step, consecutive_dead
        nonlocal snapshots, snapshot_steps, snapshot_times
        fresh = World(world.cfg, world.force_rule, world.logic_rule, world.contact_rule, world.signal_rule)
        world.positions[:] = fresh.positions
        world.velocities[:] = fresh.velocities
        world.bits[:] = fresh.bits
        world.bonds[:] = fresh.bonds
        world.signal[:] = fresh.signal
        world.ready_time[:] = fresh.ready_time
        world.bond_age[:] = fresh.bond_age
        world.step_index = 0
        world.last_logic_flips = 0
        world.last_bond_events = 0
        window_flips = 0
        window_bond_events = 0
        window_start_step = 0
        consecutive_dead = 0
        snapshots = []
        snapshot_steps = []
        snapshot_times = []

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
        elif event.key in ("r", "R"):
            reset_world()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal window_flips, window_bond_events, window_start_step, consecutive_dead
        if not paused["value"]:
            stats = world.step(steps_per_frame)
            window_flips += stats.logic_flips
            window_bond_events += world.last_bond_events
            if world.step_index % snapshot_interval == 0:
                snapshots.append(world.positions.copy())
                snapshot_steps.append(world.step_index)
                snapshot_times.append(stats.time)
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

        # Once per window, evaluate all death criteria together.
        if not paused["value"] and world.step_index - window_start_step >= die_window:
            pending = int(world.signal.sum())
            dead = (
                stats.mean_speed < die_speed_threshold
                and window_flips == 0
                and window_bond_events == 0
                and pending == 0
            )
            consecutive_dead = consecutive_dead + 1 if dead else 0
            window_flips = 0
            window_bond_events = 0
            window_start_step = world.step_index
            if consecutive_dead >= die_consecutive:
                _save_death(world, version, die_dir, fig, snapshots, snapshot_steps, snapshot_times)
                animation.event_source.stop()
                plt.close(fig)
                return scatter, status

        return scatter, status

    interval = 1000.0 / world.cfg.fps
    animation = FuncAnimation(fig, update, interval=interval, blit=False, cache_frame_data=False)
    # Keep a live reference for backends that otherwise garbage-collect animations.
    fig._morphonand_animation = animation  # type: ignore[attr-defined]
    plt.show()


def _save_death(
    world: World,
    version: str,
    die_dir: Path,
    fig,
    snapshots: list[np.ndarray],
    snapshot_steps: list[int],
    snapshot_times: list[float],
) -> Path:
    die_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(die_dir / f"{version}.png", dpi=160, bbox_inches="tight")

    n = world.positions.shape[0]
    record = {
        "final_xy_bit": np.column_stack([world.positions, world.bits]),
        "snapshots": np.asarray(snapshots) if snapshots else np.empty((0, n, 2)),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=int),
        "snapshot_times": np.asarray(snapshot_times, dtype=float),
    }
    path = die_dir / f"{version}.npy"
    np.save(path, record, allow_pickle=True)
    return path


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
