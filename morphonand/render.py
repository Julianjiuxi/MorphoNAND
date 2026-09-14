from __future__ import annotations

from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from .world import World


BIT_COLORS = np.array(["#33A1FD", "#FF5A5F"], dtype=object)

VERSION_BY_MODE = {
    "periodic": "v0.1.0",
    "contact": "v0.1.1",
    "signal": "v0.1.2",
}


def version_label(world: World) -> str:
    cfg = world.cfg
    if cfg.logic_mode == "signal" and cfg.batch_events:
        return "v0.1.4"
    if cfg.bond_stiffness > 0.0:
        return "v0.1.3"
    return VERSION_BY_MODE.get(cfg.logic_mode, "v0.1.0")


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


def _bond_segments(world: World) -> tuple[list, list, int]:
    """Split the undirected bond graph into (normal, active, active_gate_count).

    A bond is "active" when it is one of the two oldest bonds feeding a gate,
    or the single bond of a degree-1 NOT gate (only when allow_not is on).
    """
    bonds = world.bonds
    bond_age = world.bond_age
    allow_not = world.cfg.allow_not
    n = bonds.shape[0]
    normal = np.triu(bonds, 1)
    active = np.zeros((n, n), dtype=bool)
    active_gates = 0
    for i in range(n):
        neighbors = np.flatnonzero(bonds[i])
        if neighbors.size == 0:
            continue
        if neighbors.size == 1:
            if allow_not:
                active[i, neighbors[0]] = True
                active_gates += 1
        else:
            by_age = np.argsort(-bond_age[i, neighbors])
            a, b = neighbors[by_age[0]], neighbors[by_age[1]]
            active[i, a] = True
            active[i, b] = True
            active_gates += 1
    active = np.triu(active | active.T, 1)

    pos = world.positions
    normal_segs = [[pos[i], pos[j]] for i, j in zip(*np.nonzero(normal))]
    active_segs = [[pos[i], pos[j]] for i, j in zip(*np.nonzero(active))]
    return normal_segs, active_segs, active_gates


def show(world: World, steps_per_frame: int = 1) -> None:
    version = version_label(world)
    fig, ax = _setup_axes(world, version)
    scatter = ax.scatter(
        world.positions[:, 0],
        world.positions[:, 1],
        s=world.cfg.marker_size,
        c=BIT_COLORS[world.bits],
        edgecolors="none",
        alpha=0.9,
        zorder=3,
    )
    status = ax.text(0.01, 0.98, "", transform=ax.transAxes, fontsize=9, verticalalignment="top")

    # Bond graph layers: physical bonds and active NAND-input bonds.
    normal_coll = LineCollection([], colors="#888888", linewidths=0.8, alpha=0.6, zorder=1)
    active_coll = LineCollection([], colors="#FFD166", linewidths=2.5, alpha=0.95, zorder=2)
    ax.add_collection(normal_coll)
    ax.add_collection(active_coll)

    # Activity overlays: pending signal halos and just-flipped particles.
    signal_scatter = ax.scatter(
        [], [], s=world.cfg.marker_size * 3.0, facecolors="none",
        edgecolors="#FFD166", linewidths=1.5, alpha=0.9, zorder=4,
    )
    flip_scatter = ax.scatter(
        [], [], s=world.cfg.marker_size * 2.5, facecolors="none",
        edgecolors="#FFFFFF", linewidths=2.0, alpha=1.0, zorder=5,
    )

    trails: deque[np.ndarray] | None = None
    trail_artist = None
    if world.cfg.trail_length > 0:
        trails = deque(maxlen=world.cfg.trail_length)
        trail_artist = ax.scatter([], [], s=2, alpha=0.12, zorder=0)

    paused = {"value": False}
    layers = {"bond": True, "logic": True, "signal": True}

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
    bit_snapshots: list[np.ndarray] = []
    bond_snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    snapshot_times: list[float] = []
    evals_history: list[int] = []
    flips_history: list[int] = []
    signals_history: list[int] = []
    bond_counts: list[int] = []
    cum_evals = 0
    cum_flips = 0
    cum_signals = 0

    def reset_world() -> None:
        nonlocal window_flips, window_bond_events, window_start_step, consecutive_dead
        nonlocal snapshots, bit_snapshots, bond_snapshots, snapshot_steps, snapshot_times
        nonlocal evals_history, flips_history, signals_history, bond_counts
        nonlocal cum_evals, cum_flips, cum_signals
        fresh = World(world.cfg, world.force_rule, world.logic_rule, world.contact_rule, world.signal_rule)
        world.positions[:] = fresh.positions
        world.velocities[:] = fresh.velocities
        world.bits[:] = fresh.bits
        world.bonds[:] = fresh.bonds
        world.signal[:] = fresh.signal
        world.ready_time[:] = fresh.ready_time
        world.refractory_until[:] = fresh.refractory_until
        world.bond_age[:] = fresh.bond_age
        world.step_index = 0
        world.last_logic_flips = 0
        world.last_bond_events = 0
        world.last_evals = 0
        world.last_signals = 0
        window_flips = 0
        window_bond_events = 0
        window_start_step = 0
        consecutive_dead = 0
        snapshots = []
        bit_snapshots = []
        bond_snapshots = []
        snapshot_steps = []
        snapshot_times = []
        evals_history = []
        flips_history = []
        signals_history = []
        bond_counts = []
        cum_evals = 0
        cum_flips = 0
        cum_signals = 0

    def on_key(event):
        key = (event.key or "").lower()
        if event.key == " ":
            paused["value"] = not paused["value"]
        elif key == "r":
            reset_world()
        elif key == "b":
            layers["bond"] = not layers["bond"]
        elif key == "l":
            layers["logic"] = not layers["logic"]
        elif key == "s":
            layers["signal"] = not layers["signal"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal window_flips, window_bond_events, window_start_step, consecutive_dead
        nonlocal cum_evals, cum_flips, cum_signals
        prev_bits = world.bits.copy()
        if not paused["value"]:
            stats = world.step(steps_per_frame)
            window_flips += stats.logic_flips
            window_bond_events += world.last_bond_events
            cum_evals += world.last_evals
            cum_flips += stats.logic_flips
            cum_signals += world.last_signals
            if world.step_index % snapshot_interval == 0:
                snapshots.append(world.positions.copy())
                bit_snapshots.append(world.bits.copy())
                bond_snapshots.append(world.bonds.copy())
                snapshot_steps.append(world.step_index)
                snapshot_times.append(stats.time)
                evals_history.append(cum_evals)
                flips_history.append(cum_flips)
                signals_history.append(cum_signals)
                bond_counts.append(int(world.bonds.sum()) // 2)
        else:
            stats = world.stats()

        scatter.set_offsets(world.positions)
        scatter.set_color(BIT_COLORS[world.bits])

        normal_segs, active_segs, active_gates = _bond_segments(world)
        normal_coll.set_segments(normal_segs if layers["bond"] else [])
        active_coll.set_segments(active_segs if layers["logic"] else [])

        if layers["signal"]:
            signal_scatter.set_offsets(world.positions[world.signal])
        else:
            signal_scatter.set_offsets(np.empty((0, 2)))
        flip_scatter.set_offsets(world.positions[world.bits != prev_bits])

        n_bonds = int(world.bonds.sum()) // 2
        status.set_text(
            f"step={stats.step}  ones={stats.fraction_ones:.3f}  "
            f"bonds={n_bonds}  active_gates={active_gates}  evals={world.last_evals}  "
            f"flips={stats.logic_flips}  signals={world.last_signals}  "
            f"mean|v|={stats.mean_speed:.3f}  "
            f"[space=pause, r=reset, b/l/s=layers]"
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
                history = {
                    "snapshots": snapshots,
                    "bit_snapshots": bit_snapshots,
                    "bond_snapshots": bond_snapshots,
                    "snapshot_steps": snapshot_steps,
                    "snapshot_times": snapshot_times,
                    "evals_history": evals_history,
                    "flips_history": flips_history,
                    "signals_history": signals_history,
                    "bond_counts": bond_counts,
                }
                _save_death(world, version, die_dir, fig, history)
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
    history: dict,
) -> Path:
    die_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(die_dir / f"{version}.png", dpi=160, bbox_inches="tight")

    n = world.positions.shape[0]
    record = {
        "final_xy_bit": np.column_stack([world.positions, world.bits]),
        "final_bonds": world.bonds,
        "snapshots": np.asarray(history["snapshots"]) if history["snapshots"] else np.empty((0, n, 2)),
        "bit_snapshots": np.asarray(history["bit_snapshots"], dtype=np.uint8) if history["bit_snapshots"] else np.empty((0, n), dtype=np.uint8),
        "bond_snapshots": np.asarray(history["bond_snapshots"], dtype=bool) if history["bond_snapshots"] else np.empty((0, n, n), dtype=bool),
        "snapshot_steps": np.asarray(history["snapshot_steps"], dtype=int),
        "snapshot_times": np.asarray(history["snapshot_times"], dtype=float),
        "evals_history": np.asarray(history["evals_history"], dtype=int),
        "flips_history": np.asarray(history["flips_history"], dtype=int),
        "signals_history": np.asarray(history["signals_history"], dtype=int),
        "bond_counts": np.asarray(history["bond_counts"], dtype=int),
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
