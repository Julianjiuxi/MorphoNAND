from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .analysis import ClusterTracker, cluster_metrics
from .config import SimConfig
from .world import World


SERIES_FIELDS = [
    "step", "time", "ones", "K", "A", "B",
    "f_giant", "n_clusters", "mean_size", "mean_rg",
    "mean_anisotropy", "mean_lifetime", "mean_com_speed", "size_dist",
]

SUMMARY_FIELDS = [
    "rho", "v0", "seed", "N", "L", "status", "final_step", "n_samples",
    "mean_A", "mean_K", "mean_B", "mean_f_giant", "mean_n_clusters",
    "mean_mean_size", "mean_mean_rg", "mean_mean_anisotropy",
    "mean_mean_lifetime", "mean_mean_com_speed",
]


def run_trial(
    cfg: SimConfig,
    rho: float,
    v0: float,
    seed: int,
    max_steps: int,
    measure_interval: int,
    die_window: int,
    die_consecutive: int,
    die_speed_threshold: float,
    trial_dir: Path,
) -> dict:
    world = World(cfg)
    n = cfg.n_particles
    dt = cfg.dt
    box_size = cfg.box_size
    tracker = ClusterTracker()

    die_flips = 0
    die_bond_events = 0
    die_window_start = 0
    consecutive_dead = 0

    meas_flips = 0
    meas_bond_events = 0

    series: list[dict] = []
    status = "chaos"
    final_step = max_steps

    while world.step_index < max_steps:
        stats = world.step(1)
        die_flips += stats.logic_flips
        die_bond_events += world.last_bond_events
        meas_flips += stats.logic_flips
        meas_bond_events += world.last_bond_events

        if world.step_index - die_window_start >= die_window:
            pending = int(world.signal.sum())
            dead = (
                stats.mean_speed < die_speed_threshold
                and die_flips == 0
                and die_bond_events == 0
                and pending == 0
            )
            consecutive_dead = consecutive_dead + 1 if dead else 0
            die_flips = 0
            die_bond_events = 0
            die_window_start = world.step_index
            if consecutive_dead >= die_consecutive:
                status = "died"
                final_step = world.step_index
                break

        if world.step_index % measure_interval == 0:
            window_dt = measure_interval * dt
            a = meas_flips / (n * window_dt)
            b = (meas_bond_events / 2.0) / (n * window_dt)
            k = float(np.mean(0.5 * np.sum(world.velocities ** 2, axis=1)))

            cm = cluster_metrics(world.bonds, world.positions, box_size)
            tr = tracker.update(cm["clusters"], world.positions, box_size, window_dt)
            big = [i for i, c in enumerate(cm["clusters"]) if len(c) >= 2]
            mean_lifetime = float(np.mean([tr["lifetimes"][i] for i in big])) if big else 0.0
            mean_com_speed = float(np.mean([tr["com_speed"][i] for i in big])) if big else 0.0

            series.append({
                "step": world.step_index,
                "time": stats.time,
                "ones": stats.fraction_ones,
                "K": k,
                "A": a,
                "B": b,
                "f_giant": cm["f_giant"],
                "n_clusters": cm["n_clusters"],
                "mean_size": cm["mean_size"],
                "mean_rg": cm["mean_rg"],
                "mean_anisotropy": cm["mean_anisotropy"],
                "mean_lifetime": mean_lifetime,
                "mean_com_speed": mean_com_speed,
                "size_dist": json.dumps(cm["size_dist"]),
            })
            meas_flips = 0
            meas_bond_events = 0

    label = f"rho{rho:g}_v{v0:g}_seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / f"{label}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SERIES_FIELDS)
        writer.writeheader()
        writer.writerows(series)

    summary: dict = {
        "rho": rho,
        "v0": v0,
        "seed": seed,
        "N": n,
        "L": box_size,
        "status": status,
        "final_step": final_step,
        "n_samples": len(series),
    }
    for key in ["A", "K", "B", "f_giant", "n_clusters", "mean_size",
                "mean_rg", "mean_anisotropy", "mean_lifetime", "mean_com_speed"]:
        summary[f"mean_{key}"] = float(np.mean([s[key] for s in series])) if series else float("nan")
    return summary


def _trial_worker(args):
    (cfg, rho, v0, seed, max_steps, measure_interval,
     die_window, die_consecutive, die_speed_threshold, trial_dir) = args
    return run_trial(
        cfg, rho, v0, seed, max_steps, measure_interval,
        die_window, die_consecutive, die_speed_threshold, trial_dir,
    )


def sweep(
    cfg: SimConfig,
    rhos: list[float],
    v0s: list[float],
    seeds: list[int],
    max_steps: int,
    measure_interval: int,
    die_window: int,
    die_consecutive: int,
    die_speed_threshold: float,
    out_dir: Path,
    jobs: int = 1,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = out_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for rho in rhos:
        box_size = float(np.sqrt(cfg.n_particles / rho))
        for v0 in v0s:
            for seed in seeds:
                trial_cfg = replace(cfg, box_size=box_size, initial_speed=v0, seed=seed)
                tasks.append((
                    trial_cfg, rho, v0, seed, max_steps, measure_interval,
                    die_window, die_consecutive, die_speed_threshold, trial_dir,
                ))

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        def emit(row: dict) -> None:
            writer.writerow(row)
            fh.flush()
            print(
                f"rho={row['rho']:g} v0={row['v0']:g} seed={row['seed']} -> {row['status']} "
                f"(step {row['final_step']}, A={row['mean_A']:.4f}, K={row['mean_K']:.4f})"
            )

        if jobs > 1:
            with multiprocessing.Pool(jobs) as pool:
                for row in pool.imap_unordered(_trial_worker, tasks):
                    emit(row)
        else:
            for task in tasks:
                emit(_trial_worker(task))

    return summary_path


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MorphoNAND phase-sweep experiment runner")
    p.add_argument("--config", default="configs/v0.1.4.json", help="Base JSON config")
    p.add_argument("--rho", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--v0", default="0.01,0.05,0.1,0.2,0.4,0.8,1.2")
    p.add_argument("--seeds", default="20", help="Comma list of seeds, or a single count N meaning seeds 0..N-1")
    p.add_argument("--max-steps", type=int, default=8000, help="Hard step ceiling; survivors are labelled 'chaos'")
    p.add_argument("--measure-interval", type=int, default=100)
    p.add_argument("--die-window", type=int, default=500)
    p.add_argument("--die-consecutive", type=int, default=3)
    p.add_argument("--die-speed-threshold", type=float, default=0.005)
    p.add_argument("--out", default="exp_result")
    p.add_argument("--jobs", type=int, default=0, help="Parallel workers (0 = auto = CPU count)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = SimConfig.from_json(args.config)

    rhos = _floats(args.rho)
    v0s = _floats(args.v0)
    seed_text = args.seeds.strip()
    if seed_text.isdigit():
        seeds = list(range(int(seed_text)))
    else:
        seeds = _ints(seed_text)

    jobs = args.jobs or os.cpu_count() or 1
    path = sweep(
        cfg, rhos, v0s, seeds,
        args.max_steps, args.measure_interval,
        args.die_window, args.die_consecutive, args.die_speed_threshold,
        Path(args.out), jobs,
    )
    print(f"\nWrote summary to {path}")


if __name__ == "__main__":
    main()
