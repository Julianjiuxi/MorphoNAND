from __future__ import annotations

import argparse
from dataclasses import replace

from .config import SimConfig
from .render import save_snapshot, show
from .world import World


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MorphoNAND v0: 1-bit embodied computation sandbox")
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--particles", type=int, help="Override particle count")
    p.add_argument("--seed", type=int, help="Override random seed")
    p.add_argument("--no-logic", action="store_true", help="Disable NAND updates; mechanics only")
    p.add_argument("--steps-per-frame", type=int, default=1)
    p.add_argument("--snapshot", help="Save a PNG instead of opening an interactive window")
    p.add_argument("--warmup", type=int, default=800, help="Steps before snapshot")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = SimConfig.from_json(args.config) if args.config else SimConfig()
    if args.particles is not None:
        cfg = replace(cfg, n_particles=args.particles)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.no_logic:
        cfg = replace(cfg, logic_enabled=False)
    cfg.validate()

    world = World(cfg)
    if args.snapshot:
        out = save_snapshot(world, args.snapshot, warmup_steps=args.warmup)
        print(out)
    else:
        show(world, steps_per_frame=max(1, args.steps_per_frame))


if __name__ == "__main__":
    main()
