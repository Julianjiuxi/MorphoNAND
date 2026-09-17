from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


ROWS = load("exp_result/summary.csv")
RHOS = sorted({float(r["rho"]) for r in ROWS})
V0S = sorted({float(r["v0"]) for r in ROWS})

rho_arr = np.array(RHOS)
v0_arr = np.array(V0S)

# 每个 (rho, v0) 格子 -> 该格子下所有 seed 的 trial
cells: dict[tuple[float, float], list[dict]] = defaultdict(list)
for r in ROWS:
    cells[(float(r["rho"]), float(r["v0"]))].append(r)


def agg(fn) -> np.ndarray:
    Z = np.full((len(v0_arr), len(rho_arr)), np.nan)
    for j, v0 in enumerate(v0_arr):
        for i, rho in enumerate(rho_arr):
            rows = cells.get((rho, v0), [])
            if rows:
                Z[j, i] = fn(rows)
    return Z


def fmean(key: str) -> np.ndarray:
    return agg(lambda rs: np.mean([float(r[key]) for r in rs]))


P_chaos = agg(lambda rs: np.mean([r["status"] == "chaos" for r in rs]))
mean_A = fmean("mean_A")
mean_K = fmean("mean_K")
mean_B = fmean("mean_B")
mean_f_giant = fmean("mean_f_giant")
mean_final = fmean("final_step")


def lin_edges(vals: np.ndarray) -> np.ndarray:
    mids = (vals[:-1] + vals[1:]) / 2.0
    return np.concatenate([[vals[0] - (vals[1] - vals[0]) / 2], mids,
                           [vals[-1] + (vals[-1] - vals[-2]) / 2]])


def log_edges(vals: np.ndarray) -> np.ndarray:
    lv = np.log10(vals)
    mids = (lv[:-1] + lv[1:]) / 2.0
    return np.concatenate([[lv[0] - (lv[1] - lv[0]) / 2], mids,
                           [lv[-1] + (lv[-1] - lv[-2]) / 2]])


rho_edges = lin_edges(rho_arr)
v0_edges = log_edges(v0_arr)

panels = [
    ("P(chaos)  survival prob", P_chaos, "RdYlGn", (0.0, 1.0)),
    ("mean_A  logic flip rate", mean_A, "viridis", None),
    ("mean_K  kinetic energy", mean_K, "viridis", None),
    ("mean_B  bond-event rate", mean_B, "viridis", None),
    ("mean_f_giant  largest cluster", mean_f_giant, "viridis", None),
    ("mean final_step", mean_final, "viridis", None),
]

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, (title, Z, cmap, clim) in zip(axes.flat, panels):
    pc = ax.pcolormesh(rho_edges, v0_edges, Z, cmap=cmap, shading="flat")
    if clim is not None:
        pc.set_clim(*clim)
    ax.set_title(title)
    ax.set_xlabel("rho (density)")
    ax.set_ylabel("v0 (initial speed)")
    ax.set_xticks(rho_arr)
    ax.set_xticklabels([f"{v:g}" for v in rho_arr])
    ax.set_yticks(np.log10(v0_arr))
    ax.set_yticklabels([f"{v:g}" for v in v0_arr])
    fig.colorbar(pc, ax=ax)

    # 在存活相图上叠加 died/chaos 的分界（P=0.5 等值线）
    if title.startswith("P(chaos)"):
        X, Y = np.meshgrid(rho_arr, np.log10(v0_arr))
        ax.contour(X, Y, P_chaos, levels=[0.5], colors="black",
                   linewidths=1.5, linestyles="--")

fig.suptitle("MorphoNAND phase diagram (mean over 20 seeds per cell)")
fig.tight_layout()
out = "exp_result/phase_diagram.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)

print("\nP(chaos) matrix (rows = v0, cols = rho):")
print("      rho:", " ".join(f"{v:g}" for v in rho_arr))
for j, v0 in enumerate(v0_arr):
    print(f"v0={v0:g}:", " ".join(f"{P_chaos[j, i]:.2f}" for i in range(len(rho_arr))))

print("\nsurvival P(chaos) averaged over v0, per rho:")
for i, rho in enumerate(rho_arr):
    print(f"rho={rho:g}: {np.nanmean(P_chaos[:, i]):.3f}")

print("\nsurvival P(chaos) averaged over rho, per v0:")
for j, v0 in enumerate(v0_arr):
    print(f"v0={v0:g}: {np.nanmean(P_chaos[j, :]):.3f}")
