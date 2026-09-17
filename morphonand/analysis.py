from __future__ import annotations

import numpy as np

Array = np.ndarray


def connected_components(bonds: Array) -> list[list[int]]:
    """Union-find over the undirected bond graph; returns every component,
    including isolated (size-1) particles."""
    n = bonds.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.nonzero(np.triu(bonds, 1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def unwrap_cluster(positions: Array, idx: Array, box_size: float) -> Array:
    """Minimum-image unwrap relative to the first member, so a cluster that
    straddles a periodic boundary gets contiguous local coordinates."""
    pts = positions[idx]
    ref = pts[0]
    disp = pts - ref
    disp = disp - box_size * np.round(disp / box_size)
    return ref + disp


def periodic_centroid(positions: Array, box_size: float) -> np.ndarray:
    """Circular-mean centroid that is well-defined on the torus."""
    theta = 2.0 * np.pi * positions / box_size
    angle = np.arctan2(np.sin(theta).mean(axis=0), np.cos(theta).mean(axis=0))
    return (angle % (2.0 * np.pi)) * box_size / (2.0 * np.pi)


def radius_of_gyration(pts: Array) -> float:
    com = pts.mean(axis=0)
    d = pts - com
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def anisotropy(pts: Array) -> float:
    """alpha = (lambda1 - lambda2) / (lambda1 + lambda2) in [0, 1].

    0 -> round blob, 1 -> collapsed worm/chain.
    """
    com = pts.mean(axis=0)
    d = pts - com
    cov = (d.T @ d) / len(pts)
    evals = np.linalg.eigvalsh(cov)
    s = evals[0] + evals[1]
    if s < 1e-12:
        return 0.0
    return float((evals[1] - evals[0]) / s)


def cluster_metrics(bonds: Array, positions: Array, box_size: float) -> dict:
    clusters = connected_components(bonds)
    sizes = np.array([len(c) for c in clusters], dtype=int)
    n_clusters = len(clusters)
    f_giant = float(sizes.max() / len(positions)) if n_clusters else 0.0
    mean_size = float(sizes.mean()) if n_clusters else 0.0

    size_dist: dict[int, int] = {}
    for s in sizes.tolist():
        size_dist[s] = size_dist.get(s, 0) + 1

    rgs: list[float] = []
    alphas: list[float] = []
    for c in clusters:
        if len(c) >= 2:
            pts = unwrap_cluster(positions, np.asarray(c), box_size)
            rgs.append(radius_of_gyration(pts))
            alphas.append(anisotropy(pts))

    return {
        "clusters": clusters,
        "sizes": sizes,
        "n_clusters": n_clusters,
        "f_giant": f_giant,
        "mean_size": mean_size,
        "size_dist": size_dist,
        "mean_rg": float(np.mean(rgs)) if rgs else 0.0,
        "mean_anisotropy": float(np.mean(alphas)) if alphas else 0.0,
    }


def jaccard(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class ClusterTracker:
    """Track cluster identity across samples via greedy Jaccard matching.

    Yields, per current cluster, a lifetime (simulation time since it was first
    observed) and the recent centre-of-mass speed (periodic minimum image).
    """

    def __init__(self, match_threshold: float = 0.3) -> None:
        self.match_threshold = match_threshold
        self.prev: list[set[int]] = []
        self.prev_com: list[np.ndarray] = []
        self.ages: list[int] = []

    def update(
        self,
        clusters: list[list[int]],
        positions: Array,
        box_size: float,
        sample_dt: float,
    ) -> dict:
        curr = [set(c) for c in clusters]
        curr_com = [periodic_centroid(positions[np.asarray(list(c))], box_size) for c in clusters]

        matched_curr: list[int | None] = [None] * len(curr)
        matched_prev = [False] * len(self.prev)
        curr_age = [1] * len(curr)
        curr_speed = [0.0] * len(curr)

        pairs: list[tuple[float, int, int]] = []
        for pi, pset in enumerate(self.prev):
            for ci, cset in enumerate(curr):
                j = jaccard(pset, cset)
                if j >= self.match_threshold:
                    pairs.append((j, pi, ci))
        pairs.sort(key=lambda t: t[0], reverse=True)

        for _j, pi, ci in pairs:
            if matched_prev[pi] or matched_curr[ci] is not None:
                continue
            matched_prev[pi] = True
            matched_curr[ci] = pi
            curr_age[ci] = self.ages[pi] + 1
            d = curr_com[ci] - self.prev_com[pi]
            d = d - box_size * np.round(d / box_size)
            curr_speed[ci] = float(np.linalg.norm(d)) / sample_dt

        self.prev = curr
        self.prev_com = curr_com
        self.ages = curr_age

        return {
            "lifetimes": [a * sample_dt for a in curr_age],
            "com_speed": curr_speed,
        }
