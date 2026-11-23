"""Draw a subset of points with a *chosen* temperature distribution.

Random subsampling reproduces the field's natural histogram, so the rare-but-
important regions (hot spots at the LED junction, cold far-field) end up with a
handful of points and the model never learns them. These samplers let you
decide how many points each temperature band contributes.

Allocation modes (``bin_weights``):
  proportional  w_k ∝ n_k            -> same as random sampling (alpha = 1)
  power         w_k ∝ n_k ** alpha   -> alpha=0 uniform, 1 proportional, 0.5 sqrt
  uniform       w_k ∝ 1              -> equal points per temperature band
  custom        w_k from your JSON   -> explicit target shape

Allocation is exact, not probabilistic: bins that cannot supply their quota
hand the deficit back and it is re-spread over bins that still have headroom.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SampleSpec:
    n: int = 60_000
    bins: int = 20
    mode: str = "power"          # proportional | power | uniform | custom
    alpha: float = 0.4           # only for mode="power"
    edges: list | None = None    # explicit bin edges, else linspace over [min,max]
    weights: list | None = None  # only for mode="custom", len == len(edges)-1
    binning: str = "quantile"  # equal_width | quantile
    spatial_voxel: float = 0.0   # >0: spread picks across voxels of this size
    seed: int = 0
    all_solids: bool | None = None
    all_pcb: bool | None = None
    all_chip: bool | None = None
    sim_number: int | None = None


# @dataclass
# class Sampler:
#     n: int = 200000
#     all_solids: bool
#     all_pcb: bool
#     all_chip: bool
#     sim_number: int


# --------------------------------------------------------------------------- #
def make_edges(t: np.ndarray, spec: SampleSpec) -> np.ndarray:
    if spec.edges is not None:
        return np.asarray(spec.edges, dtype=float)
    if spec.binning == "quantile":
        q = np.linspace(0, 1, spec.bins + 1)
        e = np.unique(np.quantile(t, q))
        if len(e) < 2:
            e = np.array([t.min(), t.max() + 1e-9])
        return e
    lo, hi = float(np.min(t)), float(np.max(t))
    return np.linspace(lo, hi + 1e-9 * max(1.0, abs(hi)), spec.bins + 1)


def bin_index(t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(t, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


def bin_weights(counts: np.ndarray, spec: SampleSpec) -> np.ndarray:
    k = len(counts)
    if spec.mode == "proportional":
        w = counts.astype(float)
    elif spec.mode == "uniform":
        w = np.ones(k, dtype=float)
    elif spec.mode == "power":
        w = np.power(counts.astype(float), spec.alpha)
    elif spec.mode == "custom":
        if spec.weights is None or len(spec.weights) != k:
            raise ValueError(f"custom mode needs {k} weights, got {spec.weights}")
        w = np.asarray(spec.weights, dtype=float)
    else:
        raise ValueError(f"unknown mode {spec.mode!r}")
    w = np.where(counts > 0, w, 0.0)
    if w.sum() <= 0:
        raise ValueError("all bin weights are zero")
    return w / w.sum()


def allocate(counts: np.ndarray, n_total: int, w: np.ndarray) -> np.ndarray:
    """Water-filling allocation: honour weights, respect per-bin supply."""
    counts = counts.astype(int)
    n_total = int(min(n_total, counts.sum()))
    alloc = np.zeros(len(counts), dtype=int)
    remaining, active = n_total, w.copy()

    for _ in range(200):
        if remaining <= 0 or active.sum() <= 0:
            break
        headroom = counts - alloc
        active = np.where(headroom > 0, active, 0.0)
        if active.sum() <= 0:
            break
        share = active / active.sum()
        take = np.minimum(np.floor(share * remaining).astype(int), headroom)
        if take.sum() == 0:  # remainder smaller than the number of bins
            order = np.argsort(-share)
            for b in order:
                if remaining <= 0:
                    break
                if headroom[b] > 0:
                    alloc[b] += 1
                    remaining -= 1
            continue
        alloc += take
        remaining -= int(take.sum())
    return alloc


# --------------------------------------------------------------------------- #
def _voxel_spread(idx: np.ndarray, xyz: np.ndarray, n: int, voxel: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Pick n indices spread over space: round-robin across occupied voxels."""
    if n >= len(idx):
        return idx
    keys = np.floor(xyz[idx] / voxel).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    order = rng.permutation(len(idx))
    group, idx = group[order], idx[order]
    sort = np.argsort(group, kind="stable")
    group, idx = group[sort], idx[sort]
    starts = np.searchsorted(group, np.arange(group.max() + 1))
    ends = np.r_[starts[1:], len(group)]
    picked, ring = [], 0
    while len(picked) < n:
        progressed = False
        for s, e in zip(starts, ends):
            if s + ring < e:
                picked.append(idx[s + ring])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        ring += 1
    return np.array(picked, dtype=int)


def sample(df: pd.DataFrame, spec: SampleSpec, target: str = "temperature"):
    """Return (subset_dataframe, report_dict)."""
    rng = np.random.default_rng(spec.seed)
    if len(df) == 0:
        raise ValueError(
            "nothing to sample from - the split left this region empty. "
            "Check the split settings (a block size larger than the geometry "
            "is the usual cause).")
    t = df[target].to_numpy(float)
    if not np.isfinite(t).all():
        raise ValueError(f"{target!r} contains NaN or inf")
    edges = make_edges(t, spec)
    b = bin_index(t, edges)
    counts = np.bincount(b, minlength=len(edges) - 1)
    w = bin_weights(counts, spec)
    alloc = allocate(counts, spec.n, w)

    xyz = df[["x", "y", "z"]].to_numpy(float) if spec.spatial_voxel > 0 else None
    chosen = []
    for k, want in enumerate(alloc):
        if want <= 0:
            continue
        pool = np.flatnonzero(b == k)
        if want >= len(pool):
            chosen.append(pool)
        elif xyz is not None:
            chosen.append(_voxel_spread(pool, xyz, want, spec.spatial_voxel, rng))
        else:
            chosen.append(rng.choice(pool, size=want, replace=False))
    sel = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=int)

    report = {
        "requested": spec.n,
        "selected": int(len(sel)),
        "mode": spec.mode,
        "alpha": spec.alpha,
        "edges": edges.tolist(),
        "available_per_bin": counts.tolist(),
        "selected_per_bin": alloc.tolist(),
        "starved_bins": [
            {"bin": k, "lo": float(edges[k]), "hi": float(edges[k + 1]),
             "wanted": int(round(w[k] * spec.n)), "available": int(counts[k])}
            for k in range(len(counts))
            if counts[k] > 0 and alloc[k] < round(w[k] * spec.n) - 1
        ],
        "empty_bins": [k for k in range(len(counts)) if counts[k] == 0],
    }
    return df.iloc[sel].reset_index(drop=True), report


def density_weights(t: np.ndarray, bins: int = 20, alpha: float = 1.0,
                    clip: float = 20.0) -> np.ndarray:
    """Per-point loss weights ∝ 1/density(T)**alpha, normalised to mean 1.

    An alternative to resampling: keep every point, but tell the loss that rare
    temperatures matter more. Often better than aggressive resampling because
    nothing is thrown away.
    """
    edges = np.linspace(t.min(), t.max() + 1e-9, bins + 1)
    b = bin_index(t, edges)
    counts = np.bincount(b, minlength=bins).astype(float)
    dens = counts[b] / counts.sum()
    w = np.power(np.maximum(dens, 1e-12), -alpha)
    w = np.clip(w / w.mean(), 1.0 / clip, clip)
    return w / w.mean()


def simple_sample(X:np.ndarray , sampleSpec: Sampler):
    X = np.asarray(X, np.float32)
    print(X)
    

