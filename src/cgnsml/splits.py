"""Train/test splits.

A warning worth internalising: if every point comes from one solved field, a
random point-wise split is *not* a generalisation test. Neighbouring mesh nodes
are millimetres apart and nearly identical, so the test set is effectively
visible during training and the reported error will be optimistically low.

  random  -> interpolation check only. Fine as a sanity metric, not as evidence.
  block   -> hold out contiguous voxel blocks. Measures whether the model fills
             unseen gaps rather than memorising nodes. Use this as your headline.
  zone    -> hold out whole zones/parts (e.g. train on PCB+housing, test on LED).
  case    -> hold out whole simulation runs. The only true design-generalisation
             test; requires loading several CGNS files with a `case` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split(df: pd.DataFrame, how: str = "block", test_frac: float = 0.2,
          seed: int = 0, block: float = 5.0, holdout: list | None = None,
          stratify_on: str | None = "temperature", bins: int = 20):
    rng = np.random.default_rng(seed)
    n = len(df)

    if how == "random":
        idx = rng.permutation(n)
        cut = int(n * (1 - test_frac))
        return idx[:cut], idx[cut:]

    if how == "stratified":
        t = df[stratify_on].to_numpy(float)
        edges = np.linspace(t.min(), t.max() + 1e-9, bins + 1)
        b = np.clip(np.digitize(t, edges[1:-1]), 0, bins - 1)
        tr, te = [], []
        for k in np.unique(b):
            pool = rng.permutation(np.flatnonzero(b == k))
            cut = int(len(pool) * (1 - test_frac))
            tr.append(pool[:cut])
            te.append(pool[cut:])
        return np.concatenate(tr), np.concatenate(te)

    if how == "block":
        xyz = df[["x", "y", "z"]].to_numpy(float)
        extent = xyz.max(0) - xyz.min(0)
        span = float(extent.max())

        # A block size given in the wrong unit (5.0 on a geometry measured in
        # metres) puts every point in one block and the holdout comes out empty.
        # Auto-size instead of failing: aim for roughly 12 blocks per axis.
        auto = span / 12.0
        if not block or block <= 0:
            block = auto
            print(f"  [block] auto size {block:.4g} "
                  f"(geometry span {span:.4g})")
        else:
            probe = np.floor((xyz - xyz.min(0)) / block).astype(np.int64)
            if len(np.unique(probe, axis=0)) < 10:
                print(f"  ! block={block:g} yields under 10 blocks over a span of "
                      f"{span:.4g} - are your coordinates in metres, not mm? "
                      f"using {auto:.4g} instead")
                block = auto

        keys = np.floor((xyz - xyz.min(0)) / block).astype(np.int64)
        _, gid = np.unique(keys, axis=0, return_inverse=True)
        n_groups = gid.max() + 1
        if n_groups < 2:
            raise ValueError(
                f"block split produced a single block over a span of {span:.4g}. "
                f"Set split.block to something well under the geometry size, or "
                f"use --split random.")

        groups = rng.permutation(n_groups)
        sizes = np.bincount(gid, minlength=n_groups)[groups]
        # take whole blocks until the target fraction is reached - always >= 1
        k = int(np.searchsorted(np.cumsum(sizes), test_frac * n)) + 1
        mask = np.isin(gid, groups[:min(k, n_groups - 1)])
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    if how in ("zone", "case", "ptype"):
        col = "ptype" if how == "ptype" else how
        if col not in df.columns:
            raise ValueError(f"column {col!r} not in the table")
        vals = df[col].astype(str).to_numpy()
        if holdout:
            mask = np.isin(vals, np.array(holdout, dtype=str))
        else:
            uniq = rng.permutation(np.unique(vals))
            keep, acc = [], 0
            for u in uniq:
                if acc >= test_frac * n:
                    break
                keep.append(u)
                acc += int((vals == u).sum())
            mask = np.isin(vals, keep)
        if mask.all() or not mask.any():
            raise ValueError(f"{how} holdout selected {mask.sum()} of {n} rows")
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    raise ValueError(f"unknown split {how!r}")

