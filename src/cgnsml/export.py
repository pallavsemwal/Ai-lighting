"""Export the loaded table as tensors, in the layout the training notebooks use.


Writes one file per group (air / solid / chip / ...), each a dict of arrays:


    {"centroids": (n,3), "temperature": (n,), "heat_gen": (n,),
     "alpha": (n,), "velocity": (n,3), "zone": (n,), "case": (n,)}


Keys are only present when the underlying zones actually carry them - solids
have no velocity, and inventing a zero column there would quietly teach a model
that solids are stagnant fluid.
"""
from __future__ import annotations


import os


import numpy as np
import pandas as pd


VECTORS = {"centroids": ("x", "y", "z"), "velocity": ("vx", "vy", "vz")}
SCALARS = ["temperature", "heat_gen", "alpha", "k_eq", "density", "cp", "vmag", ]




def to_dict(df: pd.DataFrame, float32: bool = True) -> dict:
    dt = np.float32 if float32 else np.float64
    out: dict = {}
    for name, comps in VECTORS.items():
        if all(c in df.columns for c in comps):
            block = np.stack([df[c].to_numpy() for c in comps], axis=1)
            if not np.isnan(block).all():
                out[name] = block.astype(dt)
    for c in SCALARS:
        if c in df.columns and not df[c].isna().all():
            out[c] = df[c].to_numpy().astype(dt)
    for c in ("zone", "ptype", "case"):
        if c in df.columns:
            out[c] = df[c].astype(str).to_numpy()
    out["n"] = len(df)
    return out




def write(df: pd.DataFrame, outdir: str, fmt: str = "pt",
          group_by: str | None = "ptype", float32: bool = True) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    groups = {"all": df}
    if group_by and group_by in df.columns:
        groups = {str(g): sub for g, sub in df.groupby(group_by, observed=True)}


    written = []
    for name, sub in groups.items():
        payload = to_dict(sub, float32)
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        path = os.path.join(outdir, f"{slug}.{fmt}")
        if fmt == "pt":
            import torch
            torch.save(payload, path)
        elif fmt == "npz":
            np.savez_compressed(path, **{k: v for k, v in payload.items()
                                         if k != "n"})
        else:
            raise ValueError(f"unknown format {fmt!r}; use pt or npz")
        keys = [k for k in payload if k != "n"]
        print(f"  {path}  {len(sub):,} points  {keys}")
        written.append(path)
    return written





