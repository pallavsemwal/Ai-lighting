from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path



def distribution(full_t, sampled_t, outdir, bins=60, name="distribution.png"):
    """Before/after histogram - the plot that justifies the whole sampler."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(full_t, bins=bins, color="#8899aa")
    ax[0].set_title(f"full field  (n={len(full_t):,})")
    ax[0].set_xlabel("temperature")
    ax[0].set_yscale("log")
    ax[1].hist(sampled_t, bins=bins, color="#cc5522")
    ax[1].set_title(f"sampled  (n={len(sampled_t):,})")
    ax[1].set_xlabel("temperature")
    for a in ax:
        a.set_ylabel("points (log)")
        a.grid(alpha=0.25)
    return _save(fig, outdir, name)


def coverage(full_t, sampled_t, outdir, bins=30, name="coverage.png"):
    """Fraction of each band that made it into the sample."""
    edges = np.linspace(full_t.min(), full_t.max() + 1e-9, bins + 1)
    a, _ = np.histogram(full_t, edges)
    b, _ = np.histogram(sampled_t, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(centers, np.where(a > 0, b / np.maximum(a, 1), 0),
           width=np.diff(edges) * 0.9, color="#2277aa")
    ax.set_xlabel("temperature")
    ax.set_ylabel("sampled / available")
    ax.set_title("per-band sampling rate")
    ax.grid(alpha=0.25)
    return _save(fig, outdir, name)


def parity(y_true, y_pred, outdir, name="parity.png", c=None, clabel=""):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    s = ax.scatter(y_true, y_pred, s=3, alpha=0.25, c=c, cmap="viridis")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("truth")
    ax.set_ylabel("prediction")
    ax.set_title("parity")
    ax.grid(alpha=0.25)
    if c is not None:
        fig.colorbar(s, ax=ax, label=clabel)
    return _save(fig, outdir, name)


def error_vs_temperature(y_true, y_pred, outdir, bins=20, name="error_vs_temp.png"):
    err = np.abs(y_pred - y_true)
    edges = np.linspace(y_true.min(), y_true.max() + 1e-9, bins + 1)
    idx = np.clip(np.digitize(y_true, edges[1:-1]), 0, bins - 1)
    data = [err[idx == k] for k in range(bins)]
    pos = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(9, 4))
    keep = [k for k in range(bins) if len(data[k]) > 3]
    if keep:
        ax.boxplot([data[k] for k in keep], positions=pos[keep],
                   widths=np.diff(edges).mean() * 0.7, showfliers=False,
                   manage_ticks=False)
    ax.set_xlabel("true temperature")
    ax.set_ylabel("absolute error")
    ax.set_title("error by temperature band")
    ax.grid(alpha=0.25)
    return _save(fig, outdir, name)


def residual_hist(y_true, y_pred, outdir, name="residuals.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_pred - y_true, bins=80, color="#556677")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("prediction - truth")
    ax.set_ylabel("count")
    ax.set_title("residuals")
    ax.grid(alpha=0.25)
    return _save(fig, outdir, name)


def spatial_error(xyz, err, outdir, name="spatial_error.png", max_points=40000):
    """Where in the geometry the model is wrong - three orthographic views."""
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, err = xyz[sel], err[sel]
    vmax = float(np.percentile(err, 99))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (i, j, t) in zip(axes, [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        s = ax.scatter(xyz[:, i], xyz[:, j], c=err, s=2, cmap="inferno",
                       vmin=0, vmax=vmax)
        ax.set_title(f"{t} - |error|")
        ax.set_aspect("equal", "datalim")
    fig.colorbar(s, ax=axes, shrink=0.85, label="absolute error")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def training_curve(history, outdir, name="training.png"):
    if not history:
        return None
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, [h["train"] for h in history], label="train")
    ax.plot(ep, [h["val"] for h in history], label="val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, outdir, name)


def export_vtk(xyz, fields: dict, path: str):
    """Legacy ASCII VTK point cloud - opens directly in ParaView, no deps."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = len(xyz)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\ncgnsml\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {n} float\n")
        np.savetxt(fh, xyz, fmt="%.6g")
        fh.write(f"\nCELLS {n} {2 * n}\n")
        np.savetxt(fh, np.column_stack([np.ones(n, int), np.arange(n)]), fmt="%d")
        fh.write(f"\nCELL_TYPES {n}\n")
        np.savetxt(fh, np.ones(n, int), fmt="%d")
        fh.write(f"\nPOINT_DATA {n}\n")
        for name, arr in fields.items():
            fh.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, np.asarray(arr).ravel(), fmt="%.6g")
    return path
