
from __future__ import annotations


import numpy as np
import pandas as pd




def basic(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = y_pred - y_true
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "p95_abs": float(np.percentile(np.abs(e), 95)),
        "max_abs": float(np.max(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": float(1 - ss_res / ss_tot),
    }




def by_bins(y_true, y_pred, values, edges, label="bin") -> pd.DataFrame:
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if not m.any():
            rows.append({label: k, "lo": edges[k], "hi": edges[k + 1], "n": 0})
            continue
        r = basic(y_true[m], y_pred[m])
        r.update({label: k, "lo": float(edges[k]), "hi": float(edges[k + 1])})
        rows.append(r)
    cols = [label, "lo", "hi", "n", "mae", "rmse", "p95_abs", "max_abs", "bias"]
    return pd.DataFrame(rows).reindex(columns=cols)




def by_group(y_true, y_pred, groups) -> pd.DataFrame:
    rows = []
    for g in pd.unique(groups):
        m = np.asarray(groups) == g
        r = basic(y_true[m], y_pred[m])
        r["group"] = str(g)
        rows.append(r)
    return pd.DataFrame(rows)[["group", "n", "mae", "rmse", "p95_abs", "max_abs", "bias", "r2"]]




def report(y_true, y_pred, temps=None, groups=None, bins=12, tail_q=0.99) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {"overall": basic(y_true, y_pred)}
    if temps is not None:
        temps = np.asarray(temps).ravel()
        edges = np.linspace(temps.min(), temps.max() + 1e-9, bins + 1)
        out["by_temperature"] = by_bins(y_true, y_pred, temps, edges, "temp_bin")
        hot = temps >= np.quantile(temps, tail_q)
        cold = temps <= np.quantile(temps, 1 - tail_q)
        out["hot_tail"] = basic(y_true[hot], y_pred[hot])
        out["cold_tail"] = basic(y_true[cold], y_pred[cold])
        # worst band drives the useful headline number
        bt = out["by_temperature"]
        nz = bt[bt["n"] > 0]
        if len(nz):
            out["worst_band"] = nz.loc[nz["mae"].idxmax()].to_dict()
    if groups is not None:
        out["by_type"] = by_group(y_true, y_pred, groups)
    return out




def format_report(rep: dict, unit: str = "K") -> str:
    lines = []
    o = rep["overall"]
    lines.append(f"overall   n={o['n']:,}  MAE={o['mae']:.3f}{unit}  "
                 f"RMSE={o['rmse']:.3f}{unit}  P95={o['p95_abs']:.3f}{unit}  "
                 f"max={o['max_abs']:.3f}{unit}  R2={o['r2']:.4f}")
    if "hot_tail" in rep:
        h, c = rep["hot_tail"], rep["cold_tail"]
        lines.append(f"hot 1%    n={h['n']:,}  MAE={h['mae']:.3f}{unit}  max={h['max_abs']:.3f}{unit}")
        lines.append(f"cold 1%   n={c['n']:,}  MAE={c['mae']:.3f}{unit}  max={c['max_abs']:.3f}{unit}")
    if "worst_band" in rep:
        w = rep["worst_band"]
        lines.append(f"worst band [{w['lo']:.1f}, {w['hi']:.1f}]{unit}  "
                     f"n={int(w['n']):,}  MAE={w['mae']:.3f}{unit}")
    if "by_temperature" in rep:
        lines.append("\nper temperature band")
        lines.append(rep["by_temperature"].to_string(index=False,
                     float_format=lambda v: f"{v:.3f}"))
    if "by_type" in rep:
        lines.append("\nper part type")
        lines.append(rep["by_type"].to_string(index=False,
                     float_format=lambda v: f"{v:.3f}"))
    return "\n".join(lines)



