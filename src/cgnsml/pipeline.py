"""Glue: config in, run directory out.

Order of operations matters and is deliberate:

    load full field  ->  split into train/test REGIONS  ->  sample inside each

Sampling first and splitting second would let a block holdout leak, because the
subset no longer covers whole blocks. Splitting first also lets the test set
keep the field's natural temperature distribution while the training set is
reweighted - which is what you want: reweight what the model *sees*, measure on
what the world *is*.

Every run writes a self-describing folder (config, sampling report, metrics,
plots, model, predictions) so results stay reproducible and comparable.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import cgns_io, metrics, models, sampling, splits, viz
from .features import Encoder, FeatureSpec

DEFAULTS = {
    "data": {"path": None, "cache": None, "cache_dir": None,
             "type_source": "zone", "zone_type_map": {}, "zones": [],
             "field_map": {}, "solution": "", "float32": True,
             "sum_fields": None, "derived": ["k_eq", "alpha"],
             # multi-case: path may be a list, a glob, or comma-separated
             "conditions": {}, "points_per_case": 0},
    "sample": {
        "n_train": 50000, "n_test": 10000,
        "bins": 20, "mode": "power", "alpha": 0.4,
        "binning": "equal_width", "edges": None, "weights": None,
        "spatial_voxel": 0.0, "seed": 0,
        # native -> test keeps the field's real histogram (report honestly)
        # same   -> test uses the training distribution (debug rare bands)
        "test_mode": "native",
        "per_case": True,   # equal share per case when several are loaded
    },
    "split": {"how": "block", "test_frac": 0.2, "block": 5.0,
              "holdout": None, "seed": 0, "val_cases": 0},
    "features": {"inputs": ["x", "y", "z"], "targets": ["temperature"],
                 "use_type": True, "knn": 0, "scale": "standard"},
    "model": {"name": "hgb", "params": {}, "loss_weights": None,
              "loss_weight_alpha": 1.0, "scale_targets": True},
    "output": {"dir": "runs/latest", "vtk": False},
}


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_read_spec(cfg: dict) -> cgns_io.ReadSpec:
    d = cfg["data"]
    fm = dict(cgns_io.DEFAULT_FIELD_MAP)
    fm.update(d.get("field_map") or {})
    return cgns_io.ReadSpec(field_map=fm, type_source=d.get("type_source", "zone"),
                            zone_type_map=d.get("zone_type_map") or {},
                            zones=d.get("zones") or [],
                            solution=d.get("solution", ""),
                            float32=d.get("float32", True),
                            points_per_case=d.get("points_per_case", 0),
                            sum_fields=d.get("sum_fields") if d.get("sum_fields")
                            is not None else dict(cgns_io.DEFAULT_SUM_FIELDS),
                            derived=tuple(d.get("derived", ("k_eq", "alpha"))))


def _spec(cfg: dict, n: int, mode: str | None = None) -> sampling.SampleSpec:
    s = cfg["sample"]
    return sampling.SampleSpec(
        n=n, bins=s["bins"], mode=mode or s["mode"], alpha=s["alpha"],
        edges=s["edges"], weights=s["weights"], binning=s["binning"],
        spatial_voxel=s["spatial_voxel"], seed=s["seed"])


def load_full(cfg: dict):
    """One file or many. Many gets a `case` column and per-case conditions."""
    d, target = cfg["data"], cfg["features"]["targets"][0]
    files = cgns_io.expand_paths(d["path"])
    if len(files) == 1 and not d.get("conditions") and not d.get("cache_dir"):
        return cgns_io.load_any(files[0], build_read_spec(cfg), d.get("cache"))
    cache_dir = d.get("cache_dir") or (
        os.path.dirname(d["cache"]) if d.get("cache") else None)
    return cgns_io.load_cases(files, build_read_spec(cfg), cache_dir,
                              d.get("conditions"), target,
                              seed=cfg["sample"]["seed"])


def load_and_prepare(cfg: dict):
    """Return (full, train_df, test_df, sampling_reports)."""
    full = load_full(cfg)
    target = cfg["features"]["targets"][0]
    # print(f"[data] {len(full):,} points  "
    #       f"{target} in [{full[target].min():.2f}, {full[target].max():.2f}]  "
    #       f"types={sorted(full['ptype'].astype(str).unique())[:8]}")
    sp = cfg["split"]
    tr_region, te_region = splits.split(full, how=sp["how"],
                                        test_frac=sp["test_frac"], seed=sp["seed"],
                                        block=sp["block"], holdout=sp["holdout"],
                                        stratify_on=target)
    # print(f"[split] {sp['how']}: train region={len(tr_region):,}  "
    #       f"test region={len(te_region):,}")
    if len(tr_region) == 0 or len(te_region) == 0:
        raise ValueError(
            f"split {sp['how']!r} produced an empty region "
            f"(train={len(tr_region)}, test={len(te_region)}). "
            f"For block splits set split.block to 0 to auto-size it.")

    s = cfg["sample"]
    per_case = s.get("per_case", True) and "case" in full.columns
    test_mode = "proportional" if s.get("test_mode", "native") == "native" else None
    train_df, rep_tr = _sample_region(full.iloc[tr_region], cfg, s["n_train"],
                                      None, target, per_case)
    test_df, rep_te = _sample_region(full.iloc[te_region], cfg, s["n_test"],
                                     test_mode, target, per_case)

    print(f"[sample] train={len(train_df):,} (mode={rep_tr['mode']}, "
          f"alpha={rep_tr['alpha']})  test={len(test_df):,} (mode={rep_te['mode']})")
    for st in rep_tr["starved_bins"]:
        print(f"  ! band [{st['lo']:.1f}, {st['hi']:.1f}] wanted {st['wanted']:,}"
              f" but only {st['available']:,} exist")
    if rep_tr["empty_bins"]:
        print(f"  ! {len(rep_tr['empty_bins'])} temperature bands are empty in "
              f"the training region")
    return full, train_df, test_df, {"train": rep_tr, "test": rep_te}


def _sample_region(region: pd.DataFrame, cfg: dict, n: int, mode, target: str,
                   per_case: bool):
    s = cfg["sample"]
    take_all = [str(p) for p in s.get("take_all", [])]
    if not take_all:
        return _sample_region_core(region, cfg, n, mode, target, per_case)
    region = region.reset_index(drop=True)

    # several cases: handle each one on its own, n points per case
    if per_case and region["case"].astype(str).nunique() > 1:
        case_col = region["case"].astype(str)
        parts, reports = [], []
        for c in sorted(case_col.unique()):
            got, rep = _sample_region(region[case_col == c], cfg, n, mode,
                                      target, False)
            rep["case"] = c
            parts.append(got)
            reports.append(rep)
        out = pd.concat(parts, ignore_index=True)
        merged = dict(reports[0])
        merged.update(
            selected=int(len(out)), requested=n * len(reports),
            per_case={r["case"]: r["selected"] for r in reports},
            starved_bins=[dict(b, case=r["case"])
                          for r in reports for b in r["starved_bins"]],
            empty_bins=sorted({b for r in reports for b in r["empty_bins"]}),
        )
        return out, merged

    # one case: keep all take_all points, fill the rest of n from the others
    ptype = region["ptype"].astype(str)
    keep = region[ptype.isin(take_all)]
    fill_from = [str(p) for p in s.get("fill_from", [])]
    fill_mask = ptype.isin(fill_from) if fill_from else ~ptype.isin(take_all)
    rest = region[fill_mask]
    n_rest = n - len(keep)
    if n_rest <= 0:
        raise ValueError(
            f"take_all types {take_all} have {len(keep):,} points in one case, "
            f"which is >= the requested {n:,}")
    if len(rest) == 0:
        raise ValueError("No points to fill from")
    got, rep = _sample_region_core(rest, cfg, n_rest, mode, target, False)

    out = pd.concat([keep, got], ignore_index=True)
    rep["requested"] = n
    rep["selected"] = int(len(out))
    rep["taken_all"] = {k: int(v) for k, v in
                        keep["ptype"].astype(str).value_counts().items()}
    return out, rep
 
def _sample_region_core(region: pd.DataFrame, cfg: dict, n: int, mode, target: str,
                   per_case: bool):
    """Sample a region. With several cases, give each an equal share so a large
    case cannot crowd the others out of the training set."""
    region = region.reset_index(drop=True)
    if not per_case:
        return sampling.sample(region, _spec(cfg, n, mode), target)

    cases = [c for c in region["case"].astype(str).unique()]
    if len(cases) <= 1:
        return sampling.sample(region, _spec(cfg, n, mode), target)

    share = max(1, n)
    parts, reports = [], []
    for c in sorted(cases):
        sub = region[region["case"].astype(str) == c].reset_index(drop=True)
        got, rep = sampling.sample(sub, _spec(cfg, share, mode), target)
        rep["case"] = c
        parts.append(got)
        reports.append(rep)
    out = pd.concat(parts, ignore_index=True)
    merged = {
        "selected": int(len(out)), "requested": n,
        "mode": reports[0]["mode"], "alpha": reports[0]["alpha"],
        "per_case": {r["case"]: r["selected"] for r in reports},
        "starved_bins": [dict(b, case=r["case"])
                         for r in reports for b in r["starved_bins"]],
        "empty_bins": sorted({b for r in reports for b in r["empty_bins"]}),
        "edges": reports[0]["edges"],
        "available_per_bin": reports[0]["available_per_bin"],
        "selected_per_bin": reports[0]["selected_per_bin"],
    }
    return out, merged



def _accepts(fn, name: str, strict: bool = False) -> bool:
    """Does fn take a keyword `name`? strict=False also accepts **kwargs."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return (not strict) and any(p.kind is p.VAR_KEYWORD for p in params.values())




def _case_ids(df: pd.DataFrame):
    return df["case"].astype(str).to_numpy() if "case" in df.columns else None




def _carve_val(train_df: pd.DataFrame, cfg: dict):
    """Move split.val_cases whole training simulations into a validation set."""
    k = int(cfg["split"].get("val_cases") or 0)
    if k <= 0:
        return train_df, None
    if "case" not in train_df.columns:
        raise ValueError("split.val_cases needs several simulations "
                         "(multi-file data.path, so rows carry a `case` column)")
    cases = sorted(train_df["case"].astype(str).unique())
    if k >= len(cases):
        raise ValueError(f"split.val_cases={k} but only {len(cases)} training "
                         f"simulation(s): {cases}")
    rng = np.random.default_rng(cfg["split"].get("seed", 0))
    val = sorted(rng.choice(cases, size=k, replace=False).tolist())
    m = train_df["case"].astype(str).isin(val).to_numpy()
    print(f"[split] validation simulation(s) for early stopping: {val}")
    return train_df[~m].reset_index(drop=True), train_df[m].reset_index(drop=True)




def run(cfg: dict, do_train: bool = True) -> dict:
    t0 = time.time()
    outdir = cfg["output"]["dir"]
    os.makedirs(outdir, exist_ok=True)
    target = cfg["features"]["targets"][0]

    full, train_df, test_df, reps = load_and_prepare(cfg)
    json.dump(reps, open(os.path.join(outdir, "sampling.json"), "w"), indent=2)
    viz.distribution(full[target].to_numpy(), train_df[target].to_numpy(), outdir)
    viz.coverage(full[target].to_numpy(), train_df[target].to_numpy(), outdir)

    if not do_train:
        train_df.to_csv(os.path.join(outdir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(outdir, "test.csv"), index=False)
        print(f"[done] datasets written to {outdir}")
        return {"sampling": reps, "outdir": outdir}

    # ---- features ------------------------------------------------------- #
    train_df, val_df = _carve_val(train_df, cfg)
    enc = Encoder(FeatureSpec(**cfg["features"])).fit(train_df)
    Xtr, ytr = enc.transform(train_df)
    Xte, yte = enc.transform(test_df)
    scale_y = cfg["model"].get("scale_targets", True)
    ytr_m = enc.scale_y(ytr) if scale_y else ytr
    yte_m = enc.scale_y(yte) if scale_y else yte
    print(f"[features] {Xtr.shape[1]} columns: {enc.columns}")
    g_tr, g_te = _case_ids(train_df), _case_ids(test_df)
    if val_df is not None:
        Xva, yva = enc.transform(val_df)
        eval_set = (Xva, enc.scale_y(yva) if scale_y else yva)
        g_ev = _case_ids(val_df)
    else:
        eval_set, g_ev = (Xte, yte_m), g_te

    # ---- optional loss weights (alternative to resampling) --------------- #
    w = None
    if (cfg["model"].get("loss_weights") == "inverse_density"
            and cfg["sample"]["mode"] != "proportional"):
        print("  ! you are resampling AND weighting the loss by rarity - the two "
              "compound and usually hurt.\n"
              "    Pick one: either --dist power ... or --dist proportional "
              "--loss-weights inverse_density")
    if cfg["model"].get("loss_weights") == "inverse_density":
        w = sampling.density_weights(train_df[target].to_numpy(float),
                                     bins=cfg["sample"]["bins"],
                                     alpha=cfg["model"].get("loss_weight_alpha", 1.0))
        print(f"[weights] inverse-density, range [{w.min():.2f}, {w.max():.2f}]")

    # ---- train ---------------------------------------------------------- #
    params = dict(cfg["model"].get("params") or {})
    if params.get("branch_cols") and isinstance(params["branch_cols"][0], str):
        names = params["branch_cols"]
        idx = [enc.columns.index(nm) for nm in names if nm in enc.columns]
        unknown = [nm for nm in names if nm not in enc.columns]
        if unknown:
            raise ValueError(f"branch_cols {unknown} are not feature columns. "
                             f"Available: {enc.columns}")
        params["branch_cols"] = idx
        print(f"[branch] {names} -> columns {idx}")
    # [point_models] resolve *_cols params given by column name
    for _key, _val in list(params.items()):
        if (_key.endswith("_cols") and isinstance(_val, (list, tuple)) and _val
                and all(isinstance(_v, str) for _v in _val)):
            _missing = [_v for _v in _val if _v not in enc.columns]
            if _missing:
                raise ValueError(f"{_key} {_missing} are not feature columns. "
                                 f"Available: {enc.columns}")
            params[_key] = [enc.columns.index(_v) for _v in _val]
            print(f"[cols] {_key}: {list(_val)} -> {params[_key]}")
    model = models.get_model(cfg["model"]["name"], **params)
    print(params)
    # model=models.get_model('pointnet_thermal',**params)
    print(f"[train] {cfg['model']['name']} on {len(Xtr):,} points")
    t1 = time.time()
    fit_kw = {}
    if _accepts(model.fit, "sample_weight"):
        fit_kw["sample_weight"] = w
    if _accepts(model.fit, "eval_set"):
        fit_kw["eval_set"] = eval_set
        if val_df is None:
            print("  ! eval_set is the TEST set (early stopping / best epoch "
                  "see it). Set split.val_cases: 1 for honest test scores.")
    if _accepts(model.fit, "groups", strict=True):
        fit_kw["groups"] = g_tr
    if _accepts(model.fit, "eval_groups", strict=True):
        fit_kw["eval_groups"] = g_ev
    model.fit(Xtr, ytr_m, **fit_kw)
    fit_s = time.time() - t1
    models.save_model(model, os.path.join(outdir, "model.bin"))

    # ---- evaluate ------------------------------------------------------- #
    _pred_groups = _accepts(model.predict, "groups", strict=True)

    def _pred(X, g=None):
        p = model.predict(X, groups=g) if _pred_groups else model.predict(X)
        return enc.unscale_y(p) if scale_y else p

    yt, yp = yte[:, 0], _pred(Xte, g_te)[:, 0]
    rep = metrics.report(yt, yp, temps=yt,
                         groups=test_df["ptype"].astype(str).to_numpy()
                         if "ptype" in test_df else None)
    print("\n" + metrics.format_report(rep))

    tr_basic = metrics.basic(ytr[:, 0], _pred(Xtr, g_tr)[:, 0])
    rep["train_overall"] = tr_basic
    print(f"\ntrain MAE={tr_basic['mae']:.3f}   "
          f"generalisation gap={rep['overall']['mae'] - tr_basic['mae']:+.3f}")

    # ---- artefacts ------------------------------------------------------ #
    err = np.abs(yp - yt)
    viz.parity(yt, yp, outdir, c=err, clabel="|error|")
    viz.error_vs_temperature(yt, yp, outdir)
    viz.residual_hist(yt, yp, outdir)
    viz.spatial_error(test_df[["x", "y", "z"]].to_numpy(float), err, outdir)
    if getattr(model, "history", None):
        viz.training_curve(model.history, outdir)

    out = test_df.copy()
    out["pred"], out["abs_error"] = yp, err
    out.to_csv(os.path.join(outdir, "predictions.csv"), index=False)
    if cfg["output"].get("vtk"):
        viz.export_vtk(test_df[["x", "y", "z"]].to_numpy(float),
                       {"truth": yt, "pred": yp, "abs_error": err},
                       os.path.join(outdir, "predictions.vtk"))

    summary = {
        "config": cfg, "sampling": reps, "fit_seconds": fit_s,
        "total_seconds": time.time() - t0,
        "metrics": {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in rep.items()},
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"),
              indent=2, default=str)
    rep["by_temperature"].to_csv(
        os.path.join(outdir, "metrics_by_temperature.csv"), index=False)
    print(f"\n[done] {time.time() - t0:.1f}s -> {outdir}")
    return summary


def sweep(cfg: dict, model_names: list[str]) -> pd.DataFrame:
    """Same data, several models. The only fair way to compare architectures."""
    rows = []
    base_dir = cfg["output"]["dir"]
    for name in model_names:
        c = deep_merge(cfg, {"model": {"name": name},
                             "output": {"dir": os.path.join(base_dir, name)}})
        if name != cfg["model"]["name"]:
            c["model"] = dict(c["model"], params={})
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        try:
            s = run(c)
        except Exception as exc:  # keep the sweep alive
            print(f"  !! {name} failed: {exc}")
            rows.append({"model": name, "error": str(exc)})
            continue
        o = s["metrics"]["overall"]
        h = s["metrics"].get("hot_tail", {})
        rows.append({"model": name, "mae": o["mae"], "rmse": o["rmse"],
                     "p95": o["p95_abs"], "max": o["max_abs"], "r2": o["r2"],
                     "hot1pct_mae": h.get("mae"), "fit_s": round(s["fit_seconds"], 1)})
    df = pd.DataFrame(rows)
    if "mae" in df.columns:
        df = df.sort_values("mae")
    os.makedirs(base_dir, exist_ok=True)
    df.to_csv(os.path.join(base_dir, "leaderboard.csv"), index=False)
    print("\n" + df.to_string(index=False))
    return df

