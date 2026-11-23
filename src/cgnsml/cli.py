"""Command line interface.

    cgnsml inspect  file.cgns                       # what's inside?
    cgnsml prepare  -c cfg.yaml                     # sample + split only
    cgnsml train    -c cfg.yaml --model fourier     # full pipeline
    cgnsml sweep    -c cfg.yaml --models hgb,rf,fourier,siren
    cgnsml report   runs/latest                     # re-print metrics
    cgnsml models                                   # list registered models

Any config key can be overridden on the command line; CLI wins over YAML,
YAML wins over defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import cgns_io, models, pipeline


def _overrides(a) -> dict:
    """Map flat CLI flags onto the nested config."""
    o = {"data": {}, "sample": {}, "split": {}, "features": {}, "model": {},
         "output": {}}
    if a.__dict__.get("input"):
        o["data"]["path"] = a.input
    if a.__dict__.get("cache"):
        o["data"]["cache"] = a.cache
    if a.__dict__.get("zones"):
        o["data"]["zones"] = a.zones.split(",")
    if a.__dict__.get("solution"):
        o["data"]["solution"] = a.solution
    if a.__dict__.get("cache_dir"):
        o["data"]["cache_dir"] = a.cache_dir
    if a.__dict__.get("points_per_case") is not None:
        o["data"]["points_per_case"] = a.points_per_case
    for src, dst in [("n_train", "n_train"), ("n_test", "n_test"),
                     ("bins", "bins"), ("dist", "mode"), ("alpha", "alpha"),
                     ("binning", "binning"), ("voxel", "spatial_voxel"),
                     ("seed", "seed"), ("test_mode", "test_mode")]:
        v = a.__dict__.get(src)
        if v is not None:
            o["sample"][dst] = v
    if a.__dict__.get("per_case"):
        o["sample"]["per_case"] = True
    if a.__dict__.get("edges"):
        o["sample"]["edges"] = [float(x) for x in a.edges.split(",")]
    if a.__dict__.get("weights"):
        o["sample"]["weights"] = [float(x) for x in a.weights.split(",")]
    for src, dst in [("split", "how"), ("test_frac", "test_frac"),
                     ("block", "block")]:
        v = a.__dict__.get(src)
        if v is not None:
            o["split"][dst] = v
    if a.__dict__.get("holdout"):
        o["split"]["holdout"] = a.holdout.split(",")
    if a.__dict__.get("inputs"):
        o["features"]["inputs"] = a.inputs.split(",")
    if a.__dict__.get("targets"):
        o["features"]["targets"] = a.targets.split(",")
    if a.__dict__.get("knn") is not None:
        o["features"]["knn"] = a.knn
    if a.__dict__.get("no_type"):
        o["features"]["use_type"] = False
    if a.__dict__.get("model"):
        o["model"]["name"] = a.model
    if a.__dict__.get("params"):
        o["model"]["params"] = json.loads(a.params)
    if a.__dict__.get("loss_weights"):
        o["model"]["loss_weights"] = a.loss_weights
    if a.__dict__.get("out"):
        o["output"]["dir"] = a.out
    if a.__dict__.get("vtk"):
        o["output"]["vtk"] = True
    return o


def build_cfg(a) -> dict:
    cfg = pipeline.deep_merge(pipeline.DEFAULTS, pipeline.load_config(a.__dict__.get("config")))
    cfg = pipeline.deep_merge(cfg, _overrides(a))
    # params in the config belong to the config's model. Switching model on the
    # command line without new params must not carry branch_cols onto a tree.
    if a.__dict__.get("model") and not a.__dict__.get("params"):
        cfg["model"]["params"] = {}
    if not cfg["data"]["path"]:
        sys.exit("error: no input file (use -i/--input or set data.path in the config)")
    return cfg


def _common(p, needs_data=True):
    if needs_data:
        p.add_argument("-i", "--input",
                       help="path to .cgns (or .npz); a glob or comma-separated "
                            "list loads several cases")
        p.add_argument("--cache", help="npz cache path; written on first read")
        p.add_argument("--zones", help="only read these zones, comma separated")
        p.add_argument("--solution", help="FlowSolution to use (substring, or 'last')")
        p.add_argument("--cache-dir", dest="cache_dir",
                       help="directory for per-case caches (multi-case mode)")
        p.add_argument("--points-per-case", type=int, dest="points_per_case",
                       help="decimate each case on read (memory guard)")
    p.add_argument("-c", "--config", help="YAML config")
    p.add_argument("-o", "--out", help="run output directory")


def _sampling_args(p):
    g = p.add_argument_group("sampling")
    g.add_argument("--n-train", type=int, dest="n_train")
    g.add_argument("--n-test", type=int, dest="n_test")
    g.add_argument("--bins", type=int)
    g.add_argument("--dist", choices=["proportional", "power", "uniform", "custom"],
                   help="target temperature distribution for the training set")
    g.add_argument("--alpha", type=float,
                   help="power-mode exponent: 0=uniform per band, 1=natural")
    g.add_argument("--binning", choices=["equal_width", "quantile"])
    g.add_argument("--edges", help="explicit bin edges, comma separated")
    g.add_argument("--weights", help="custom per-bin weights, comma separated")
    g.add_argument("--voxel", type=float, dest="voxel",
                   help="spread picks across voxels of this size (model units)")
    g.add_argument("--test-mode", dest="test_mode", choices=["native", "same"])
    g.add_argument("--seed", type=int)

    s = p.add_argument_group("split")
    s.add_argument("--split", choices=["block", "random", "stratified", "zone",
                                       "ptype", "case"])
    s.add_argument("--test-frac", type=float, dest="test_frac")
    s.add_argument("--block", type=float, help="spatial block size for --split block")
    s.add_argument("--holdout", help="explicit zone/type/case names to hold out")
    s.add_argument("--per-case", dest="per_case", action="store_true",
                   help="equal sampling share per case (default on)")

    f = p.add_argument_group("features")
    f.add_argument("--inputs", help="comma separated input columns")
    f.add_argument("--targets", help="comma separated target columns")
    f.add_argument("--knn", type=int, help="add k-NN geometry features")
    f.add_argument("--no-type", action="store_true", dest="no_type")


def main(argv=None):
    ap = argparse.ArgumentParser("cgnsml", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="print the CGNS tree")
    p.add_argument("input")
    p.add_argument("--depth", type=int, default=6)

    p = sub.add_parser("summary", help="per-zone grid location, sizes and fields")
    p.add_argument("input")

    p = sub.add_parser("stats", help="field summary and temperature histogram")
    _common(p)
    p.add_argument("--bins", type=int, default=20)

    p = sub.add_parser("export", help="write the loaded table as .pt / .npz tensors")
    _common(p)
    p.add_argument("--format", choices=["pt", "npz"], default="pt")
    p.add_argument("--group-by", dest="group_by", default="ptype",
                   choices=["ptype", "zone", "case", "none"],
                   help="one file per group value (default: ptype)")

    p = sub.add_parser("prepare", help="sample + split, write train/test CSV")
    _common(p)
    _sampling_args(p)

    p = sub.add_parser("train", help="sample, split, train, evaluate, plot")
    _common(p)
    _sampling_args(p)
    p.add_argument("--model", help="registry name or module.path:ClassName")
    p.add_argument("--params", help="JSON of model kwargs")
    p.add_argument("--loss-weights", dest="loss_weights",
                   choices=["inverse_density"],
                   help="weight the loss by rarity instead of resampling")
    p.add_argument("--vtk", action="store_true", help="also export a ParaView file")

    p = sub.add_parser("sweep", help="train several models on identical data")
    _common(p)
    _sampling_args(p)
    p.add_argument("--models", required=True, help="comma separated model names")
    p.add_argument("--vtk", action="store_true")

    p = sub.add_parser("report", help="re-print the metrics of a finished run")
    p.add_argument("rundir")

    sub.add_parser("models", help="list registered models")

    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        print(cgns_io.inspect(a.input, a.depth))
        return 0

    if a.cmd == "summary":
        print(cgns_io.summarize(a.input))
        return 0

    if a.cmd == "models":
        for name in models.available():
            cls = models.REGISTRY[name]
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:12s} {doc}")
        print("\n  any importable class works too:  --model my_pkg.models:MyNet")
        return 0

    if a.cmd == "report":
        s = json.load(open(os.path.join(a.rundir, "summary.json")))
        import pandas as pd

        from . import metrics as M
        rep = dict(s["metrics"])
        for k in ("by_temperature", "by_type"):
            if k in rep:
                rep[k] = pd.DataFrame(rep[k])
        print(M.format_report(rep))
        return 0

    cfg = build_cfg(a)

    if a.cmd == "stats":
        df = pipeline.load_full(cfg)
        import numpy as np
        t = df[cfg["features"]["targets"][0]].to_numpy(float)
        print('this is df:' , df.describe(include="all").to_string())
        edges = np.linspace(t.min(), t.max() + 1e-9, a.bins + 1)
        counts, _ = np.histogram(t, edges)
        print(f"\ntemperature histogram ({a.bins} bands)")
        peak = counts.max()
        for k, c in enumerate(counts):
            bar = "#" * int(60 * c / peak) if peak else ""
            print(f"  [{edges[k]:8.2f} {edges[k+1]:8.2f}) {c:9,d} {bar}")
        rare = [k for k, c in enumerate(counts) if 0 < c < 0.002 * len(t)]
        if rare:
            print(f"\n{len(rare)} band(s) hold under 0.2% of points - these are the "
                  f"ones random sampling loses. Try --dist power --alpha 0.3")
        return 0

    if a.cmd == "export":
        from . import export as _export
        df = pipeline.load_full(cfg)
        outdir = cfg["output"]["dir"]
        gb = None if a.group_by == "none" else a.group_by
        print(f"[export] {len(df):,} points -> {outdir}")
        _export.write(df, outdir, a.format, gb,
                      cfg["data"].get("float32", True))
        return 0

    if a.cmd == "prepare":
        pipeline.run(cfg, do_train=False)
        return 0

    if a.cmd == "train":
        pipeline.run(cfg)
        return 0

    if a.cmd == "sweep":
        pipeline.sweep(cfg, [m.strip() for m in a.models.split(",")])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

