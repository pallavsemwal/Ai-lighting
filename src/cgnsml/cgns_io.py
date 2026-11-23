"""Read a CGNS/HDF5 file into a flat point table, with an on-disk cache.


CGNS files are HDF5 underneath. Every CGNS node is an HDF5 group carrying a
``label`` attribute (Zone_t, FlowSolution_t, DataArray_t, ...) and an optional
child dataset named ``" data"`` (note the leading space) holding the payload.


The critical detail this reader gets right: **a FlowSolution can live at cell
centres rather than at vertices.** When it does, GridCoordinates has one length
(number of vertices) and the solution arrays have another (number of cells), and
the two cannot be zipped together. Star-CCM+ exports write the matching cell
positions as CentroidX/Y/Z *inside* the FlowSolution - those are the coordinates
to use. Lengths are checked, never truncated.
"""
from __future__ import annotations


import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Sequence


import numpy as np
import pandas as pd


try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


_DATA = " data"


#: coordinates when GridLocation is Vertex
COORDS_VERTEX = {"x": ["CoordinateX"], "y": ["CoordinateY"], "z": ["CoordinateZ"]}
#: coordinates when GridLocation is CellCenter
COORDS_CELL = {"x": ["CentroidX", "Centroid_0"],
               "y": ["CentroidY", "Centroid_1"],
               "z": ["CentroidZ", "Centroid_2"]}


#: solution fields -> candidate DataArray names, checked in order
DEFAULT_FIELD_MAP = {
    "temperature": ["Temperature", "StaticTemperature", "TotalTemperature"],
    "vx": ["VelocityX", "Velocity_0"],
    "vy": ["VelocityY", "Velocity_1"],
    "vz": ["VelocityZ", "Velocity_2"],
    "vmag": ["VelocityMagnitude"],
    "kx": ["Conductivity_ExportX", "ThermalConductivityX"],
    "ky": ["Conductivity_ExportY", "ThermalConductivityY"],
    "kz": ["Conductivity_ExportZ", "ThermalConductivityZ"],
    "density": ["Density"],
    "cp": ["SpecificHeat", "SpecificHeatCapacity"],
    "ptype": ["MaterialId", "ComponentId", "PartId"],
}


#: fields summed across every DataArray whose name contains a needle.
#: Star-CCM+ writes one array per heat source, so the total dissipation in a
#: zone is their sum. Newer exports name them *Heat_data*; older ones used
#: UserFieldFunction_*. A value may be one needle or a list of needles.
DEFAULT_SUM_FIELDS = {"heat_gen": ["Heat_data", "UserFieldFunction"]}




# --------------------------------------------------------------------------- #
# low-level node helpers
# --------------------------------------------------------------------------- #
def _attr_str(node, key: str) -> str:
    v = node.attrs.get(key)
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("ascii", "ignore").strip("\x00").strip()
    arr = np.asarray(v)
    if arr.dtype.kind in "iuS":
        return arr.tobytes().decode("ascii", "ignore").strip("\x00").strip()
    return str(v)




def label_of(node) -> str:
    return _attr_str(node, "label")




def payload(node):
    """Return the node's ``" data"`` dataset (lazy h5py handle) or None."""
    return node[_DATA] if _DATA in node else None




def node_text(node) -> str:
    """Decode a char-array node such as GridLocation_t or ZoneType_t."""
    d = payload(node)
    if d is None:
        return ""
    a = np.asarray(d[()])
    if a.dtype.kind in "iu" and a.dtype.itemsize == 1:
        return a.tobytes().decode("ascii", "ignore").strip("\x00").strip()
    if a.dtype.kind in "SU":
        return str(np.asarray(a).astype(str).ravel()[0]).strip("\x00").strip()
    return ""




def walk(node, path: str = "") -> Iterator[tuple[str, "h5py.Group"]]:
    """Depth-first walk over CGNS nodes, skipping HDF5 bookkeeping children."""
    yield (path or "/"), node
    for key in node.keys():
        if key.startswith(" "):
            continue
        child = node[key]
        if isinstance(child, h5py.Group):
            yield from walk(child, f"{path}/{key}")




def find(root, label: str) -> list[tuple[str, "h5py.Group"]]:
    return [(p, n) for p, n in walk(root) if label_of(n) == label]




def children_by_label(node, label: str) -> dict[str, "h5py.Group"]:
    """Direct children only - this is what keeps DimensionalExponents_t out."""
    out = {}
    for k in node.keys():
        if k.startswith(" "):
            continue
        c = node[k]
        if isinstance(c, h5py.Group) and label_of(c) == label:
            out[k] = c
    return out




def _read_array(node) -> np.ndarray | None:
    d = payload(node)
    if d is None:
        return None
    if d.dtype.kind not in "fiu" or d.dtype.itemsize == 1:
        return None
    return np.asarray(d[()]).ravel()




def _match(candidates: Sequence[str], available: dict, fuzzy: bool = False,
           note: list | None = None):
    """Exact (case-insensitive) match. Fuzzy 'contains' only when asked, and it
    reports what it picked - silent fuzzy matching on names like 'X' would
    happily grab VelocityX."""
    lowered = {k.lower(): k for k in available}
    for cand in candidates:
        if cand.lower() in lowered:
            return available[lowered[cand.lower()]]
    if fuzzy:
        for cand in candidates:
            for k_low, k in lowered.items():
                if cand.lower() in k_low:
                    if note is not None:
                        note.append(f"fuzzy match {cand!r} -> {k!r}")
                    return available[k]
    return None




# --------------------------------------------------------------------------- #
# inspection
# --------------------------------------------------------------------------- #
def _require_h5py():
    if h5py is None:
        raise RuntimeError("h5py is required to read CGNS files: pip install h5py")




def inspect(path: str, max_depth: int = 6) -> str:
    """Human-readable dump of the CGNS tree. Run this first on a new file."""
    _require_h5py()
    lines = [f"{path}  ({os.path.getsize(path) / 1e9:.2f} GB)"]
    with h5py.File(path, "r") as f:
        for node_path, node in walk(f):
            depth = node_path.count("/")
            if depth > max_depth:
                continue
            lab = label_of(node)
            if not lab or lab == "CGNSLibraryVersion_t":
                continue
            d = payload(node)
            shape = f"  shape={d.shape} dtype={d.dtype}" if d is not None else ""
            lines.append("  " * depth + f"{node_path.split('/')[-1]} [{lab}]{shape}")
    return "\n".join(lines)




def summarize(path: str) -> str:
    """Per-zone summary: grid location, point count, available fields.


    Read this before configuring anything - it tells you whether the solution is
    at vertices or cell centres, and which fields each zone actually carries.
    """
    _require_h5py()
    rows = []
    with h5py.File(path, "r") as f:
        for zpath, znode in find(f, "Zone_t"):
            zname = zpath.split("/")[-1]
            for sname, snode in children_by_label(znode, "FlowSolution_t").items():
                loc = "Vertex"
                for g in children_by_label(snode, "GridLocation_t").values():
                    loc = node_text(g) or loc
                arrays = children_by_label(snode, "DataArray_t")
                lens = {k: payload(v).shape[0] for k, v in arrays.items()
                        if payload(v) is not None}
                n = max(lens.values()) if lens else 0
                gc = {}
                for g in children_by_label(znode, "GridCoordinates_t").values():
                    gc = children_by_label(g, "DataArray_t")
                nv = payload(next(iter(gc.values()))).shape[0] if gc else 0
                row = {"zone": zname, "solution": sname, "location": loc,
                       "n_solution": n, "n_vertices": nv,
                       "fields": ", ".join(sorted(arrays))}
                # per-zone target range: this is how you find out which zone
                # actually carries the variation you want to model
                tnode = _match(["Temperature", "StaticTemperature"], arrays)
                if tnode is not None:
                    t = _read_array(tnode)
                    row.update(T_min=float(t.min()), T_max=float(t.max()),
                               T_span=float(t.max() - t.min()))
                for c in "XYZ":
                    cn = _match([f"Centroid{c}", f"Coordinate{c}"], arrays) or \
                         _match([f"Coordinate{c}"], gc)
                    if cn is not None:
                        a = _read_array(cn)
                        row[f"{c.lower()}_span"] = float(a.max() - a.min())
                rows.append(row)
    df = pd.DataFrame(rows)
    cols = [c for c in ["zone", "solution", "location", "n_solution",
                        "n_vertices", "T_min", "T_max", "T_span"]
            if c in df.columns]
    out = [df[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.2f}")]
    gcols = [c for c in ["x_span", "y_span", "z_span"] if c in df.columns]
    if gcols:
        out.append("\ngeometry span per zone (sets the scale for split.block)")
        out.append(df[["zone"] + gcols].to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))
    out.append("")
    for r in rows:
        out.append(f"{r['zone']}/{r['solution']}: {r['fields']}")
    if "T_span" in df.columns and len(df) > 1:
        hot = df.loc[df["T_span"].idxmax()]
        cold = df[df["T_span"] < 0.05 * df["T_span"].max()]
        if len(cold):
            out.append(f"\n! zone(s) {list(cold['zone'])} vary by under 5% of "
                       f"what {hot['zone']} does ({hot['T_span']:.1f} K). "
                       f"Filtering them in isolation leaves almost no signal "
                       f"to learn.")
    total = int(df["n_solution"].sum()) if len(df) else 0
    out.append(f"\ntotal usable points: {total:,}")
    if len(df):
        for _, r in df.iterrows():
            out.append(f"  {r['zone']:>8s}: {r['n_solution']:>10,}  "
                       f"({100 * r['n_solution'] / total:5.1f}%)")
    return "\n".join(out)




# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
@dataclass
class ReadSpec:
    """How to pull a flat point table out of a CGNS file."""


    field_map: dict = field(default_factory=lambda: dict(DEFAULT_FIELD_MAP))
    #: "zone" -> part type is the zone name; "field" -> read it from a DataArray
    type_source: str = "zone"
    #: optional {zone_name_substring: label} used when type_source == "zone"
    zone_type_map: dict = field(default_factory=dict)
    #: only read these zones (substring match); empty = all
    zones: list = field(default_factory=list)
    #: which FlowSolution to take when a zone has several (substring, or "last")
    solution: str = ""
    required: tuple = ("x", "y", "z", "temperature")
    #: store as float32 - halves memory on multi-million-point meshes
    float32: bool = True
    #: decimate each case to this many points on read (0 = keep all). A memory
    #: guard: ten 5-million-point cases will not fit in RAM at once.
    points_per_case: int = 0
    #: {output column: substring or [substrings]} - every matching DataArray
    #: is summed
    sum_fields: dict = field(default_factory=lambda: dict(DEFAULT_SUM_FIELDS))
    #: value for a summed column in a zone with no matching arrays. 0.0 means
    #: "no heat source here" (PCB without sources, Air, housing), which keeps
    #: heat_gen defined in every zone so it can be a model input. None leaves
    #: the column absent there (NaN after zones are stacked).
    sum_fill: float | None = 0.0
    #: derived columns to compute where the inputs exist: k_eq, alpha
    derived: tuple = ("k_eq", "alpha")




def read_cgns(path: str, spec: ReadSpec | None = None) -> pd.DataFrame:
    """Flatten every solution point of every zone into one DataFrame."""
    _require_h5py()
    spec = spec or ReadSpec()
    frames: list[pd.DataFrame] = []


    with h5py.File(path, "r") as f:
        zones = find(f, "Zone_t")
        if not zones:
            raise ValueError("no Zone_t nodes found - run `cgnsml inspect`")


        for zpath, znode in zones:
            zname = zpath.split("/")[-1]
            if spec.zones and not any(z.lower() in zname.lower() for z in spec.zones):
                continue


            sols = children_by_label(znode, "FlowSolution_t")
            if not sols:
                print(f"  ! zone {zname}: no FlowSolution, skipped")
                continue
            if spec.solution == "last":
                sname = sorted(sols)[-1]
            elif spec.solution:
                hits = [k for k in sols if spec.solution.lower() in k.lower()]
                if not hits:
                    print(f"  ! zone {zname}: no solution matching "
                          f"{spec.solution!r}, skipped")
                    continue
                sname = sorted(hits)[0]
            else:
                sname = sorted(sols)[0]
            snode = sols[sname]


            # grid location decides where coordinates come from
            loc = "Vertex"
            for g in children_by_label(snode, "GridLocation_t").values():
                loc = node_text(g) or loc


            sol_arrays = children_by_label(snode, "DataArray_t")
            lens = {k: payload(v).shape[0] for k, v in sol_arrays.items()
                    if payload(v) is not None}
            if not lens:
                print(f"  ! zone {zname}: solution {sname} has no arrays, skipped")
                continue
            n = max(set(lens.values()), key=list(lens.values()).count)


            grid_arrays: dict = {}
            for g in children_by_label(znode, "GridCoordinates_t").values():
                grid_arrays.update(children_by_label(g, "DataArray_t"))


            notes: list[str] = []
            if loc.lower().startswith("cellcent"):
                coord_map, coord_pool = COORDS_CELL, sol_arrays
            else:
                coord_map, coord_pool = COORDS_VERTEX, grid_arrays


            cols: dict[str, np.ndarray] = {}
            for col, cands in coord_map.items():
                node = _match(cands, coord_pool)
                if node is not None:
                    cols[col] = _read_array(node)


            missing_xyz = [c for c in "xyz" if c not in cols]
            if missing_xyz and loc.lower().startswith("cellcent"):
                # cell-centred solution with no centroids: coordinates would have
                # to be averaged from the connectivity, which we do not guess at
                raise ValueError(
                    f"zone {zname}: solution is CellCenter but no Centroid"
                    f"{missing_xyz} arrays are present. Re-export from the solver "
                    f"with cell centroids, or export a vertex-centred solution.")
            if missing_xyz:
                print(f"  ! zone {zname}: missing coordinates {missing_xyz}, skipped")
                continue


            for col, cands in spec.field_map.items():
                if col == "ptype" and spec.type_source != "field":
                    continue
                node = _match(cands, sol_arrays, fuzzy=False, note=notes)
                if node is None:
                    continue
                arr = _read_array(node)
                if arr is not None:
                    cols[col] = arr


            # summed fields: one DataArray per heat source -> total dissipation
            for out_name, needles in (spec.sum_fields or {}).items():
                if isinstance(needles, str):
                    needles = [needles]
                low = [s.lower() for s in needles]
                hits = sorted(k for k in sol_arrays
                              if any(s in k.lower() for s in low))
                acc = np.zeros(n, dtype=np.float64)
                used, skipped = [], []
                for h in hits:
                    a = _read_array(sol_arrays[h])
                    if a is not None and len(a) == n:
                        acc += a
                        used.append(h)
                    else:
                        skipped.append(h)
                if used:
                    cols[out_name] = acc
                    notes.append(f"{out_name} = sum of {len(used)} arrays: "
                                 f"{', '.join(used)}")
                elif spec.sum_fill is not None:
                    cols[out_name] = np.full(n, float(spec.sum_fill))
                    notes.append(f"{out_name} = {spec.sum_fill:g} (no "
                                 f"{'/'.join(needles)} arrays in this zone)")
                if skipped:
                    notes.append(f"! {out_name}: ignored {skipped} "
                                 f"(not a numeric array of length {n:,})")


            # derived physics: per-cell equivalent conductivity and diffusivity
            if "k_eq" in spec.derived and all(c in cols for c in ("kx", "ky", "kz")):
                # per cell, NOT a single mean over the whole zone - conductivity
                # varies by orders of magnitude between copper and FR4
                cols["k_eq"] = (cols["kx"] + cols["ky"] + cols["kz"]) / 3.0
            if "alpha" in spec.derived and "k_eq" in cols and \
                    all(c in cols for c in ("density", "cp")):
                denom = cols["density"] * cols["cp"]
                cols["alpha"] = np.divide(cols["k_eq"], denom,
                                          out=np.full(n, np.nan),
                                          where=denom > 0)
                notes.append("alpha = k_eq / (density * cp), per cell")


            # hard length check - never truncate mismatched arrays into a table
            bad = {k: len(v) for k, v in cols.items() if len(v) != n}
            if bad:
                raise ValueError(
                    f"zone {zname}: length mismatch against {n:,} solution points: "
                    f"{bad}. This usually means the solution is at cell centres "
                    f"while the coordinates are at vertices.")


            missing = [c for c in spec.required if c not in cols]
            if missing:
                print(f"  ! zone {zname}: missing {missing}, skipped")
                continue


            df = pd.DataFrame(cols)
            if spec.float32:
                for c in df.columns:
                    df[c] = df[c].astype(np.float32)
            df["zone"] = zname


            if spec.type_source == "zone" or "ptype" not in df.columns:
                lab = zname
                for needle, mapped in spec.zone_type_map.items():
                    if needle.lower() in zname.lower():
                        lab = mapped
                        break
                df["ptype"] = lab


            frames.append(df)
            absent = [c for c in spec.field_map if c not in cols and c != "ptype"]
            extra = f"  (no {', '.join(absent)})" if absent else ""
            print(f"  + {zname}/{sname}: {n:,} {loc} points{extra}")
            for msg in notes:
                print(f"      {msg}")


    if not frames:
        raise ValueError("no zone produced usable data")


    out = pd.concat(frames, ignore_index=True)
    all_nan = [c for c in out.columns
               if out[c].dtype.kind == "f" and out[c].isna().all()]
    if all_nan:
        out = out.drop(columns=all_nan)
    partial = [c for c in out.columns
               if out[c].dtype.kind == "f" and out[c].isna().any()]
    if partial:
        print(f"  ! {partial} are missing in some zones (NaN there) - do not use "
              f"them as model inputs unless you filter to zones that have them")
    out["ptype"] = out["ptype"].astype(str).astype("category")
    out["zone"] = out["zone"].astype(str).astype("category")
    return out




# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def expand_paths(path) -> list[str]:
    """Accept a single path, a list, a glob, or a comma-separated string."""
    import glob as _glob


    if isinstance(path, str):
        parts = [p.strip() for p in path.split(",")] if "," in path else [path]
    else:
        parts = list(path)
    out: list[str] = []
    for p in parts:
        hits = sorted(_glob.glob(p)) if any(c in p for c in "*?[") else [p]
        if not hits:
            raise FileNotFoundError(f"no file matches {p!r}")
        out.extend(hits)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq




def case_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]




def _provenance(source: str, spec: "ReadSpec | None" = None) -> str:
    """Identity of the CGNS file a cache was built from."""
    st = os.stat(source)
    return json.dumps({
        "source": os.path.basename(source),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "zones": sorted(spec.zones) if spec and spec.zones else [],
        "solution": spec.solution if spec else "",
        "points_per_case": spec.points_per_case if spec else 0,
        # settings that change which columns the table has: a cache built
        # with different ones must be rebuilt, not reused
        "field_map": spec.field_map if spec else {},
        "sum_fields": spec.sum_fields if spec else {},
        "sum_fill": spec.sum_fill if spec else 0.0,
        "derived": list(spec.derived) if spec else [],
        "zone_type_map": spec.zone_type_map if spec else {},
        "type_source": spec.type_source if spec else "zone",
    }, sort_keys=True, default=str)




def save_table(df: pd.DataFrame, path: str, source: str | None = None,
               spec: "ReadSpec | None" = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    num = df.select_dtypes(include=[np.number])
    codes, levels = {}, {}
    for c in df.columns:
        if c in num.columns:
            continue
        vals = df[c].astype(str).values
        uniq, idx = np.unique(vals, return_inverse=True)
        codes[f"__cat__{c}"] = idx.astype(np.int32)
        levels[c] = uniq.tolist()
    np.savez_compressed(
        path,
        __columns__=np.array(list(num.columns)),
        __levels__=np.array(json.dumps(levels)),
        __source__=np.array(_provenance(source, spec) if source else ""),
        **{c: num[c].to_numpy() for c in num.columns},
        **codes,
    )




def load_table(path: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    levels = json.loads(str(z["__levels__"]))
    df = pd.DataFrame({c: z[c] for c in z["__columns__"].tolist()})
    for c, lv in levels.items():
        df[c] = pd.Categorical.from_codes(z[f"__cat__{c}"], categories=lv)
    return df




def load_any(path: str, spec: ReadSpec | None = None, cache: str | None = None) -> pd.DataFrame:
    """Load from cache if it exists, otherwise parse the CGNS file and cache it."""
    if path.endswith(".npz"):
        return load_table(path)
    if cache and os.path.exists(cache):
        # A cache built from a different CGNS file (or a different zone filter)
        # must never be reused silently - that produces a confident answer about
        # the wrong data. Rebuild instead.
        want = _provenance(path, spec)
        try:
            have = str(np.load(cache, allow_pickle=False)["__source__"])
        except (KeyError, OSError, ValueError):
            have = ""
        if have == want:
            print(f"[cache] {cache}")
            return load_table(cache)
        old = json.loads(have) if have else {}
        new = json.loads(want)
        changed = sorted(k for k in new if old.get(k) != new[k])
        print(f"[cache] stale - built from {old.get('source', 'an unknown file')}"
              f" (changed: {', '.join(changed) or 'unknown'})"
              f", rebuilding for {os.path.basename(path)}")
    print(f"[read] {path}")
    df = read_cgns(path, spec)
    if cache:
        save_table(df, cache, source=path, spec=spec)
        print(f"[cache] wrote {cache} ({os.path.getsize(cache) / 1e6:.0f} MB)")
    return df




# --------------------------------------------------------------------------- #
# multiple cases
# --------------------------------------------------------------------------- #
def load_cases(paths, spec: ReadSpec | None = None, cache_dir: str | None = None,
               conditions: dict | None = None, target: str = "temperature",
               seed: int = 0) -> pd.DataFrame:
    """Load several CGNS files into one table with a ``case`` column.


    Each file is cached separately, so adding an eleventh case does not force a
    re-read of the first ten. ``conditions`` maps a case name to the scalars that
    describe that simulation (power, ambient temperature, ...); they become
    constant columns and are what a DeepONet branch network consumes.
    """
    spec = spec or ReadSpec()
    files = expand_paths(paths)
    conditions = conditions or {}
    frames = []


    for path in files:
        case = case_name(path)
        cache = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache = os.path.join(cache_dir, f"{case}.npz")
        print(f"[case] {case}")
        df = load_any(path, spec, cache)


        if spec.points_per_case and len(df) > spec.points_per_case:
            from .sampling import SampleSpec, sample as _sample
            df, _ = _sample(df, SampleSpec(n=spec.points_per_case, mode="power",
                                           alpha=0.5, seed=seed), target)
            print(f"    decimated to {len(df):,} points")


        df["case"] = case
        cond = _resolve_conditions(case, conditions)
        for k, v in cond.items():
            df[k] = np.float32(v) if spec.float32 else float(v)
        if cond:
            print(f"    conditions: {cond}")
        elif conditions:
            print(f"    ! no conditions matched case {case!r} - it will look "
                  f"identical to every other case to a branch network")
        frames.append(df)


    out = pd.concat(frames, ignore_index=True)
    out["case"] = out["case"].astype("category")
    out["ptype"] = out["ptype"].astype(str).astype("category")


    missing = [c for c in out.columns
               if out[c].dtype.kind == "f" and out[c].isna().any()]
    if missing:
        print(f"  ! {missing} are absent from some cases - a condition column "
              f"that is not defined for every case cannot be a model input")
    print(f"[cases] {len(files)} cases, {len(out):,} points total")
    for c, n in out["case"].value_counts().sort_index().items():
        print(f"    {c}: {n:,}")
    return out




def _resolve_conditions(case: str, conditions: dict) -> dict:
    """Exact case-name key first, then substring keys."""
    if case in conditions:
        return dict(conditions[case])
    for key, vals in conditions.items():
        if key.lower() in case.lower():
            return dict(vals)
    return {}


