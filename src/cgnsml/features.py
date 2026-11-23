"""Turn the point table into an (X, y) matrix pair.


Kept deliberately simple: everything a model needs is a float matrix, so any
estimator with fit/predict works without knowing about CGNS or point clouds.
"""
from __future__ import annotations


from dataclasses import dataclass, field


import numpy as np
import pandas as pd




@dataclass
class FeatureSpec:
    inputs: list = field(default_factory=lambda: ["x", "y", "z", 'alpha', 'k_eq', 'density', 'cp', 'zone','ptype' , 'heat_gen'])
    targets: list = field(default_factory=lambda: ["temperature"])
    use_type: bool = True
    #: k-nearest-neighbour geometry features (local density, mean distance,
    #: offset to centroid). Heat conduction is local, so these help a lot for
    #: models that have no notion of neighbourhood (trees, plain MLPs).
    knn: int = 0
    scale: str = "standard"   # standard | minmax | none




@dataclass
class Encoder:
    spec: FeatureSpec
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    y_mu: np.ndarray | None = None
    y_sd: np.ndarray | None = None
    type_levels: list = field(default_factory=list)
    columns: list = field(default_factory=list)


    # -- fit -------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> "Encoder":
        if self.spec.use_type and "ptype" in df.columns:
            self.type_levels = sorted(df["ptype"].astype(str).unique().tolist())
        X = self._raw(df)
        if self.spec.scale == "standard":
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        elif self.spec.scale == "minmax":
            self.mu, self.sd = X.min(0), (X.max(0) - X.min(0)) + 1e-12
        else:
            self.mu, self.sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        # never rescale the one-hot block
        n_oh = len(self.type_levels)
        if n_oh:
            self.mu[-n_oh:] = 0.0
            self.sd[-n_oh:] = 1.0
        y = df[self.spec.targets].to_numpy(float)
        self.y_mu, self.y_sd = y.mean(0), y.std(0) + 1e-12
        return self


    # -- transform -------------------------------------------------------- #
    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        parts, cols = [], []
        base = df[self.spec.inputs].to_numpy(float)
        parts.append(base)
        cols += list(self.spec.inputs)


        if self.spec.knn > 0:
            f, names = knn_features(df, k=self.spec.knn)
            parts.append(f)
            cols += names


        if self.type_levels:
            vals = df["ptype"].astype(str).to_numpy()
            oh = np.zeros((len(df), len(self.type_levels)))
            for j, lv in enumerate(self.type_levels):
                oh[:, j] = (vals == lv)
            parts.append(oh)
            cols += [f"type={lv}" for lv in self.type_levels]


        self.columns = cols
        return np.column_stack(parts)


    def transform(self, df: pd.DataFrame):
        X = (self._raw(df) - self.mu) / self.sd
        y = df[self.spec.targets].to_numpy(float) if \
            all(t in df.columns for t in self.spec.targets) else None
        return X, y


    def scale_y(self, y):
        return (y - self.y_mu) / self.y_sd


    def unscale_y(self, y):
        return y * self.y_sd + self.y_mu




def knn_features(df: pd.DataFrame, k: int = 8):
    """Local geometry descriptors from a KD-tree over the point cloud."""
    from scipy.spatial import cKDTree


    xyz = df[["x", "y", "z"]].to_numpy(float)
    tree = cKDTree(xyz)
    d, idx = tree.query(xyz, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    centroid = xyz[idx].mean(1)
    feats = np.column_stack([
        d.mean(1), d.max(1), d.std(1),
        np.linalg.norm(centroid - xyz, axis=1),
        centroid - xyz,
    ])
    names = ["knn_dmean", "knn_dmax", "knn_dstd", "knn_offset",
             "knn_dx", "knn_dy", "knn_dz"]
    return feats, names

