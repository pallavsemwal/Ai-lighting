"""Baselines. Run these before any deep model - on a fixed geometry they are
often within a few percent of a neural net and train in seconds."""
from __future__ import annotations

import numpy as np

from . import register


class _SkWrap:
    _factory = None

    def __init__(self, **params):
        self.params = params
        self.model = None
        self._multi = False

    def _build(self, n_out: int):
        from sklearn.multioutput import MultiOutputRegressor
        est = self._factory(**self.params)
        if n_out > 1 and not self._supports_multi:
            self._multi = True
            return MultiOutputRegressor(est)
        return est

    _supports_multi = True

    def fit(self, X, y, sample_weight=None, eval_set=None):
        y = np.asarray(y)
        n_out = y.shape[1] if y.ndim > 1 else 1
        self.model = self._build(n_out)
        target = y.ravel() if n_out == 1 else y
        try:
            print(target)
            self.model.fit(X, target, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(X, target)
        return self

    def predict(self, X):
        p = self.model.predict(X)
        return p.reshape(len(X), -1)


@register("rf")
class RandomForest(_SkWrap):
    """Strong, fast, gives feature importances. No extrapolation ability."""

    def __init__(self, n_estimators=300, max_depth=None, n_jobs=-1,
                 min_samples_leaf=2, random_state=0, **kw):
        from sklearn.ensemble import RandomForestRegressor
        self._factory = RandomForestRegressor
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         n_jobs=n_jobs, min_samples_leaf=min_samples_leaf,
                         random_state=random_state, **kw)


@register("hgb")
class HistGradientBoosting(_SkWrap):
    """Best accuracy-per-second baseline. Single output at a time."""

    _supports_multi = False

    def __init__(self, max_iter=500, learning_rate=0.08, max_leaf_nodes=63,
                 l2_regularization=1.0, random_state=0, **kw):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self._factory = HistGradientBoostingRegressor
        super().__init__(max_iter=max_iter, learning_rate=learning_rate,
                         max_leaf_nodes=max_leaf_nodes,
                         l2_regularization=l2_regularization,
                         random_state=random_state, **kw)


@register("knn")
class KNNBaseline(_SkWrap):
    """Nearest-neighbour interpolation. If a model cannot beat this on a random
    split, that split is leaking and the model has learnt nothing."""

    def __init__(self, n_neighbors=8, weights="distance", n_jobs=-1, **kw):
        from sklearn.neighbors import KNeighborsRegressor
        self._factory = KNeighborsRegressor
        super().__init__(n_neighbors=n_neighbors, weights=weights,
                         n_jobs=n_jobs, **kw)


@register("ridge")
class RidgeBaseline(_SkWrap):
    def __init__(self, alpha=1.0, **kw):
        from sklearn.linear_model import Ridge
        self._factory = Ridge
        super().__init__(alpha=alpha, **kw)