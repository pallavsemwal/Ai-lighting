"""Model registry.

Any object with this interface is usable by the CLI:

    class MyModel:
        def __init__(self, **params): ...
        def fit(self, X, y, sample_weight=None, eval_set=None): ...
        def predict(self, X) -> np.ndarray: ...
        def save(self, path): ...          # optional
        @classmethod
        def load(cls, path): ...           # optional

Refer to it either by registered name (``--model hgb``) or by import path
(``--model my_package.models:PointTransformer``). Nothing else in the pipeline
knows what the model is.
"""
from __future__ import annotations

import importlib
import pickle
from typing import Callable

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def get_model(spec: str, **params):
    """``spec`` is a registry key or ``module.path:ClassName``."""
    if ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
    else:
        _load_builtins()
        # print('REGISTRY:',REGISTRY)
        if spec not in REGISTRY:
            raise KeyError(f"unknown model {spec!r}; known: {sorted(REGISTRY)}")
        cls = REGISTRY[spec]
    return cls(**params)


def available() -> list[str]:
    _load_builtins()
    return sorted(REGISTRY)


def _load_builtins():
    from ....models import sklearn_models  # noqa: F401
    try:
        from . import torch_models  # noqa: F401
        from ....models import pointnet_thermal  # noqa: F401
        from . import point_models  # noqa: F401  # [point_models]
        from . import transolver
        print('Models imported')
    except ImportError as e:
        print('error importing models', e)
        pass


def save_model(model, path: str) -> None:
    if hasattr(model, "save"):
        model.save(path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(model, fh)


def load_model(path: str, cls=None):
    if cls is not None and hasattr(cls, "load"):
        return cls.load(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)