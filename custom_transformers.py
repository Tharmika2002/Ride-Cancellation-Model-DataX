# custom_transformers.py
"""
Safe default implementations for custom transformers / functions referenced by a pickled pipeline.

- Classes inherit from sklearn's BaseEstimator + TransformerMixin
- transform(...) is identity (passes data through) until you replace TODOs with your real logic
- Also registers class/function aliases under __main__ to satisfy pickles saved from notebooks

Replace TODO blocks with your actual feature engineering code when ready.
"""

from __future__ import annotations
import sys
import types
from typing import Iterable, Optional, List, Union, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------- small utils ----------
def _as_dataframe(X: Union[pd.DataFrame, np.ndarray, list], columns: Optional[List[str]] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    df = pd.DataFrame(X)
    if columns and len(columns) == df.shape[1]:
        df.columns = columns
    return df


# ---------- identity-ish helper functions (for FunctionTransformer pickles) ----------
def identity_fn(X: Any, *args, **kwargs):
    """Return input unchanged (placeholder for FunctionTransformer target)."""
    return X

def preprocess(X: Any):
    """Placeholder preprocess function; replace with your real preprocessing."""
    return X

def build_features(X: Any):
    """Placeholder feature-building function; replace with your real logic."""
    return X


# ---------- common transformer shells you can customize ----------
class MyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Drop-in, customizable feature engineer.
    Replace the TODOs with your actual feature code used at training time.
    """
    def __init__(self, cols_to_create: Optional[List[str]] = None):
        self.cols_to_create = cols_to_create or []

    def fit(self, X, y=None):
        X = _as_dataframe(X)
        # TODO: learn any statistics you used (means, encoders, etc.)
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        # TODO: add/modify columns exactly as you did in training
        # Example (commented):
        # if "hour_of_day" not in X and "booking_time" in X:
        #     X["hour_of_day"] = pd.to_datetime(X["booking_time"]).dt.hour
        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Another common name people use in notebooks."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        # TODO: your feature construction
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[Iterable[str]] = None, errors: str = "ignore"):
        self.columns = list(columns) if columns is not None else []
        self.errors = errors  # "ignore" or "raise"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        drop = [c for c in self.columns if c in X.columns]
        if self.errors == "raise":
            missing = set(self.columns) - set(X.columns)
            if missing:
                raise KeyError(f"Missing columns to drop: {missing}")
        return X.drop(columns=drop, errors="ignore")


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts simple date/time features from a timestamp column.
    Configure with the column name you used in training.
    """
    def __init__(self, ts_col: str = "booking_time"):
        self.ts_col = ts_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        if self.ts_col in X.columns:
            s = pd.to_datetime(X[self.ts_col], errors="coerce")
            X["hour_of_day"] = s.dt.hour
            X["day_of_week"] = s.dt.dayofweek
            X["is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype(int)
            # Optional band:
            def _band(h):
                if pd.isna(h): return np.nan
                h = int(h)
                if 5 <= h <= 11: return "Morning"
                if 12 <= h <= 16: return "Afternoon"
                if 17 <= h <= 21: return "Evening"
                return "Night"
            X["time_band"] = s.dt.hour.map(_band)
        return X


class CategoryFixer(BaseEstimator, TransformerMixin):
    """
    Ensures key categorical columns exist & are strings (safe defaults).
    """
    def __init__(self, cat_cols: Optional[List[str]] = None):
        self.cat_cols = cat_cols or ["Pickup Location", "Drop Location", "vehicle_type", "payment_method", "time_band"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        for c in self.cat_cols:
            if c not in X.columns:
                X[c] = ""
            X[c] = X[c].astype(str)
        return X


class RateEncoder(BaseEstimator, TransformerMixin):
    """
    Placeholder target-rate/lookup encoder.
    If in training you merged precomputed rates, replicate that here.
    """
    def __init__(self, default_pickup_rate: float = 0.0, default_drop_rate: float = 0.0):
        self.default_pickup_rate = default_pickup_rate
        self.default_drop_rate = default_drop_rate
        self.pickup_map_ = {}
        self.drop_map_ = {}

    def fit(self, X, y=None):
        # TODO: if you learned maps from training data, compute & store them here.
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        # If maps are empty, just ensure columns exist (identity behavior).
        if "pickup_cancel_rate" not in X.columns:
            X["pickup_cancel_rate"] = self.default_pickup_rate
        if "drop_cancel_rate" not in X.columns:
            X["drop_cancel_rate"] = self.default_drop_rate
        return X


class PairFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Placeholder encoder for pickup-drop pair frequency.
    """
    def __init__(self, default_freq: float = 0.0):
        self.default_freq = default_freq
        self.freq_map_ = {}

    def fit(self, X, y=None):
        # TODO: learn and store pair frequencies if you did so in training
        return self

    def transform(self, X):
        X = _as_dataframe(X)
        if "pickup_drop_pair_freq" not in X.columns:
            X["pickup_drop_pair_freq"] = self.default_freq
        return X


# ---------- Optional: expose names under __main__ to satisfy notebook-saved pickles ----------
def _register_aliases_in_main(names: Iterable[str]):
    main = sys.modules.get("__main__")
    if main is None:
        main = types.ModuleType("__main__")
        sys.modules["__main__"] = main
    g = globals()
    for name in names:
        if hasattr(main, name):
            continue
        if name in g:
            setattr(main, name, g[name])

# Common names people often used during training in notebooks:
_alias_names = [
    "MyFeatureEngineer",
    "FeatureBuilder",
    "ColumnDropper",
    "DateTimeFeatures",
    "CategoryFixer",
    "RateEncoder",
    "PairFrequencyEncoder",
    # function targets
    "identity_fn",
    "preprocess",
    "build_features",
]
_register_aliases_in_main(_alias_names)
