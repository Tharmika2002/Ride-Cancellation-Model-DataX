import os
import sys
import json
from datetime import datetime
import types
from typing import List, Tuple

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Version banner & pre-imports ----------
import sklearn, numpy, pandas  # noqa: F401

st.caption(
    f"Python {sys.version.split()[0]} | "
    f"sklearn {sklearn.__version__} | "
    f"numpy {numpy.__version__} | "
    f"pandas {pandas.__version__}"
)

# Common pickled classes from popular libs
try:
    import imblearn.pipeline  # registers imblearn.pipeline.Pipeline
except Exception:
    pass
try:
    import xgboost.sklearn  # registers xgboost.sklearn.XGBClassifier
except Exception:
    pass

# Try to import user custom transformers if they exist
try:
    import custom_transformers  # noqa: F401
except Exception:
    pass
# ---------------------------------------------------


# ---------- Paths ----------
MODEL_PATH = "final_model_pipeline.pkl"
LOOKUP_CANDIDATES = ["lookup_stats.json", "models/lookup_stats.json"]


# ---------- Helpers ----------
def _first_existing(paths: List[str]):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _list_pickle_globals(path: str) -> List[Tuple[str, str]]:
    """Return (module, class) pairs referenced by the pickle (for debugging)."""
    import pickletools
    with open(path, "rb") as f:
        data = f.read()
    out = set()
    for op, arg, _ in pickletools.genops(data):
        if op.name == "GLOBAL":
            try:
                mod, name = arg.split(" ")
                out.add((mod, name))
            except Exception:
                pass
    return sorted(out)


def _ensure_module(name: str):
    """Ensure a module exists in sys.modules (create if missing)."""
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _auto_stub_missing_classes(globs: List[Tuple[str, str]]) -> List[str]:
    """
    For any classes under __main__ or custom_transformers that are missing,
    create a no-op sklearn-style transformer stub so unpickling can proceed.
    """
    from sklearn.base import BaseEstimator, TransformerMixin

    created = []
    for mod, cls in globs:
        if mod not in ("__main__", "custom_transformers"):
            # we only auto-stub likely user-defined modules; others should be real
            continue

        module_obj = _ensure_module(mod)
        if hasattr(module_obj, cls):
            continue  # already present

        # Dynamically create a no-op estimator/transformer
        def _make_stub(name):
            # simple identity transformer
            class _Stub(BaseEstimator, TransformerMixin):
                def __init__(self, *args, **kwargs):
                    # accept anything; store for repr
                    self._args = args
                    self._kwargs = kwargs

                def fit(self, X, y=None):
                    return self

                def transform(self, X):
                    # identity pass-through
                    return X

                # if used as final estimator
                def predict(self, X):
                    raise RuntimeError(
                        f"Auto-stub '{name}' cannot predict. Replace with real implementation."
                    )

            _Stub.__name__ = name
            return _Stub

        setattr(module_obj, cls, _make_stub(cls))
        created.append(f"{mod}.{cls}")

    return created


# ---------- Loaders ----------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please add final_model_pipeline.pkl to the repo root.")
        st.stop()

    # Inspect required classes before load
    globs = _list_pickle_globals(MODEL_PATH)

    # Try to auto-stub any missing custom classes (under __main__ or custom_transformers)
    created = _auto_stub_missing_classes(globs)
    if created:
        with st.expander("⚠️ Auto-created NO-OP stubs for these missing classes"):
            for name in created:
                st.write(name)
        st.info(
            "Replace these stubs with your real code later: create custom_transformers.py and "
            "define the exact classes listed above, then remove the auto-stub logic."
        )

    # Attempt to load
    try:
        return joblib.load(MODEL_PATH)

    except AttributeError:
        # If still failing, show everything the pickle expects
        st.error("❌ Could not unpickle the model: a required class/module isn't importable.")
        with st.expander("🔎 Debug: classes referenced by the model (module.Class)"):
            for mod, name in globs:
                st.write(f"{mod}.{name}")
        st.info(
            "Fix:\n"
            "• If you see entries like __main__.MyFeatureEngineer or custom_transformers.MyFeatureEngineer:\n"
            "  - Create custom_transformers.py in the repo with that class implementation\n"
            "  - Keep `import custom_transformers` near the top of this file\n"
            "• If you see imblearn.pipeline.Pipeline or xgboost.sklearn.XGBClassifier, those are already pre-imported above.\n"
            "• Ensure the ML stack versions match training (pin in requirements.txt)."
        )
        st.stop()


@st.cache_resource
def load_lookup():
    p = _first_existing(LOOKUP_CANDIDATES)
    if not p:
        st.error("❌ Lookup stats not found. Add lookup_stats.json at repo root or models/lookup_stats.json.")
        st.stop()
    with open(p, "r") as f:
        return json.load(f)


# ---------- App init ----------
model = load_pipeline()
stats = load_lookup()


# ---------- Helpers ----------
def get_classes(m):
    # direct estimator (has classes_)
    if hasattr(m, "classes_") and m.classes_ is not None:
        return list(m.classes_)
    # try named_steps['classifier'] (sklearn Pipeline style)
    steps = getattr(m, "named_steps", None)
    if isinstance(steps, dict):
        clf = steps.get("classifier")
        if clf is not None and hasattr(clf, "classes_"):
            return list(clf.classes_)
    return None


classes_ = get_classes(model)


# ---------- UI ----------
st.title("🚖 Ride Booking Outcome Predictor (DataX)")
st.header("Enter Booking Details")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Booking Date")
    time = st.time_input("Booking Time")
    pickup = st.text_input("Pickup Area (e.g., Area-1)")
    customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, step=0.1)
    vehicle_type = st.selectbox("Vehicle Type", ["Auto", "Mini", "Sedan", "SUV", "Bike"])
with col2:
    drop = st.text_input("Drop Area (e.g., Area-10)")
    driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0, step=0.1)
    payment_method = st.selectbox("Payment Method", ["Cash", "Card", "Wallet", "UPI"])

# Time features
dt = datetime.combine(date, time)
hour = dt.hour
day_of_week = dt.weekday()
is_weekend = int(day_of_week in (5, 6))


def assign_time_band(h):
    if 5 <= h <= 11:
        return "Morning"
    if 12 <= h <= 16:
        return "Afternoon"
    if 17 <= h <= 21:
        return "Evening"
    return "Night"


time_band = assign_time_band(hour)


# Lookup engineered features (computed on train split during training)
def get_lookup_rates(pu, dr, _stats):
    pu_rate = _stats["pickup_cancel_rate"].get(pu, _stats["defaults"]["pickup_cancel_rate"])
    dr_rate = _stats["drop_cancel_rate"].get(dr, _stats["defaults"]["drop_cancel_rate"])
    pair_key = f"{pu}|||{dr}"
    pair_f = _stats["pair_freq"].get(pair_key, _stats["defaults"]["pair_freq"])
    return pu_rate, dr_rate, pair_f


pickup_cancel_rate, drop_cancel_rate, pickup_drop_pair_freq = get_lookup_rates(pickup, drop, stats)

# Build single-row input matching training schema EXACTLY
X_input = pd.DataFrame([{
    # numeric
    "hour_of_day": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "pickup_cancel_rate": pickup_cancel_rate,
    "drop_cancel_rate": drop_cancel_rate,
    "pickup_drop_pair_freq": pickup_drop_pair_freq,
    "customer_rating": float(customer_rating),
    "driver_rating": float(driver_rating),
    # categorical
    "time_band": time_band,
    "Pickup Location": pickup,
    "Drop Location": drop,
    "vehicle_type": vehicle_type,
    "payment_method": payment_method,
}])


st.subheader("Prediction")
if st.button("Predict Booking Status"):
    try:
        pred = model.predict(X_input)[0]
        st.success(f"✅ Predicted Booking Status: **{pred}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            labels = classes_ if classes_ is not None and len(classes_) == len(probs) else [str(i) for i in range(len(probs))]
            fig, ax = plt.subplots()
            ax.bar(labels, probs)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Prediction Confidence")
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure the input schema matches the training pipeline and any auto-stubbed classes are replaced by real implementations.")
