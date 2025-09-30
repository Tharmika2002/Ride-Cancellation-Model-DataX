import os
import sys
import json
import types
import hashlib
from datetime import datetime
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
    """Return (module, symbol) pairs referenced by the pickle (for debugging)."""
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

def _auto_stub_functions_any_module(globs: List[Tuple[str, str]]) -> List[str]:
    """Create identity function stubs for ANY missing module.symbol."""
    created = []
    for mod, sym in globs:
        module_obj = _ensure_module(mod)
        if hasattr(module_obj, sym):
            continue
        # identity function
        def _fn(*args, **kwargs):
            if args:
                return args[0]
            return None
        _fn.__name__ = sym  # works even for '<lambda>'
        setattr(module_obj, sym, _fn)
        created.append(f"{mod}.{sym} (function)")
    return created

def _auto_stub_classes_any_module(globs: List[Tuple[str, str]]) -> List[str]:
    """Create no-op sklearn-style transformer class stubs for ANY missing module.symbol."""
    from sklearn.base import BaseEstimator, TransformerMixin
    created = []
    for mod, cls in globs:
        module_obj = _ensure_module(mod)
        if hasattr(module_obj, cls):
            continue

        def _make_stub(name):
            class _Stub(BaseEstimator, TransformerMixin):
                def __init__(self, *args, **kwargs): pass
                def fit(self, X, y=None): return self
                def transform(self, X): return X
                def predict(self, X):
                    raise RuntimeError(
                        f"Auto-stub '{name}' cannot predict. Replace with the real implementation."
                    )
            _Stub.__name__ = name
            return _Stub

        setattr(module_obj, cls, _make_stub(cls))
        created.append(f"{mod}.{cls} (class)")
    return created

def _file_info(path: str) -> str:
    size = os.path.getsize(path) if os.path.exists(path) else -1
    sha = ""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()[:16]
    except Exception:
        pass
    return f"{size} bytes | sha256:{sha}"

# ---------- Loaders ----------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please add final_model_pipeline.pkl to the repo root.")
        st.stop()

    st.write(f"**Model file:** `{MODEL_PATH}`  ({_file_info(MODEL_PATH)})")

    globs = _list_pickle_globals(MODEL_PATH)

    # Always show what the pickle expects
    with st.expander("🔎 Model references (module.symbol) found in pickle"):
        st.code("\n".join(f"{m}.{n}" for m, n in globs) or "(none)")

    # Pass 0: try load directly
    try:
        return joblib.load(MODEL_PATH)
    except AttributeError as e:
        st.warning(f"First load attempt failed (AttributeError): {e!r}")

    # Pass 1: add function stubs anywhere missing, then try again
    created_fns = _auto_stub_functions_any_module(globs)
    if created_fns:
        with st.expander("⚠️ Auto-created identity FUNCTIONS (temporary stubs)"):
            for name in created_fns:
                st.write(name)
    try:
        return joblib.load(MODEL_PATH)
    except AttributeError as e:
        st.warning(f"Second load attempt (after function stubs) failed: {e!r}")

    # Pass 2: also add class stubs anywhere missing, then try again
    created_clss = _auto_stub_classes_any_module(globs)
    if created_clss:
        with st.expander("⚠️ Auto-created no-op CLASSES (temporary stubs)"):
            for name in created_clss:
                st.write(name)
    try:
        return joblib.load(MODEL_PATH)
    except AttributeError as e:
        st.error("❌ Could not unpickle the model: a required class/module isn't importable.")
        st.code(repr(e))  # <-- show the exact missing symbol
        st.info(
            "Create **custom_transformers.py** and implement the real functions/classes named above "
            "(matching the module and symbol names), and keep `import custom_transformers` near the top.\n\n"
            "Also verify the model file is correct (not a Git LFS pointer). See file size & sha256 above."
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
    if hasattr(m, "classes_") and m.classes_ is not None:
        return list(m.classes_)
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
        st.info("Ensure input feature names match training and replace any auto-stubbed symbols with real implementations in custom_transformers.py.")
